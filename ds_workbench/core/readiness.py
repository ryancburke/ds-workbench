"""
The readiness policy: one place that decides whether a set of check results is ready.

Used by preflight.check(), the team Sessions page (team_sessions) and the Workbench readiness
strip, so they can never disagree. Pure logic over skill result dicts — no model calls.

Rules:
  - a mandatory check that never ran → incomplete (never ready)
  - a check that failed ({"error": ...}) → incomplete, with the error; never treated as clean
  - fail closed: a missing or unrecognised verdict value makes the check failed, and a finding
    with an unknown or missing severity counts as a caution
  - critical findings → blocked
  - warnings → caution
  - only when every mandatory check ran and nothing is blocking or cautioned → ready_for_review

Precedence when several apply: blocked > incomplete > caution > ready_for_review. A known
blocker is the most urgent thing to show; missing and failed checks are still listed.

Availability confirmations: a person (never the model) can confirm that a flagged leakage column
is available at prediction time. A confirmation applies only to the exact (column, dataset,
prediction point) it was made for; the confirmed finding then shows as info with who confirmed
it, when and why, and no longer blocks. Confirmations are passed to assess() separately — any
"confirmed" field inside a skill result is ignored.
"""
from __future__ import annotations

from datetime import datetime, timezone

# Checks a modelling workflow must run before anything is ready
MODELLING_CHECKS = ("data_quality", "leakage_detector")
# A causal analysis (no model training) needs the data checked and the effect analysed
CAUSAL_CHECKS = ("data_quality", "causal_inference")
_MODELLING_SKILLS = {"leakage_detector", "feature_engineering", "model_selector"}

STATUSES = ("blocked", "incomplete", "caution", "ready_for_review")   # most to least urgent
STATUS_LABEL = {"blocked": "Blocked", "incomplete": "Incomplete", "caution": "Caution",
                "ready_for_review": "Ready for review"}
CHECK_LABEL = {"data_quality": "Data quality", "leakage_detector": "Leakage",
               "feature_engineering": "Features", "model_selector": "Model",
               "causal_inference": "Causal", "join_auditor": "Join", "mlops_review": "MLOps",
               "experiment_tracker": "Experiment", "data_verification": "Data verification"}


def mandatory_for(results: dict) -> tuple[str, ...]:
    """Mandatory checks for what this work is: causal-only work needs DQ + the causal check;
    everything else is treated as modelling work (DQ + leakage)."""
    ran = {k for k, v in results.items() if v is not None}
    if "causal_inference" in ran and not (ran & _MODELLING_SKILLS):
        return CAUSAL_CHECKS
    return MODELLING_CHECKS


def required_checks(results: dict, declared=None) -> tuple[str, ...]:
    """Mandatory checks for this work.

    Without a declaration: inferred from what ran (mandatory_for, unchanged behaviour).
    With one (e.g. from an investigation plan): the declared checks plus a floor set by what
    actually ran — any modelling skill requires DQ + leakage, a causal analysis requires DQ +
    causal. A declaration can add checks but never remove that floor, so declaring less can't
    make unchecked modelling work ready."""
    if declared is None:
        return mandatory_for(results)
    declared = tuple(dict.fromkeys(declared))
    unknown = [c for c in declared if c not in CHECK_LABEL]
    if not declared or unknown:
        raise ValueError(f"required checks must be a non-empty list of known checks "
                         f"({', '.join(CHECK_LABEL)}); got {list(declared)}")
    ran = {k for k, v in results.items() if v is not None}
    floor = (MODELLING_CHECKS if ran & _MODELLING_SKILLS else ()) + \
            (CAUSAL_CHECKS if "causal_inference" in ran else ())
    return tuple(dict.fromkeys([*declared, *floor]))


def _item(source: str, issue: str, column: str = "—", action: str = "") -> dict:
    return {"source": source, "column": column or "—", "issue": issue[:160], "action": action[:160]}


# Finding severities, by meaning. Anything else — including a missing severity — is treated as a
# caution: an unknown severity is never silently ignored.
_BLOCKING_SEVERITIES = {"critical"}
_CAUTION_SEVERITIES = {"warning", "concern"}
_BENIGN_SEVERITIES = {"info", "suggestion", "positive", "note"}

# Skill-level verdict field → (blocking, caution, clean) values, as each skill returns them.
# A missing or unrecognised value makes the check failed (fail closed), never clean.
VERDICT_RULES = {
    "leakage_detector":   ("verdict",         {"leakage_detected"}, {"suspicious"},    {"clean"}),
    "join_auditor":       ("overall_verdict", {"dangerous"},        {"review_needed"}, {"safe"}),
    "model_selector":     ("data_readiness",  {"insufficient"},     {"needs_prep"},    {"ready"}),
    "causal_inference":   ("verdict",         {"not_identified"},   {"caution"},       {"valid"}),
    "experiment_tracker": ("verdict",         {"red_flags"},        {"needs_work"},    {"promising"}),
    "data_verification":  ("verdict",         {"failed"},           {"verified_with_issues"}, {"verified"}),
}


# ── Availability confirmations (a person's judgment, scoped to one context) ──────

def _norm(text) -> str:
    return " ".join(str(text or "").split()).casefold()


def make_confirmation(column: str, *, dataset: str, prediction_point: str, note: str, user: str,
                      ts: str | None = None) -> dict:
    """A person's record that `column` in `dataset` is available at `prediction_point`.
    Every part is required: a confirmation without its context could be reused where it's wrong."""
    for name, value in (("column", column), ("dataset", dataset), ("prediction_point", prediction_point),
                        ("note", note), ("user", user)):
        if not str(value or "").strip():
            raise ValueError(f"confirming availability needs a {name.replace('_', ' ')}")
    return {"column": str(column).strip(), "dataset": str(dataset).strip(),
            "prediction_point": str(prediction_point).strip(), "note": str(note).strip(),
            "user": str(user).strip(),
            "ts": ts or datetime.now(timezone.utc).isoformat(timespec="seconds")}


def confirmation_for(confirmations, column, dataset, prediction_point) -> dict | None:
    """The confirmation for exactly this (column, dataset, prediction point), if any. A missing
    dataset or prediction point never matches."""
    if not (_norm(column) and _norm(dataset) and _norm(prediction_point)):
        return None
    key = (_norm(column), _norm(dataset), _norm(prediction_point))
    for c in reversed(list(confirmations or [])):      # latest wins
        if isinstance(c, dict) and c.get("note") and c.get("user") and \
                (_norm(c.get("column")), _norm(c.get("dataset")), _norm(c.get("prediction_point"))) == key:
            return c
    return None


def check_state(skill: str, result: dict | None, confirmations: list | None = None) -> dict:
    """State of one check: missing, failed, blocked, caution or clean, with its reasons.
    Fails closed: an error, an unrecognised verdict or an unknown severity never reads as clean.
    `confirmations` (leakage only) turn matching findings into confirmed info items."""
    if result is None:
        return {"skill": skill, "state": "missing", "blockers": [], "cautions": [], "confirmed": []}
    if not isinstance(result, dict) or result.get("error"):
        err = result.get("error") if isinstance(result, dict) else f"unexpected result: {type(result).__name__}"
        return {"skill": skill, "state": "failed", "error": str(err), "blockers": [], "cautions": [],
                "confirmed": []}

    blockers, cautions, confirmed = [], [], []
    for f in result.get("findings", []) or []:
        if not isinstance(f, dict):
            continue
        issue = f.get("evidence") or f.get("description") or f.get("observation") or f.get("issue_type", "")
        action = f.get("recommendation") or f.get("fix") or ""
        severity = f.get("severity")
        c = (confirmation_for(confirmations, f.get("column"), result.get("dataset"), result.get("prediction_point"))
             if skill == "leakage_detector" and severity not in _BENIGN_SEVERITIES else None)
        if c is not None:
            confirmed.append({**_item(skill, issue, f.get("column"), action), "severity": "info",
                              "was": severity, "confirmed_by": c["user"], "confirmed_at": c["ts"],
                              "note": c["note"], "prediction_point": c["prediction_point"]})
            continue
        if severity in _BLOCKING_SEVERITIES:
            blockers.append(_item(skill, issue, f.get("column"), action))
        elif severity in _CAUTION_SEVERITIES:
            cautions.append(_item(skill, issue, f.get("column"), action))
        elif severity not in _BENIGN_SEVERITIES:
            cautions.append(_item(skill, f"unrecognised severity {severity!r}: {issue}", f.get("column"), action))

    error = None
    if skill in VERDICT_RULES:
        key, blocking, cautioning, clean = VERDICT_RULES[skill]
        value = result.get(key)
        label = f"{key.replace('_', ' ')}: {str(value).replace('_', ' ')}"
        # A verdict stands for its findings: once every finding behind it is confirmed by a
        # person, the verdict alone doesn't block (or caution) again
        if value in blocking:
            if not (blockers or confirmed):
                blockers.append(_item(skill, label))
        elif value in cautioning:
            if not (blockers or cautions or confirmed):
                cautions.append(_item(skill, label))
        elif value not in clean:
            error = f"unrecognised verdict: {key}={value!r}"

    if error:
        # Keep any findings so a known blocker still shows, but the check is never clean
        return {"skill": skill, "state": "failed", "error": error, "blockers": blockers, "cautions": cautions,
                "confirmed": confirmed}
    state = "blocked" if blockers else "caution" if cautions else "clean"
    return {"skill": skill, "state": state, "blockers": blockers, "cautions": cautions, "confirmed": confirmed}


def assess(results: dict, mandatory: tuple[str, ...] | None = None, confirmations: list | None = None,
           declared: list | tuple | None = None) -> dict:
    """Readiness of a set of results ({skill: result dict or None}).

    `mandatory` fixes the required checks exactly (preflight's training check uses it);
    otherwise they come from required_checks(results, declared). `confirmations` are people's
    availability confirmations (make_confirmation); they never come from a skill result."""
    mandatory = tuple(mandatory) if mandatory is not None else required_checks(results, declared)
    checks = {s: check_state(s, results.get(s), confirmations) for s in dict.fromkeys([*mandatory, *results])}

    missing = [s for s in mandatory if checks[s]["state"] == "missing"]
    failed = [{"skill": s, "error": c["error"]} for s, c in checks.items() if c["state"] == "failed"]
    blockers = [b for c in checks.values() for b in c["blockers"]]
    cautions = [w for c in checks.values() for w in c["cautions"]]
    confirmed = [x for c in checks.values() for x in c["confirmed"]]

    if blockers:
        status = "blocked"
    elif missing or failed:
        status = "incomplete"
    elif cautions:
        status = "caution"
    else:
        status = "ready_for_review"

    return {
        "status": status, "label": STATUS_LABEL[status], "mandatory": list(mandatory),
        "checks": checks, "missing": missing, "failed": failed,
        "blockers": blockers, "cautions": cautions, "confirmed": confirmed,
        "summary": summary(status, missing, failed, blockers, cautions),
    }


def summary(status: str, missing: list, failed: list, blockers: list, cautions: list) -> str:
    gaps = []
    if missing:
        gaps.append("not run: " + ", ".join(CHECK_LABEL.get(s, s) for s in missing))
    if failed:
        gaps.append("failed: " + ", ".join(CHECK_LABEL.get(f["skill"], f["skill"]) for f in failed))
    gap_text = f" Also {'; '.join(gaps)}." if gaps else ""
    if status == "blocked":
        n = len(blockers)
        return f"{n} blocking issue{'s' if n != 1 else ''} must be resolved first.{gap_text}"
    if status == "incomplete":
        return f"Not ready: required checks are incomplete ({'; '.join(gaps)})."
    if status == "caution":
        n = len(cautions)
        return f"All required checks ran; {n} issue{'s' if n != 1 else ''} to review before proceeding."
    return "All required checks ran with nothing blocking. Ready for a reviewer to confirm."


assess_readiness = assess  # exported name

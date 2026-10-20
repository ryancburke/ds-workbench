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
"""
from __future__ import annotations

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
               "experiment_tracker": "Experiment"}


def mandatory_for(results: dict) -> tuple[str, ...]:
    """Mandatory checks for what this work is: causal-only work needs DQ + the causal check;
    everything else is treated as modelling work (DQ + leakage)."""
    ran = {k for k, v in results.items() if v is not None}
    if "causal_inference" in ran and not (ran & _MODELLING_SKILLS):
        return CAUSAL_CHECKS
    return MODELLING_CHECKS


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
}


def check_state(skill: str, result: dict | None) -> dict:
    """State of one check: missing, failed, blocked, caution or clean, with its reasons.
    Fails closed: an error, an unrecognised verdict or an unknown severity never reads as clean."""
    if result is None:
        return {"skill": skill, "state": "missing", "blockers": [], "cautions": []}
    if not isinstance(result, dict) or result.get("error"):
        err = result.get("error") if isinstance(result, dict) else f"unexpected result: {type(result).__name__}"
        return {"skill": skill, "state": "failed", "error": str(err), "blockers": [], "cautions": []}

    blockers, cautions = [], []
    for f in result.get("findings", []) or []:
        if not isinstance(f, dict):
            continue
        issue = f.get("evidence") or f.get("description") or f.get("observation") or f.get("issue_type", "")
        action = f.get("recommendation") or f.get("fix") or ""
        severity = f.get("severity")
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
        if value in blocking:
            if not blockers:
                blockers.append(_item(skill, label))
        elif value in cautioning:
            if not (blockers or cautions):
                cautions.append(_item(skill, label))
        elif value not in clean:
            error = f"unrecognised verdict: {key}={value!r}"

    if error:
        # Keep any findings so a known blocker still shows, but the check is never clean
        return {"skill": skill, "state": "failed", "error": error, "blockers": blockers, "cautions": cautions}
    state = "blocked" if blockers else "caution" if cautions else "clean"
    return {"skill": skill, "state": state, "blockers": blockers, "cautions": cautions}


def assess(results: dict, mandatory: tuple[str, ...] | None = None) -> dict:
    """Readiness of a set of results ({skill: result dict or None})."""
    mandatory = tuple(mandatory) if mandatory is not None else mandatory_for(results)
    checks = {s: check_state(s, results.get(s)) for s in dict.fromkeys([*mandatory, *results])}

    missing = [s for s in mandatory if checks[s]["state"] == "missing"]
    failed = [{"skill": s, "error": c["error"]} for s, c in checks.items() if c["state"] == "failed"]
    blockers = [b for c in checks.values() for b in c["blockers"]]
    cautions = [w for c in checks.values() for w in c["cautions"]]

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
        "blockers": blockers, "cautions": cautions, "summary": summary(status, missing, failed, blockers, cautions),
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

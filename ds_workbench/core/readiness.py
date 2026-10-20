"""
The readiness policy: one place that decides whether a set of check results is ready.

Used by preflight.check(), the team Sessions page (team_sessions) and the Workbench readiness
strip, so they can never disagree. Pure logic over skill result dicts — no model calls.

Rules:
  - a mandatory check that never ran → incomplete (never ready)
  - a check that failed ({"error": ...}) → incomplete, with the error; never treated as clean
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


def check_state(skill: str, result: dict | None) -> dict:
    """State of one check: missing, failed, blocked, caution or clean, with its reasons."""
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
        if f.get("severity") == "critical":
            blockers.append(_item(skill, issue, f.get("column"), action))
        elif f.get("severity") in ("warning", "concern"):
            cautions.append(_item(skill, issue, f.get("column"), action))

    # Skill-level verdicts that mean "blocked" or "caution" even without itemised findings
    verdict_rules = {
        "leakage_detector":   ("verdict", {"leakage_detected"}, {"suspicious"}),
        "join_auditor":       ("overall_verdict", {"dangerous"}, {"review_needed"}),
        "model_selector":     ("data_readiness", {"insufficient"}, {"needs_prep"}),
        "causal_inference":   ("verdict", {"not_identified"}, {"caution"}),
        "experiment_tracker": ("verdict", {"red_flags"}, {"needs_work"}),
    }
    if skill in verdict_rules:
        key, blocking, cautioning = verdict_rules[skill]
        value = result.get(key)
        if value in blocking and not blockers:
            blockers.append(_item(skill, f"{key.replace('_', ' ')}: {str(value).replace('_', ' ')}"))
        elif value in cautioning and not (blockers or cautions):
            cautions.append(_item(skill, f"{key.replace('_', ' ')}: {str(value).replace('_', ' ')}"))

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

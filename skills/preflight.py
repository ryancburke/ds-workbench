"""
Pre-flight training readiness check.

Synthesises findings from data_quality and leakage_detector into a single
go / caution / blocked verdict — no API call, instant.

Usage (standalone):
    dq  = data_quality.assess(df, "orders")
    ld  = leakage_detector.check(df, "churned")
    pf  = preflight.check(dq, ld)

Usage (with session — pulls results automatically):
    pf  = preflight.check(session=session)
"""
from __future__ import annotations
from ds_workbench.core.readiness import MODELLING_CHECKS, assess
from skills.result import SkillResult


def check(
    dq_result=None,
    leakage_result=None,
    session=None,
) -> SkillResult:
    """Return a training-readiness verdict by combining DQ and leakage findings.

    At least one of dq_result, leakage_result, or session must be provided.
    When session is given, results are pulled from it automatically unless
    overridden by explicit arguments.
    """
    if session is not None:
        if dq_result is None:
            dq_result = session.get_result("data_quality")
        if leakage_result is None:
            leakage_result = session.get_result("leakage_detector")

    if dq_result is None and leakage_result is None:
        return SkillResult(
            {"error": "Provide at least one of dq_result, leakage_result, or session."},
            skill="preflight",
        )

    # The readiness policy decides; a missing or failed check is never treated as passed
    readiness = assess({"data_quality": dq_result, "leakage_detector": leakage_result},
                       mandatory=MODELLING_CHECKS)
    checks = readiness["checks"]
    dq_score = dq_result.get("overall_score") if isinstance(dq_result, dict) and not dq_result.get("error") else None

    cleared = []
    if checks["data_quality"]["state"] == "clean":
        cleared.append(f"Data quality{f' score {dq_score}/100' if dq_score is not None else ''} — no blocking issues")
    if checks["leakage_detector"]["state"] == "clean":
        cleared.append("No leakage found — feature set is safe to use")

    return SkillResult(
        {
            "verdict":           readiness["status"],
            "blockers":          [{"source": b["source"], "column": b["column"], "reason": b["issue"],
                                   "action": b["action"]} for b in readiness["blockers"]],
            "cautions":          [{"source": c["source"], "column": c["column"], "reason": c["issue"]}
                                  for c in readiness["cautions"]],
            "missing":           readiness["missing"],
            "failed":            readiness["failed"],
            "cleared":           cleared,
            "summary":           readiness["summary"],
            "dq_score":          dq_score,
            "dq_available":      checks["data_quality"]["state"] not in ("missing", "failed"),
            "leakage_available": checks["leakage_detector"]["state"] not in ("missing", "failed"),
        },
        skill="preflight",
    )

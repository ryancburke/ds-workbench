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

    dq_findings  = list(dq_result.get("findings",  []) if dq_result  else [])
    dq_score     = dq_result.get("overall_score") if dq_result else None
    ld_findings  = list(leakage_result.get("findings", []) if leakage_result else [])
    ld_verdict   = leakage_result.get("verdict", "clean") if leakage_result else None

    dq_criticals  = [f for f in dq_findings if f.get("severity") == "critical"]
    dq_warnings   = [f for f in dq_findings if f.get("severity") == "warning"]
    ld_criticals  = [f for f in ld_findings if f.get("severity") == "critical"]
    ld_warnings   = [f for f in ld_findings if f.get("severity") == "warning"]

    # ── Build blockers and cautions ───────────────────────────────────────────

    blockers = []
    for f in ld_criticals:
        blockers.append({
            "source":  "leakage",
            "column":  f.get("column", "?"),
            "reason":  f.get("evidence", f.get("description", ""))[:120],
            "action":  f.get("recommendation", "")[:140],
        })
    for f in dq_criticals:
        blockers.append({
            "source":  "data_quality",
            "column":  f.get("column", "?"),
            "reason":  f.get("description", "")[:120],
            "action":  f.get("recommendation", "")[:140],
        })

    cautions = []
    for f in ld_warnings:
        cautions.append({
            "source":  "leakage",
            "column":  f.get("column", "?"),
            "reason":  f.get("description", "")[:120],
        })
    for f in dq_warnings[:4]:
        cautions.append({
            "source":  "data_quality",
            "column":  f.get("column", "?"),
            "reason":  f.get("description", "")[:120],
        })

    # ── Verdict ───────────────────────────────────────────────────────────────

    if blockers:
        verdict = "blocked"
    elif cautions:
        verdict = "caution"
    else:
        verdict = "ready"

    # ── Cleared items ─────────────────────────────────────────────────────────

    cleared = []
    if leakage_result is not None and not ld_criticals:
        cleared.append("No critical leakage found — feature set is safe to use")
    if dq_score is not None and dq_score >= 75 and not dq_criticals:
        cleared.append(f"Data quality score {dq_score}/100 — no blocking issues")
    if leakage_result is None:
        cleared.append("Leakage check not yet run — run leakage_detector before training")
    if dq_result is None:
        cleared.append("Data quality check not yet run — run data_quality before training")

    # ── Summary sentence ──────────────────────────────────────────────────────

    if verdict == "blocked":
        n = len(blockers)
        summary = (
            f"{n} blocking issue{'s' if n != 1 else ''} must be resolved before training. "
            "See the action items below."
        )
    elif verdict == "caution":
        n = len(cautions)
        summary = (
            f"No hard blockers, but {n} issue{'s' if n != 1 else ''} warrant review "
            "before you commit to training."
        )
    else:
        summary = (
            "Data and feature set look clean. "
            "Proceed to feature engineering and model selection."
        )

    return SkillResult(
        {
            "verdict":            verdict,
            "blockers":           blockers,
            "cautions":           cautions,
            "cleared":            cleared,
            "summary":            summary,
            "dq_score":           dq_score,
            "dq_available":       dq_result       is not None,
            "leakage_available":  leakage_result  is not None,
        },
        skill="preflight",
    )

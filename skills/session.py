"""
WorkflowSession — optional context object for multi-skill DS workflows.

Pass the same session instance to each skill call. Skills will:
  - Warn in their notebook output if recommended prerequisites haven't run yet.
  - Automatically read prior-skill findings so you don't have to pass them explicitly.
  - Register their own result so downstream skills can use it.

Usage in a Databricks notebook::

    from ds_workbench import WorkflowSession, data_quality, leakage_detector
    from ds_workbench import feature_engineering, model_selector

    session = WorkflowSession()

    dq = data_quality.assess(df, "sales", session=session)
    ld = leakage_detector.check(df, "churn", session=session)   # reads DQ automatically
    fe = feature_engineering.suggest(df, "churn", session=session)  # reads LD automatically
    ms = model_selector.select(df, "churn", session=session)    # reads DQ + LD automatically

Skills used without a session continue to work exactly as before.
"""
from __future__ import annotations

# skill → skills that should run before it
_PREREQS: dict[str, list[str]] = {
    "leakage_detector":    ["data_quality"],
    "feature_engineering": ["data_quality", "leakage_detector"],
    "model_selector":      ["data_quality", "leakage_detector"],
}

# Suggested one-liner per skill for the warning hint
_HINT: dict[str, str] = {
    "data_quality":     "data_quality.assess(df, table_name, session=session)",
    "leakage_detector": "leakage_detector.check(df, target_col, session=session)",
    "eda":              "eda.analyze(df, table_name, session=session)",
    "join_auditor":     "join_auditor.audit(left_df, right_df, keys, session=session)",
}


class WorkflowSession:
    """
    Lightweight context object that links skill calls in a DS workflow.

    Skills register their results here and read prior findings from it,
    enabling automatic cross-skill context and prerequisite warnings.

    Args:
        context: Plain-English description of the business question or task
            (e.g. "Investigating Q3 revenue shortfall by region").
            Skills that accept a ``context`` parameter will use this
            automatically when no explicit value is passed.

    Example::

        session = WorkflowSession(context="Predicting customer churn for Q4 retention campaign")
        dq = data_quality.assess(df, "customers", session=session)
        eda_result = eda.analyze(df, "customers", session=session)  # focuses on churn drivers
    """

    def __init__(self, context: str = "") -> None:
        self._results: dict[str, object] = {}
        self.context = context

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, skill: str, result) -> None:
        """Record a completed skill result. Called automatically by each skill."""
        self._results[skill] = result

    # ── Lookup ────────────────────────────────────────────────────────────────

    def has_run(self, skill: str) -> bool:
        return skill in self._results

    def get_result(self, skill: str):
        """Return the most recent result for a skill, or None."""
        return self._results.get(skill)

    def get_findings(self, skill: str) -> list:
        """Return the findings list from a prior skill run, or []."""
        r = self._results.get(skill)
        return list(r.get("findings", [])) if r else []

    # ── Prerequisite warnings ─────────────────────────────────────────────────

    def prerequisite_warnings(self, current_skill: str) -> list[str]:
        """
        Return a list of HTML warning strings for missing prerequisites.
        Empty list when all prerequisites have been satisfied.
        Called by each skill before it runs.
        """
        required = _PREREQS.get(current_skill, [])
        missing  = [s for s in required if not self.has_run(s)]
        if not missing:
            return []

        warnings = []
        current_label = current_skill.replace("_", " ").title()
        for skill in missing:
            friendly = skill.replace("_", " ").title()
            hint     = _HINT.get(skill, f"{skill}.(…, session=session)")
            warnings.append(
                '<div style="background:#fffde7;border-left:4px solid #f9a825;'
                'padding:8px 12px;font-size:12px">'
                f'<strong style="color:#e65100">⚡ Run {friendly} first</strong><br>'
                '<span style="color:#5d4037;font-size:11px">'
                f'{current_label} produces better results with {friendly} context. '
                f'Add before this call: '
                f'<code style="background:#fef9e7;padding:1px 5px;border-radius:3px">'
                f'{hint}</code>'
                '</span></div>'
            )
        return warnings

    # ── Repr ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        ran = list(self._results.keys())
        ctx = f" | context: '{self.context}'" if self.context else ""
        return f"WorkflowSession({len(ran)} completed: {', '.join(ran) or 'none'}{ctx})"

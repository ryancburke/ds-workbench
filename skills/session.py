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

    def __init__(self, context: str = "", *, datasets: list[str] | None = None, persist: bool = True,
                 prediction_point: str = "", column_notes: dict[str, str] | None = None,
                 required_checks: list[str] | None = None) -> None:
        """
        Args:
            context: The business question (shown on the team Sessions page).
            datasets: Tables this session is about (used to compare with earlier sessions).
            persist: Save the session for the team Sessions page (best effort; never raises).
            prediction_point: When the model will score, e.g. "1st of each month, predicting
                churn in the next 30 days". Leakage checks use it to reason about timing.
            column_notes: When/how columns are recorded, e.g. {"tickets_mtd": "opened 1st-14th"}.
            required_checks: Checks this work must pass, e.g. ["data_quality", "causal_inference"]
                for a causal question. Default: inferred from what runs. Modelling skills always
                add data quality + leakage (see ds_workbench.core.readiness.required_checks).
        """
        from ds_workbench.core.readiness import required_checks as _validate
        import uuid
        self._results: dict[str, object] = {}
        self.context = context
        self.id = f"nb_{uuid.uuid4().hex[:10]}"
        self.datasets = list(datasets or [])
        self.persist = persist
        self.prediction_point = (prediction_point or "").strip()
        self.column_notes = dict(column_notes or {})
        self.confirmations: list[dict] = []
        self.spec: dict | None = None
        self.required_checks = list(required_checks) if required_checks is not None else None
        if self.required_checks is not None:
            _validate({}, self.required_checks)      # fail fast on an unknown or empty list
        self._runs: list[dict] = []

    # ── Problem spec (problem_framer.frame drafts it; a person confirms it) ────

    def set_spec(self, spec: dict) -> None:
        """Attach a ProblemSpec (called by problem_framer.frame). Its required checks become
        this session's declared checks; its question becomes the context if none was set."""
        if spec.get("error"):
            return
        self.spec = dict(spec)
        self.required_checks = list(spec.get("required_checks") or []) or None
        if not self.context:
            self.context = spec.get("question", "")
        if self.persist:
            self._save()

    def confirm_spec(self, note: str = "") -> dict:
        """Confirm the problem spec as the definition of this work (recommended before data
        discovery). Records who confirmed it and when; the model can never confirm a spec."""
        from ds_workbench.core.problem_spec import confirm
        from skills import usage
        from skills.api_retry import _call_ctx
        if not self.spec:
            raise ValueError("No problem spec yet: run problem_framer.frame(question, session=session) first")
        self.spec = confirm(self.spec, user=usage.current_user(dict(_call_ctx.get())), note=note)
        if self.persist:
            self._save()
        return self.spec

    # ── Availability confirmations (the user's judgment, never the model's) ───

    def confirm_available(self, columns, *, note: str, dataset: str = "") -> list[dict]:
        """Record that you have checked these columns exist at the prediction point.

        A confirmed column no longer blocks readiness; its leakage finding shows as info with
        your name, the time and your note. The confirmation applies only to this dataset and
        this session's prediction point: if either changes, the finding blocks again.

        Args:
            columns: A column name or a list of them.
            note: Required — how you know (e.g. "snapshot taken at 00:00 on the 1st; verified
                in the pipeline DAG").
            dataset: The table; defaults to the session's dataset when it has exactly one.

        Example::

            session.confirm_available("tickets_mtd", note="Counted up to the prediction date; see ETL job 42")
        """
        from ds_workbench.core.readiness import make_confirmation
        from skills import usage
        from skills.api_retry import _call_ctx

        if not self.prediction_point:
            raise ValueError("Set a prediction point first (WorkflowSession(prediction_point=...)): "
                             "availability only means something relative to when the model scores.")
        if not dataset:
            ld = self.get_result("leakage_detector") or {}
            dataset = ld.get("dataset") or (self.datasets[0] if len(self.datasets) == 1 else "")
        if not dataset:
            raise ValueError("Say which dataset these columns are in: confirm_available(..., dataset='...')")
        user = usage.current_user(dict(_call_ctx.get()))
        new = [make_confirmation(c, dataset=dataset, prediction_point=self.prediction_point,
                                 note=note, user=user)
               for c in ([columns] if isinstance(columns, str) else list(columns))]
        self.confirmations.extend(new)
        if self.persist:
            self._save()
        return new

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, skill: str, result) -> None:
        """Record a completed skill result. Called automatically by each skill."""
        from datetime import datetime, timezone
        from skills.api_retry import _call_ctx
        self._results[skill] = result
        # Skills register from inside their run, so the run_id is in the call context
        run_id = result.get("run_id") or _call_ctx.get().get("run_id")
        self._runs.append({"skill": skill, "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                           "status": "error" if result.get("error") else "complete",
                           "run_id": run_id, "label": skill})
        if self.persist:
            self._save()

    def _save(self) -> None:
        try:
            from skills import team_sessions, usage
            from skills.api_retry import _call_ctx
            team_sessions.save_session(
                self.id, owner=usage.current_user(dict(_call_ctx.get())), question=self.context,
                datasets=self.datasets, results=dict(self._results), runs=self._runs, source="notebook",
                prediction_point=self.prediction_point, confirmations=self.confirmations,
                required_checks=self.required_checks, spec=self.spec)
        except Exception:
            pass  # persistence must never break an analysis

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

    # ── Session digest ────────────────────────────────────────────────────────

    def digest(self):
        """Return a single consolidated SkillResult summarising all skill runs.

        Surfaces blockers, cautions, cleared checks, and a recommended next
        step.  No API call — instant synthesis of already-computed results.

        Example::

            dq = data_quality.assess(df, "customers", session=session)
            ld = leakage_detector.check(df, "churn",   session=session)
            ms = model_selector.select(df, "churn",    session=session)
            session.digest()   # ← renders consolidated view
        """
        from skills.result import SkillResult

        if not self._results:
            return SkillResult(
                {"error": "No skills have been run in this session yet."},
                skill="session_digest",
            )
        data = build_digest(dict(self._results), confirmations=self.confirmations,
                            required_checks=self.required_checks)
        return SkillResult(data, skill="session_digest")

    # ── Repr ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        ran = list(self._results.keys())
        ctx = f" | context: '{self.context}'" if self.context else ""
        return f"WorkflowSession({len(ran)} completed: {', '.join(ran) or 'none'}{ctx})"


# ── Standalone digest builder (used by WorkflowSession.digest and Dash card) ──

def build_digest(results: dict, confirmations: list | None = None,
                 required_checks: list | None = None) -> dict:
    """Synthesise a dict of {skill: result} into a structured digest.

    ``results`` is a plain-dict mapping of skill name → result dict; ``confirmations`` are
    people's availability confirmations (WorkflowSession.confirm_available).
    Returns a dict suitable for SkillResult(…, skill='session_digest').
    """
    from ds_workbench.core.readiness import CHECK_LABEL, assess
    readiness = assess(results, confirmations=confirmations, declared=required_checks)
    confirmed_cols = {c["column"] for c in readiness["confirmed"]}
    # A failed check is never narrated as cleared: drop it before building the narrative
    failed = {f["skill"] for f in readiness["failed"]}
    results = {k: v for k, v in results.items() if k not in failed and v is not None}
    skills_run = list(results.keys())
    blockers: list[dict] = []
    cautions: list[dict] = []
    cleared:  list[dict] = []

    # ── Leakage ──────────────────────────────────────────────────────────────
    ld = results.get("leakage_detector") or {}
    if ld:
        for f in ld.get("findings", []):
            if f.get("column") in confirmed_cols:
                continue            # a person confirmed it's available: listed under confirmed
            entry = {
                "source": "leakage",
                "column": f.get("column", "?"),
                "issue":  (f.get("evidence") or f.get("description", ""))[:110],
                "action": f.get("recommendation", "")[:110],
            }
            if f.get("severity") == "critical":
                blockers.append(entry)
            elif f.get("severity") == "warning":
                cautions.append({k: v for k, v in entry.items() if k != "action"})
        if ld.get("verdict") == "clean":
            cleared.append({"skill": "leakage_detector", "summary": "No leakage detected"})

    # ── Data Quality ─────────────────────────────────────────────────────────
    dq = results.get("data_quality") or {}
    if dq:
        score = dq.get("overall_score", 0)
        dq_blocked = False
        for f in dq.get("findings", []):
            entry = {
                "source": "data_quality",
                "column": f.get("column", "?"),
                "issue":  f.get("description", "")[:110],
                "action": f.get("recommendation", "")[:110],
            }
            if f.get("severity") == "critical":
                blockers.append(entry)
                dq_blocked = True
            elif f.get("severity") == "warning":
                cautions.append({k: v for k, v in entry.items() if k != "action"})
        if not dq_blocked:
            cleared.append({"skill": "data_quality", "summary": f"{score}/100 — no critical issues"})

    # ── Join Auditor ──────────────────────────────────────────────────────────
    ja = results.get("join_auditor") or {}
    if ja:
        verdict = ja.get("overall_verdict", "safe")
        fo = ja.get("row_count_change", {}).get("fan_out_factor", 1.0)
        if verdict == "dangerous":
            blockers.append({
                "source": "join_auditor", "column": "—",
                "issue":  f"Dangerous join: ×{fo:.2f} fan-out",
                "action": "Investigate join keys before training",
            })
        elif verdict == "review_needed":
            cautions.append({
                "source": "join_auditor", "column": "—",
                "issue":  f"Join needs review: ×{fo:.2f} fan-out",
            })
        else:
            cleared.append({"skill": "join_auditor", "summary": f"Safe — ×{fo:.2f} fan-out"})

    # ── Model Selector ────────────────────────────────────────────────────────
    ms = results.get("model_selector") or {}
    if ms:
        rd   = ms.get("data_readiness", "needs_prep")
        recs = ms.get("recommendations", [])
        top  = recs[0] if recs else {}
        name = top.get("display_name") or top.get("model_family", "")
        task = ms.get("task_type", "").replace("_", " ")
        notes = ms.get("readiness_notes", [])
        if rd == "ready":
            cleared.append({"skill": "model_selector",
                            "summary": (f"{name} — strong fit for {task}" if name else f"Ready · {task}")})
        elif rd == "needs_prep":
            cautions.append({
                "source": "model_selector", "column": "—",
                "issue": notes[0][:110] if notes else "Data needs preparation before training",
            })
        else:
            blockers.append({
                "source": "model_selector", "column": "—",
                "issue": "Insufficient data for training",
                "action": notes[0][:110] if notes else "Address readiness notes",
            })

    # ── MLOps ─────────────────────────────────────────────────────────────────
    ml = results.get("mlops_review") or results.get("mlops_reviewer") or {}
    if ml:
        score = ml.get("readiness_score", 0)
        crits = [f for f in ml.get("findings", []) if f.get("severity") == "critical"]
        if crits:
            blockers.append({
                "source": "mlops_review", "column": "—",
                "issue":  f"MLOps: {len(crits)} critical issue{'s' if len(crits) != 1 else ''}",
                "action": crits[0].get("fix", "")[:110],
            })
        else:
            cleared.append({"skill": "mlops_review", "summary": f"Readiness {score}/100"})

    # ── EDA ───────────────────────────────────────────────────────────────────
    eda = results.get("eda") or {}
    if eda:
        n_tgt = len(eda.get("potential_targets", []))
        n_ins = len(eda.get("key_insights", []))
        cleared.append({"skill": "eda",
                        "summary": f"{n_tgt} ML target{'s' if n_tgt != 1 else ''} · {n_ins} insight{'s' if n_ins != 1 else ''}"})

    # ── Feature Engineering ───────────────────────────────────────────────────
    fe = results.get("feature_engineering") or {}
    if fe:
        n_feat = len(fe.get("features", []))
        n_drop = len(fe.get("dropped_cols", []))
        cleared.append({"skill": "feature_engineering",
                        "summary": f"{n_feat} features planned · {n_drop} dropped"})

    # ── Overall verdict: from the readiness policy, never computed here ───────
    overall = readiness["status"]

    # ── Next step ─────────────────────────────────────────────────────────────
    ld_blockers = [b for b in blockers if b["source"] == "leakage"]
    dq_blockers = [b for b in blockers if b["source"] == "data_quality"]
    if readiness["failed"]:
        f = readiness["failed"][0]
        next_step = f"Re-run {CHECK_LABEL.get(f['skill'], f['skill'])} — it failed: {f['error'][:100]}"
    elif readiness["missing"] and not blockers:
        next_step = f"Run {CHECK_LABEL.get(readiness['missing'][0], readiness['missing'][0])} — required before this is ready"
    elif ld_blockers:
        n = len(ld_blockers)
        cols = ", ".join(b["column"] for b in ld_blockers[:3] if b["column"] != "—")
        next_step = (f"Drop {n} leaking column{'s' if n > 1 else ''}{f' ({cols})' if cols else ''}, or confirm "
                     "availability with session.confirm_available(...), then re-run leakage_detector")
    elif dq_blockers:
        n = len(dq_blockers)
        next_step = f"Fix {n} critical data quality issue{'s' if n > 1 else ''} before training"
    elif blockers:
        next_step = blockers[0].get("action") or "Resolve blockers before training"
    elif cautions:
        next_step = "Review cautions, then proceed to model selection"
    elif "leakage_detector" not in skills_run:
        next_step = "Run leakage_detector to check for target leakage"
    elif "model_selector" not in skills_run:
        next_step = "Run model_selector to complete pre-training checks"
    else:
        next_step = "All checks clear — proceed to feature engineering and training"

    return {
        "skills_run":      skills_run,
        "overall_verdict": overall,
        "missing":         readiness["missing"],
        "failed":          readiness["failed"],
        "blockers":        blockers,
        "cautions":        cautions,
        "cleared":         cleared,
        "confirmed":       readiness["confirmed"],
        "leakage_caveat":  ld.get("caveat", ""),
        "next_step":       next_step,
    }

"""Session-level Workbench panels: readiness strip, pipeline health, guided workflow,
synthesis, cross-skill conflicts and the investigation timeline."""
import dash_bootstrap_components as dbc
from dash import html

from workbench.constants import JA_VERDICT_CLR, SKILL_META
from workbench.renderers.ld import _LD_VERDICT_COLOR


# ── Timeline helper ───────────────────────────────────────────────────────────

def _timeline_row(run: dict, skill: str) -> html.Tr:
    m  = SKILL_META.get(skill, {})
    s  = run.get("summary", {})
    st = run.get("status", "complete")

    skill_cell = html.Td(dbc.Badge(
        [html.I(className=f"bi {m.get('icon','')} me-1"), m.get("label", skill)],
        color=m.get("color", "secondary"), className="px-2 py-1",
    ))
    label_cell = html.Td(html.Strong(run.get("label", "—")), style={"minWidth": "140px"})

    if st == "error":
        summary_cell = html.Td(html.Span(run.get("error", "")[:80], className="text-danger small"))
    elif skill == "data_quality":
        score = s.get("score", "?")
        sc = "success" if isinstance(score, int) and score >= 75 else \
             "warning" if isinstance(score, int) and score >= 50 else "danger"
        summary_cell = html.Td(html.Div([
            dbc.Badge(f"{score}/100", color=sc, className="me-1"),
            html.Span(f"{s.get('critical',0)} crit · {s.get('warning',0)} warn",
                      className="text-muted small"),
        ], className="d-flex align-items-center gap-1"))
    elif skill == "eda":
        summary_cell = html.Td(html.Span(
            f"{s.get('charts',0)} charts · {s.get('insights',0)} insights · "
            f"{s.get('targets',0)} ML targets", className="text-muted small"))
    elif skill == "join_auditor":
        verdict = s.get("verdict", "review_needed")
        vc = JA_VERDICT_CLR.get(verdict, "secondary")
        summary_cell = html.Td(html.Div([
            dbc.Badge(verdict.replace("_", " ").title(), color=vc, className="me-1"),
            html.Span(f"×{s.get('fan_out_factor',1):.2f}", className="text-muted small"),
        ], className="d-flex align-items-center gap-1"))
    elif skill == "nl_to_spark":
        preview = s.get("sql_preview", "")[:60]
        summary_cell = html.Td(
            html.Code(preview, style={"fontSize": "0.72rem", "color": "#7dd3fc"}))
    elif skill == "mlops_review":
        score = s.get("score", "?")
        sc = "success" if isinstance(score, int) and score >= 70 else \
             "warning" if isinstance(score, int) and score >= 40 else "danger"
        summary_cell = html.Td(html.Div([
            dbc.Badge(f"{score}/100", color=sc, className="me-1"),
            html.Span(f"{s.get('critical',0)} crit", className="text-muted small"),
        ], className="d-flex align-items-center gap-1"))
    elif skill == "model_selector":
        task = s.get("task_type", "").replace("_", " ").title()
        top  = s.get("top_model", "")
        summary_cell = html.Td(html.Div([
            dbc.Badge(task, color="secondary",
                      style={"border": "1px solid #a78bfa", "color": "#a78bfa",
                             "fontSize": "0.65rem"}, className="me-1"),
            html.Span(top[:40], className="text-muted small"),
        ], className="d-flex align-items-center gap-1"))
    elif skill == "leakage_detector":
        verdict = s.get("verdict", "clean")
        risk    = s.get("risk_level", "none")
        vc = _LD_VERDICT_COLOR.get(verdict, "secondary")
        summary_cell = html.Td(html.Div([
            dbc.Badge(verdict.replace("_", " ").title(), color=vc, className="me-1"),
            html.Span(f"risk: {risk}", className="text-muted small"),
        ], className="d-flex align-items-center gap-1"))
    elif skill == "feature_engineering":
        n_f = s.get("n_features", 0)
        n_d = s.get("n_dropped", 0)
        summary_cell = html.Td(html.Span(
            f"{n_f} features · {n_d} dropped", className="text-muted small"))
    else:
        summary_cell = html.Td()

    ts_cell = html.Td(html.Span(run.get("timestamp", ""), className="text-muted small"),
                      style={"width": "55px"})
    return html.Tr([ts_cell, skill_cell, label_cell, summary_cell])


# ── Cross-tool helpers ────────────────────────────────────────────────────────

def _readiness_strip(store: dict) -> html.Div:
    """Compact always-visible training-readiness bar for the session card."""
    dq_runs = [r for r in store.get("data_quality",     []) if r.get("status") == "complete"]
    ld_runs = [r for r in store.get("leakage_detector", []) if r.get("status") == "complete"]
    ms_runs = [r for r in store.get("model_selector",   []) if r.get("status") == "complete"]

    def _chip(text: str, color: str) -> html.Span:
        ico = {
            "success":   "bi-check-circle-fill text-success",
            "warning":   "bi-exclamation-triangle-fill text-warning",
            "danger":    "bi-x-circle-fill text-danger",
            "secondary": "bi-circle text-muted",
        }.get(color, "bi-circle text-muted")
        return html.Span(
            [html.I(className=f"bi {ico} me-1"), text],
            className=f"small {'text-muted' if color == 'secondary' else ''}",
        )

    # DQ chip
    if dq_runs:
        s = dq_runs[-1].get("summary", {})
        score, crit = s.get("score", 0), s.get("critical", 0)
        if crit:          dq_color, dq_txt = "danger",    f"DQ {score}/100 · {crit} crit"
        elif score < 75:  dq_color, dq_txt = "warning",   f"DQ {score}/100"
        else:             dq_color, dq_txt = "success",   f"DQ {score}/100"
    else:
        dq_color, dq_txt = "secondary", "DQ — not run"

    # Leakage chip
    if ld_runs:
        v = ld_runs[-1].get("result", {}).get("verdict", "clean")
        if v == "leakage_detected": ld_color, ld_txt = "danger",  "Leakage detected"
        elif v == "suspicious":     ld_color, ld_txt = "warning", "Leakage suspicious"
        else:                       ld_color, ld_txt = "success", "Leakage clean"
    else:
        ld_color, ld_txt = "secondary", "Leakage — not run"

    # Model chip
    if ms_runs:
        rd   = ms_runs[-1].get("result", {}).get("data_readiness", "needs_prep")
        recs = ms_runs[-1].get("result", {}).get("recommendations", [])
        top  = recs[0].get("display_name", recs[0].get("model_family", "")) if recs else ""
        if rd == "ready":        ms_color, ms_txt = "success", ("Model ready" + (f" · {top}" if top else ""))
        elif rd == "needs_prep": ms_color, ms_txt = "warning", "Model needs prep"
        else:                    ms_color, ms_txt = "danger",  "Model insufficient"
    else:
        ms_color, ms_txt = "secondary", "Model — not run"

    # Overall verdict
    all_run  = bool(dq_runs and ld_runs and ms_runs)
    colors   = [c for c in [dq_color, ld_color, ms_color] if c != "secondary"]
    if not any([dq_runs, ld_runs, ms_runs]):
        v_color, v_label = "secondary", "PENDING"
    elif all_run:
        if all(c == "success" for c in colors):
            v_color, v_label = "success", "READY TO TRAIN"
        elif "danger" in colors:
            v_color, v_label = "danger",  "NOT READY"
        else:
            v_color, v_label = "warning", "CAUTION"
    else:
        v_color = "danger" if "danger" in colors else "warning"
        v_label = "BLOCKED" if v_color == "danger" else "IN PROGRESS"

    return html.Div([
        html.Span("Training readiness", className="text-muted small fw-semibold me-3",
                  style={"whiteSpace": "nowrap"}),
        _chip(dq_txt, dq_color),
        html.Span("·", className="text-muted small mx-1"),
        _chip(ld_txt, ld_color),
        html.Span("·", className="text-muted small mx-1"),
        _chip(ms_txt, ms_color),
        dbc.Badge(v_label, color=v_color, pill=True,
                  className="ms-auto", style={"fontSize": "0.65rem"}),
    ], className="d-flex align-items-center flex-wrap gap-1 py-2 mt-2 mb-1",
       style={"borderTop": "1px solid #252545", "borderBottom": "1px solid #252545"})


def _pipeline_health_card(store: dict) -> html.Div | None:
    dq_runs = [r for r in store.get("data_quality",        []) if r.get("status") == "complete"]
    ja_runs = [r for r in store.get("join_auditor",        []) if r.get("status") == "complete"]
    ld_runs = [r for r in store.get("leakage_detector",    []) if r.get("status") == "complete"]
    ms_runs = [r for r in store.get("model_selector",      []) if r.get("status") == "complete"]
    ml_runs = [r for r in store.get("mlops_review",        []) if r.get("status") == "complete"]

    if not any([dq_runs, ja_runs, ld_runs, ms_runs, ml_runs]):
        return None

    items   = []
    all_ok  = True
    blockers = []

    if dq_runs:
        s     = dq_runs[-1].get("summary", {})
        score = s.get("score", 0)
        crit  = s.get("critical", 0)
        color = "danger" if crit or score < 50 else "warning" if score < 75 else "success"
        icon  = "bi-shield-x" if color == "danger" else "bi-shield-exclamation" if color == "warning" else "bi-shield-check"
        if color != "success":
            all_ok = False
            if color == "danger":
                blockers.append(f"DQ: {crit} critical issue{'s' if crit != 1 else ''}")
        items.append(dbc.Col([
            html.Div("Data Quality", className="text-muted small"),
            html.Div([html.I(className=f"bi {icon} text-{color} me-1"),
                      dbc.Badge(f"{score}/100", color=color)]),
        ], width="auto"))

    if ja_runs:
        verdict = ja_runs[-1].get("summary", {}).get("verdict", "review_needed")
        color   = JA_VERDICT_CLR.get(verdict, "secondary")
        if verdict != "safe":
            all_ok = False
            if verdict == "dangerous":
                blockers.append("Join: dangerous fan-out or data loss")
        items.append(dbc.Col([
            html.Div("Join Health", className="text-muted small"),
            dbc.Badge(verdict.replace("_", " ").title(), color=color),
        ], width="auto"))

    if ld_runs:
        ld_res  = ld_runs[-1].get("result", {})
        verdict = ld_res.get("verdict", "clean")
        risk    = ld_res.get("risk_level", "none")
        color   = {"leakage_detected": "danger", "suspicious": "warning", "clean": "success"}.get(verdict, "secondary")
        if verdict != "clean":
            all_ok = False
            if verdict == "leakage_detected":
                blockers.append("Leakage: critical features must be removed")
        items.append(dbc.Col([
            html.Div("Leakage Risk", className="text-muted small"),
            dbc.Badge(verdict.replace("_", " ").title(), color=color),
        ], width="auto"))

    if ms_runs:
        ms_res    = ms_runs[-1].get("result", {})
        readiness = ms_res.get("data_readiness", "needs_prep")
        top_model = ""
        recs = ms_res.get("recommendations", [])
        if recs:
            top_model = recs[0].get("display_name", recs[0].get("model_family", ""))
        color = {"ready": "success", "needs_prep": "warning", "insufficient": "danger"}.get(readiness, "secondary")
        if readiness != "ready":
            all_ok = False
        items.append(dbc.Col([
            html.Div("Model Readiness", className="text-muted small"),
            html.Div([
                dbc.Badge(readiness.replace("_", " ").title(), color=color),
                html.Span(f"  {top_model}", className="text-muted small ms-1") if top_model else html.Span(),
            ]),
        ], width="auto"))

    if ml_runs:
        ml_res = ml_runs[-1].get("result", {})
        score  = ml_res.get("readiness_score", 0)
        ml_crits = [f for f in ml_res.get("findings", []) if f.get("severity") == "critical"]
        color  = "danger" if ml_crits or score < 40 else "warning" if score < 70 else "success"
        if color != "success":
            all_ok = False
            if color == "danger":
                blockers.append(f"MLOps: {len(ml_crits)} blocker{'s' if len(ml_crits) != 1 else ''}")
        items.append(dbc.Col([
            html.Div("MLOps", className="text-muted small"),
            dbc.Badge(f"{score}/100", color=color),
        ], width="auto"))

    # ── Training-readiness verdict (DQ + leakage + model selector only) ─────────
    dq_s      = dq_runs[-1].get("summary", {}) if dq_runs else {}
    dq_ok     = bool(dq_runs) and dq_s.get("score", 0) >= 75 and dq_s.get("critical", 0) == 0
    ld_ok     = bool(ld_runs) and ld_runs[-1].get("result", {}).get("verdict") == "clean"
    ms_ok     = bool(ms_runs) and ms_runs[-1].get("result", {}).get("data_readiness") == "ready"
    prereqs_run = bool(dq_runs and ld_runs and ms_runs)
    training_ready = dq_ok and ld_ok and ms_ok

    if prereqs_run:
        if training_ready:
            t_color, t_icon, t_label = "success", "bi-check-circle-fill", "READY TO TRAIN"
            t_reasons: list[str] = []
        else:
            t_color, t_icon, t_label = "danger", "bi-x-octagon-fill", "NOT READY TO TRAIN"
            t_reasons = []
            if not dq_ok:
                s = dq_s.get("score", 0)
                c = dq_s.get("critical", 0)
                t_reasons.append(f"Data Quality: {s}/100 score" + (f", {c} critical issue{'s' if c != 1 else ''}" if c else ""))
            if not ld_ok:
                v = ld_runs[-1].get("result", {}).get("verdict", "?")
                t_reasons.append(f"Leakage: {v.replace('_', ' ')}")
            if not ms_ok:
                rd = ms_runs[-1].get("result", {}).get("data_readiness", "?")
                t_reasons.append(f"Model readiness: {rd.replace('_', ' ')}")
    else:
        missing = [n for ran, n in [(dq_runs, "Data Quality"), (ld_runs, "Leakage Detector"),
                                     (ms_runs, "Model Selector")] if not ran]
        t_color, t_icon, t_label = "secondary", "bi-hourglass-split", "TRAINING READINESS PENDING"
        t_reasons = [f"Still needed: {', '.join(missing)}"] if missing else []

    verdict_hero = dbc.Alert(
        [
            html.Strong([html.I(className=f"bi {t_icon} me-2"), t_label],
                        className="d-block mb-1" if t_reasons else ""),
            html.Ul([html.Li(r, className="small") for r in t_reasons],
                    className="mb-0 ps-3") if t_reasons else html.Span(),
        ],
        color=t_color, className="py-2 mb-3",
    )

    blocker_row = html.Div(
        [html.Span(f"⚠ {b}", className="text-warning small me-3") for b in blockers],
        className="mt-2",
    ) if blockers else html.Div()

    return dbc.Card([
        dbc.CardBody([
            verdict_hero,
            dbc.Row(items, align="center", className="g-3"),
            blocker_row,
        ]),
    ], style={
        "border": f"1px solid {'#19875440' if training_ready else '#ffffff15'}",
        "background": f"{'rgba(25,135,84,0.05)' if training_ready else 'rgba(255,255,255,0.02)'}",
    }, className="mb-2")


def _guided_workflow(store: dict) -> html.Div:
    pipeline = [
        ("data_quality",        "bi-shield-check",          "Data Quality",       "Assess data before modeling"),
        ("eda",                 "bi-graph-up",               "EDA",                "Understand distributions"),
        ("join_auditor",        "bi-link-45deg",             "Join Audit",         "Validate join keys"),
        ("leakage_detector",    "bi-exclamation-diamond",    "Leakage Check",      "Detect target leakage"),
        ("feature_engineering", "bi-wrench-adjustable",      "Feature Engineering","Build the feature pipeline"),
        ("model_selector",      "bi-cpu",                    "Model Selection",    "Choose the right algorithm"),
        ("mlops_review",        "bi-gear-fill",              "MLOps Review",       "Production readiness check"),
    ]
    ld_done = any(r.get("status") == "complete"
                  for r in store.get("leakage_detector", []))

    items = []
    for skill, icon, label, desc in pipeline:
        runs = store.get(skill, [])
        done = any(r.get("status") == "complete" for r in (runs if isinstance(runs, list) else []))
        order_warn = skill == "feature_engineering" and done and not ld_done
        items.append(html.Div([
            html.I(className=f"bi {'bi-check-circle-fill text-success' if done else 'bi-circle text-muted'} me-2",
                   style={"fontSize": "0.75rem"}),
            html.Span(label, className=f"small {'text-muted' if done else 'fw-semibold'}"),
            html.Span(f" — {desc}", className="text-muted small") if not done else html.Span(),
            dbc.Badge("⚠ run leakage first", color="warning", text_color="dark",
                      className="ms-2", style={"fontSize": "0.6rem"}) if order_warn else html.Span(),
        ], className="mb-1"))
    return html.Div([
        html.Div("Workflow Progress", className="text-muted small mb-2 fw-semibold",
                 style={"letterSpacing": "0.06em"}),
        html.Div(items),
    ])


def _synthesis_card(store: dict) -> html.Div | None:
    """Session digest — appears when 2+ skills have run."""
    from skills.session import build_digest

    SKILL_KEYS_ORDERED = [
        "data_quality", "leakage_detector", "join_auditor", "model_selector",
        "mlops_review", "eda", "feature_engineering",
    ]
    results = {}
    n_run = 0
    for skill in SKILL_KEYS_ORDERED:
        runs = [r for r in store.get(skill, []) if r.get("status") == "complete"]
        if runs:
            results[skill] = runs[-1].get("result", {})
            n_run += 1

    if n_run < 2:
        return None

    d = build_digest(results)
    verdict  = d["overall_verdict"]
    blockers = d["blockers"]
    cautions = d["cautions"]
    cleared  = d["cleared"]
    next_step = d["next_step"]

    vc = {"blocked": "danger", "caution": "warning",
          "ready": "success", "in_progress": "secondary"}.get(verdict, "secondary")
    vl = {"blocked": "BLOCKED", "caution": "CAUTION",
          "ready": "READY TO TRAIN", "in_progress": "IN PROGRESS"}.get(verdict, "PENDING")

    # ── Blockers ─────────────────────────────────────────────────────────────
    blocker_rows = []
    for b in blockers:
        src   = b["source"].replace("_", " ")
        color = "danger" if b["source"] in ("leakage", "leakage_detector") else "warning"
        blocker_rows.append(html.Tr([
            html.Td(dbc.Badge(src, color=color, style={"fontSize":"0.65rem"})),
            html.Td(html.Code(b["column"], style={"fontSize":"0.75rem"})),
            html.Td([
                html.Span(b["issue"], className="small"),
                html.Br() if b.get("action") else html.Span(),
                html.Span(f"→ {b['action']}", className="text-muted small") if b.get("action") else html.Span(),
            ]),
        ], style={"borderLeft": "3px solid #dc3545"}))

    blockers_section = html.Div([
        html.Div([
            html.I(className="bi bi-x-octagon-fill text-danger me-1"),
            html.Strong(f"Fix before training ({len(blockers)})", className="small text-danger"),
        ], className="mb-1"),
        dbc.Table([html.Tbody(blocker_rows)], size="sm", bordered=False,
                  style={"backgroundColor": "transparent", "fontSize": "0.8rem"}),
    ], className="mb-3") if blockers else html.Div()

    # ── Cautions ─────────────────────────────────────────────────────────────
    caution_rows = []
    for c in cautions:
        src = c["source"].replace("_", " ")
        caution_rows.append(html.Tr([
            html.Td(dbc.Badge(src, color="warning", text_color="dark",
                              style={"fontSize": "0.65rem"})),
            html.Td(html.Code(c["column"], style={"fontSize": "0.75rem"})),
            html.Td(html.Span(c["issue"], className="small")),
        ], style={"borderLeft": "3px solid #ffc107"}))

    cautions_section = html.Div([
        html.Div([
            html.I(className="bi bi-exclamation-triangle text-warning me-1"),
            html.Strong(f"Review ({len(cautions)})", className="small text-warning"),
        ], className="mb-1"),
        dbc.Table([html.Tbody(caution_rows)], size="sm", bordered=False,
                  style={"backgroundColor": "transparent", "fontSize": "0.8rem"}),
    ], className="mb-3") if cautions else html.Div()

    # ── Cleared ───────────────────────────────────────────────────────────────
    cleared_items = [
        html.Span([
            dbc.Badge(c["skill"].replace("_", " "), color="success",
                      style={"fontSize": "0.6rem"}, className="me-1"),
            html.Span(c["summary"], className="text-muted", style={"fontSize": "0.75rem"}),
        ], className="me-3")
        for c in cleared
    ]
    cleared_section = html.Div([
        html.Div([
            html.I(className="bi bi-check-circle-fill text-success me-1"),
            html.Strong("Cleared", className="small text-success me-2"),
            *cleared_items,
        ], className="d-flex align-items-center flex-wrap gap-1"),
    ], className="mb-3") if cleared else html.Div()

    # ── Next step ─────────────────────────────────────────────────────────────
    next_section = dbc.Alert([
        html.I(className="bi bi-arrow-right-circle me-2"),
        html.Strong("Next: "),
        next_step,
    ], color="success" if verdict == "ready" else "warning" if verdict == "caution" else "secondary",
       className="py-2 mb-0 small") if next_step else html.Div()

    accent = {"danger": "#dc3545", "warning": "#ffc107", "success": "#198754"}.get(vc, "#6c757d")
    body_children = (
        [dbc.Row([dbc.Col(blockers_section, md=6), dbc.Col(cautions_section, md=6)]),
         cleared_section, next_section]
        if blocker_rows else
        [cautions_section, cleared_section, next_section]
    )

    return dbc.Card([
        dbc.CardHeader([
            html.I(className="bi bi-journal-text me-2"),
            html.Strong("Session Digest"),
            html.Span(f" · {n_run} skill{'s' if n_run != 1 else ''} run",
                      className="text-muted small ms-1"),
            dbc.Badge(vl, color=vc, pill=True, className="ms-2",
                      style={"fontSize": "0.65rem"}),
        ], className="d-flex align-items-center"),
        dbc.CardBody(body_children),
    ], className="mb-3", style={"border": "1px solid #3a3a5a",
                                 "borderLeft": f"4px solid {accent}"})


def _cross_skill_conflicts(store: dict) -> html.Div | None:
    conflicts = []
    dq_runs = [r for r in store.get("data_quality",       []) if r.get("status") == "complete"]
    ml_runs = [r for r in store.get("mlops_review",       []) if r.get("status") == "complete"]
    ld_runs = [r for r in store.get("leakage_detector",   []) if r.get("status") == "complete"]
    fe_runs = [r for r in store.get("feature_engineering",[]) if r.get("status") == "complete"]
    ja_runs = [r for r in store.get("join_auditor",       []) if r.get("status") == "complete"]
    nl_runs = [r for r in store.get("nl_to_spark",        []) if r.get("status") == "complete"]

    if dq_runs and ld_runs:
        dq_findings = dq_runs[-1].get("result", {}).get("findings", [])
        ld_findings = ld_runs[-1].get("result", {}).get("findings", [])
        dq_cols = {f.get("column") for f in dq_findings
                   if f.get("column") and f.get("severity") in ("critical", "warning")}
        ld_cols = {f.get("column") for f in ld_findings
                   if f.get("column") and f.get("severity") in ("critical", "warning")}
        overlap = sorted(dq_cols & ld_cols)
        if overlap:
            col_list = ", ".join(f"`{c}`" for c in overlap[:5])
            more = f" (+{len(overlap)-5} more)" if len(overlap) > 5 else ""
            conflicts.append({
                "icon": "bi-exclamation-diamond-fill", "color": "danger",
                "title": "Columns flagged by both DQ and Leakage",
                "detail": f"{col_list}{more} — these have data quality AND leakage issues. Highest priority to resolve before training.",
            })

    if dq_runs and ml_runs:
        dq_crit  = dq_runs[-1].get("summary", {}).get("critical", 0)
        ml_score = ml_runs[-1].get("summary", {}).get("score", 0)
        if dq_crit and ml_score < 70:
            conflicts.append({
                "icon": "bi-shield-x", "color": "danger",
                "title": "DQ issues may block MLOps",
                "detail": f"{dq_crit} critical DQ finding(s) + MLOps readiness {ml_score}/100 — resolve data quality before deployment.",
            })

    if fe_runs and not ld_runs:
        conflicts.append({
            "icon": "bi-exclamation-diamond", "color": "warning",
            "title": "Feature engineering without leakage check",
            "detail": "Feature engineering was run without prior leakage detection. Features may encode target leakage.",
        })
    elif fe_runs and ld_runs and ld_runs[-1].get("summary", {}).get("verdict") == "leakage_detected":
        conflicts.append({
            "icon": "bi-exclamation-octagon-fill", "color": "danger",
            "title": "Features engineered despite detected leakage",
            "detail": "Leakage was detected but feature engineering proceeded. Review features for leaked signals.",
        })

    if ja_runs and nl_runs:
        ja_verdict = ja_runs[-1].get("summary", {}).get("verdict", "safe")
        if ja_verdict == "dangerous":
            conflicts.append({
                "icon": "bi-link-45deg", "color": "warning",
                "title": "Queries run on a dangerous join",
                "detail": "NL queries were generated after the Join Auditor flagged this join as dangerous. Results may be inflated by fan-out.",
            })

    if not conflicts:
        return None

    items = [dbc.Alert(
        [html.I(className=f"bi {c['icon']} me-2"),
         html.Strong(c["title"] + ": "),
         c["detail"]],
        color=c["color"], className="py-2 mb-2",
    ) for c in conflicts]

    return dbc.Card([
        dbc.CardHeader([html.I(className="bi bi-lightning-charge-fill me-2 text-warning"),
                        html.Strong("Cross-Skill Conflicts")]),
        dbc.CardBody(items),
    ], className="mb-3", style={"borderColor": "#dc354540"})


# ══════════════════════════════════════════════════════════════════════════════
# Callbacks
# ══════════════════════════════════════════════════════════════════════════════

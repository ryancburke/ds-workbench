"""Workbench renderers for Model Selector results (result dict → Dash components)."""
import dash_bootstrap_components as dbc
from dash import html
from workbench.renderers._safe import safe_render


# ── Model Selector renderer ───────────────────────────────────────────────────

_MS_STATUS_COLOR  = {"met": "success", "violated": "danger", "uncertain": "secondary"}
_MS_STATUS_ICON   = {"met": "bi-check-circle-fill text-success",
                     "violated": "bi-x-circle-fill text-danger",
                     "uncertain": "bi-question-circle text-muted"}
_MS_SUIT_COLOR    = {"strong": "success", "moderate": "warning", "weak": "secondary"}
_MS_READY_COLOR   = {"ready": "success", "needs_prep": "warning", "insufficient": "danger"}


@safe_render
def _ms_render(result: dict) -> html.Div:
    task        = result.get("task_type", "unknown").replace("_", " ").title()
    readiness   = result.get("data_readiness", "needs_prep")
    rdy_color   = _MS_READY_COLOR.get(readiness, "secondary")
    t_analysis  = result.get("target_analysis", {})
    assumptions = result.get("assumption_checks", [])
    recs        = result.get("recommendations", [])
    rdy_notes   = result.get("readiness_notes", [])

    # ── Verdict hero ──────────────────────────────────────────────────────────
    _top_model = recs[0].get("display_name", recs[0].get("model_family", "")) if recs else ""
    _violated  = [a for a in assumptions if a.get("status") == "violated"]
    if readiness == "insufficient" or len(_violated) >= 2:
        _v_color, _v_icon, _v_label = "danger",  "bi-x-octagon-fill",            "INSUFFICIENT DATA"
    elif readiness == "needs_prep" or _violated:
        _v_color, _v_icon, _v_label = "warning", "bi-exclamation-triangle-fill", "PREP REQUIRED"
    else:
        _v_color, _v_icon, _v_label = "success", "bi-check-circle-fill",         "DATA SUFFICIENT"
    verdict_hero = dbc.Alert([
        html.Strong([html.I(className=f"bi {_v_icon} me-2"), f"Model Selector: {_v_label}"],
                    className="d-block mb-1"),
        html.Span([html.Strong("Task: "), task, "  ·  ", html.Strong("Top pick: "), _top_model or "—"],
                  className="small"),
    ], color=_v_color, className="py-2 mb-3")

    # ── Header strip ──────────────────────────────────────────────────────────
    header = dbc.Row([
        dbc.Col(html.Div([
            html.I(className="bi bi-cpu me-2", style={"color": "#a78bfa"}),
            dbc.Badge(task, color="secondary",
                      style={"border": "1px solid #a78bfa", "color": "#a78bfa",
                             "fontSize": "0.8rem"}),
        ]), width="auto"),
        dbc.Col(html.Div([
            html.Span("Data readiness: ", className="text-muted small me-1"),
            dbc.Badge(readiness.replace("_", " ").title(), color=rdy_color,
                      style={"fontSize": "0.75rem"}),
        ]), width="auto"),
    ], align="center", className="mb-3")

    # ── Target summary ────────────────────────────────────────────────────────
    target_card = dbc.Card(dbc.CardBody([
        html.Strong("Target Analysis", className="d-block mb-1 small text-muted"),
        html.P(t_analysis.get("summary", ""), className="mb-0 small"),
    ]), className="mb-3", style={"border": "1px solid #2a2a4a"})

    # ── Assumption checks ─────────────────────────────────────────────────────
    assumption_rows = []
    for a in assumptions:
        st     = a.get("status", "uncertain")
        icon   = _MS_STATUS_ICON.get(st, "bi-circle text-muted")
        badge  = dbc.Badge(st.title(), color=_MS_STATUS_COLOR.get(st, "secondary"),
                           style={"fontSize": "0.62rem"}, className="me-2")
        assumption_rows.append(html.Tr([
            html.Td(html.I(className=f"bi {icon}"), style={"width": "24px"}),
            html.Td([badge, html.Strong(a.get("assumption", ""), className="small")]),
            html.Td(a.get("evidence", ""), className="small text-muted"),
            html.Td(a.get("affected_models", ""), className="small text-muted fst-italic"),
        ]))

    assumption_card = dbc.Card([
        dbc.CardHeader(html.Strong("Assumption Checks")),
        dbc.CardBody(
            dbc.Table(
                [html.Thead(html.Tr([html.Th(""), html.Th("Assumption"),
                                     html.Th("Evidence"), html.Th("Affects")]))] +
                [html.Tbody(assumption_rows)],
                hover=True, responsive=True, size="sm",
            ) if assumption_rows else html.P("No assumptions checked.", className="text-muted mb-0"),
            className="p-0",
        ),
    ], className="mb-3")

    # ── Model recommendations ─────────────────────────────────────────────────
    rec_cards = []
    for i, rec in enumerate(recs):
        suit      = rec.get("suitability", "moderate")
        suit_clr  = _MS_SUIT_COLOR.get(suit, "secondary")
        caveats   = rec.get("caveats", [])
        code      = rec.get("starter_code", "")
        collapse_id = f"ms-code-{i}"

        rec_cards.append(dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(html.Div([
                        html.Span(f"#{rec.get('rank',i+1)} ", className="text-muted me-1"),
                        html.Strong(rec.get("display_name", rec.get("model_family", "")),
                                    style={"fontSize": "0.95rem"}),
                    ]), width=8),
                    dbc.Col(
                        dbc.Badge(suit.title(), color=suit_clr,
                                  style={"fontSize": "0.7rem"}),
                        width=4, className="text-end",
                    ),
                ], align="center", className="mb-2"),
                html.P(rec.get("rationale", ""), className="small mb-2"),
                html.Div([
                    html.Ul([html.Li(c, className="small text-warning") for c in caveats],
                            className="mb-1"),
                ]) if caveats else html.Div(),
                html.Div([
                    dbc.Button("Show starter code", id={"type": "ms-code-toggle", "index": i},
                               size="sm", color="link",
                               className="p-0 small", style={"color": "#a78bfa"}),
                    dbc.Collapse(
                        html.Pre(code, className="mt-2 mb-0",
                                 style={"fontSize": "0.72rem", "background": "#12122a",
                                        "padding": "0.75rem", "borderRadius": "4px"}),
                        id={"type": "ms-code-collapse", "index": i},
                        is_open=False,
                    ),
                ]) if code else html.Div(),
            ]),
        ], className="mb-2", style={"border": "1px solid #2a2a4a" if suit != "strong"
                                    else "1px solid #a78bfa40"}))

    recs_section = html.Div([
        html.H6("Recommended Models", className="text-muted small mb-2 fw-semibold",
                style={"letterSpacing": "0.08em"}),
        html.Div(rec_cards),
    ], className="mb-3")

    # ── Readiness notes ───────────────────────────────────────────────────────
    notes_section = dbc.Alert([
        html.Strong("Before training: "),
        html.Ul([html.Li(n, className="small") for n in rdy_notes], className="mb-0 mt-1"),
    ], color="warning", className="py-2") if rdy_notes else html.Div()

    # ── Decision: the four choices for the next cell, before any reasoning ────
    decision = result.get("decision") or {}
    decision_rows = [
        html.Tr([html.Td(html.Strong(label), style={"width": "110px"}), html.Td(decision[key])])
        for label, key in (("Model", "model"), ("Imbalance", "imbalance_strategy"),
                           ("Metric", "primary_metric"), ("Validation", "validation"))
        if decision.get(key)
    ]
    decision_table = dbc.Table(html.Tbody(decision_rows), size="sm", bordered=False,
                               className="mb-3") if decision_rows else html.Div()

    return html.Div([verdict_hero, decision_table, header, target_card, assumption_card,
                     recs_section, notes_section])


"""Workbench renderers for Experiment Tracker results (result dict → Dash components)."""
import dash_bootstrap_components as dbc
from dash import html
from workbench.renderers._safe import safe_render


# ── Experiment Tracker renderer ──────────────────────────────────────────────

_ET_VERDICT_COLOR = {"promising": "success", "needs_work": "warning", "red_flags": "danger"}
_ET_SEV_COLOR     = {"positive": "success", "note": "info", "concern": "warning", "critical": "danger"}
_ET_DIM_ICON      = {
    "performance":          "bi-graph-up",
    "overfitting":          "bi-arrows-expand",
    "class_imbalance":      "bi-distribute-vertical",
    "data_leakage":         "bi-exclamation-diamond-fill",
    "feature_quality":      "bi-table",
    "deployment_readiness": "bi-box-arrow-up-right",
}


@safe_render
def _et_render(result: dict) -> html.Div:
    verdict    = result.get("verdict", "needs_work")
    interp     = result.get("interpretation", "")
    findings   = result.get("findings", [])
    next_steps = result.get("next_steps", [])
    suggested  = result.get("suggested_experiments", [])

    sev_rank = {"critical": 0, "concern": 1, "note": 2, "positive": 3}
    findings = sorted(findings, key=lambda f: sev_rank.get(f.get("severity", "note"), 2))

    verdict_color = _ET_VERDICT_COLOR.get(verdict, "secondary")
    verdict_label = verdict.replace("_", " ").upper()
    v_icon = ("bi-check-circle-fill"       if verdict == "promising"  else
              "bi-exclamation-triangle-fill" if verdict == "needs_work" else
              "bi-x-octagon-fill")

    hero = dbc.Alert([
        html.Div([html.I(className=f"bi {v_icon} me-2"), html.Strong(verdict_label)],
                 className="fs-6 mb-2"),
        html.P(interp, className="mb-0 small"),
    ], color=verdict_color, className="mb-3")

    finding_rows = []
    for f in findings:
        sev      = f.get("severity", "note")
        dim      = f.get("dimension", "performance")
        icon_cls = _ET_DIM_ICON.get(dim, "bi-dot")
        row_style = ({"borderLeft": f"3px solid var(--bs-{_ET_SEV_COLOR.get(sev,'secondary')})"}
                     if sev in ("critical", "concern") else {})
        finding_rows.append(html.Tr([
            html.Td(dbc.Badge(sev, color=_ET_SEV_COLOR.get(sev, "secondary")),
                    style={"width": "90px"}),
            html.Td([html.I(className=f"bi {icon_cls} me-1 small"),
                     html.Span(dim.replace("_", " "), className="small text-muted")],
                    style={"width": "160px"}),
            html.Td([
                html.Div(f.get("observation", ""), className="small"),
                html.Div(f.get("recommendation", ""), className="small text-muted mt-1"),
            ]),
        ], style=row_style))

    findings_table = dbc.Table(
        [html.Tbody(finding_rows)], size="sm", className="mb-3", borderless=True,
    ) if finding_rows else html.Div()

    next_card = dbc.Card([
        dbc.CardHeader(html.Strong("Next Steps")),
        dbc.CardBody(html.Ol(
            [html.Li(s, className="small mb-1") for s in next_steps],
            className="mb-0 ps-3",
        )),
    ], className="mb-3") if next_steps else html.Div()

    exp_card = dbc.Card([
        dbc.CardHeader(html.Strong("Suggested Experiments")),
        dbc.CardBody(html.Ul(
            [html.Li(s, className="small mb-1") for s in suggested],
            className="mb-0",
        )),
    ]) if suggested else html.Div()

    return html.Div([hero, findings_table, next_card, exp_card])


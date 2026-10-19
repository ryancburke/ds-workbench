"""Workbench renderers for Causal Inference results (result dict → Dash components)."""
import dash_bootstrap_components as dbc
from dash import html

_VERDICT = {
    "valid":          ("success", "bi-check-circle-fill",         "VALID — the estimate can be acted on"),
    "caution":        ("warning", "bi-exclamation-triangle-fill", "CAUTION — usable with the caveats below"),
    "not_identified": ("danger",  "bi-x-octagon-fill",            "NOT IDENTIFIED — don't act on any estimate yet"),
}
_SEV_COLOR = {"critical": "danger", "warning": "warning", "info": "secondary"}
_SEV_ORDER = {"critical": 0, "warning": 1, "info": 2}


def _effect_line(result: dict) -> html.Div:
    est = result.get("estimate") or {}
    if est.get("effect") is None:
        return html.Div()
    excluded = est.get("excluded_covariates") or []
    detail = (f"95% CI {est['ci_low']:.4g} to {est['ci_high']:.4g} · p = {est['p_value']:.3g} · "
              f"{est.get('method', '').replace('_', ' ')}"
              + (f" · not adjusting for {', '.join(excluded)}" if excluded else ""))
    if result.get("verdict") == "not_identified":
        head = html.Span([html.Strong("Not a causal estimate"), f" (for reference only): {est['effect']:+.4g}"])
    else:
        head = html.Strong(f"Effect: {est['effect']:+.4g}", style={"fontSize": "1.05rem"})
    return html.Div([head, html.Span(f"  ({detail})", className="small text-muted")], className="mb-2")


def _ci_render(result: dict) -> html.Div:
    if result.get("error"):
        return dbc.Alert(result["error"], color="danger", className="py-2")
    color, icon, label = _VERDICT.get(result.get("verdict"), ("secondary", "bi-question-circle", "UNKNOWN"))
    design = result.get("design", "").replace("_", " ")
    strategy = result.get("identification_strategy", "").replace("_", " ")
    hero = dbc.Alert([
        html.Strong([html.I(className=f"bi {icon} me-2"), label], className="d-block mb-1"),
        html.Span([html.Strong("Design: "), design, "  ·  ", html.Strong("Identification: "), strategy],
                  className="small"),
    ], color=color, className="py-2 mb-2")

    findings = sorted(result.get("findings", []), key=lambda f: _SEV_ORDER.get(f.get("severity"), 3))
    rows = [html.Tr([
        html.Td(dbc.Badge(f.get("severity", "").upper(), color=_SEV_COLOR.get(f.get("severity"), "secondary"),
                          style={"fontSize": "0.65rem"})),
        html.Td(f.get("check", "").replace("_", " "), className="small"),
        html.Td(html.Code(f.get("column") or "—", style={"fontSize": "0.75rem"})),
        html.Td([html.Div(f.get("evidence", ""), className="small"),
                 html.Div(f.get("recommendation", ""), className="small text-muted")]),
    ]) for f in findings]
    findings_table = dbc.Table(
        [html.Thead(html.Tr([html.Th("Sev"), html.Th("Check"), html.Th("Column"), html.Th("Evidence / what to do")])),
         html.Tbody(rows)], size="sm", hover=True, responsive=True, className="mb-2") if rows else html.Div()

    balance = (result.get("evidence") or {}).get("balance", [])[:6]
    balance_table = dbc.Table(
        [html.Thead(html.Tr([html.Th("Covariate"), html.Th("|SMD|"), html.Th("|r| with outcome"),
                             html.Th("Adjusted effect without it")])),
         html.Tbody([html.Tr([html.Td(html.Code(b["covariate"], style={"fontSize": "0.75rem"})),
                              html.Td(f"{b['abs_smd']:.3f}", className="text-warning" if b["abs_smd"] > 0.1 else ""),
                              html.Td(f"{b['abs_corr_with_outcome']:.3f}"),
                              html.Td(f"{b['adjusted_effect_without_it']:+.4g}"
                                      if b.get("adjusted_effect_without_it") is not None else "—")])
                     for b in balance])],
        size="sm", className="mb-0") if balance else None

    lists = []
    if result.get("assumptions"):
        lists.append(html.Div([html.Strong("Relies on: ", className="small"),
                               html.Ul([html.Li(a, className="small") for a in result["assumptions"]], className="mb-1")]))
    if result.get("next_steps"):
        lists.append(html.Div([html.Strong("Next steps: ", className="small"),
                               html.Ul([html.Li(s, className="small") for s in result["next_steps"]], className="mb-1")]))

    return html.Div([
        hero,
        _effect_line(result),
        html.P(result.get("summary", ""), className="small mb-2"),
        findings_table,
        *lists,
        dbc.Accordion([dbc.AccordionItem(balance_table, title="Covariate balance")],
                      start_collapsed=True, className="mt-2") if balance_table is not None else html.Div(),
    ])

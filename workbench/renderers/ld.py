"""Workbench renderers for Leakage Detector results (result dict → Dash components)."""
import dash_bootstrap_components as dbc
from dash import html

from workbench.constants import DQ_SEV_COLOR
from workbench.renderers._safe import safe_render


# ── Leakage Detector helpers ──────────────────────────────────────────────────

_LD_VERDICT_COLOR = {"leakage_detected": "danger", "suspicious": "warning", "clean": "success"}
_LD_RISK_COLOR    = {"high": "danger", "medium": "warning", "low": "info", "none": "success"}
_LD_SEV_COLOR     = {"critical": "danger", "warning": "warning", "info": "info"}
_LD_ROW_STYLE     = {"critical": {"borderLeft": "3px solid #dc3545"},
                     "warning":  {"borderLeft": "3px solid #ffc107"}}


def _ld_verdict_hero(result: dict) -> dbc.Alert:
    verdict  = result.get("verdict", "clean")
    risk     = result.get("risk_level", "none")
    findings = result.get("findings", [])
    critical = [f for f in findings if f.get("severity") == "critical"]
    color    = _LD_VERDICT_COLOR.get(verdict, "secondary")
    label_map = {"leakage_detected": "LEAKAGE DETECTED", "suspicious": "SUSPICIOUS", "clean": "CLEAN"}
    icon_map  = {"leakage_detected": "bi-x-octagon-fill", "suspicious": "bi-exclamation-circle-fill",
                 "clean": "bi-check-circle-fill"}
    label = label_map.get(verdict, verdict.replace("_", " ").upper())
    icon  = icon_map.get(verdict, "bi-circle")
    children = [
        html.Strong([html.I(className=f"bi {icon} me-2"), f"Leakage: {label}  "],
                    className="d-inline"),
        dbc.Badge(f"risk: {risk}", color=_LD_RISK_COLOR.get(risk, "secondary"),
                  className="ms-1", style={"fontSize": "0.75rem"}),
    ]
    if critical:
        children.append(html.Ul(
            [html.Li(f.get("description", "")[:80], className="small") for f in critical[:3]],
            className="mb-0 mt-1 ps-3",
        ))
    return dbc.Alert(children, color=color, className="py-2 mb-3")


def _ld_top_blockers(findings: list) -> dbc.Card | None:
    critical = [f for f in findings if f.get("severity") == "critical"]
    if not critical:
        return None
    rows = []
    for f in critical[:5]:
        corr = f.get("correlation_with_target")
        rows.append(html.Tr([
            html.Td(html.Code(f.get("column", "—"), style={"fontSize": "0.78rem"})),
            html.Td(f.get("leakage_type", "").replace("_", " ").title(),
                    className="text-muted small"),
            html.Td(f"{corr:.3f}" if corr is not None else "—",
                    className="small font-monospace"),
            html.Td(f.get("description", ""), className="small"),
            html.Td(f.get("recommendation", ""), className="small text-muted"),
        ], style={"borderLeft": "3px solid #dc3545"}))
    return dbc.Card([
        dbc.CardHeader([
            html.I(className="bi bi-exclamation-octagon-fill text-danger me-2"),
            html.Strong("Critical Leakage — Remove Before Training"),
            html.Span(f" ({len(critical)} feature{'s' if len(critical) != 1 else ''})",
                      className="text-muted small ms-2"),
        ]),
        dbc.CardBody(
            dbc.Table(
                [html.Thead(html.Tr([html.Th("Column"), html.Th("Type"),
                                     html.Th("Corr"), html.Th("Evidence"), html.Th("Action")]))] +
                [html.Tbody(rows)],
                hover=True, responsive=True, size="sm",
            ),
            className="p-0",
        ),
    ], className="mb-3", style={"borderColor": "#dc354540"})


def _ld_dq_xref(ld_findings: list, dq_findings: list) -> dbc.Card | None:
    if not ld_findings or not dq_findings:
        return None
    dq_map = {f.get("column", ""): f for f in dq_findings if f.get("column")}
    matches = [
        (f, dq_map[f["column"]])
        for f in ld_findings
        if f.get("column") in dq_map
    ]
    if not matches:
        return None
    rows = []
    for ld_f, dq_f in matches[:6]:
        ld_sev = ld_f.get("severity", "info")
        dq_sev = dq_f.get("severity", "info")
        rows.append(html.Tr([
            html.Td(html.Code(ld_f.get("column", ""), style={"fontSize": "0.78rem"})),
            html.Td(dbc.Badge(f"LD: {ld_sev}", color=_LD_SEV_COLOR.get(ld_sev, "secondary"),
                              className="me-1 small")),
            html.Td(dbc.Badge(f"DQ: {dq_sev}", color=DQ_SEV_COLOR.get(dq_sev, "secondary"),
                              className="small")),
            html.Td(ld_f.get("description", "")[:60], className="small"),
            html.Td(dq_f.get("description", "")[:60], className="small text-muted"),
        ]))
    return dbc.Card([
        dbc.CardHeader([
            html.I(className="bi bi-link-45deg me-2 text-warning"),
            html.Strong("DQ ↔ Leakage Cross-Reference"),
            html.Span(" — these columns appear in both scans", className="text-muted small ms-2"),
        ]),
        dbc.CardBody(
            dbc.Table(
                [html.Thead(html.Tr([html.Th("Column"), html.Th("Leakage"), html.Th("DQ"),
                                     html.Th("Leakage Finding"), html.Th("DQ Finding")]))] +
                [html.Tbody(rows)],
                hover=True, responsive=True, size="sm",
            ),
            className="p-0",
        ),
    ], className="mb-3", style={"borderColor": "#ffc10740"})


@safe_render
def _ld_render(result: dict, dq_findings: list | None = None) -> html.Div:
    findings = result.get("findings", [])
    summary  = result.get("summary", "")

    sorted_f = sorted(findings,
                      key=lambda f: {"critical": 0, "warning": 1, "info": 2}.get(
                          f.get("severity", "info"), 2))
    rows = []
    for f in sorted_f:
        sev  = f.get("severity", "info")
        corr = f.get("correlation_with_target")
        rows.append(html.Tr([
            html.Td(dbc.Badge(sev.upper(), color=_LD_SEV_COLOR.get(sev, "secondary"))),
            html.Td(html.Code(f.get("column", "—"), style={"fontSize": "0.78rem"})),
            html.Td(f.get("leakage_type", "").replace("_", " ").title(),
                    className="text-muted small"),
            html.Td(f"{corr:.3f}" if corr is not None else "—",
                    className="small font-monospace"),
            html.Td([
                html.Div(f.get("description", ""), className="small"),
                html.Div(f.get("recommendation", ""), className="small text-muted mt-1"),
            ]),
        ], style=_LD_ROW_STYLE.get(sev, {})))

    top_blockers = _ld_top_blockers(findings)
    dq_xref      = _ld_dq_xref(findings, dq_findings or [])

    return html.Div([
        _ld_verdict_hero(result),
        top_blockers or html.Div(),
        dq_xref      or html.Div(),
        html.P(summary, className="small mb-3") if summary else html.Div(),
        dbc.Card([
            dbc.CardHeader(html.Strong(f"All Findings ({len(findings)})")),
            dbc.CardBody(
                dbc.Table(
                    [html.Thead(html.Tr([html.Th("Sev"), html.Th("Column"), html.Th("Type"),
                                         html.Th("Corr"), html.Th("Detail")]))] +
                    [html.Tbody(rows)],
                    hover=True, responsive=True, size="sm",
                ) if rows else html.P("No leakage detected — dataset looks clean.",
                                      className="text-success mb-0"),
                className="p-0",
            ),
        ]),
    ])


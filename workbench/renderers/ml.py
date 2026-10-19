"""Workbench renderers for MLOps Review results (result dict → Dash components)."""
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import dcc, html

from workbench.constants import (
    DQ_SEV_COLOR, ML_CAT_ICON, ML_FEAS_COLOR, ML_FEAS_ICON, ML_SEV_COLOR,
)
from workbench.renderers._safe import safe_render


# ── MLOps helpers ─────────────────────────────────────────────────────────────

def _ml_gauge(score: int) -> go.Figure:
    color = "#dc3545" if score < 40 else "#ffc107" if score < 70 else "#198754"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        gauge={"axis": {"range": [0, 100], "tickcolor": "#888"}, "bar": {"color": color},
               "bgcolor": "#1e1e2e",
               "steps": [{"range": [0, 40],  "color": "#3d1515"},
                         {"range": [40, 70],  "color": "#3d3215"},
                         {"range": [70, 100], "color": "#153d1e"}]},
        title={"text": "Production Readiness", "font": {"color": "#ccc", "size": 13}},
        number={"font": {"color": color, "size": 48}},
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=180,
                      margin={"t": 40, "b": 0, "l": 10, "r": 10})
    return fig


def _ml_feasibility_card(feasibility: dict) -> dbc.Card:
    verdict  = feasibility.get("verdict", "marginal")
    color    = ML_FEAS_COLOR.get(verdict, "secondary")
    icon_cls = ML_FEAS_ICON.get(verdict, "bi-circle")
    label    = {"ready": "Ready to Model", "marginal": "Marginal", "insufficient": "Not Ready"}
    gaps     = feasibility.get("key_gaps", [])
    return dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Div("Data & Feature Feasibility", className="text-muted small mb-1"),
                    html.Div([
                        html.I(className=f"bi {icon_cls} me-2 fs-5"),
                        dbc.Badge(label.get(verdict, verdict.title()), color=color,
                                  className="p-2", style={"fontSize": "0.9rem"}),
                    ], className="d-flex align-items-center"),
                ], width="auto"),
                dbc.Col(html.P(feasibility.get("assessment", ""), className="mb-0 small mt-1")),
            ], align="center", className="mb-2"),
            html.Ul([html.Li(g, className="small mb-1") for g in gaps],
                    className="mb-0 mt-2") if gaps else html.Div(),
        ]),
    ], color=color, outline=True, className="mb-3", style={"borderWidth": "2px"})


def _ml_findings_table(findings: list) -> list:
    blocking = [f for f in findings if f.get("severity") in ("critical", "warning")]
    advisory = [f for f in findings if f.get("severity") == "suggestion"]
    blocking = sorted(blocking, key=lambda f: {"critical": 0, "warning": 1}.get(f.get("severity"), 1))
    rows = []
    for section_label, section_icon, section_f in [
        ("Blocking Issues", "bi-exclamation-octagon-fill text-danger", blocking),
        ("Advisory",        "bi-lightbulb text-info",                  advisory),
    ]:
        if not section_f:
            continue
        rows.append(html.Tr(html.Td(
            [html.I(className=f"bi {section_icon} me-2"), html.Strong(section_label)],
            colSpan=3, className="small py-1",
            style={"background": "rgba(255,255,255,0.04)",
                   "borderTop": "1px solid rgba(255,255,255,0.1)"},
        )))
        for f in section_f:
            sev    = f.get("severity", "suggestion")
            border = {"critical": "3px solid #dc3545", "warning": "3px solid #ffc107"}.get(sev, "")
            rows.append(html.Tr([
                html.Td(dbc.Badge(sev.upper(), color=ML_SEV_COLOR.get(sev, "secondary"))),
                html.Td([html.I(className=f"bi {ML_CAT_ICON.get(f.get('category',''),'bi-dot')} me-1"),
                         f.get("category", "").replace("_", " ").title()],
                        className="text-muted small"),
                html.Td([
                    html.Div(f.get("description", ""), className="small"),
                    html.Code(f.get("line_reference", ""), className="d-block mt-1",
                              style={"fontSize": "0.72rem", "color": "#aaa"})
                    if f.get("line_reference") else html.Div(),
                    html.Div(f.get("fix", ""), className="small text-muted mt-1"),
                ]),
            ], style={"borderLeft": border} if border else {}))
    return rows


def _ml_verdict_hero(result: dict) -> dbc.Alert:
    score    = result.get("readiness_score", 0)
    findings = result.get("findings", [])
    critical = [f for f in findings if f.get("severity") == "critical"]
    if critical or score < 40:
        color, label, icon = "danger",  "NOT READY",        "bi-x-octagon-fill"
    elif score < 70:
        color, label, icon = "warning", "NEEDS REVIEW",     "bi-exclamation-circle-fill"
    else:
        color, label, icon = "success", "PRODUCTION READY", "bi-check-circle-fill"
    children = [
        html.Strong([html.I(className=f"bi {icon} me-2"), f"MLOps: {label}  "],
                    className="d-inline"),
        dbc.Badge(f"{score}/100", color=color, className="ms-1", style={"fontSize": "0.75rem"}),
    ]
    if critical:
        children.append(html.Ul(
            [html.Li(f.get("description", "")[:80], className="small") for f in critical[:3]],
            className="mb-0 mt-1 ps-3",
        ))
    return dbc.Alert(children, color=color, className="py-2 mb-3")


@safe_render
def _ml_render(result: dict, session_store: dict | None = None) -> html.Div:
    findings    = result.get("findings", [])
    counts      = {s: sum(1 for f in findings if f.get("severity") == s)
                   for s in ("critical", "warning", "suggestion")}
    snippet     = result.get("refactored_snippet")
    feasibility = result.get("feasibility")
    rows        = _ml_findings_table(findings)

    # Cross-reference DQ columns into MLOps findings
    dq_xref = None
    if session_store:
        dq_runs = [r for r in session_store.get("data_quality", []) if r.get("status") == "complete"]
        if dq_runs:
            dq_cols: dict[str, tuple[str, str]] = {}
            for run in dq_runs:
                for f in run.get("result", {}).get("findings", []):
                    col = f.get("column", "")
                    if col and col not in dq_cols:
                        dq_cols[col] = (f.get("severity", "info"), f.get("description", ""))
            matches = []
            for f in findings:
                haystack = (f.get("description","") + " " + f.get("fix","")).lower()
                for col, (sev, dq_desc) in dq_cols.items():
                    if col.lower() in haystack:
                        matches.append((col, sev, dq_desc[:60], f.get("description","")[:70]))
            if matches:
                xref_rows = [html.Tr([
                    html.Td(html.Code(col, style={"fontSize": "0.78rem"})),
                    html.Td(dbc.Badge(sev.upper(), color=DQ_SEV_COLOR.get(sev, "secondary"),
                                      className="small")),
                    html.Td(dq_d, className="small text-muted"),
                    html.Td(ml_d, className="small"),
                ]) for col, sev, dq_d, ml_d in matches[:5]]
                dq_xref = dbc.Card([
                    dbc.CardHeader([
                        html.I(className="bi bi-link-45deg me-2 text-warning"),
                        html.Strong("DQ Cross-Reference"),
                        html.Span(" — columns flagged in Data Quality also appear in MLOps findings",
                                  className="text-muted small ms-2"),
                    ]),
                    dbc.CardBody(
                        dbc.Table([
                            html.Thead(html.Tr([html.Th("Column"), html.Th("DQ Severity"),
                                                html.Th("DQ Issue"), html.Th("MLOps Finding")])),
                            html.Tbody(xref_rows),
                        ], hover=True, responsive=True, size="sm"),
                        className="p-0",
                    ),
                ], className="mb-3", style={"borderColor": "#ffc10740"})

    return html.Div([
        _ml_verdict_hero(result),
        _ml_feasibility_card(feasibility) if feasibility else html.Div(),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=_ml_gauge(result.get("readiness_score", 0)),
                              config={"displayModeBar": False}), width=4),
            dbc.Col([
                html.P(result.get("summary", ""), className="lead mt-3 small"),
                html.Div([
                    dbc.Badge([html.I(className="bi bi-x-circle me-1"),
                               f"{counts['critical']} Critical"],
                              color="danger", className="me-2 p-2"),
                    dbc.Badge([html.I(className="bi bi-exclamation-triangle me-1"),
                               f"{counts['warning']} Warnings"],
                              color="warning", text_color="dark", className="me-2 p-2"),
                    dbc.Badge([html.I(className="bi bi-lightbulb me-1"),
                               f"{counts['suggestion']} Suggestions"],
                              color="info", text_color="dark", className="p-2"),
                ], className="mt-3"),
            ], width=8),
        ], className="mb-3"),
        dq_xref or html.Div(),
        dbc.Card([
            dbc.CardHeader(html.Strong("Findings")),
            dbc.CardBody(
                dbc.Table(
                    [html.Thead(html.Tr([html.Th("Severity"), html.Th("Category"),
                                        html.Th("Issue & Fix")]))] +
                    [html.Tbody(rows)],
                    hover=True, responsive=True, size="sm",
                ) if rows else html.P("Code looks production-ready!", className="text-success mb-0"),
                className="p-0",
            ),
        ], className="mb-3"),
        dbc.Card([
            dbc.CardHeader(html.Strong("Quick Win — Refactored Snippet")),
            dbc.CardBody(html.Pre(snippet, className="mb-0", style={"fontSize": "0.78rem"})),
        ]) if snippet else html.Div(),
    ])


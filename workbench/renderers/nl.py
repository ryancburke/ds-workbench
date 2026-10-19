"""Workbench renderers for NL → PySpark results (result dict → Dash components)."""
import dash_bootstrap_components as dbc
from dash import html


# ── NL → PySpark helpers ──────────────────────────────────────────────────────

def _nl_render(result: dict, dq_critical: bool = False) -> html.Div:
    assumptions = result.get("assumptions", [])
    perf_notes  = result.get("performance_notes", [])
    explanation = result.get("explanation", "")

    # Extract first sentence as the one-line intent
    _intent = explanation.split(".")[0].strip() + "." if explanation else ""

    # ── Query intent hero ────────────────────────────────────────────────────
    intent_children = [
        html.Strong([html.I(className="bi bi-chat-square-text me-2"), "Query Intent: "],
                    className="d-inline"),
        html.Span(_intent, className="small"),
    ]
    if dq_critical:
        intent_children.append(html.Div(
            [html.I(className="bi bi-shield-exclamation me-1 text-danger"),
             html.Span("Critical DQ findings in session — results may be unreliable.",
                       className="small text-danger")],
            className="mt-1",
        ))
    if assumptions:
        intent_children.append(html.Ul(
            [html.Li([html.I(className="bi bi-lightbulb me-1 text-warning"), a],
                     className="small") for a in assumptions],
            className="mb-0 mt-1 ps-3",
        ))
    intent_card = dbc.Alert(intent_children,
                            color="danger" if dq_critical else "secondary",
                            className="py-2 mb-3")

    return html.Div([
        intent_card,
        dbc.Card([
            dbc.CardHeader(html.Strong("Spark SQL")),
            dbc.CardBody(html.Pre(result.get("sql_query", ""), className="mb-0",
                                  style={"fontSize": "0.8rem"})),
        ], className="mb-3"),
        dbc.Card([
            dbc.CardHeader(html.Strong("PySpark (DataFrame API)")),
            dbc.CardBody(html.Pre(result.get("pyspark_code", ""), className="mb-0",
                                  style={"fontSize": "0.8rem"})),
        ], className="mb-3"),
        dbc.Card([
            dbc.CardHeader(html.Strong("Full Explanation")),
            dbc.CardBody(html.P(explanation, className="mb-0 small")),
        ], className="mb-3"),
        dbc.Card([
            dbc.CardHeader(html.Strong("Performance Notes")),
            dbc.CardBody(html.Ul([html.Li(n, className="small") for n in perf_notes],
                                 className="mb-0")),
        ], className="mb-3") if perf_notes else html.Div(),
    ])


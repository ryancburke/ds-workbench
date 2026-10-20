import time
from datetime import datetime

import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html
from dash.exceptions import PreventUpdate

dash.register_page(__name__, path="/mlops-review", name="MLOps Review", order=5)

SEVERITY_COLOR   = {"critical": "danger", "warning": "warning", "suggestion": "info"}
FEASIBILITY_COLOR = {"ready": "success", "marginal": "warning", "insufficient": "danger"}
FEASIBILITY_ICON  = {"ready": "bi-check-circle-fill text-success",
                     "marginal": "bi-exclamation-circle text-warning",
                     "insufficient": "bi-x-circle-fill text-danger"}
CATEGORY_ICON  = {
    "hardcoded_config":   "bi-lock",
    "missing_logging":    "bi-journal-x",
    "no_error_handling":  "bi-shield-x",
    "inefficient_spark":  "bi-lightning",
    "reproducibility":    "bi-arrow-repeat",
    "missing_tests":      "bi-check2-square",
    "code_structure":     "bi-code-square",
    "security":           "bi-incognito",
    "documentation":      "bi-file-text",
}

PLACEHOLDER = """\
from pyspark.sql import SparkSession
import pandas as pd

spark = SparkSession.builder.getOrCreate()

df = spark.read.parquet("/mnt/data/transactions/2024-01-15/")
df = df.filter(df.date >= "2023-12-15")
df = df.withColumn("revenue", df.amount * 1.082)

result = df.toPandas()
for i, row in result.iterrows():
    result.at[i, "category"] = "high" if row["revenue"] > 500 else "low"

result.to_csv("/tmp/output.csv")
print("done")
"""


def layout():
    return html.Div([
        html.H4([html.I(className="bi bi-gear-fill me-2"), "MLOps Readiness Review"],
                className="mb-1"),
        html.P("Paste notebook code — get a production readiness report before handing off.",
               className="text-muted mb-4"),

        dbc.Card(dbc.CardBody([
            html.Label("Notebook Code"),
            dbc.Textarea(
                id="mlops-code",
                placeholder=PLACEHOLDER,
                rows=12,
                style={"fontFamily": "monospace", "fontSize": "0.78rem",
                       "backgroundColor": "#12122a", "color": "#cdd6f4",
                       "border": "1px solid #3a3a5a"},
            ),
            dbc.Button([html.I(className="bi bi-play-fill me-1"), "Review"],
                       id="mlops-run-btn", color="danger", className="w-100 mt-3"),
        ]), className="mb-4"),

        dcc.Loading(html.Div(id="mlops-runs-container"), type="circle", color="#dc3545"),
    ])


# ── Result renderer ────────────────────────────────────────────────────────────

def _score_gauge(score: int) -> go.Figure:
    color = "#dc3545" if score < 40 else "#ffc107" if score < 70 else "#198754"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        gauge={"axis": {"range": [0, 100], "tickcolor": "#888"}, "bar": {"color": color},
               "bgcolor": "#1e1e2e",
               "steps": [{"range": [0, 40], "color": "#3d1515"},
                          {"range": [40, 70], "color": "#3d3215"},
                          {"range": [70, 100], "color": "#153d1e"}]},
        title={"text": "Production Readiness", "font": {"color": "#ccc", "size": 13}},
        number={"font": {"color": color, "size": 52}},
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=200,
                      margin={"t": 40, "b": 0, "l": 20, "r": 20})
    return fig


def _feasibility_card(feasibility: dict) -> dbc.Card:
    verdict   = feasibility.get("verdict", "marginal")
    color     = FEASIBILITY_COLOR.get(verdict, "secondary")
    icon_cls  = FEASIBILITY_ICON.get(verdict, "bi-circle")
    label     = {"ready": "Ready to Model", "marginal": "Marginal", "insufficient": "Not Ready"}
    gaps      = feasibility.get("key_gaps", [])

    return dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Div("Data & Feature Feasibility", className="text-muted small mb-1"),
                    html.Div([
                        html.I(className=f"bi {icon_cls} me-2 fs-5"),
                        dbc.Badge(label.get(verdict, verdict.title()),
                                  color=color, className="p-2",
                                  style={"fontSize": "0.9rem"}),
                    ], className="d-flex align-items-center"),
                ], width="auto"),
                dbc.Col([
                    html.P(feasibility.get("assessment", ""), className="mb-0 small mt-1"),
                ]),
            ], align="center", className="mb-2"),
            html.Div([
                html.Ul([html.Li(g, className="small mb-1") for g in gaps],
                        className="mb-0 mt-2"),
            ]) if gaps else html.Div(),
        ]),
    ], color=color, outline=True, className="mb-3",
       style={"borderWidth": "2px"})


def _findings_table(findings: list) -> list:
    """Group findings into Blocking (critical/warning) then Advisory (suggestion), with headers."""
    blocking   = [f for f in findings if f.get("severity") in ("critical", "warning")]
    advisory   = [f for f in findings if f.get("severity") == "suggestion"]
    sev_rank   = {"critical": 0, "warning": 1}
    blocking   = sorted(blocking, key=lambda f: sev_rank.get(f.get("severity"), 1))

    rows = []
    for section_label, section_icon, section_findings in [
        ("Blocking Issues",  "bi-exclamation-octagon-fill text-danger", blocking),
        ("Advisory",         "bi-lightbulb text-info",                  advisory),
    ]:
        if not section_findings:
            continue
        rows.append(html.Tr(
            html.Td(
                [html.I(className=f"bi {section_icon} me-2"), html.Strong(section_label)],
                colSpan=3,
                className="small py-1",
                style={"background": "rgba(255,255,255,0.04)",
                       "borderTop": "1px solid rgba(255,255,255,0.1)"},
            )
        ))
        for f in section_findings:
            sev = f.get("severity", "suggestion")
            border = {"critical": "3px solid #dc3545", "warning": "3px solid #ffc107"}.get(sev, "")
            rows.append(html.Tr([
                html.Td(dbc.Badge(sev.upper(), color=SEVERITY_COLOR.get(sev, "secondary"))),
                html.Td([html.I(className=f"bi {CATEGORY_ICON.get(f['category'],'bi-dot')} me-1"),
                         f["category"].replace("_", " ").title()],
                        className="text-muted small"),
                html.Td([
                    html.Div(f["description"], className="small"),
                    html.Code(f.get("line_reference", ""), className="d-block mt-1",
                              style={"fontSize": "0.72rem", "color": "#aaa"})
                    if f.get("line_reference") else html.Div(),
                    html.Div(f["fix"], className="small text-muted mt-1"),
                ]),
            ], style={"borderLeft": border} if border else {}))
    return rows


def _render_results(result: dict) -> html.Div:
    findings    = result.get("findings", [])
    counts      = {s: sum(1 for f in findings if f["severity"] == s)
                   for s in ("critical", "warning", "suggestion")}
    snippet     = result.get("refactored_snippet")
    feasibility = result.get("feasibility")
    rows        = _findings_table(findings)

    return html.Div([
        # ── Feasibility — the primary question ────────────────────────────────
        _feasibility_card(feasibility) if feasibility else html.Div(),

        # ── Score + summary ───────────────────────────────────────────────────
        dbc.Row([
            dbc.Col(dcc.Graph(figure=_score_gauge(result.get("readiness_score", 0)),
                              config={"displayModeBar": False}), width=4),
            dbc.Col([
                html.P(result.get("summary", ""), className="lead mt-3"),
                html.Div([
                    dbc.Badge([html.I(className="bi bi-x-circle me-1"), f"{counts['critical']} Critical"],
                              color="danger", className="me-2 p-2"),
                    dbc.Badge([html.I(className="bi bi-exclamation-triangle me-1"), f"{counts['warning']} Warnings"],
                              color="warning", text_color="dark", className="me-2 p-2"),
                    dbc.Badge([html.I(className="bi bi-lightbulb me-1"), f"{counts['suggestion']} Suggestions"],
                              color="info", text_color="dark", className="p-2"),
                ], className="mt-3"),
            ], width=8),
        ], className="mb-3"),

        # ── Findings — blocking before advisory ──────────────────────────────
        dbc.Card([
            dbc.CardHeader(html.Strong("Findings")),
            dbc.CardBody(
                dbc.Table(
                    [html.Thead(html.Tr([html.Th("Severity"), html.Th("Category"), html.Th("Issue & Fix")]))] +
                    [html.Tbody(rows)],
                    hover=True, responsive=True, size="sm",
                ) if rows else html.P("Code looks production-ready!", className="text-success mb-0"),
                className="p-0",
            ),
        ], className="mb-3"),

        # ── Quick win snippet ─────────────────────────────────────────────────
        dbc.Card([
            dbc.CardHeader(html.Strong("Quick Win — Refactored Snippet")),
            dbc.CardBody(html.Pre(snippet, className="mb-0")),
        ]) if snippet else html.Div(),
    ])


# ── Callbacks ──────────────────────────────────────────────────────────────────

@callback(
    Output("runs-store", "data", allow_duplicate=True),
    Input("mlops-run-btn", "n_clicks"),
    State("mlops-code", "value"),
    State("runs-store", "data"),
    prevent_initial_call=True,
)
def run_mlops(_, code, store):
    if not _:  # a click, not a page load: Dash Pages can fire this when the page mounts
        raise PreventUpdate
    from skills.mlops_reviewer import review

    if not code or not code.strip():
        raise PreventUpdate

    label = (code.strip().splitlines()[0] or "notebook")[:40]
    try:
        result = review(code)
        findings = result.get("findings", [])
        run = {
            "id": f"mlops_{int(time.time() * 1000)}",
            "label": label,
            "timestamp": datetime.now().strftime("%H:%M"),
            "status": "complete",
            "result": result,
            "summary": {
                "score":    result.get("readiness_score", 0),
                "critical": sum(1 for f in findings if f["severity"] == "critical"),
                "warning":  sum(1 for f in findings if f["severity"] == "warning"),
            },
        }
    except PreventUpdate:
        raise
    except Exception as exc:
        run = {"id": f"mlops_{int(time.time()*1000)}", "label": label,
               "timestamp": datetime.now().strftime("%H:%M"),
               "status": "error", "error": str(exc), "summary": {}}

    updated = dict(store)
    updated["mlops_review"] = store.get("mlops_review", []) + [run]
    return updated


@callback(
    Output("mlops-runs-container", "children"),
    Input("runs-store", "data"),
)
def render_mlops_runs(store):
    from components.run_card import make_run_card
    runs = store.get("mlops_review", [])
    if not runs:
        return html.P("No runs yet — paste notebook code and click Review.",
                      className="text-muted text-center py-4")
    cards = []
    for run in reversed(runs):
        content = _render_results(run["result"]) if run["status"] == "complete" \
            else dbc.Alert(run.get("error", "Unknown error"), color="danger")
        cards.append(make_run_card(run, "mlops_review", content))
    return cards

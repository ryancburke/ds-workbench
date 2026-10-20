import base64
import io
import time
from datetime import datetime

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html
from dash.exceptions import PreventUpdate

dash.register_page(__name__, path="/data-quality", name="Data Quality", order=1)

SEVERITY_COLOR = {"critical": "danger", "warning": "warning", "info": "info"}


def layout():
    return html.Div([
        html.H4([html.I(className="bi bi-shield-check me-2"), "Data Quality Audit"],
                className="mb-1"),
        html.P("Scan a dataset for nulls, duplicates, type issues, and format inconsistencies.",
               className="text-muted mb-4"),

        # ── Config card ───────────────────────────────────────────────────────
        dbc.Card(dbc.CardBody(
            dbc.Row([
                dbc.Col([
                    html.Label("Data Source"),
                    dbc.Tabs([
                        dbc.Tab([
                            dbc.Select(id="dq-sample-select", className="mt-2", value="orders_dirty",
                                options=[
                                    {"label": "Orders — Dirty  (nulls, mixed-case, duplicates)", "value": "orders_dirty"},
                                    {"label": "Sales — Clean   (product, pricing, discounts)",   "value": "sales_clean"},
                                    {"label": "HR Attrition    (salary outliers, missing scores)","value": "hr_attrition"},
                                    {"label": "E-commerce      (sessions, conversions, channel)", "value": "ecommerce"},
                                ]),
                        ], label="Sample", tab_id="tab-sample"),
                        dbc.Tab([
                            dbc.Input(id="dq-table-input", placeholder="catalog.schema.table_name",
                                      className="mt-2"),
                        ], label="Databricks", tab_id="tab-table"),
                        dbc.Tab([
                            dcc.Upload(id="dq-upload",
                                children=html.Div(["Drop CSV or ", html.A("browse")]),
                                className="upload-zone mt-2",
                                style={"padding": "10px", "textAlign": "center", "fontSize": "0.85rem"}),
                            html.Div(id="dq-upload-filename", className="text-muted small mt-1"),
                        ], label="Upload", tab_id="tab-upload"),
                    ], id="dq-source-tabs", active_tab="tab-sample"),
                ], width=5),
                dbc.Col([
                    html.Label("Sample Size"),
                    dcc.Slider(id="dq-sample-size", min=100, max=5000, step=100, value=1000,
                               marks={100: "100", 1000: "1k", 5000: "5k"},
                               tooltip={"placement": "bottom", "always_visible": False}),
                ], width=4, className="d-flex flex-column justify-content-center"),
                dbc.Col([
                    dbc.Button([html.I(className="bi bi-play-fill me-1"), "Run"],
                               id="dq-run-btn", color="primary", className="w-100 mt-3"),
                ], width=3, className="d-flex align-items-end"),
            ], className="g-3"),
        ), className="mb-4"),

        # ── Runs ──────────────────────────────────────────────────────────────
        dcc.Loading(html.Div(id="dq-runs-container"), type="circle", color="#0d6efd"),
        dcc.Store(id="dq-upload-store"),
    ])


# ── Result renderer ────────────────────────────────────────────────────────────

def _score_gauge(score: int) -> go.Figure:
    color = "#dc3545" if score < 50 else "#ffc107" if score < 75 else "#198754"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        gauge={"axis": {"range": [0, 100], "tickcolor": "#888"}, "bar": {"color": color},
               "bgcolor": "#1e1e2e",
               "steps": [{"range": [0, 50], "color": "#3d1515"},
                          {"range": [50, 75], "color": "#3d3215"},
                          {"range": [75, 100], "color": "#153d1e"}]},
        title={"text": "Quality Score", "font": {"color": "#ccc", "size": 13}},
        number={"font": {"color": color, "size": 52}},
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=200,
                      margin={"t": 40, "b": 0, "l": 20, "r": 20})
    return fig


def _null_chart(profile: dict) -> go.Figure:
    cols = list(profile["columns"].keys())
    pcts = [profile["columns"][c]["null_pct"] for c in cols]
    fig = px.bar(x=pcts, y=cols, orientation="h", labels={"x": "Null %", "y": ""},
                 color=pcts, color_continuous_scale=["#198754", "#ffc107", "#dc3545"],
                 range_color=[0, 100])
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#ccc", coloraxis_showscale=False,
                      height=max(160, len(cols) * 26), title="Null Rates by Column",
                      margin={"t": 40, "b": 10, "l": 120, "r": 10},
                      yaxis={"categoryorder": "total ascending"})
    fig.update_traces(marker_line_width=0)
    return fig


_SEV_RANK = {"critical": 0, "warning": 1, "info": 2}
_SEV_HEADER = {
    "critical": ("bi-exclamation-octagon-fill text-danger", "Critical Issues"),
    "warning":  ("bi-exclamation-triangle text-warning",    "Warnings"),
    "info":     ("bi-info-circle text-info",                "Info"),
}
_ROW_STYLE = {
    "critical": {"borderLeft": "3px solid #dc3545"},
    "warning":  {"borderLeft": "3px solid #ffc107"},
    "info":     {},
}


def _findings_table(findings: list) -> list:
    """Return grouped, severity-sorted table rows with section headers."""
    sorted_f = sorted(findings, key=lambda f: _SEV_RANK.get(f.get("severity", "info"), 2))
    rows, last_sev = [], None
    for f in sorted_f:
        sev = f.get("severity", "info")
        if sev != last_sev:
            icon_cls, label = _SEV_HEADER.get(sev, ("bi-circle", sev.title()))
            rows.append(html.Tr(
                html.Td(
                    [html.I(className=f"bi {icon_cls} me-2"), html.Strong(label)],
                    colSpan=4,
                    className="small py-1",
                    style={"background": "rgba(255,255,255,0.04)", "borderTop": "1px solid rgba(255,255,255,0.1)"},
                )
            ))
            last_sev = sev
        rows.append(html.Tr([
            html.Td(dbc.Badge(sev.upper(), color=SEVERITY_COLOR[sev])),
            html.Td(html.Code(f.get("column", "—"), style={"fontSize": "0.8rem"})),
            html.Td(f["issue_type"].replace("_", " ").title(), className="text-muted small"),
            html.Td([
                html.Div(f["description"], className="small"),
                html.Div(f["recommendation"], className="small text-muted mt-1"),
            ]),
        ], style=_ROW_STYLE.get(sev, {})))
    return rows


def _render_results(result: dict, profile: dict) -> html.Div:
    findings = result.get("findings", [])
    counts = {s: sum(1 for f in findings if f["severity"] == s)
              for s in ("critical", "warning", "info")}
    rows = _findings_table(findings)
    return html.Div([
        dbc.Row([
            dbc.Col(dcc.Graph(figure=_score_gauge(result["overall_score"]),
                              config={"displayModeBar": False}), width=4),
            dbc.Col([
                html.P(result["summary"], className="mt-3"),
                html.Div([
                    dbc.Badge([html.I(className="bi bi-x-circle me-1"), f"{counts['critical']} Critical"],
                              color="danger", className="me-2 p-2"),
                    dbc.Badge([html.I(className="bi bi-exclamation-triangle me-1"), f"{counts['warning']} Warnings"],
                              color="warning", text_color="dark", className="me-2 p-2"),
                    dbc.Badge([html.I(className="bi bi-info-circle me-1"), f"{counts['info']} Info"],
                              color="info", text_color="dark", className="p-2"),
                ]),
            ], width=8),
        ], className="mb-3"),
        dbc.Card([
            dbc.CardHeader(html.Strong("Findings")),
            dbc.CardBody(
                dbc.Table(
                    [html.Thead(html.Tr([html.Th("Sev"), html.Th("Column"),
                                         html.Th("Issue"), html.Th("Detail")]))] +
                    [html.Tbody(rows)],
                    hover=True, responsive=True, size="sm",
                ) if rows else html.P("No issues found.", className="text-success mb-0"),
                className="p-0",
            ),
        ], className="mb-3"),
        dbc.Card([
            dbc.CardHeader(html.Strong("Null Rates")),
            dbc.CardBody(dcc.Graph(figure=_null_chart(profile),
                                   config={"displayModeBar": False})),
        ]),
    ])


# ── Callbacks ──────────────────────────────────────────────────────────────────

@callback(
    Output("dq-upload-store", "data"),
    Output("dq-upload-filename", "children"),
    Input("dq-upload", "contents"),
    State("dq-upload", "filename"),
    prevent_initial_call=True,
)
def store_upload(contents, filename):
    if not contents:
        return None, ""
    _, cs = contents.split(",")
    df = pd.read_csv(io.StringIO(base64.b64decode(cs).decode("utf-8")))
    return df.to_json(date_format="iso", orient="split"), f"{filename} ({len(df):,} rows)"


@callback(
    Output("runs-store", "data", allow_duplicate=True),
    Input("dq-run-btn", "n_clicks"),
    State("dq-source-tabs", "active_tab"),
    State("dq-sample-select", "value"),
    State("dq-table-input", "value"),
    State("dq-upload-store", "data"),
    State("dq-sample-size", "value"),
    State("runs-store", "data"),
    prevent_initial_call=True,
)
def run_dq(_, active_tab, sample_name, table_name, upload_data, sample_size, store):
    if not _:  # a click, not a page load: Dash Pages can fire this when the page mounts
        raise PreventUpdate
    from databricks_client import get_table_sample, load_sample_dataset
    from skills.data_quality import assess, build_profile

    label = "unknown"
    try:
        if active_tab == "tab-sample" and sample_name:
            df, label = load_sample_dataset(sample_name), sample_name
        elif active_tab == "tab-upload" and upload_data:
            df = pd.read_json(io.StringIO(upload_data), orient="split")
            label = "uploaded_dataset"
        elif active_tab == "tab-table" and table_name:
            df = get_table_sample(table_name.strip(), limit=sample_size or 1000)
            label = table_name.strip()
        else:
            raise PreventUpdate

        result = assess(df, table_name=label)
        profile = build_profile(df)
        findings = result.get("findings", [])
        run = {
            "id": f"dq_{int(time.time() * 1000)}",
            "label": label,
            "timestamp": datetime.now().strftime("%H:%M"),
            "status": "complete",
            "result": result,
            "profile": profile,
            "summary": {
                "score": result.get("overall_score", 0),
                "critical": sum(1 for f in findings if f["severity"] == "critical"),
                "warning":  sum(1 for f in findings if f["severity"] == "warning"),
                "info":     sum(1 for f in findings if f["severity"] == "info"),
            },
        }
    except PreventUpdate:
        raise
    except Exception as exc:
        run = {"id": f"dq_{int(time.time()*1000)}", "label": label,
               "timestamp": datetime.now().strftime("%H:%M"),
               "status": "error", "error": str(exc), "summary": {}}

    updated = dict(store)
    updated["data_quality"] = store.get("data_quality", []) + [run]
    return updated


@callback(
    Output("dq-runs-container", "children"),
    Input("runs-store", "data"),
)
def render_dq_runs(store):
    from components.run_card import make_run_card
    runs = store.get("data_quality", [])
    if not runs:
        return html.P("No runs yet — configure a dataset above and click Run.",
                      className="text-muted text-center py-4")
    cards = []
    for run in reversed(runs):
        content = _render_results(run["result"], run.get("profile", {})) \
            if run["status"] == "complete" \
            else dbc.Alert(run.get("error", "Unknown error"), color="danger")
        cards.append(make_run_card(run, "data_quality", content))
    return cards

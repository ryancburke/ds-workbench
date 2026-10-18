import base64
import io
import time
from datetime import datetime

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from dash import Input, Output, State, callback, dcc, html
from dash.exceptions import PreventUpdate

dash.register_page(__name__, path="/eda", name="EDA", order=2)


def layout():
    return html.Div([
        html.H4([html.I(className="bi bi-graph-up me-2"), "EDA Accelerator"],
                className="mb-1"),
        html.P("Auto-generate statistical insights and Plotly charts for any dataset.",
               className="text-muted mb-4"),

        dbc.Card(dbc.CardBody(
            dbc.Row([
                dbc.Col([
                    html.Label("Data Source"),
                    dbc.Tabs([
                        dbc.Tab([
                            dbc.Select(id="eda-sample-select", className="mt-2", value="sales_clean",
                                options=[
                                    {"label": "Sales — Clean   (product, pricing, discounts)",    "value": "sales_clean"},
                                    {"label": "HR Attrition    (salary, tenure, performance)",    "value": "hr_attrition"},
                                    {"label": "E-commerce      (sessions, conversions, channel)", "value": "ecommerce"},
                                    {"label": "Orders — Dirty  (nulls, mixed-case, duplicates)",  "value": "orders_dirty"},
                                ]),
                        ], label="Sample", tab_id="tab-sample"),
                        dbc.Tab([
                            dbc.Input(id="eda-table-input", placeholder="catalog.schema.table_name",
                                      className="mt-2"),
                        ], label="Databricks", tab_id="tab-table"),
                        dbc.Tab([
                            dcc.Upload(id="eda-upload",
                                children=html.Div(["Drop CSV or ", html.A("browse")]),
                                className="upload-zone mt-2",
                                style={"padding": "10px", "textAlign": "center", "fontSize": "0.85rem"}),
                            html.Div(id="eda-upload-filename", className="text-muted small mt-1"),
                        ], label="Upload", tab_id="tab-upload"),
                    ], id="eda-source-tabs", active_tab="tab-sample"),
                ], width=7),
                dbc.Col([
                    dbc.Button([html.I(className="bi bi-play-fill me-1"), "Run"],
                               id="eda-run-btn", color="success", className="w-100 mt-3"),
                ], width=3, className="d-flex align-items-end"),
            ], className="g-3"),
        ), className="mb-4"),

        html.Div(id="eda-session-ctx"),
        html.Div(id="eda-dq-gate"),
        dcc.Loading(html.Div(id="eda-runs-container"), type="circle", color="#198754"),
        dcc.Store(id="eda-upload-store"),
    ])


# ── Session context ────────────────────────────────────────────────────────────

@callback(
    Output("eda-session-ctx", "children"),
    Input("runs-store", "data"),
    prevent_initial_call=True,
)
def eda_session_ctx(store):
    from components.session_context import session_banner
    return session_banner(store)


# ── DQ gate ───────────────────────────────────────────────────────────────────

@callback(
    Output("eda-dq-gate", "children"),
    Input("eda-sample-select",  "value"),
    Input("runs-store",         "data"),
    State("eda-source-tabs",    "active_tab"),
    prevent_initial_call=True,
)
def update_eda_dq_gate(sample_name, store, active_tab):
    from components.dq_gate import dq_gate_banner
    if active_tab != "tab-sample" or not sample_name:
        return html.Div()
    return dq_gate_banner(sample_name, store, check_btn_id="eda-dq-check-btn")


@callback(
    Output("runs-store", "data", allow_duplicate=True),
    Input("eda-dq-check-btn", "n_clicks"),
    State("eda-sample-select", "value"),
    State("runs-store", "data"),
    prevent_initial_call=True,
)
def eda_run_quick_dq(_, sample_name, store):
    import time as _t
    from databricks_client import load_sample_dataset
    from skills.data_quality import assess

    if not sample_name:
        raise PreventUpdate

    df = load_sample_dataset(sample_name)
    result = assess(df, table_name=sample_name)
    findings = result.get("findings", [])
    run = {
        "id":        f"dq_{int(_t.time() * 1000)}",
        "label":     sample_name,
        "timestamp": datetime.now().strftime("%H:%M"),
        "status":    "complete",
        "result":    result,
        "summary": {
            "score":    result.get("overall_score", 0),
            "critical": sum(1 for f in findings if f["severity"] == "critical"),
            "warning":  sum(1 for f in findings if f["severity"] == "warning"),
            "info":     sum(1 for f in findings if f["severity"] == "info"),
        },
    }
    updated = dict(store)
    updated["data_quality"] = store.get("data_quality", []) + [run]
    return updated


# ── Helpers ────────────────────────────────────────────────────────────────────

def _dark(fig, height=280):
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#ccc", height=height,
                      margin={"t": 40, "b": 20, "l": 20, "r": 20})
    return fig


def _build_chart(rec: dict, df: pd.DataFrame) -> dcc.Graph | None:
    ct, x, y, title = rec.get("chart_type"), rec.get("x_column"), rec.get("y_column"), rec.get("title", "")
    try:
        if ct == "histogram" and x and x in df.columns:
            fig = px.histogram(df, x=x, title=title)
        elif ct == "scatter" and x and y and x in df.columns and y in df.columns:
            fig = px.scatter(df, x=x, y=y, title=title, opacity=0.55)
        elif ct == "bar" and x and x in df.columns:
            vc = df[x].value_counts().head(20).reset_index()
            vc.columns = [x, "count"]
            fig = px.bar(vc, x=x, y="count", title=title)
        elif ct == "box" and y and y in df.columns:
            fig = px.box(df, y=y, title=title)
        elif ct == "heatmap":
            num = df.select_dtypes(include="number")
            if len(num.columns) < 2:
                return None
            fig = px.imshow(num.corr().round(3), title=title,
                            color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        elif ct == "violin" and y and y in df.columns:
            fig = px.violin(df, y=y, title=title, box=True)
        else:
            return None
        return dcc.Graph(figure=_dark(fig), config={"displayModeBar": False})
    except Exception:
        return None


def _render_results(result: dict, df_json: str) -> html.Div:
    try:
        df = pd.read_json(io.StringIO(df_json), orient="split")
    except Exception:
        df = pd.DataFrame()

    charts = [c for rec in result.get("recommended_charts", [])
              if (c := _build_chart(rec, df)) is not None]
    targets = result.get("potential_targets", [])

    return html.Div([
        dbc.Card([
            dbc.CardHeader(html.Strong("Overview")),
            dbc.CardBody(html.P(result.get("narrative", ""), className="mb-0")),
        ], className="mb-3"),

        dbc.Card([
            dbc.CardHeader(html.Strong("Key Insights")),
            dbc.CardBody(html.Ul(
                [html.Li(i, className="mb-1") for i in result.get("key_insights", [])],
                className="mb-0",
            )),
        ], className="mb-3"),

        dbc.Card([
            dbc.CardHeader(html.Strong(f"Charts ({len(charts)})")),
            dbc.CardBody(
                dbc.Row([dbc.Col(c, width=6, className="mb-3") for c in charts])
                if charts else html.P("No charts generated.", className="text-muted mb-0"),
            ),
        ], className="mb-3") if charts else html.Div(),

        dbc.Card([
            dbc.CardHeader(html.Strong("Potential ML Targets")),
            dbc.CardBody(
                dbc.Table(
                    [html.Thead(html.Tr([html.Th("Column"), html.Th("Task"), html.Th("Rationale")]))] +
                    [html.Tbody([html.Tr([
                        html.Td(html.Code(t["column"])),
                        html.Td(dbc.Badge(t["task_type"], color="secondary")),
                        html.Td(t["rationale"], className="small"),
                    ]) for t in targets])],
                    hover=True, responsive=True, size="sm",
                ) if targets else html.P("No obvious ML targets identified.", className="text-muted mb-0"),
                className="p-0",
            ),
        ]) if targets else html.Div(),
    ])


# ── Callbacks ──────────────────────────────────────────────────────────────────

@callback(
    Output("eda-upload-store", "data"),
    Output("eda-upload-filename", "children"),
    Input("eda-upload", "contents"),
    State("eda-upload", "filename"),
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
    Input("eda-run-btn", "n_clicks"),
    State("eda-source-tabs", "active_tab"),
    State("eda-sample-select", "value"),
    State("eda-table-input", "value"),
    State("eda-upload-store", "data"),
    State("runs-store", "data"),
    prevent_initial_call=True,
)
def run_eda(_, active_tab, sample_name, table_name, upload_data, store):
    from databricks_client import get_table_sample, load_sample_dataset
    from skills.eda import analyze

    label = "unknown"
    try:
        if active_tab == "tab-sample" and sample_name:
            df, label = load_sample_dataset(sample_name), sample_name
        elif active_tab == "tab-upload" and upload_data:
            df = pd.read_json(io.StringIO(upload_data), orient="split")
            label = "uploaded_dataset"
        elif active_tab == "tab-table" and table_name:
            df = get_table_sample(table_name.strip(), limit=2000)
            label = table_name.strip()
        else:
            raise PreventUpdate

        result = analyze(df, table_name=label)
        run = {
            "id": f"eda_{int(time.time() * 1000)}",
            "label": label,
            "timestamp": datetime.now().strftime("%H:%M"),
            "status": "complete",
            "result": result,
            "df_json": df.to_json(date_format="iso", orient="split"),
            "summary": {
                "charts":   len(result.get("recommended_charts", [])),
                "insights": len(result.get("key_insights", [])),
                "targets":  len(result.get("potential_targets", [])),
            },
        }
    except PreventUpdate:
        raise
    except Exception as exc:
        run = {"id": f"eda_{int(time.time()*1000)}", "label": label,
               "timestamp": datetime.now().strftime("%H:%M"),
               "status": "error", "error": str(exc), "summary": {}}

    updated = dict(store)
    updated["eda"] = store.get("eda", []) + [run]
    return updated


@callback(
    Output("eda-runs-container", "children"),
    Input("runs-store", "data"),
)
def render_eda_runs(store):
    from components.run_card import make_run_card
    runs = store.get("eda", [])
    if not runs:
        return html.P("No runs yet — configure a dataset above and click Run.",
                      className="text-muted text-center py-4")
    cards = []
    for run in reversed(runs):
        content = _render_results(run["result"], run.get("df_json", "{}")) \
            if run["status"] == "complete" \
            else dbc.Alert(run.get("error", "Unknown error"), color="danger")
        cards.append(make_run_card(run, "eda", content))
    return cards

"""Workbench callbacks for the NL → PySpark section: run the skill and render its runs."""
import io
import time
from datetime import datetime

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import callback, html, Input, Output, State
from dash.exceptions import PreventUpdate

from workbench.renderers.nl import _nl_render


# ── NL → PySpark ──────────────────────────────────────────────────────────────

@callback(
    Output("wb-nl-tables", "data"),
    Input("wb-nl-dataset", "value"),
    prevent_initial_call=True,
)
def wb_nl_load_tables(dataset):
    from databricks_client import load_sample_dataset

    if not dataset:
        raise PreventUpdate
    df = load_sample_dataset(dataset)
    return {dataset: df.to_json(date_format="iso", orient="split")}


@callback(
    Output("runs-store", "data", allow_duplicate=True),
    Input("wb-nl-run-btn",  "n_clicks"),
    State("wb-nl-question", "value"),
    State("wb-nl-tables",   "data"),
    State("wb-nl-dataset",  "value"),
    State("runs-store",     "data"),
    prevent_initial_call=True,
)
def wb_run_nl(_, question, tables, dataset, store):
    from skills.nl_to_spark import generate

    if not question or not question.strip():
        raise PreventUpdate

    session = store.get("session") or {}
    context = session.get("problem_statement", "")

    # Load table if store is empty
    if not tables and dataset:
        from databricks_client import load_sample_dataset
        df     = load_sample_dataset(dataset)
        tables = {dataset: df.to_json(date_format="iso", orient="split")}

    if not tables:
        raise PreventUpdate

    label  = (question[:40] + "…") if len(question) > 40 else question
    tables_df = {name: pd.read_json(io.StringIO(js), orient="split")
                 for name, js in tables.items()}
    result = generate(question, tables_df, context=context)
    sql    = result.get("sql_query", "")
    run = {
        "id":        f"nl_{int(time.time() * 1000)}",
        "label":     label,
        "timestamp": datetime.now().strftime("%H:%M"),
        "status":    "complete",
        "result":    result,
        "summary":   {"sql_preview": sql.split("\n")[0] if sql else ""},
    }
    updated = dict(store)
    updated["nl_to_spark"] = store.get("nl_to_spark", []) + [run]
    return updated


@callback(
    Output("wb-nl-results", "children"),
    Input("runs-store", "data"),
)
def wb_render_nl(store):
    from components.run_card import make_run_card

    runs = store.get("nl_to_spark", [])
    if not runs:
        return html.P("No runs yet — enter a question and click Run.",
                      className="text-muted small py-3")

    completed     = [r for r in runs if r.get("status") == "complete"]
    compare_panel = html.Div()
    if len(completed) >= 2:
        r1, r2 = completed[-2], completed[-1]
        compare_panel = dbc.Card([
            dbc.CardHeader([html.I(className="bi bi-columns-gap me-2"),
                            html.Strong("Last 2 Queries — Side by Side")]),
            dbc.CardBody(dbc.Row([
                dbc.Col([
                    html.Div(html.Code(r1.get("label",""), className="small text-info"),
                             className="mb-1"),
                    html.Pre(r1["result"].get("sql_query",""), className="mb-0",
                             style={"fontSize": "0.72rem", "maxHeight": "180px",
                                    "overflow": "auto", "background": "#12122a",
                                    "padding": "0.5rem", "borderRadius": "4px"}),
                ], width=6),
                dbc.Col([
                    html.Div(html.Code(r2.get("label",""), className="small text-info"),
                             className="mb-1"),
                    html.Pre(r2["result"].get("sql_query",""), className="mb-0",
                             style={"fontSize": "0.72rem", "maxHeight": "180px",
                                    "overflow": "auto", "background": "#12122a",
                                    "padding": "0.5rem", "borderRadius": "4px"}),
                ], width=6),
            ])),
        ], className="mb-3", style={"borderColor": "#0dcaf040"})

    dq_critical = any(
        r.get("summary", {}).get("critical", 0) > 0
        for r in store.get("data_quality", [])
        if r.get("status") == "complete"
    )

    cards = [compare_panel]
    for run in reversed(runs):
        if run["status"] == "complete":
            send_btn = dbc.Button(
                [html.I(className="bi bi-gear-fill me-1"), "Send to MLOps Review"],
                id={"type": "nl-to-mlops-btn", "index": run["id"]},
                size="sm", color="danger", outline=True,
                className="mb-3",
                style={"fontSize": "0.75rem"},
            )
            content = html.Div([send_btn, _nl_render(run["result"], dq_critical=dq_critical)])
        else:
            content = dbc.Alert(run.get("error", "Unknown error"), color="danger")
        cards.append(make_run_card(run, "nl_to_spark", content))
    return cards


@callback(
    Output("wb-ml-code",  "value"),
    Output("wb-accordion", "active_item"),
    Input({"type": "nl-to-mlops-btn", "index": dash.ALL}, "n_clicks"),
    State("runs-store", "data"),
    prevent_initial_call=True,
)
def nl_send_to_mlops(n_clicks_list, store):
    if not any(n for n in n_clicks_list if n):
        raise PreventUpdate
    ctx = dash.callback_context
    if not ctx.triggered_id:
        raise PreventUpdate
    run_id = ctx.triggered_id["index"]
    runs   = store.get("nl_to_spark", [])
    run    = next((r for r in runs if r["id"] == run_id), None)
    if not run or run.get("status") != "complete":
        raise PreventUpdate
    code = run["result"].get("pyspark_code", "")
    # Expand the accordion to MLOps Review
    return code, ["dq", "eda", "ms", "ld", "fe", "ja", "nl", "ml"]


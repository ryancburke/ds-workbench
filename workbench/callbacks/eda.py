"""Workbench callbacks for the EDA section: run the skill and render its runs."""
import time
from datetime import datetime

import dash_bootstrap_components as dbc
from dash import callback, html, Input, Output, State
from dash.exceptions import PreventUpdate

from workbench.renderers.eda import _eda_render


# ── EDA ────────────────────────────────────────────────────────────────────────

@callback(
    Output("runs-store", "data", allow_duplicate=True),
    Input("wb-eda-run-btn", "n_clicks"),
    State("wb-eda-dataset", "value"),
    State("runs-store",     "data"),
    prevent_initial_call=True,
)
def wb_run_eda(_, dataset, store):
    if not _:  # a click, not a page load: Dash Pages can fire this when the page mounts
        raise PreventUpdate
    from databricks_client import load_sample_dataset
    from skills.eda import analyze

    if not dataset:
        raise PreventUpdate

    df     = load_sample_dataset(dataset)
    result = analyze(df, table_name=dataset)
    run = {
        "id":        f"eda_{int(time.time() * 1000)}",
        "label":     dataset,
        "timestamp": datetime.now().strftime("%H:%M"),
        "status":    "complete",
        "result":    result,
        "df_json":   df.to_json(date_format="iso", orient="split"),
        "summary": {
            "charts":   len(result.get("recommended_charts", [])),
            "insights": len(result.get("key_insights", [])),
            "targets":  len(result.get("potential_targets", [])),
        },
    }
    updated = dict(store)
    updated["eda"] = store.get("eda", []) + [run]
    return updated


@callback(
    Output("wb-eda-results", "children"),
    Input("runs-store", "data"),
)
def wb_render_eda(store):
    from components.run_card import make_run_card

    runs = store.get("eda", [])
    if not runs:
        return html.P("No runs yet — select a dataset and click Run EDA.",
                      className="text-muted small py-3")
    cards = []
    for run in reversed(runs):
        content = _eda_render(run["result"], run.get("df_json", "{}")) \
            if run["status"] == "complete" \
            else dbc.Alert(run.get("error", "Unknown error"), color="danger")
        cards.append(make_run_card(run, "eda", content))
    return cards


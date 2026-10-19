"""Workbench callbacks for the Feature Engineering section: run the skill and render its runs."""
import time
from datetime import datetime

import dash_bootstrap_components as dbc
from dash import callback, html, Input, Output, State
from dash.exceptions import PreventUpdate

from workbench.renderers.fe import _fe_render


# ── Feature Engineering ────────────────────────────────────────────────────────

@callback(
    Output("wb-fe-target", "options"),
    Output("wb-fe-target", "value"),
    Input("wb-fe-dataset", "value"),
    prevent_initial_call=True,
)
def wb_fe_load_cols(dataset):
    from databricks_client import load_sample_dataset
    if not dataset:
        raise PreventUpdate
    df   = load_sample_dataset(dataset)
    opts = [{"label": c, "value": c} for c in df.columns]
    default = df.columns[-1]
    for col in df.columns:
        if df[col].nunique() <= 10 and df[col].dtype not in ("float64", "float32"):
            default = col
            break
    return opts, default


@callback(
    Output("runs-store", "data", allow_duplicate=True),
    Input("wb-fe-run-btn", "n_clicks"),
    State("wb-fe-dataset",  "value"),
    State("wb-fe-target",   "value"),
    State("runs-store",     "data"),
    prevent_initial_call=True,
)
def wb_run_fe(_, dataset, target_col, store):
    from databricks_client import load_sample_dataset
    from skills.feature_engineering import suggest

    if not dataset or not target_col:
        raise PreventUpdate

    session = store.get("session") or {}
    context = session.get("problem_statement", "")
    label   = f"{dataset} → {target_col}"

    ld_same = [r for r in store.get("leakage_detector", [])
               if r.get("status") == "complete"
               and dataset.lower() in r.get("label", "").lower()]
    leakage_warning  = None
    sequence_warning = False
    if ld_same:
        ld_res = ld_same[-1]["result"]
        if ld_res.get("verdict") == "leakage_detected":
            crits = [f for f in ld_res.get("findings", []) if f.get("severity") == "critical"]
            leakage_warning = crits[0].get("description", "Leakage detected in dataset.") if crits \
                              else "Leakage detected in dataset."
    else:
        sequence_warning = True

    try:
        df     = load_sample_dataset(dataset)
        result = suggest(df, target_col, context=context)
        run = {
            "id":               f"fe_{int(time.time() * 1000)}",
            "label":            label,
            "timestamp":        datetime.now().strftime("%H:%M"),
            "status":           "complete",
            "result":           result,
            "leakage_warning":  leakage_warning,
            "sequence_warning": sequence_warning,
            "summary": {
                "n_features": len(result.get("features", [])),
                "n_dropped":  len(result.get("dropped_cols", [])),
            },
        }
    except Exception as exc:
        run = {"id": f"fe_{int(time.time()*1000)}", "label": label,
               "timestamp": datetime.now().strftime("%H:%M"),
               "status": "error", "error": str(exc), "summary": {}}

    updated = dict(store)
    updated["feature_engineering"] = store.get("feature_engineering", []) + [run]
    return updated


@callback(
    Output("wb-fe-results", "children"),
    Input("runs-store", "data"),
)
def wb_render_fe(store):
    from components.run_card import make_run_card
    runs = store.get("feature_engineering", [])
    if not runs:
        return html.P("Select a dataset and target column, then click Plan Features.",
                      className="text-muted small py-3")
    cards = []
    for run in reversed(runs):
        if run["status"] == "complete":
            content = _fe_render(
                run["result"],
                leakage_warning=run.get("leakage_warning"),
                sequence_warning=run.get("sequence_warning", False),
            )
        else:
            content = dbc.Alert(run.get("error", "Unknown error"), color="danger")
        cards.append(make_run_card(run, "feature_engineering", content))
    return cards


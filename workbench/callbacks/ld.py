"""Workbench callbacks for the Leakage Detector section: run the skill and render its runs."""
import time
from datetime import datetime

import dash_bootstrap_components as dbc
from dash import callback, html, Input, Output, State
from dash.exceptions import PreventUpdate

from workbench.renderers.ld import _ld_render


# ── Leakage Detector ──────────────────────────────────────────────────────────

@callback(
    Output("wb-ld-target", "options"),
    Output("wb-ld-target", "value"),
    Input("wb-ld-dataset", "value"),
    prevent_initial_call=True,
)
def wb_ld_load_cols(dataset):
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
    Input("wb-ld-run-btn", "n_clicks"),
    State("wb-ld-dataset",  "value"),
    State("wb-ld-target",   "value"),
    State("runs-store",     "data"),
    prevent_initial_call=True,
)
def wb_run_ld(_, dataset, target_col, store):
    if not _:  # a click, not a page load: Dash Pages can fire this when the page mounts
        raise PreventUpdate
    from databricks_client import load_sample_dataset
    from skills.leakage_detector import check

    if not dataset or not target_col:
        raise PreventUpdate

    session = store.get("session") or {}
    context = session.get("problem_statement", "")
    label   = f"{dataset} → {target_col}"
    try:
        df     = load_sample_dataset(dataset)
        result = check(df, target_col, context=context)
        run = {
            "id":        f"ld_{int(time.time() * 1000)}",
            "label":     label,
            "timestamp": datetime.now().strftime("%H:%M"),
            "status":    "complete",
            "result":    result,
            "summary": {
                "verdict":     result.get("verdict", "clean"),
                "risk_level":  result.get("risk_level", "none"),
                "n_findings":  len(result.get("findings", [])),
            },
        }
    except Exception as exc:
        run = {"id": f"ld_{int(time.time()*1000)}", "label": label,
               "timestamp": datetime.now().strftime("%H:%M"),
               "status": "error", "error": str(exc), "summary": {}}

    updated = dict(store)
    updated["leakage_detector"] = store.get("leakage_detector", []) + [run]
    return updated


@callback(
    Output("wb-ld-results", "children"),
    Input("runs-store", "data"),
)
def wb_render_ld(store):
    from components.run_card import make_run_card
    runs = store.get("leakage_detector", [])
    if not runs:
        return html.P("Select a dataset and target column, then click Check Leakage.",
                      className="text-muted small py-3")
    dq_runs   = [r for r in store.get("data_quality", []) if r.get("status") == "complete"]
    dq_findings = dq_runs[-1]["result"].get("findings", []) if dq_runs else []
    cards = []
    for run in reversed(runs):
        content = _ld_render(run["result"], dq_findings=dq_findings) \
            if run["status"] == "complete" \
            else dbc.Alert(run.get("error", "Unknown error"), color="danger")
        cards.append(make_run_card(run, "leakage_detector", content))
    return cards


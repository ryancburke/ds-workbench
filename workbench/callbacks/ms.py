"""Workbench callbacks for the Model Selector section: run the skill and render its runs."""
import time
from datetime import datetime

import dash_bootstrap_components as dbc
from dash import callback, html, Input, Output, State
from dash.exceptions import PreventUpdate

from workbench.renderers.ms import _ms_render


# ── Model Selector ─────────────────────────────────────────────────────────────

@callback(
    Output("wb-ms-target", "options"),
    Output("wb-ms-target", "value"),
    Input("wb-ms-dataset", "value"),
    prevent_initial_call=True,
)
def wb_ms_load_cols(dataset):
    from databricks_client import load_sample_dataset
    if not dataset:
        raise PreventUpdate
    df   = load_sample_dataset(dataset)
    opts = [{"label": c, "value": c} for c in df.columns]
    # Suggest a sensible default target: last column, or first binary/categorical column
    default = df.columns[-1]
    for col in df.columns:
        if df[col].nunique() <= 10 and df[col].dtype not in ("float64", "float32"):
            default = col
            break
    return opts, default


@callback(
    Output("runs-store", "data", allow_duplicate=True),
    Input("wb-ms-run-btn",  "n_clicks"),
    State("wb-ms-dataset",  "value"),
    State("wb-ms-target",   "value"),
    State("runs-store",     "data"),
    prevent_initial_call=True,
)
def wb_run_ms(_, dataset, target_col, store):
    from databricks_client import load_sample_dataset
    from skills.model_selector import select

    if not dataset or not target_col:
        raise PreventUpdate

    session = store.get("session") or {}
    problem = session.get("problem_statement", "")

    dq_runs     = store.get("data_quality", [])
    latest_dq   = next((r for r in reversed(dq_runs) if r.get("status") == "complete"), None)
    dq_findings = (latest_dq or {}).get("result", {}).get("findings") or None

    ld_runs          = store.get("leakage_detector", [])
    latest_ld        = next((r for r in reversed(ld_runs) if r.get("status") == "complete"), None)
    leakage_findings = (latest_ld or {}).get("result", {}).get("findings") or None

    label = f"{dataset} → {target_col}"
    try:
        df     = load_sample_dataset(dataset)
        result = select(df, target_col, problem_context=problem,
                        dq_findings=dq_findings, leakage_findings=leakage_findings)
        recs         = result.get("recommendations", [])
        top          = recs[0].get("display_name", "") if recs else ""
        t_analysis   = result.get("target_analysis", {}) or {}
        top_rationale = recs[0].get("rationale", "") if recs else ""
        run = {
            "id":        f"ms_{int(time.time() * 1000)}",
            "label":     label,
            "timestamp": datetime.now().strftime("%H:%M"),
            "status":    "complete",
            "result":    result,
            "summary": {
                "task_type":        result.get("task_type", ""),
                "top_model":        top,
                "top_rationale":    top_rationale[:120] if top_rationale else "",
                "readiness":        result.get("data_readiness", ""),
                "n_recs":           len(recs),
                "class_balance":    t_analysis.get("class_balance_issue", False),
                "imbalance_ratio":  t_analysis.get("imbalance_ratio"),
            },
        }
    except PreventUpdate:
        raise
    except Exception as exc:
        run = {"id": f"ms_{int(time.time()*1000)}", "label": label,
               "timestamp": datetime.now().strftime("%H:%M"),
               "status": "error", "error": str(exc), "summary": {}}

    updated = dict(store)
    updated["model_selector"] = store.get("model_selector", []) + [run]
    return updated


@callback(
    Output("wb-ms-results", "children"),
    Input("runs-store", "data"),
)
def wb_render_ms(store):
    from components.run_card import make_run_card
    runs = store.get("model_selector", [])
    if not runs:
        return html.P("Select a dataset and target column, then click Select Models.",
                      className="text-muted small py-3")
    cards = []
    for run in reversed(runs):
        content = _ms_render(run["result"]) if run["status"] == "complete" \
            else dbc.Alert(run.get("error", "Unknown error"), color="danger")
        cards.append(make_run_card(run, "model_selector", content))
    return cards


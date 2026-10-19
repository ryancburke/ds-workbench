"""Workbench callbacks for the Experiment Tracker section: run the skill and render its runs."""
import json

import dash
import dash_bootstrap_components as dbc
from dash import callback, Input, Output, State

from workbench.renderers.et import _et_render


# ── Experiment Tracker ────────────────────────────────────────────────────────

@callback(
    Output("wb-et-output", "children"),
    Input("wb-et-run-btn", "n_clicks"),
    State("wb-et-metrics", "value"),
    State("runs-store", "data"),
    prevent_initial_call=True,
)
def wb_run_et(n_clicks, metrics_json, store):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate

    if not metrics_json or not metrics_json.strip():
        return dbc.Alert("Paste your training metrics JSON above first.", color="warning")

    try:
        metrics = json.loads(metrics_json)
    except json.JSONDecodeError as exc:
        return dbc.Alert(f"Invalid JSON: {exc}", color="danger")

    session = (store or {}).get("session") or {}
    context = session.get("problem_statement", "")

    try:
        from skills.experiment_tracker import interpret
        result = interpret(metrics, context=context)
        return _et_render(result)
    except Exception as exc:
        return dbc.Alert(f"Error interpreting metrics: {exc}", color="danger")


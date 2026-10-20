"""Workbench callbacks for the Causal Inference section: run the skill and render its runs."""
import time
from datetime import datetime

import dash_bootstrap_components as dbc
from dash import callback, html, Input, Output, State
from dash.exceptions import PreventUpdate

from workbench.constants import CAUSAL_PRESETS
from workbench.renderers.ci import _ci_render

_NONE = "__none__"


@callback(
    Output("wb-ci-treatment", "options"),
    Output("wb-ci-treatment", "value"),
    Output("wb-ci-outcome", "options"),
    Output("wb-ci-outcome", "value"),
    Output("wb-ci-pre", "options"),
    Output("wb-ci-pre", "value"),
    Output("wb-ci-design", "value"),
    Output("wb-ci-time", "options"),
    Output("wb-ci-time", "value"),
    Output("wb-ci-unit", "options"),
    Output("wb-ci-unit", "value"),
    Output("wb-ci-start", "value"),
    Input("wb-ci-dataset", "value"),
    prevent_initial_call=True,
)
def wb_ci_load_cols(dataset):
    from databricks_client import load_sample_dataset
    if not dataset:
        raise PreventUpdate
    df = load_sample_dataset(dataset)
    two_valued = [c for c in df.columns if df[c].nunique() == 2]
    numeric = [c for c in df.columns if df[c].dtype.kind in "fiub"]
    preset = CAUSAL_PRESETS.get(dataset, {})
    treatment = preset.get("treatment") or (two_valued[0] if two_valued else None)
    outcome = preset.get("outcome") or (numeric[-1] if numeric else None)
    return ([{"label": c, "value": c} for c in two_valued], treatment,
            [{"label": c, "value": c} for c in numeric], outcome,
            [{"label": "None", "value": _NONE}] + [{"label": c, "value": c} for c in numeric],
            preset.get("pre") or _NONE,
            preset.get("design", "unsure"),
            [{"label": c, "value": c} for c in df.columns], preset.get("time"),
            [{"label": "None", "value": _NONE}] + [{"label": c, "value": c} for c in df.columns],
            preset.get("unit") or _NONE,
            preset.get("start", ""))


@callback(
    Output("runs-store", "data", allow_duplicate=True),
    Input("wb-ci-run-btn", "n_clicks"),
    State("wb-ci-dataset", "value"),
    State("wb-ci-treatment", "value"),
    State("wb-ci-outcome", "value"),
    State("wb-ci-pre", "value"),
    State("wb-ci-design", "value"),
    State("wb-ci-time", "value"),
    State("wb-ci-unit", "value"),
    State("wb-ci-start", "value"),
    State("runs-store", "data"),
    prevent_initial_call=True,
)
def wb_run_ci(_, dataset, treatment, outcome, pre, design, time_col, unit_col, start, store):
    if not _:  # a click, not a page load: Dash Pages can fire this when the page mounts
        raise PreventUpdate
    from databricks_client import load_sample_dataset
    from skills.causal_inference import analyze, did

    if not dataset or not treatment or not outcome:
        raise PreventUpdate
    context = ((store.get("session") or {}).get("problem_statement", ""))
    label = f"{dataset}: {treatment} → {outcome}"
    run_id = f"ci_{int(time.time() * 1000)}"
    randomized = {"randomized": True, "observational": False}.get(design)

    df = load_sample_dataset(dataset)
    if design == "did":
        if not time_col or not start:
            raise PreventUpdate
        result = did(df, outcome, treatment, time_col, start,
                     unit_col=None if unit_col in (None, _NONE) else unit_col, context=context)
    else:
        result = analyze(df, treatment, outcome, randomized=randomized,
                         pre_outcome_col=None if pre in (None, _NONE) else pre, context=context)
    if result.get("error"):
        run = {"id": run_id, "label": label, "timestamp": datetime.now().strftime("%H:%M"),
               "status": "error", "error": result["error"], "summary": {}}
    else:
        est = result.get("estimate") or {}
        run = {"id": run_id, "label": label, "timestamp": datetime.now().strftime("%H:%M"),
               "status": "complete", "result": dict(result),
               "summary": {"verdict": result.get("verdict"), "effect": est.get("effect")}}

    updated = dict(store)
    updated["causal_inference"] = store.get("causal_inference", []) + [run]
    return updated


@callback(
    Output("wb-ci-results", "children"),
    Input("runs-store", "data"),
)
def wb_render_ci(store):
    from components.run_card import make_run_card
    runs = store.get("causal_inference", [])
    if not runs:
        return html.P("Pick a dataset, check the treatment and outcome columns, then click Analyze Effect.",
                      className="text-muted small py-3")
    cards = []
    for run in reversed(runs):
        content = (_ci_render(run["result"]) if run["status"] == "complete"
                   else dbc.Alert(run.get("error", "Unknown error"), color="danger"))
        cards.append(make_run_card(run, "causal_inference", content))
    return cards

"""Workbench callbacks for the MLOps Review section: run the skill and render its runs."""
import time
from datetime import datetime

import dash_bootstrap_components as dbc
from dash import callback, html, Input, Output, State
from dash.exceptions import PreventUpdate

from workbench.renderers.ml import _ml_render


# ── MLOps Review ───────────────────────────────────────────────────────────────

@callback(
    Output("runs-store", "data", allow_duplicate=True),
    Input("wb-ml-run-btn", "n_clicks"),
    State("wb-ml-code",    "value"),
    State("runs-store",    "data"),
    prevent_initial_call=True,
)
def wb_run_ml(_, code, store):
    if not _:  # a click, not a page load: Dash Pages can fire this when the page mounts
        raise PreventUpdate
    from skills.mlops_reviewer import review

    if not code or not code.strip():
        raise PreventUpdate

    label    = (code.strip().splitlines()[0] or "notebook")[:40]
    result   = review(code)
    findings = result.get("findings", [])
    run = {
        "id":        f"mlops_{int(time.time() * 1000)}",
        "label":     label,
        "timestamp": datetime.now().strftime("%H:%M"),
        "status":    "complete",
        "result":    result,
        "summary": {
            "score":    result.get("readiness_score", 0),
            "critical": sum(1 for f in findings if f.get("severity") == "critical"),
            "warning":  sum(1 for f in findings if f.get("severity") == "warning"),
        },
    }
    updated = dict(store)
    updated["mlops_review"] = store.get("mlops_review", []) + [run]
    return updated


@callback(
    Output("wb-ml-results", "children"),
    Input("runs-store", "data"),
)
def wb_render_ml(store):
    from components.run_card import make_run_card

    runs = store.get("mlops_review", [])
    if not runs:
        return html.P("No runs yet — paste notebook code and click Review.",
                      className="text-muted small py-3")
    cards = []
    for run in reversed(runs):
        content = _ml_render(run["result"], session_store=store) \
            if run["status"] == "complete" \
            else dbc.Alert(run.get("error", "Unknown error"), color="danger")
        cards.append(make_run_card(run, "mlops_review", content))
    return cards


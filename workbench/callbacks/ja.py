"""Workbench callbacks for the Join Auditor section: run the skill and render its runs."""
import time
from datetime import datetime

import dash_bootstrap_components as dbc
from dash import callback, html, Input, Output, State
from dash.exceptions import PreventUpdate

from workbench.renderers.ja import _ja_render


# ── Join Auditor ───────────────────────────────────────────────────────────────

@callback(
    Output("runs-store", "data", allow_duplicate=True),
    Input("wb-ja-run-btn",  "n_clicks"),
    State("wb-ja-scenario", "value"),
    State("runs-store",     "data"),
    prevent_initial_call=True,
)
def wb_run_ja(_, scenario, store):
    from databricks_client import load_join_scenario, SAMPLE_JOIN_SCENARIOS
    from skills.join_auditor import audit

    if not scenario:
        raise PreventUpdate

    left_df, right_df, join_key = load_join_scenario(scenario)
    label = SAMPLE_JOIN_SCENARIOS.get(scenario, {}).get("label", scenario)

    join_keys = [k.strip() for k in join_key.split(",")] if isinstance(join_key, str) else join_key
    result   = audit(left_df, right_df, join_keys=join_keys)
    findings = result.get("findings", [])
    run = {
        "id":        f"ja_{int(time.time() * 1000)}",
        "label":     label,
        "timestamp": datetime.now().strftime("%H:%M"),
        "status":    "complete",
        "result":    result,
        "summary": {
            "verdict":         result.get("overall_verdict", "review_needed"),
            "fan_out_factor":  result.get("row_count_change", {}).get("fan_out_factor", 1.0),
            "critical":        sum(1 for f in findings if f.get("severity") == "critical"),
            "warning":         sum(1 for f in findings if f.get("severity") == "warning"),
        },
    }
    updated = dict(store)
    updated["join_auditor"] = store.get("join_auditor", []) + [run]
    return updated


@callback(
    Output("wb-ja-results", "children"),
    Input("runs-store", "data"),
)
def wb_render_ja(store):
    from components.run_card import make_run_card

    runs = store.get("join_auditor", [])
    if not runs:
        return html.P("No runs yet — select a join scenario and click Run Audit.",
                      className="text-muted small py-3")
    cards = []
    for run in reversed(runs):
        content = _ja_render(run["result"]) \
            if run["status"] == "complete" \
            else dbc.Alert(run.get("error", "Unknown error"), color="danger")
        cards.append(make_run_card(run, "join_auditor", content))
    return cards


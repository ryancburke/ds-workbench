"""Workbench callbacks for the Data Quality section: run the skill and render its runs."""
import time
from datetime import datetime

import dash_bootstrap_components as dbc
from dash import callback, html, Input, Output, State
from dash.exceptions import PreventUpdate

from workbench.renderers.dq import _dq_render


# ── Data Quality ───────────────────────────────────────────────────────────────

@callback(
    Output("runs-store", "data", allow_duplicate=True),
    Input("wb-dq-run-btn", "n_clicks"),
    State("wb-dq-dataset", "value"),
    State("runs-store",    "data"),
    prevent_initial_call=True,
)
def wb_run_dq(_, dataset, store):
    if not _:  # a click, not a page load: Dash Pages can fire this when the page mounts
        raise PreventUpdate
    from databricks_client import load_sample_dataset
    from skills.data_quality import assess

    if not dataset:
        raise PreventUpdate

    df       = load_sample_dataset(dataset)
    result   = assess(df, table_name=dataset)
    findings = result.get("findings", [])
    run = {
        "id":        f"dq_{int(time.time() * 1000)}",
        "label":     dataset,
        "timestamp": datetime.now().strftime("%H:%M"),
        "status":    "complete",
        "result":    result,
        "profile":   result.get("profile", {}),
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


@callback(
    Output("wb-dq-results", "children"),
    Input("runs-store", "data"),
)
def wb_render_dq(store):
    from components.run_card import make_run_card

    runs = store.get("data_quality", [])
    if not runs:
        return html.P("No runs yet — select a dataset and click Run DQ.",
                      className="text-muted small py-3")
    completed = [r for r in runs if r.get("status") == "complete"]
    compare_panel = html.Div()
    if len(completed) >= 2:
        r1, r2 = completed[-2], completed[-1]
        s1, s2 = r1.get("summary", {}), r2.get("summary", {})
        def _score_badge(s):
            sc = s.get("score", 0)
            color = "success" if sc >= 75 else "warning" if sc >= 50 else "danger"
            return dbc.Badge(f"{sc}/100", color=color, className="me-1")
        compare_panel = dbc.Card([
            dbc.CardHeader([html.I(className="bi bi-columns-gap me-2"),
                            html.Strong("Last 2 DQ Runs — Side by Side")]),
            dbc.CardBody(dbc.Row([
                dbc.Col([
                    html.Div([html.Code(r1.get("label",""), className="small text-info me-2"),
                              _score_badge(s1),
                              html.Span(f"{s1.get('critical',0)} crit · {s1.get('warning',0)} warn",
                                        className="text-muted small")],
                             className="mb-2"),
                    html.Div(r1.get("result", {}).get("summary",""), className="small text-muted"),
                ], width=6),
                dbc.Col([
                    html.Div([html.Code(r2.get("label",""), className="small text-info me-2"),
                              _score_badge(s2),
                              html.Span(f"{s2.get('critical',0)} crit · {s2.get('warning',0)} warn",
                                        className="text-muted small")],
                             className="mb-2"),
                    html.Div(r2.get("result", {}).get("summary",""), className="small text-muted"),
                ], width=6),
            ])),
        ], className="mb-3", style={"borderColor": "#0d6efd40"})

    cards = [compare_panel]
    for run in reversed(runs):
        profile = run.get("profile") or run.get("result", {}).get("profile", {})
        content = _dq_render(run["result"], profile) \
            if run["status"] == "complete" \
            else dbc.Alert(run.get("error", "Unknown error"), color="danger")
        cards.append(make_run_card(run, "data_quality", content))
    return cards


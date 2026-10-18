import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dcc, html
from dash.exceptions import PreventUpdate

dash.register_page(__name__, path="/agent-sim", name="Agent Sim", order=6)

# ── Layout ─────────────────────────────────────────────────────────────────────

def layout():
    from agents.simulation import SCENARIOS, PERSONAS
    return html.Div([
        dcc.Store(id="sim-store", storage_type="session",
                  data={"status": "idle", "transcript": [], "persona_states": []}),
        dcc.Interval(id="sim-interval", interval=600, disabled=True, n_intervals=0),

        # ── Header ────────────────────────────────────────────────────────────
        html.H4([html.I(className="bi bi-people-fill me-2"), "Agent Simulation"],
                className="mb-1"),
        html.P(
            "Three DS personas receive a business scenario and autonomously decide which skills "
            "to run, interpret the results, and deliver a final recommendation.",
            className="text-muted mb-4",
        ),

        # ── Persona strip ─────────────────────────────────────────────────────
        dbc.Row([
            dbc.Col(_persona_card(p), width=4) for p in PERSONAS
        ], className="mb-4"),

        # ── Scenario selector ─────────────────────────────────────────────────
        html.Div(id="sim-scenario-panel", children=_scenario_panel(SCENARIOS)),

        # ── Timeline ──────────────────────────────────────────────────────────
        html.Div(id="sim-status-bar"),
        dcc.Loading(
            html.Div(id="sim-transcript"),
            type="circle",
            color="#6478ff",
        ),

        # ── Debriefs ──────────────────────────────────────────────────────────
        html.Div(id="sim-debriefs"),

        # ── Restart ───────────────────────────────────────────────────────────
        html.Div(id="sim-restart-area"),
    ])


def _persona_card(p: dict) -> dbc.Card:
    return dbc.Card(
        dbc.CardBody([
            html.Div([
                html.I(className=f"bi {p['icon']} me-2 fs-5",
                       style={"color": p["color"]}),
                html.Span(p["name"], className="fw-bold"),
            ], className="mb-1"),
            html.Div(p["role"], className="small text-muted mb-2"),
            html.Div(
                # First sentence of 'approach' as a tagline
                p["approach"].split(".")[0] + ".",
                className="small", style={"color": "#888", "lineHeight": "1.4"},
            ),
        ]),
        style={"backgroundColor": "#16162a",
               "borderLeft": f"3px solid {p['color']}",
               "border": f"1px solid #252545",
               "borderLeftWidth": "3px",
               "borderLeftColor": p["color"]},
    )


def _scenario_panel(scenarios: list) -> html.Div:
    return html.Div([
        html.P("SCENARIO", className="text-muted mb-2",
               style={"fontSize": "0.65rem", "letterSpacing": "0.12em"}),
        dbc.Row([
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.Div([
                            html.I(className=f"bi {s['icon']} me-2",
                                   style={"color": s["color"]}),
                            html.Strong(s["title"]),
                        ], className="mb-2"),
                        html.P(s["brief"][:160] + "…", className="small text-muted mb-3"),
                        dbc.Button(
                            [html.I(className="bi bi-play-fill me-1"), "Run Simulation"],
                            id={"type": "sim-start-btn", "index": s["id"]},
                            color="primary",
                            outline=True,
                            size="sm",
                            className="w-100",
                        ),
                    ]),
                    style={"backgroundColor": "#16162a", "border": "1px solid #252545",
                           "cursor": "pointer"},
                ),
                width=4,
            )
            for s in scenarios
        ], className="g-3 mb-4"),
    ])


# ── Callbacks ──────────────────────────────────────────────────────────────────

@callback(
    Output("sim-store", "data"),
    Output("sim-interval", "disabled"),
    Output("sim-scenario-panel", "children"),
    Input({"type": "sim-start-btn", "index": dash.ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def start_simulation(n_clicks_list):
    from dash import ctx
    from agents.simulation import SCENARIOS, init_sim_state

    if not any(n for n in n_clicks_list if n):
        raise PreventUpdate

    triggered = ctx.triggered[0]["prop_id"]
    import json
    scenario_id = json.loads(triggered.rsplit(".", 1)[0])["index"]

    scenario = next(s for s in SCENARIOS if s["id"] == scenario_id)
    sim_state = init_sim_state(scenario_id)

    status_bar = html.Div([
        html.Div([
            html.I(className=f"bi {scenario['icon']} me-2",
                   style={"color": scenario["color"]}),
            html.Strong(f"Simulation: {scenario['title']}"),
            html.Span(" — agents are working…", className="text-muted ms-1 small"),
        ], className="d-flex align-items-center"),
    ], className="mb-3 p-2",
       style={"backgroundColor": "#1a1a30", "borderRadius": "6px",
              "border": "1px solid #252545"})

    return sim_state, False, status_bar


@callback(
    Output("sim-store", "data", allow_duplicate=True),
    Output("runs-store", "data", allow_duplicate=True),
    Output("sim-interval", "disabled", allow_duplicate=True),
    Input("sim-interval", "n_intervals"),
    State("sim-store", "data"),
    State("runs-store", "data"),
    prevent_initial_call=True,
)
def tick_simulation(n_intervals, sim_state, runs_store):
    if not sim_state or sim_state.get("status") != "running":
        raise PreventUpdate

    from agents.simulation import run_sim_step

    updated_state, new_runs, _benches = run_sim_step(sim_state)

    # Merge new run records into runs-store
    updated_runs = dict(runs_store)
    for skill_key, records in new_runs.items():
        updated_runs[skill_key] = updated_runs.get(skill_key, []) + records

    interval_disabled = updated_state.get("status") != "running"
    return updated_state, updated_runs, interval_disabled


@callback(
    Output("sim-transcript", "children"),
    Output("sim-debriefs", "children"),
    Output("sim-restart-area", "children"),
    Input("sim-store", "data"),
)
def render_sim(sim_state):
    if not sim_state or not sim_state.get("transcript"):
        return html.Div(), html.Div(), html.Div()

    transcript = sim_state.get("transcript", [])
    status     = sim_state.get("status", "idle")

    # Split transcript into skill steps and debriefs
    skill_steps  = [s for s in transcript if s.get("skill")]
    debrief_steps = [s for s in transcript if s.get("debrief")]

    timeline = []
    for step in skill_steps:
        timeline.append(_render_step(step))

    debrief_section = html.Div()
    if debrief_steps:
        debrief_section = html.Div([
            html.Hr(className="my-4"),
            html.P("FINAL RECOMMENDATIONS", className="text-muted mb-3",
                   style={"fontSize": "0.65rem", "letterSpacing": "0.12em"}),
            dbc.Row([
                dbc.Col(_render_debrief(step), width=4)
                for step in debrief_steps
            ], className="g-3"),
        ])

    restart_area = html.Div()
    if status == "complete":
        from agents.simulation import SCENARIOS
        restart_area = html.Div([
            html.Hr(className="my-4"),
            html.Div([
                html.I(className="bi bi-check-circle-fill me-2 text-success"),
                html.Span("Simulation complete. ", className="text-success fw-bold"),
                html.Span(
                    "Runs have been added to each skill page — navigate using the sidebar to explore results.",
                    className="text-muted small",
                ),
            ], className="d-flex align-items-center mb-3"),
            _scenario_panel(SCENARIOS),
        ])

    return timeline, debrief_section, restart_area


# ── Render helpers ─────────────────────────────────────────────────────────────

SKILL_META = {
    "data_quality": {"icon": "bi-shield-check",    "color": "#0d6efd", "label": "Data Quality Audit"},
    "eda":          {"icon": "bi-graph-up",         "color": "#198754", "label": "EDA"},
    "join_auditor": {"icon": "bi-link-45deg",       "color": "#ffc107", "label": "Join Auditor"},
    "nl_to_spark":  {"icon": "bi-translate",        "color": "#0dcaf0", "label": "NL → PySpark"},
    "mlops_review": {"icon": "bi-gear-fill",        "color": "#dc3545", "label": "MLOps Review"},
}


def _step_outcome_badges(skill: str, summary: dict) -> list:
    """Outcome badges derived from the skill summary — shown inline with the skill chip."""
    if skill == "data_quality":
        score = summary.get("score", 0)
        sc    = "success" if score >= 75 else "warning" if score >= 50 else "danger"
        n_c   = summary.get("critical", 0)
        label = "BLOCKING" if n_c > 0 else ("NEEDS REVIEW" if score < 75 else "CLEAR")
        lc    = "danger" if n_c > 0 else ("warning" if score < 75 else "success")
        return [
            dbc.Badge(label, color=lc, className="me-1"),
            dbc.Badge(f"{score}/100", color=sc, className="me-1"),
            html.Span(f"{n_c} crit · {summary.get('warning', 0)} warn",
                      className="text-muted small"),
        ]
    if skill == "leakage_detector":
        verdict = summary.get("verdict", "clean")
        vc = {"leakage_detected": "danger", "suspicious": "warning", "clean": "success"}.get(verdict, "secondary")
        label   = {"leakage_detected": "LEAKAGE DETECTED", "suspicious": "SUSPICIOUS",
                   "clean": "CLEAN"}.get(verdict, verdict.upper())
        return [
            dbc.Badge(label, color=vc, className="me-1"),
            html.Span(f"risk: {summary.get('risk_level','?')} · "
                      f"{summary.get('critical',0)} critical",
                      className="text-muted small"),
        ]
    if skill == "model_selector":
        top = summary.get("top_model", "")
        return [
            dbc.Badge(summary.get("task_type", "?").replace("_", " ").title(),
                      color="secondary", className="me-1"),
            html.Span(top[:40], className="text-muted small"),
        ]
    if skill == "mlops_review":
        score = summary.get("score", 0)
        sc    = "success" if score >= 70 else "warning" if score >= 40 else "danger"
        label = "PRODUCTION READY" if score >= 70 else ("NEEDS REVIEW" if score >= 40 else "NOT READY")
        return [
            dbc.Badge(label, color=sc, className="me-1"),
            dbc.Badge(f"{score}/100", color=sc, className="me-1"),
        ]
    if skill == "join_auditor":
        verdict = summary.get("verdict", "review_needed")
        vc = {"safe": "success", "review_needed": "warning", "dangerous": "danger"}.get(verdict, "secondary")
        return [
            dbc.Badge(verdict.replace("_", " ").title(), color=vc, className="me-1"),
            html.Span(f"×{summary.get('fan_out_factor', 1.0):.2f} fan-out",
                      className="text-muted small"),
        ]
    if skill == "eda":
        return [html.Span(
            f"{summary.get('insights',0)} insights · {summary.get('targets',0)} ML targets",
            className="text-muted small")]
    if skill == "feature_engineering":
        return [html.Span(
            f"{summary.get('n_features',0)} features · {summary.get('n_dropped',0)} dropped",
            className="text-muted small")]
    return []


def _render_step(step: dict) -> html.Div:
    color   = step["persona_color"]
    sm      = SKILL_META.get(step["skill"], {"icon": "bi-play", "color": "#888", "label": step["skill"]})
    outcome = _step_outcome_badges(step["skill"], step.get("summary", {}))

    return html.Div([
        # ── Who + when ────────────────────────────────────────────────────────
        html.Div([
            html.I(className=f"bi {step['persona_icon']} me-2",
                   style={"color": color, "fontSize": "1rem"}),
            html.Span(step["persona_name"], className="fw-bold me-2"),
            html.Span(step["persona_role"], className="text-muted small me-2"),
            html.Span(f"· {step['timestamp']}", className="text-muted small"),
        ], className="d-flex align-items-center mb-2"),

        # ── Thinking bubble ───────────────────────────────────────────────────
        html.Div([
            html.P(
                f'"{step["thinking"]}"',
                className="mb-0 fst-italic",
                style={"color": "#c8c8e8", "lineHeight": "1.6"},
            ),
        ], className="ps-3 mb-3",
           style={"borderLeft": f"3px solid {color}", "paddingLeft": "12px"}),

        # ── Skill chip + outcome ───────────────────────────────────────────────
        html.Div([
            dbc.Badge([
                html.I(className=f"bi {sm['icon']} me-1"),
                f"{sm['label']}  ·  {step['dataset']}",
            ], color="dark", className="me-2 p-2",
               style={"border": f"1px solid {sm['color']}", "color": sm["color"]}),
            *outcome,
        ], className="d-flex align-items-center flex-wrap gap-1 mb-2"),

        # ── Interpretation ────────────────────────────────────────────────────
        html.Div([
            html.P(
                f'"{step["interpretation"]}"',
                className="small mb-0",
                style={"color": "#b0b0c8", "lineHeight": "1.5"},
            ),
        ], className="ps-3",
           style={"borderLeft": f"2px solid #2a2a4a"}),

    ], className="mb-4 pb-3",
       style={"borderBottom": "1px solid #1e1e38"})


def _render_debrief(step: dict) -> dbc.Card:
    color = step["persona_color"]
    return dbc.Card(
        dbc.CardBody([
            html.Div([
                html.I(className=f"bi {step['persona_icon']} me-2",
                       style={"color": color}),
                html.Span(step["persona_name"], className="fw-bold"),
            ], className="mb-1"),
            html.Div(step["persona_role"], className="small text-muted mb-3"),
            html.P(
                step["debrief"],
                className="small mb-0",
                style={"lineHeight": "1.6", "color": "#c8c8e8"},
            ),
        ]),
        style={
            "backgroundColor": "#16162a",
            "border": f"1px solid {color}40",
            "borderTop": f"3px solid {color}",
        },
    )

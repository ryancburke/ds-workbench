import dash
import dash_bootstrap_components as dbc
from dash import ALL, Input, Output, State, callback, dcc, html
from dash.exceptions import PreventUpdate

dash.register_page(__name__, path="/feedback", name="Feedback", order=7)

PRIORITY_COLOR  = {"critical": "danger", "high": "warning", "medium": "info", "low": "secondary"}
SKILL_LABEL     = {
    "data_quality": "Data Quality", "eda": "EDA",
    "join_auditor": "Join Auditor", "nl_to_spark": "NL → PySpark",
    "mlops_review": "MLOps Review", "agent_sim": "Agent Sim", "general": "General",
}
ISSUE_ICON = {
    "missing_feature":    "bi-plus-circle",
    "ux_friction":        "bi-speedometer2",
    "visualization_gap":  "bi-bar-chart-line",
    "workflow_improvement":"bi-arrow-repeat",
    "data_insight":       "bi-lightbulb",
}
SCENARIO_CHOICES = [
    {"label": "Q3 Revenue Shortfall",  "value": "q3_revenue"},
    {"label": "Customer Churn Risk",   "value": "churn_risk"},
    {"label": "New Pipeline Handoff",  "value": "pipeline_handoff"},
]
ALL_SCENARIO_IDS = [s["value"] for s in SCENARIO_CHOICES]


def layout():
    return html.Div([
        dcc.Store(id="fb-selected-rec"),
        dcc.Store(id="fb-refresh-trigger", data=0),
        dcc.Interval(id="fb-poll-interval", interval=8000, n_intervals=0),

        # ── Header ────────────────────────────────────────────────────────────
        dbc.Row([
            dbc.Col([
                html.H4([html.I(className="bi bi-arrow-repeat me-2"), "Workbench Feedback Loop"],
                        className="mb-1"),
                html.P(
                    "Agent simulations run in the background and critique the workbench after every run. "
                    "Approve items to draft a code change, then apply it with one click.",
                    className="text-muted mb-0",
                ),
            ]),
            dbc.Col([
                html.Div(id="fb-header-stats", className="text-end"),
            ], width="auto", className="d-flex align-items-center"),
        ], className="mb-4 align-items-start"),

        # ── Run now panel ─────────────────────────────────────────────────────
        dbc.Card(dbc.CardBody([
            html.Label("Run simulations to generate new feedback", className="small text-muted mb-2"),
            dbc.Row([
                dbc.Col([
                    dbc.Checklist(
                        id="fb-run-scenarios",
                        options=SCENARIO_CHOICES,
                        value=ALL_SCENARIO_IDS,
                        inline=True,
                        className="small",
                        style={"gap": "1rem"},
                    ),
                ], width=7),
                dbc.Col([
                    dbc.ButtonGroup([
                        dbc.Button(
                            [html.I(className="bi bi-play-fill me-1"), "Run Selected"],
                            id="fb-run-btn", color="primary", size="sm",
                        ),
                        dbc.Button(
                            [html.I(className="bi bi-skip-forward-fill me-1"), "Run All 3"],
                            id="fb-run-all-btn", color="secondary", size="sm", outline=True,
                        ),
                    ]),
                ], width=5, className="d-flex align-items-center justify-content-end"),
            ], align="center"),
            html.Div(id="fb-run-status", className="mt-2"),
        ]), className="mb-4",
           style={"backgroundColor": "#16162a", "border": "1px solid #252545"}),

        # ── Recommendation list ───────────────────────────────────────────────
        dbc.Tabs([
            dbc.Tab(html.Div(id="fb-pending-list"),   label="Pending",     tab_id="pending"),
            dbc.Tab(html.Div(id="fb-drafted-list"),   label="Drafted",     tab_id="drafted"),
            dbc.Tab(html.Div(id="fb-done-list"),      label="Implemented", tab_id="implemented"),
            dbc.Tab(html.Div(id="fb-dismissed-list"), label="Dismissed",   tab_id="dismissed"),
        ], id="fb-tabs", active_tab="pending"),

        # ── Code draft modal ──────────────────────────────────────────────────
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle(html.Div(id="fb-modal-title"))),
            dbc.ModalBody([
                dcc.Loading(
                    html.Div(id="fb-modal-body"),
                    type="circle", color="#6478ff",
                ),
            ]),
            dbc.ModalFooter([
                dbc.Button("Close", id="fb-modal-close", color="secondary", outline=True),
                dbc.Button([html.I(className="bi bi-check-lg me-1"), "Apply Change"],
                           id="fb-apply-btn", color="success",
                           className="ms-2", disabled=True),
            ]),
        ], id="fb-draft-modal", size="xl", scrollable=True, is_open=False),
    ])


# ── Helpers ────────────────────────────────────────────────────────────────────

def _rec_card(rec: dict, show_actions=True) -> dbc.Card:
    votes  = rec.get("votes", 1)
    seen   = rec.get("seen_by", [])
    action_buttons = []

    if show_actions and rec.get("status") == "pending":
        action_buttons = [
            dbc.Button("Approve & Draft", id={"type": "fb-approve", "index": rec["id"]},
                       color="success", size="sm", outline=True, className="me-2"),
            dbc.Button("Dismiss", id={"type": "fb-dismiss", "index": rec["id"]},
                       color="secondary", size="sm", outline=True),
        ]
    elif rec.get("status") == "drafted":
        action_buttons = [
            dbc.Button([html.I(className="bi bi-code-slash me-1"), "View & Apply"],
                       id={"type": "fb-view-draft", "index": rec["id"]},
                       color="primary", size="sm", outline=True),
        ]

    return dbc.Card(
        dbc.CardBody([
            # ── Meta row ──────────────────────────────────────────────────────
            html.Div([
                dbc.Badge(rec.get("priority", "medium").upper(),
                          color=PRIORITY_COLOR.get(rec.get("priority", "medium"), "secondary"),
                          className="me-2"),
                dbc.Badge(SKILL_LABEL.get(rec.get("skill_affected", "general"), "?"),
                          color="dark", className="me-2",
                          style={"border": "1px solid #3a3a5a"}),
                html.I(className=f"bi {ISSUE_ICON.get(rec.get('issue_type',''), 'bi-dot')} me-1 text-muted"),
                html.Span(rec.get("issue_type", "").replace("_", " ").title(),
                          className="small text-muted me-3"),
                html.Span(
                    [html.I(className="bi bi-arrow-up me-1"),
                     html.Strong(str(votes), className="me-1"),
                     "vote" + ("s" if votes != 1 else "")],
                    className="small",
                    style={"color": "#ffc107" if votes >= 3 else "#888"},
                ),
            ], className="d-flex align-items-center flex-wrap mb-2"),

            # ── Title ─────────────────────────────────────────────────────────
            html.Div(rec.get("title", ""), className="fw-bold mb-2"),

            # ── Description ───────────────────────────────────────────────────
            html.P(rec.get("description", ""), className="small text-muted mb-2"),

            # ── User story ────────────────────────────────────────────────────
            html.Div([
                html.Span("User story: ", className="small fw-bold text-muted"),
                html.Span(rec.get("user_story", ""), className="small text-muted fst-italic"),
            ], className="mb-2"),

            # ── Implementation hint ───────────────────────────────────────────
            dbc.Alert([
                html.I(className="bi bi-wrench me-1"),
                html.Strong("Hint: "),
                rec.get("implementation_hint", ""),
            ], color="dark", className="py-1 px-2 mb-2 small",
               style={"border": "1px solid #3a3a5a"}),

            # ── Seen by ───────────────────────────────────────────────────────
            html.Div([
                html.Span("Surfaced by: ", className="small text-muted me-1"),
                html.Span(", ".join(seen[-3:]), className="small",
                          style={"color": "#888"}),
            ], className="mb-3") if seen else html.Div(),

            # ── Actions ───────────────────────────────────────────────────────
            html.Div(action_buttons),
        ]),
        className="mb-3",
        style={"backgroundColor": "#16162a", "border": "1px solid #252545"},
    )


def _empty_state(msg: str) -> html.Div:
    return html.P(msg, className="text-muted text-center py-5")


# ── Run now ────────────────────────────────────────────────────────────────────

@callback(
    Output("fb-run-status", "children"),
    Input("fb-run-btn",     "n_clicks"),
    Input("fb-run-all-btn", "n_clicks"),
    State("fb-run-scenarios", "value"),
    prevent_initial_call=True,
)
def trigger_run(_, _all, selected):
    from dash import ctx
    from agents.scheduler import trigger_bulk_run, running_scenarios

    already = running_scenarios()
    use_all = "fb-run-all-btn" in (ctx.triggered[0]["prop_id"] if ctx.triggered else "")
    ids = ALL_SCENARIO_IDS if use_all else (selected or [])

    if not ids:
        return dbc.Alert("Select at least one scenario.", color="warning",
                         className="py-1 mb-0 small")

    skipped = [s for s in ids if s in already]
    to_run  = [s for s in ids if s not in already]

    if not to_run:
        return dbc.Alert("All selected scenarios are already running.",
                         color="warning", className="py-1 mb-0 small")

    trigger_bulk_run(to_run)

    label_map = {s["value"]: s["label"] for s in SCENARIO_CHOICES}
    names = ", ".join(label_map.get(s, s) for s in to_run)
    msg_parts = [
        html.I(className="bi bi-hourglass-split me-1"),
        html.Strong(f"{len(to_run)} simulation{'s' if len(to_run) > 1 else ''} started: "),
        names,
        html.Span(" — each takes 2–5 min. Results appear automatically.",
                  className="text-muted"),
    ]
    if skipped:
        skip_names = ", ".join(label_map.get(s, s) for s in skipped)
        msg_parts += [html.Br(),
                      html.Span(f"Already running (skipped): {skip_names}",
                                className="small text-muted")]
    return dbc.Alert(msg_parts, color="info", className="py-1 mb-0 small")


# ── Poll & refresh ─────────────────────────────────────────────────────────────

@callback(
    Output("fb-refresh-trigger", "data"),
    Input("fb-poll-interval", "n_intervals"),
    State("fb-refresh-trigger", "data"),
)
def poll(_n, current):
    return (current or 0) + 1


# ── Render lists ──────────────────────────────────────────────────────────────

@callback(
    Output("fb-pending-list", "children"),
    Output("fb-drafted-list", "children"),
    Output("fb-done-list", "children"),
    Output("fb-dismissed-list", "children"),
    Output("fb-header-stats", "children"),
    Input("fb-refresh-trigger", "data"),
    prevent_initial_call=True,
)
def render_lists(_):
    from agents.scheduler import get_recommendations, get_history

    pending   = get_recommendations("pending")
    drafted   = get_recommendations("drafted")
    done      = get_recommendations("implemented")
    dismissed = get_recommendations("dismissed")
    history   = get_history()

    # Header stats
    last_run = history[0] if history else None
    from agents.scheduler import running_scenarios
    active = running_scenarios()
    label_map = {s["value"]: s["label"] for s in SCENARIO_CHOICES}
    running_badges = [
        dbc.Badge(
            [html.I(className="bi bi-arrow-repeat me-1 spin"),
             label_map.get(sid, sid)],
            color="warning", className="me-1",
        )
        for sid in active
    ]

    stats = html.Div([
        *running_badges,
        dbc.Badge(f"{len(pending)} pending", color="warning", className="me-1"),
        dbc.Badge(f"{len(drafted)} drafted", color="info", className="me-1"),
        dbc.Badge(f"{len(done)} implemented", color="success", className="me-1"),
        html.Div(
            f"Last run: {_time_ago(last_run['completed_at'])} · {last_run.get('scenario_title', last_run.get('scenario_id', ''))}"
            if last_run and last_run.get("completed_at") else "No runs yet",
            className="small text-muted mt-1",
        ),
    ], className="text-end")

    p_list = [_rec_card(r) for r in pending] or [_empty_state("No pending recommendations.")]
    d_list = [_rec_card(r) for r in drafted] or [_empty_state("No items awaiting implementation.")]
    done_l = [_rec_card(r, show_actions=False) for r in done] or [_empty_state("Nothing implemented yet.")]
    dis_l  = [_rec_card(r, show_actions=False) for r in dismissed] or [_empty_state("No dismissed items.")]

    return p_list, d_list, done_l, dis_l, stats


# ── Approve ────────────────────────────────────────────────────────────────────

@callback(
    Output("fb-refresh-trigger", "data", allow_duplicate=True),
    Input({"type": "fb-approve", "index": ALL}, "n_clicks"),
    State("fb-refresh-trigger", "data"),
    prevent_initial_call=True,
)
def approve_rec(n_clicks_list, current):
    from dash import ctx
    import json as _json
    from agents.scheduler import update_rec_status

    if not any(n for n in n_clicks_list if n):
        raise PreventUpdate

    triggered = ctx.triggered[0]["prop_id"]
    rec_id = _json.loads(triggered.rsplit(".", 1)[0])["index"]
    update_rec_status(rec_id, "approved")
    return (current or 0) + 1


# ── Dismiss ────────────────────────────────────────────────────────────────────

@callback(
    Output("fb-refresh-trigger", "data", allow_duplicate=True),
    Input({"type": "fb-dismiss", "index": ALL}, "n_clicks"),
    State("fb-refresh-trigger", "data"),
    prevent_initial_call=True,
)
def dismiss_rec(n_clicks_list, current):
    from dash import ctx
    import json as _json
    from agents.scheduler import update_rec_status

    if not any(n for n in n_clicks_list if n):
        raise PreventUpdate

    triggered = ctx.triggered[0]["prop_id"]
    rec_id = _json.loads(triggered.rsplit(".", 1)[0])["index"]
    update_rec_status(rec_id, "dismissed")
    return (current or 0) + 1


# ── Open draft modal & generate code ──────────────────────────────────────────

@callback(
    Output("fb-draft-modal", "is_open"),
    Output("fb-selected-rec", "data"),
    Output("fb-modal-title", "children"),
    Input({"type": "fb-view-draft", "index": ALL}, "n_clicks"),
    Input({"type": "fb-approve", "index": ALL}, "n_clicks"),
    Input("fb-modal-close", "n_clicks"),
    State("fb-draft-modal", "is_open"),
    prevent_initial_call=True,
)
def toggle_modal(view_clicks, approve_clicks, close_click, is_open):
    from dash import ctx
    import json as _json
    from agents.scheduler import get_rec

    tid = ctx.triggered[0]["prop_id"]

    if "fb-modal-close" in tid:
        return False, None, ""

    if not any(n for n in (view_clicks or []) + (approve_clicks or []) if n):
        raise PreventUpdate

    rec_id = _json.loads(tid.rsplit(".", 1)[0])["index"]
    rec = get_rec(rec_id)
    if not rec:
        raise PreventUpdate

    return True, rec_id, [
        html.I(className="bi bi-code-slash me-2"),
        rec.get("title", "Code Draft"),
    ]


@callback(
    Output("fb-modal-body", "children"),
    Output("fb-apply-btn", "disabled"),
    Input("fb-selected-rec", "data"),
    Input("fb-draft-modal", "is_open"),
    prevent_initial_call=True,
)
def load_or_generate_draft(rec_id, is_open):
    if not is_open or not rec_id:
        raise PreventUpdate

    from agents.scheduler import get_rec, draft_code_change

    rec = get_rec(rec_id)
    if not rec:
        raise PreventUpdate

    # If already drafted, show it directly
    if rec.get("drafted_code"):
        return _code_panel(rec), False

    # Generate draft
    try:
        file_path, new_code = draft_code_change(rec_id)
        rec = get_rec(rec_id)  # reload with draft saved
        return _code_panel(rec), False
    except Exception as exc:
        return dbc.Alert(f"Draft failed: {exc}", color="danger"), True


def _code_panel(rec: dict) -> html.Div:
    return html.Div([
        dbc.Alert([
            html.Strong(f"File: "),
            html.Code(rec.get("drafted_file", "?")),
            html.Span(" — a .bak backup will be created before applying.", className="ms-2 text-muted small"),
        ], color="dark", className="py-2 mb-3", style={"border": "1px solid #3a3a5a"}),
        html.Pre(
            rec.get("drafted_code", ""),
            style={"maxHeight": "60vh", "overflowY": "auto",
                   "fontSize": "0.75rem", "backgroundColor": "#0d0d1a",
                   "border": "1px solid #2a2a4a", "borderRadius": "4px",
                   "padding": "12px"},
        ),
    ])


# ── Apply change ───────────────────────────────────────────────────────────────

@callback(
    Output("fb-draft-modal", "is_open", allow_duplicate=True),
    Output("fb-refresh-trigger", "data", allow_duplicate=True),
    Output("fb-run-status", "children", allow_duplicate=True),
    Input("fb-apply-btn", "n_clicks"),
    State("fb-selected-rec", "data"),
    State("fb-refresh-trigger", "data"),
    prevent_initial_call=True,
)
def apply_change(_, rec_id, current):
    if not rec_id:
        raise PreventUpdate

    from agents.scheduler import apply_code_change

    try:
        file_path = apply_code_change(rec_id)
        msg = dbc.Alert(
            [html.I(className="bi bi-check-circle-fill me-1"),
             html.Strong("Applied! "),
             f"Wrote {file_path}. The app will reload automatically in debug mode."],
            color="success", className="py-1 mb-0 small",
        )
        return False, (current or 0) + 1, msg
    except Exception as exc:
        msg = dbc.Alert(
            [html.I(className="bi bi-x-circle-fill me-1"),
             html.Strong("Apply failed: "), str(exc)],
            color="danger", className="py-1 mb-0 small",
        )
        return False, current, msg


# ── Utility ────────────────────────────────────────────────────────────────────

def _time_ago(iso: str) -> str:
    try:
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - dt
        mins = int(delta.total_seconds() / 60)
        if mins < 60:
            return f"{mins}m ago"
        hours = mins // 60
        if hours < 24:
            return f"{hours}h ago"
        return f"{hours // 24}d ago"
    except Exception:
        return iso

import json

import dash
import dash_bootstrap_components as dbc
from dash import ALL, MATCH, Input, Output, State, callback, dcc, html
from dash.exceptions import PreventUpdate

import importlib

from config import config  # triggers load_dotenv
from components.sidebar import create_sidebar, SKILLS
from agents.scheduler import start_scheduler, pending_count


def _preload_skills() -> None:
    """Import every skill and core module now, on the main thread. Callbacks import skills
    lazily, and Dash serves callbacks on several threads: concurrent first imports of related
    modules can deadlock on Python's per-module import locks."""
    import ds_workbench
    import ds_workbench.core
    for module in [*ds_workbench._SKILLS.values(), "skills.session", "skills.team_sessions",
                   "skills.usage", "skills.notebook_generator", "agents.simulation",
                   *(f"ds_workbench.core.{m}" for m in sorted(ds_workbench.core._SUBMODULES))]:
        importlib.import_module(module)


_preload_skills()   # before the scheduler thread or the server start

SKILL_KEYS = [s["key"] for s in SKILLS]
EMPTY_STORE = {key: [] for key in SKILL_KEYS} | {"session": None}

app = dash.Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.DARKLY, dbc.icons.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="DS Workbench",
)
server = app.server

_scheduler = start_scheduler()


def _request_user() -> str | None:
    """Signed-in user for the usage ledger: Databricks Apps forwards it in request headers."""
    from flask import has_request_context, request
    if not has_request_context():
        return None
    return request.headers.get("X-Forwarded-Email") or request.headers.get("X-Forwarded-User")


from skills.usage import set_user_resolver  # noqa: E402
set_user_resolver(_request_user)

app.layout = dbc.Container(
    [
        dcc.Store(id="runs-store", storage_type="session", data=EMPTY_STORE),
        dcc.Interval(id="feedback-badge-interval", interval=30000, n_intervals=0),
        dbc.Row(
            [
                dbc.Col(create_sidebar(), width=2, className="p-0"),
                dbc.Col(html.Div(dash.page_container, className="p-4"), width=10),
            ],
            className="g-0 min-vh-100",
        ),
    ],
    fluid=True,
    className="bg-dark",
)


# ── Sidebar run-count badges ───────────────────────────────────────────────────

@callback(
    [Output(f"sidebar-count-{key}", "children") for key in SKILL_KEYS],
    Input("runs-store", "data"),
)
def update_sidebar_counts(store):
    return [
        dbc.Badge(n, color="secondary", pill=True, className="ms-1", style={"fontSize": "0.65rem"})
        if (n := len(store.get(key, []))) > 0 else ""
        for key in SKILL_KEYS
    ]


# ── Global run-card callbacks (pattern-matching) ───────────────────────────────

@callback(
    Output({"type": "run-collapse", "index": MATCH}, "is_open"),
    Input({"type": "run-toggle", "index": MATCH}, "n_clicks"),
    State({"type": "run-collapse", "index": MATCH}, "is_open"),
    prevent_initial_call=True,
)
def toggle_run(_, is_open):
    if not _:  # a click, not a page load: Dash Pages can fire this when the page mounts
        raise PreventUpdate
    return not is_open


@callback(
    Output("runs-store", "data"),
    Input({"type": "run-dismiss", "index": ALL}, "n_clicks"),
    State("runs-store", "data"),
    prevent_initial_call=True,
)
def dismiss_run(n_clicks_list, store):
    if not any(n for n in n_clicks_list if n):
        raise PreventUpdate
    from dash import ctx
    triggered = ctx.triggered[0]["prop_id"]
    run_id = json.loads(triggered.rsplit(".", 1)[0])["index"]
    return {
        key: [r for r in val if r["id"] != run_id] if isinstance(val, list) else val
        for key, val in store.items()
    }


# ── Clear-all / new session callback ──────────────────────────────────────────

@callback(
    Output("runs-store", "data", allow_duplicate=True),
    Input("home-clear-btn", "n_clicks"),
    prevent_initial_call=True,
)
def clear_all_runs(_):
    if not _:  # a click, not a page load: Dash Pages can fire this when the page mounts
        raise PreventUpdate
    return EMPTY_STORE


@callback(
    Output("sidebar-feedback-count", "children"),
    Input("feedback-badge-interval", "n_intervals"),
)
def update_feedback_badge(_):
    n = pending_count()
    if n == 0:
        return ""
    return dbc.Badge(n, color="warning", pill=True,
                     className="ms-1", style={"fontSize": "0.65rem"})


if __name__ == "__main__":
    app.run(debug=config.debug, port=8050)

"""Sessions: every Workbench and notebook session, most urgent first — the 30-second team view."""
from datetime import datetime, timezone

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, callback, dcc, html

dash.register_page(__name__, path="/sessions", name="Sessions", order=2)

_ATTENTION = {"red": ("danger", "Blocked"), "amber": ("warning", "Caution"),
              "green": ("success", "Ready"), "grey": ("secondary", "In progress")}
_GATE = {"clear": ("success", "✓"), "warn": ("warning", "!"), "blocked": ("danger", "✗"),
         "skipped": ("secondary", "—"), "error": ("dark", "err")}
_CARD = {"backgroundColor": "#16162a", "border": "1px solid #252545"}


def layout():
    return html.Div([
        dcc.Interval(id="sess-poll", interval=15_000, n_intervals=0),
        dbc.Row([
            dbc.Col([
                html.H4([html.I(className="bi bi-people me-2"), "Sessions"], className="mb-1"),
                html.P("Every Workbench and notebook session, most urgent first: is it safe to proceed, "
                       "which checks ran, what's blocking, what changed since the last session on the same "
                       "data, and what it cost.", className="text-muted small mb-0"),
            ]),
            dbc.Col(dbc.Checklist(id="sess-attention-only", switch=True, value=[],
                                  options=[{"label": "Needs attention only", "value": "on"}]),
                    width="auto", className="d-flex align-items-center"),
        ], className="mb-3"),
        html.Div(id="sess-body"),
    ])


def _ago(iso: str | None) -> str:
    try:
        mins = int((datetime.now(timezone.utc) - datetime.fromisoformat(iso)).total_seconds() // 60)
    except (TypeError, ValueError):
        return ""
    return f"{mins}m ago" if mins < 60 else (f"{mins // 60}h ago" if mins < 1440 else f"{mins // 1440}d ago")


def _gate_chips(gates: dict):
    from skills.team_sessions import GATE_LABEL
    return [dbc.Badge(f"{GATE_LABEL.get(g, g)} {_GATE[s][1]}", color=_GATE[s][0], className="me-1",
                      style={"fontSize": "0.65rem"}, title=f"{GATE_LABEL.get(g, g)}: {s}")
            for g, s in gates.items()]


def _delta(d):
    if d is None:
        return html.Span("first on this data", className="small text-muted")
    if d == 0:
        return html.Span("blockers unchanged", className="small text-muted")
    color = "#ffa198" if d > 0 else "#7ee787"
    return html.Span(f"{d:+d} blocker{'s' if abs(d) != 1 else ''} vs last", className="small", style={"color": color})


def _session_item(row: dict) -> dbc.AccordionItem:
    color, label = _ATTENTION[row["attention"]]
    title = html.Div([
        dbc.Badge(label, color=color, className="me-2"),
        html.Strong((row["question"] or "(no question)")[:80], className="me-2"),
        html.Span(f"{row['owner']} · {row['source']} · {_ago(row['updated_at'])}", className="small text-muted me-3"),
        *_gate_chips(row["gates"]),
        html.Span(" · ", className="text-muted mx-1"), _delta(row["blocker_delta"]),
        html.Span(f" · ${row['cost_usd']:.2f}" if row.get("cost_usd") is not None else "", className="small text-muted"),
    ], className="d-flex flex-wrap align-items-center gap-1")

    blockers = [html.Li([html.Code(b.get("source", ""), className="me-1"),
                         (b.get("column") + ": ") if b.get("column") not in (None, "—", "?") else "",
                         b.get("issue", "")], className="small") for b in row["blockers"][:5]]
    body = html.Div([
        html.Div([html.Strong("Data: ", className="small"),
                  html.Span(", ".join(row["datasets"] or []) or "—", className="small")], className="mb-2"),
        html.Div([html.Strong("Blocking", className="small text-danger"),
                  html.Ul(blockers, className="mb-2")]) if blockers else
        html.P("Nothing blocking.", className="small text-success mb-2"),
        html.Strong("What each step concluded", className="small"),
        html.Ol([html.Li([html.Code(h["skill"], className="me-2"), h["line"]], className="small")
                 for h in row["headlines"]], className="mb-1"),
        html.Div(f"{row['n_runs']} runs · {row['n_cautions']} cautions", className="small text-muted"),
    ])
    return dbc.AccordionItem(body, title=title, item_id=row["id"])


def sessions_panel(rows: list[dict], attention_only: bool = False):
    if attention_only:
        rows = [r for r in rows if r["attention"] in ("red", "amber")]
    if not rows:
        return dbc.Alert("No sessions yet. Start one on the Workbench, or in a notebook with "
                         "WorkflowSession(context=...) — both appear here automatically.", color="secondary")
    counts = {a: sum(1 for r in rows if r["attention"] == a) for a in _ATTENTION}
    summary = html.Div([dbc.Badge(f"{counts[a]} {_ATTENTION[a][1].lower()}", color=_ATTENTION[a][0], className="me-1")
                        for a in _ATTENTION if counts[a]], className="mb-2")
    return html.Div([summary, dbc.Accordion([_session_item(r) for r in rows], start_collapsed=True,
                                            always_open=True, style=_CARD)])


@callback(Output("sess-body", "children"), Input("sess-poll", "n_intervals"), Input("sess-attention-only", "value"))
def render_sessions(_, attention_only):
    from skills.team_sessions import team_overview
    return sessions_panel(team_overview(), bool(attention_only))

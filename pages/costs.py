"""Costs: estimated API spend and reliability by skill, user and model (from the usage ledger)."""
import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

dash.register_page(__name__, path="/costs", name="Costs", order=9)

_CARD = {"backgroundColor": "#16162a", "border": "1px solid #252545"}


def layout():
    return html.Div([
        dcc.Interval(id="cost-poll", interval=30_000, n_intervals=0),
        dbc.Row([
            dbc.Col([
                html.H4([html.I(className="bi bi-cash-coin me-2"), "Costs"], className="mb-1"),
                html.P("Estimated API spend from token counts at list prices (config.MODEL_PRICES), "
                       "recorded for every Claude call. Reconcile with the Anthropic Usage & Cost Admin "
                       "API for invoice figures.", className="text-muted small mb-0"),
            ]),
            dbc.Col(dbc.RadioItems(
                id="cost-range", inline=True, value=30,
                options=[{"label": "7 days", "value": 7}, {"label": "30 days", "value": 30},
                         {"label": "All", "value": 0}],
            ), width="auto", className="d-flex align-items-center"),
        ], className="mb-3"),
        html.Div(id="cost-body"),
    ])


def _kpi(label, value, hint=""):
    return dbc.Col(dbc.Card(dbc.CardBody([
        html.Div(label, className="small text-muted"),
        html.Div(value, className="fs-4 fw-semibold"),
        html.Div(hint, className="small text-muted"),
    ]), style=_CARD), xs=6, md=True)


def _money(x):
    return "—" if x is None else (f"${x:,.2f}" if x >= 1 else f"${x:.3f}")


def _table(rows, key, label):
    head = [label, "Spend", "Calls", "Runs", "$ / run", "Success", "p50 run time", "Failed calls"]
    body = []
    for r in rows:
        sr = r.get("success_rate")
        body.append(html.Tr([
            html.Td(html.Code(str(r[key]), style={"fontSize": "0.75rem"})),
            html.Td(_money(r["spend"])), html.Td(r["calls"]),
            html.Td(int(r["runs"]) if r.get("runs") is not None else "—"),
            html.Td(_money(r.get("run_cost"))),
            html.Td(f"{sr:.0%}" if sr is not None else "—",
                    className="text-warning" if sr is not None and sr < 0.95 else ""),
            html.Td(f"{r['p50_run_s']:.0f}s" if r.get("p50_run_s") is not None else "—"),
            html.Td(r["failed_calls"] or "—"),
        ]))
    return dbc.Table([html.Thead(html.Tr([html.Th(h) for h in head])), html.Tbody(body)],
                     size="sm", hover=True, responsive=True, className="mb-0")


def _daily_chart(daily):
    days = sorted({d["day"] for d in daily})
    fig = go.Figure()
    for skill in sorted({d["skill"] for d in daily}):
        vals = {d["day"]: d["cost_usd"] for d in daily if d["skill"] == skill}
        fig.add_bar(name=skill, x=days, y=[vals.get(day, 0) for day in days])
    fig.update_layout(barmode="stack", height=260, margin={"l": 40, "r": 10, "t": 10, "b": 30},
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font={"color": "#aaa", "size": 10}, legend={"orientation": "h", "y": -0.25},
                      yaxis={"title": "USD", "gridcolor": "#252545"})
    return dcc.Graph(figure=fig, config={"displayModeBar": False})


def cost_panel(s: dict | None):
    if not s:
        return dbc.Alert("No usage recorded in this period yet. Every skill call is logged to "
                         "feedback/usage.jsonl as it happens.", color="secondary")
    section = lambda title, child: dbc.Card([dbc.CardHeader(title), dbc.CardBody(child, className="p-2")],
                                            className="mb-3", style=_CARD)
    fail_share = s["failure_spend"] / s["total_spend"] if s["total_spend"] else 0
    return html.Div([
        dbc.Row([
            _kpi("Estimated spend", _money(s["total_spend"]), f"{s['calls']} calls"),
            _kpi("Skill runs", s["runs"], f"{s['users']} user{'s' if s['users'] != 1 else ''}"),
            _kpi("Cost per successful run", _money(s["cost_per_successful_run"]), "includes failed attempts"),
            _kpi("Spent on failures", _money(s["failure_spend"]), f"{fail_share:.0%} of spend"),
        ], className="g-2 mb-3"),
        section("Daily spend by skill", _daily_chart(s["daily"])),
        section("By skill", _table(s["by_skill"], "skill", "Skill")),
        dbc.Row([
            dbc.Col(section("By user", _table(s["by_user"], "user", "User")), lg=7),
            dbc.Col(section("By model", _table(s["by_model"], "model", "Model")), lg=5),
        ]),
    ])


@callback(Output("cost-body", "children"), Input("cost-poll", "n_intervals"), Input("cost-range", "value"))
def render_costs(_, days):
    from skills.usage import load, summarize
    return cost_panel(summarize(load(), days or None))

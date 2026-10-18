import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timezone, timedelta
from dash import Input, Output, State, callback, dcc, html
from dash.exceptions import PreventUpdate

dash.register_page(__name__, path="/benchmarks", name="Benchmarks", order=8)

SKILLS = [
    "data_quality", "eda", "join_auditor", "nl_to_spark", "mlops_review",
    "model_selector", "leakage_detector", "feature_engineering",
]
SKILL_LABEL = {
    "data_quality":        "Data Quality",
    "eda":                 "EDA",
    "join_auditor":        "Join Auditor",
    "nl_to_spark":         "NL → PySpark",
    "mlops_review":        "MLOps Review",
    "model_selector":      "Model Selector",
    "leakage_detector":    "Leakage Detector",
    "feature_engineering": "Feature Engineering",
}
SKILL_COLOR = {
    "data_quality":        "#6478ff",
    "eda":                 "#64c8ff",
    "join_auditor":        "#ff7eb3",
    "nl_to_spark":         "#ffc107",
    "mlops_review":        "#20c997",
    "model_selector":      "#fd7e14",
    "leakage_detector":    "#dc3545",
    "feature_engineering": "#6f42c1",
}

READINESS_THRESHOLDS = {"ready": 4.0, "maturing": 3.0}


def _readiness(score: float | None) -> tuple[str, str]:
    """Returns (label, color) for a score."""
    if score is None:
        return "No data", "secondary"
    if score >= READINESS_THRESHOLDS["ready"]:
        return "Ready", "success"
    if score >= READINESS_THRESHOLDS["maturing"]:
        return "Maturing", "warning"
    return "Needs Work", "danger"


def layout():
    return html.Div([
        dcc.Store(id="bench-refresh-trigger", data=0),
        dcc.Interval(id="bench-poll-interval", interval=10000, n_intervals=0),

        # ── Header ────────────────────────────────────────────────────────────
        dbc.Row([
            dbc.Col([
                html.H4([html.I(className="bi bi-speedometer2 me-2"), "Skill Benchmarks"],
                        className="mb-1"),
                html.P(
                    "Each time agents run a skill on synthetic data, the output is scored against "
                    "a known answer key (quantitative recall) and rated by the persona for "
                    "usefulness and clarity (qualitative). Trend over time shows whether skills "
                    "are improving. Readiness = 4.0+ overall.",
                    className="text-muted mb-0 small",
                ),
            ]),
            dbc.Col(html.Div(id="bench-header-stats"), width="auto",
                    className="d-flex align-items-center"),
        ], className="mb-4 align-items-start"),

        # ── Per-skill health grid ─────────────────────────────────────────────
        html.Div([
            html.P([
                html.Strong("Skill Health  "),
                html.Span("Current benchmark score per skill · green = ready (4.0+) · "
                          "amber = maturing (3.0–4.0) · red = needs work (<3.0) · "
                          "delta shows change vs previous run",
                          className="text-muted small"),
            ], className="mb-2"),
            html.Div(id="bench-summary-cards"),
        ], className="mb-4"),

        # ── Trend chart ───────────────────────────────────────────────────────
        dbc.Card([
            dbc.CardHeader([
                html.Span("Overall Score Trend", className="fw-semibold"),
                dbc.ButtonGroup([
                    dbc.Button("30d",  id="bench-range-30",  size="sm", color="primary",
                               outline=False, className="ms-2"),
                    dbc.Button("90d",  id="bench-range-90",  size="sm", color="secondary",
                               outline=True),
                    dbc.Button("All",  id="bench-range-all", size="sm", color="secondary",
                               outline=True),
                ], className="float-end"),
            ]),
            dbc.CardBody(
                dcc.Loading(
                    dcc.Graph(id="bench-trend-chart", config={"displayModeBar": False},
                              style={"height": "340px"}),
                    type="circle", color="#6478ff",
                )
            ),
        ], className="mb-4", style={"backgroundColor": "#16162a", "border": "1px solid #252545"}),

        # ── Per-skill detail tabs ─────────────────────────────────────────────
        dbc.Card([
            dbc.CardHeader("Skill Detail"),
            dbc.CardBody([
                dbc.Tabs(
                    [dbc.Tab(label=SKILL_LABEL[s], tab_id=s) for s in SKILLS],
                    id="bench-skill-tabs", active_tab="data_quality",
                ),
                html.Div(id="bench-skill-detail", className="mt-3"),
            ]),
        ], style={"backgroundColor": "#16162a", "border": "1px solid #252545"}),

        # Hidden store for selected range
        dcc.Store(id="bench-range-store", data=30),
    ])


# ── Poll ──────────────────────────────────────────────────────────────────────

@callback(
    Output("bench-refresh-trigger", "data"),
    Input("bench-poll-interval", "n_intervals"),
    State("bench-refresh-trigger", "data"),
)
def poll(_n, current_count):
    from agents.scheduler import get_benchmarks
    from dash.exceptions import PreventUpdate
    n = len(get_benchmarks())
    if n == current_count:
        raise PreventUpdate
    return n


# ── Range selector ────────────────────────────────────────────────────────────

@callback(
    Output("bench-range-store",  "data"),
    Output("bench-range-30",     "outline"),
    Output("bench-range-90",     "outline"),
    Output("bench-range-all",    "outline"),
    Output("bench-range-30",     "color"),
    Output("bench-range-90",     "color"),
    Output("bench-range-all",    "color"),
    Input("bench-range-30",  "n_clicks"),
    Input("bench-range-90",  "n_clicks"),
    Input("bench-range-all", "n_clicks"),
    prevent_initial_call=True,
)
def set_range(n30, n90, nall):
    from dash import ctx
    tid = ctx.triggered[0]["prop_id"]
    if "30" in tid:
        sel = 30
    elif "90" in tid:
        sel = 90
    else:
        sel = 0
    # outline=False = active, outline=True = inactive
    def _btn(days):
        active = (days == sel)
        return (not active, "primary" if active else "secondary")
    o30, c30 = _btn(30)
    o90, c90 = _btn(90)
    oall, call_ = _btn(0)
    return sel, o30, o90, oall, c30, c90, call_


# ── Main render ───────────────────────────────────────────────────────────────

@callback(
    Output("bench-summary-cards",  "children"),
    Output("bench-trend-chart",    "figure"),
    Output("bench-header-stats",   "children"),
    Input("bench-refresh-trigger", "data"),
    Input("bench-range-store",     "data"),
    prevent_initial_call=True,
)
def render_dashboard(_, days):
    from agents.scheduler import get_benchmarks

    all_benches = get_benchmarks()
    if not all_benches:
        empty_fig = _empty_figure("No benchmark data yet — run a simulation to generate scores.")
        cards = dbc.Alert("No data yet. Run a simulation from the Feedback page.", color="secondary")
        return cards, empty_fig, ""

    # Filter by time range
    if days:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        benches = [b for b in all_benches if _parse_ts(b.get("timestamp", "")) >= cutoff]
        if not benches:
            benches = all_benches  # fallback if filter is too narrow
    else:
        benches = all_benches

    # ── Summary cards ─────────────────────────────────────────────────────────
    cards = dbc.Row([
        dbc.Col(_skill_summary_card(skill, benches), xs=12, sm=6, lg=4, xl="auto",
                className="mb-3")
        for skill in SKILLS
    ])

    # ── Trend chart ───────────────────────────────────────────────────────────
    fig = _build_trend_chart(benches)

    # ── Header stats ──────────────────────────────────────────────────────────
    n_runs = len(set(
        (b["scenario_id"], b["persona_name"], b.get("timestamp", "")[:10])
        for b in all_benches
    ))
    stats = html.Div([
        dbc.Badge(f"{len(all_benches)} total evaluations", color="secondary", className="me-1"),
        dbc.Badge(f"{n_runs} sim runs", color="info"),
    ])

    return cards, fig, stats


# ── Skill detail tab ──────────────────────────────────────────────────────────

@callback(
    Output("bench-skill-detail", "children"),
    Input("bench-skill-tabs",      "active_tab"),
    Input("bench-refresh-trigger", "data"),
    Input("bench-range-store",     "data"),
    prevent_initial_call=True,
)
def render_skill_detail(skill, _, days):
    from agents.scheduler import get_benchmarks

    all_benches = get_benchmarks(skill=skill)
    if not all_benches:
        return html.P(f"No benchmark data for {SKILL_LABEL.get(skill, skill)} yet.",
                      className="text-muted text-center py-4")

    if days:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        benches = [b for b in all_benches if _parse_ts(b.get("timestamp","")) >= cutoff]
        if not benches:
            benches = all_benches
    else:
        benches = all_benches

    scores = [b["overall"] for b in benches if b.get("overall") is not None]
    latest = benches[0]
    quant  = latest.get("quantitative")
    qual   = latest.get("qualitative", {})

    readiness_label, readiness_color = _readiness(scores[0] if scores else None)
    avg_score = round(sum(scores) / len(scores), 2) if scores else None

    # Score breakdown for latest run
    breakdown_rows = []
    if quant:
        breakdown_rows.append(html.Tr([
            html.Td("Recall (known issues found)", className="small text-muted"),
            html.Td(f"{quant.get('recall', 0) * 100:.0f}%", className="small fw-bold"),
        ]))
        breakdown_rows.append(html.Tr([
            html.Td("Quantitative score (0–5)", className="small text-muted"),
            html.Td(f"{quant.get('score', '—')}", className="small fw-bold"),
        ]))
    if qual:
        breakdown_rows.append(html.Tr([
            html.Td("Usefulness (persona rating)", className="small text-muted"),
            html.Td(f"{qual.get('usefulness', '—')} / 5", className="small fw-bold"),
        ]))
        breakdown_rows.append(html.Tr([
            html.Td("Clarity (persona rating)", className="small text-muted"),
            html.Td(f"{qual.get('clarity', '—')} / 5", className="small fw-bold"),
        ]))

    # Recent evaluations table
    recent_rows = []
    for b in benches[:10]:
        q = b.get("quantitative") or {}
        ql = b.get("qualitative") or {}
        recall_txt = f"{q.get('recall', 0)*100:.0f}%" if q else "—"
        readiness_l, readiness_c = _readiness(b.get("overall"))
        recent_rows.append(html.Tr([
            html.Td(_time_ago(b.get("timestamp", "")), className="small text-muted"),
            html.Td(b.get("scenario_title", ""), className="small"),
            html.Td(b.get("persona_name",   ""), className="small"),
            html.Td(b.get("dataset",        ""), className="small"),
            html.Td(recall_txt, className="small"),
            html.Td(f"{ql.get('usefulness','—')}/{ql.get('clarity','—')}", className="small"),
            html.Td(f"{b.get('overall','—')}", className="small fw-bold"),
            html.Td(dbc.Badge(readiness_l, color=readiness_c, pill=True,
                              style={"fontSize":"0.65rem"})),
        ]))

    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Span("Latest overall score: ", className="small text-muted"),
                    html.Span(f"{scores[0]:.1f} / 5.0" if scores else "—",
                              className="fw-bold fs-5"),
                    dbc.Badge(readiness_label, color=readiness_color, pill=True,
                              className="ms-2 align-middle", style={"fontSize":"0.7rem"}),
                ], className="mb-2"),
                html.Div([
                    html.Span("Average over period: ", className="small text-muted"),
                    html.Span(f"{avg_score:.2f}" if avg_score else "—",
                              className="small fw-bold"),
                ], className="mb-3"),
                dbc.Table([html.Tbody(breakdown_rows)],
                          bordered=False, size="sm", className="mb-0",
                          style={"backgroundColor": "transparent"}),
            ], md=4),
            dbc.Col([
                dcc.Graph(
                    figure=_build_skill_sparkline(benches, skill),
                    config={"displayModeBar": False},
                    style={"height": "180px"},
                ),
            ], md=8),
        ], className="mb-3"),

        html.Hr(style={"borderColor": "#252545"}),
        html.P("Recent evaluations", className="small text-muted mb-2"),
        dbc.Table(
            [
                html.Thead(html.Tr([
                    html.Th("When",       className="small"),
                    html.Th("Scenario",   className="small"),
                    html.Th("Persona",    className="small"),
                    html.Th("Dataset",    className="small"),
                    html.Th("Recall",     className="small"),
                    html.Th("Use/Clr",    className="small"),
                    html.Th("Overall",    className="small"),
                    html.Th("Readiness",  className="small"),
                ])),
                html.Tbody(recent_rows),
            ],
            hover=True, size="sm", responsive=True,
            style={"backgroundColor": "transparent"},
        ) if recent_rows else html.P("No recent runs.", className="text-muted small"),
    ])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _skill_summary_card(skill: str, benches: list) -> dbc.Card:
    skill_benches = [b for b in benches if b.get("skill") == skill]
    scores = [b["overall"] for b in skill_benches if b.get("overall") is not None]

    latest_score = scores[0] if scores else None
    prev_score   = scores[1] if len(scores) > 1 else None
    readiness_label, readiness_color = _readiness(latest_score)

    # Health-status border color (green/amber/red)
    status_color = {
        "success": "#20c997",
        "warning": "#ffc107",
        "danger":  "#dc3545",
    }.get(readiness_color, "#444")

    score_color = status_color if latest_score is not None else "#666"

    # Delta vs previous run
    delta_html = html.Span()
    if latest_score is not None and prev_score is not None:
        delta = round(latest_score - prev_score, 1)
        if delta > 0.05:
            delta_html = html.Span(f" +{delta:.1f}", className="small text-success",
                                   title="vs previous run")
        elif delta < -0.05:
            delta_html = html.Span(f" {delta:.1f}", className="small text-danger",
                                   title="vs previous run")
        else:
            delta_html = html.Span(" →", className="small text-muted",
                                   title="no change vs previous run")

    last_run_str = _time_ago(skill_benches[0]["timestamp"]) if skill_benches else "never"

    return dbc.Card(
        dbc.CardBody([
            html.Div([
                html.Span(SKILL_LABEL[skill], className="small fw-semibold"),
                dbc.Badge(readiness_label, color=readiness_color, pill=True,
                          className="float-end", style={"fontSize": "0.6rem"}),
            ], className="mb-1 d-flex align-items-center justify-content-between"),
            html.Div([
                html.Span(
                    f"{latest_score:.1f}" if latest_score is not None else "—",
                    style={"fontSize": "1.8rem", "fontWeight": "700", "color": score_color},
                ),
                html.Span(" / 5", className="small text-muted"),
                delta_html,
            ], className="mb-1"),
            html.Div(
                f"{len(skill_benches)} run{'s' if len(skill_benches) != 1 else ''} · {last_run_str}",
                className="text-muted", style={"fontSize": "0.7rem"},
            ),
        ], style={"padding": "10px 12px"}),
        style={
            "backgroundColor": "#16162a",
            "borderTop":    "1px solid #252545",
            "borderRight":  "1px solid #252545",
            "borderBottom": "1px solid #252545",
            "borderLeft":   f"4px solid {status_color}",
            "minWidth": "150px",
            "borderRadius": "4px",
        },
    )


def _build_trend_chart(benches: list) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#d0d3e8", "size": 11},
        margin={"t": 10, "b": 40, "l": 40, "r": 10},
        legend={"bgcolor": "rgba(0,0,0,0)", "font": {"size": 10}},
        xaxis={"gridcolor": "#252545", "zerolinecolor": "#252545"},
        yaxis={"gridcolor": "#252545", "zerolinecolor": "#252545", "range": [0, 5.2],
               "tickvals": [1, 2, 3, 4, 5]},
        hovermode="x unified",
    )

    # Add readiness bands
    fig.add_hrect(y0=4.0, y1=5.2, fillcolor="rgba(32,201,151,0.07)", line_width=0)
    fig.add_hrect(y0=3.0, y1=4.0, fillcolor="rgba(255,193,7,0.07)",  line_width=0)
    fig.add_hrect(y0=0,   y1=3.0, fillcolor="rgba(220,53,69,0.07)",  line_width=0)
    fig.add_hline(y=4.0, line_dash="dot", line_color="rgba(32,201,151,0.4)",  line_width=1)
    fig.add_hline(y=3.0, line_dash="dot", line_color="rgba(255,193,7,0.4)",   line_width=1)

    for skill in SKILLS:
        skill_benches = sorted(
            [b for b in benches if b.get("skill") == skill and b.get("overall") is not None],
            key=lambda b: b.get("timestamp", ""),
        )
        if not skill_benches:
            continue

        xs = [b["timestamp"][:19].replace("T", " ") for b in skill_benches]
        ys = [b["overall"] for b in skill_benches]

        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines+markers",
            name=SKILL_LABEL[skill],
            line={"color": SKILL_COLOR[skill], "width": 2},
            marker={"size": 5},
            hovertemplate=f"<b>{SKILL_LABEL[skill]}</b><br>Score: %{{y:.2f}}<br>%{{x}}<extra></extra>",
        ))

    if not fig.data:
        fig.add_annotation(
            text="No benchmark data in this period",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font={"color": "#6a6d88"},
        )
    return fig


def _build_skill_sparkline(benches: list, skill: str) -> go.Figure:
    sorted_b = sorted(
        [b for b in benches if b.get("overall") is not None],
        key=lambda b: b.get("timestamp", ""),
    )
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#d0d3e8", "size": 10},
        margin={"t": 5, "b": 25, "l": 30, "r": 5},
        xaxis={"gridcolor": "#252545", "showticklabels": False},
        yaxis={"gridcolor": "#252545", "range": [0, 5.2], "tickvals": [1, 2, 3, 4, 5]},
        showlegend=False,
    )
    if sorted_b:
        xs = [b["timestamp"][:19].replace("T", " ") for b in sorted_b]
        ys = [b["overall"] for b in sorted_b]
        quant_ys = [b["quantitative"]["score"] if b.get("quantitative") else None for b in sorted_b]
        qual_ys  = [(b["qualitative"]["usefulness"] + b["qualitative"]["clarity"]) / 2
                    if b.get("qualitative") else None for b in sorted_b]

        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name="Overall",
                                 line={"color": SKILL_COLOR.get(skill, "#6478ff"), "width": 2},
                                 marker={"size": 5}))
        if any(v is not None for v in quant_ys):
            fig.add_trace(go.Scatter(x=xs, y=quant_ys, mode="lines", name="Quantitative",
                                     line={"color": "#aaa", "width": 1, "dash": "dot"}))
        if any(v is not None for v in qual_ys):
            fig.add_trace(go.Scatter(x=xs, y=qual_ys, mode="lines", name="Qualitative",
                                     line={"color": "#888", "width": 1, "dash": "dash"}))
        fig.add_hline(y=4.0, line_dash="dot", line_color="rgba(32,201,151,0.35)", line_width=1)

    return fig


def _empty_figure(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#6a6d88"},
        margin={"t": 20, "b": 20, "l": 20, "r": 20},
        xaxis={"visible": False}, yaxis={"visible": False},
        annotations=[{"text": msg, "xref": "paper", "yref": "paper",
                       "x": 0.5, "y": 0.5, "showarrow": False}],
    )
    return fig


def _parse_ts(ts: str):
    try:
        dt = datetime.fromisoformat(ts)
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
    except Exception:
        return datetime.min.replace(tzinfo=timezone.utc)


def _time_ago(iso: str) -> str:
    try:
        dt = _parse_ts(iso)
        mins = int((datetime.now(timezone.utc) - dt).total_seconds() / 60)
        if mins < 60:   return f"{mins}m ago"
        if mins < 1440: return f"{mins//60}h ago"
        return f"{mins//1440}d ago"
    except Exception:
        return iso

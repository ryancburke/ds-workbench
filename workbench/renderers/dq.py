"""Workbench renderers for Data Quality results (result dict → Dash components)."""
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html

from workbench.constants import DQ_ROW_STYLE, DQ_SEV_COLOR, DQ_SEV_HEADER, DQ_SEV_RANK


# ── DQ helpers ────────────────────────────────────────────────────────────────

def _dq_gauge(score: int) -> go.Figure:
    color = "#dc3545" if score < 50 else "#ffc107" if score < 75 else "#198754"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        gauge={"axis": {"range": [0, 100], "tickcolor": "#888"}, "bar": {"color": color},
               "bgcolor": "#1e1e2e",
               "steps": [{"range": [0, 50],  "color": "#3d1515"},
                         {"range": [50, 75],  "color": "#3d3215"},
                         {"range": [75, 100], "color": "#153d1e"}]},
        title={"text": "Quality Score", "font": {"color": "#ccc", "size": 13}},
        number={"font": {"color": color, "size": 48}},
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=180,
                      margin={"t": 40, "b": 0, "l": 10, "r": 10})
    return fig


def _dq_null_chart(profile: dict) -> go.Figure | None:
    try:
        cols = list(profile["columns"].keys())
        pcts = [profile["columns"][c]["null_pct"] for c in cols]
        fig = px.bar(x=pcts, y=cols, orientation="h", labels={"x": "Null %", "y": ""},
                     color=pcts, color_continuous_scale=["#198754", "#ffc107", "#dc3545"],
                     range_color=[0, 100])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#ccc", coloraxis_showscale=False,
                          height=max(120, len(cols) * 22), title="Null Rates by Column",
                          margin={"t": 30, "b": 10, "l": 110, "r": 10},
                          yaxis={"categoryorder": "total ascending"})
        fig.update_traces(marker_line_width=0)
        return fig
    except Exception:
        return None


def _dq_findings_table(findings: list) -> list:
    sorted_f = sorted(findings, key=lambda f: DQ_SEV_RANK.get(f.get("severity", "info"), 2))
    rows, last_sev = [], None
    for f in sorted_f:
        sev = f.get("severity", "info")
        if sev != last_sev:
            icon_cls, label = DQ_SEV_HEADER.get(sev, ("bi-circle", sev.title()))
            rows.append(html.Tr(html.Td(
                [html.I(className=f"bi {icon_cls} me-2"), html.Strong(label)],
                colSpan=4, className="small py-1",
                style={"background": "rgba(255,255,255,0.04)",
                       "borderTop": "1px solid rgba(255,255,255,0.1)"},
            )))
            last_sev = sev
        rows.append(html.Tr([
            html.Td(dbc.Badge(sev.upper(), color=DQ_SEV_COLOR.get(sev, "secondary"))),
            html.Td(html.Code(f.get("column", "—"), style={"fontSize": "0.78rem"})),
            html.Td(f.get("issue_type", "").replace("_", " ").title(), className="text-muted small"),
            html.Td([
                html.Div(f.get("description", ""), className="small"),
                html.Div(f.get("recommendation", ""), className="small text-muted mt-1"),
            ]),
        ], style=DQ_ROW_STYLE.get(sev, {})))
    return rows


_DQ_PRIORITY_WEIGHT = {"critical": 3, "warning": 2, "info": 1}


def _dq_priority(f: dict) -> tuple:
    """Return (sort_key, label, color) for a DQ finding."""
    sev        = f.get("severity", "info")
    rows_pct   = f.get("affected_rows_pct") or 0.0
    weight     = _DQ_PRIORITY_WEIGHT.get(sev, 1)
    score      = weight * (1 + rows_pct / 100)   # 0–4 range
    if score >= 2.5:
        return score, "High",   "danger"
    elif score >= 1.5:
        return score, "Medium", "warning"
    else:
        return score, "Low",    "secondary"


def _dq_top_blockers(findings: list) -> html.Div | None:
    blockers = [f for f in findings if f.get("severity") in ("critical", "warning")]
    if not blockers:
        return None
    # Sort by computed priority score descending
    blockers.sort(key=lambda f: -_dq_priority(f)[0])
    rows = []
    for f in blockers[:5]:
        sev   = f.get("severity", "warning")
        _, pri_label, pri_color = _dq_priority(f)
        rows.append(html.Tr([
            html.Td(dbc.Badge(sev.upper(), color=DQ_SEV_COLOR.get(sev, "secondary"),
                              style={"minWidth": "68px"})),
            html.Td(dbc.Badge(pri_label, color=pri_color, style={"minWidth": "60px"})),
            html.Td(html.Code(f.get("column", "—"), style={"fontSize": "0.78rem"})),
            html.Td(f.get("description", ""), className="small"),
            html.Td(f.get("recommendation", ""), className="small text-muted"),
        ], style=DQ_ROW_STYLE.get(sev, {})))
    return dbc.Card([
        dbc.CardHeader([
            html.I(className="bi bi-exclamation-octagon-fill text-danger me-2"),
            html.Strong("Top Blockers"),
            html.Span(" — fix these before proceeding", className="text-muted small ms-2"),
        ]),
        dbc.CardBody(
            dbc.Table(
                [html.Thead(html.Tr([html.Th("Sev"), html.Th("Priority"), html.Th("Column"),
                                     html.Th("Issue"), html.Th("Fix")]))] +
                [html.Tbody(rows)],
                hover=True, responsive=True, size="sm",
            ),
            className="p-0",
        ),
    ], className="mb-3", style={"borderColor": "#dc354540"})


def _dq_render(result: dict, profile: dict) -> html.Div:
    findings  = result.get("findings", [])
    counts    = {s: sum(1 for f in findings if f.get("severity") == s)
                 for s in ("critical", "warning", "info")}
    rows      = _dq_findings_table(findings)
    null_fig  = _dq_null_chart(profile)
    join_card = _dq_join_key_card(findings, profile) if profile else None
    col_qual  = _dq_column_quality_table(profile) if profile else None
    blockers  = _dq_top_blockers(findings)
    return html.Div([
        _dq_verdict_hero(findings),
        blockers or html.Div(),
        _dq_schema_snapshot(profile) if profile else html.Div(),
        join_card or html.Div(),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=_dq_gauge(result.get("overall_score", 0)),
                              config={"displayModeBar": False}), width=4),
            dbc.Col([
                html.P(result.get("summary", ""), className="mt-2 small"),
                html.Div([
                    dbc.Badge([html.I(className="bi bi-x-circle me-1"),
                               f"{counts['critical']} Critical"],
                              color="danger", className="me-2 p-2"),
                    dbc.Badge([html.I(className="bi bi-exclamation-triangle me-1"),
                               f"{counts['warning']} Warnings"],
                              color="warning", text_color="dark", className="me-2 p-2"),
                    dbc.Badge([html.I(className="bi bi-info-circle me-1"),
                               f"{counts['info']} Info"],
                              color="info", text_color="dark", className="p-2"),
                ], className="mt-2"),
            ], width=8),
        ], className="mb-3"),
        dbc.Card([
            dbc.CardHeader(html.Strong("Findings")),
            dbc.CardBody(
                dbc.Table(
                    [html.Thead(html.Tr([html.Th("Sev"), html.Th("Column"),
                                        html.Th("Issue"), html.Th("Detail")]))] +
                    [html.Tbody(rows)],
                    hover=True, responsive=True, size="sm",
                ) if rows else html.P("No issues found.", className="text-success mb-0"),
                className="p-0",
            ),
        ], className="mb-3"),
        col_qual or html.Div(),
        dbc.Card([
            dbc.CardHeader(html.Strong("Null Rates")),
            dbc.CardBody(dcc.Graph(figure=null_fig, config={"displayModeBar": False})),
        ]) if null_fig else html.Div(),
    ])


def _dq_schema_snapshot(profile: dict) -> dbc.Card:
    shape     = profile.get("shape", {})
    cols_dict = profile.get("columns", {})
    rows      = shape.get("rows", 0)
    n_cols    = shape.get("columns", len(cols_dict))
    dupes     = profile.get("duplicate_rows", 0)
    n_numeric = sum(1 for c in cols_dict.values() if any(t in c.get("dtype","") for t in ("int","float")))
    n_text    = sum(1 for c in cols_dict.values() if "object" in c.get("dtype",""))
    n_date    = sum(1 for c in cols_dict.values() if "datetime" in c.get("dtype",""))
    all_nulls = [c.get("null_pct", 0) for c in cols_dict.values()]
    avg_null  = round(sum(all_nulls) / len(all_nulls), 1) if all_nulls else 0

    def _stat(val, label, danger=False):
        return html.Span([
            html.Strong(str(val), className="me-1"),
            html.Span(label, className="text-muted"),
        ], className="me-4 small",
           style={"color": "#ff6b6b"} if danger else {})

    items = [
        _stat(f"{rows:,}", "rows"),
        _stat(n_cols, "columns"),
        _stat(dupes, "duplicate rows", danger=dupes > 0),
        _stat(f"{avg_null}%", "avg null rate", danger=avg_null > 10),
    ]
    if n_numeric: items.append(_stat(n_numeric, "numeric"))
    if n_text:    items.append(_stat(n_text, "text"))
    if n_date:    items.append(_stat(n_date, "datetime"))

    return dbc.Card([
        dbc.CardHeader(html.Strong("Schema Snapshot")),
        dbc.CardBody(html.Div(items, className="d-flex flex-wrap")),
    ], className="mb-3")


_ID_SUFFIXES = ("_id", "_key", "_pk", "_fk", "_code", "_ref")


def _dq_join_key_card(findings: list, profile: dict) -> dbc.Card | None:
    cols_dict = profile.get("columns", {})
    key_cols  = [c for c in cols_dict
                 if any(c.lower().endswith(p) or c.lower().startswith(p[1:] + "_")
                        for p in _ID_SUFFIXES)]
    if not key_cols:
        return None

    rows = []
    for col in key_cols:
        info       = cols_dict[col]
        null_pct   = info.get("null_pct", 0)
        unique_pct = info.get("unique_pct", 0)
        nc = "danger" if null_pct > 0 else "success"
        uc = "success" if unique_pct > 95 else "warning" if unique_pct > 50 else "danger"
        col_finds  = [f for f in findings if f.get("column") == col]
        issue_text = col_finds[0].get("issue_type","").replace("_"," ").title() if col_finds else "—"
        rows.append(html.Tr([
            html.Td(html.Code(col, style={"fontSize":"0.78rem"})),
            html.Td(dbc.Badge(f"{null_pct}% null", color=nc,
                              text_color="dark" if nc == "warning" else None,
                              className="small")),
            html.Td(dbc.Badge(f"{unique_pct:.0f}% unique", color=uc,
                              text_color="dark" if uc == "warning" else None,
                              className="small")),
            html.Td(issue_text, className="small text-muted"),
        ]))

    return dbc.Card([
        dbc.CardHeader([html.I(className="bi bi-key me-2"), html.Strong("Join Key Health")]),
        dbc.CardBody(
            dbc.Table([
                html.Thead(html.Tr([html.Th("Column"), html.Th("Null Rate"),
                                    html.Th("Uniqueness"), html.Th("Issues")])),
                html.Tbody(rows),
            ], hover=True, responsive=True, size="sm"),
            className="p-0",
        ),
    ], className="mb-3", style={"borderColor": "#0d6efd40"})


def _dq_column_quality_table(profile: dict) -> dbc.Card | None:
    cols_dict = profile.get("columns", {})
    if not cols_dict:
        return None

    groups: dict[str, list] = {"Numeric": [], "Text": [], "DateTime": [], "Boolean": []}
    for col, info in cols_dict.items():
        dtype = info.get("dtype", "")
        if any(t in dtype for t in ("int", "float")):
            groups["Numeric"].append((col, info))
        elif "datetime" in dtype:
            groups["DateTime"].append((col, info))
        elif "bool" in dtype:
            groups["Boolean"].append((col, info))
        else:
            groups["Text"].append((col, info))

    rows = []
    for group_label, members in groups.items():
        if not members:
            continue
        rows.append(html.Tr(html.Td(
            html.Strong(group_label, className="text-muted small"),
            colSpan=4, className="py-1",
            style={"background": "rgba(255,255,255,0.03)"},
        )))
        for col, info in sorted(members, key=lambda x: -x[1].get("null_pct", 0)):
            null_pct   = info.get("null_pct", 0)
            unique_pct = info.get("unique_pct", 0)
            null_style = {}
            if null_pct > 10:
                null_style = {"backgroundColor": "#3d1515", "color": "#ff6b6b"}
            elif null_pct > 1:
                null_style = {"backgroundColor": "#3d3215", "color": "#ffd93d"}
            rows.append(html.Tr([
                html.Td(html.Code(col, style={"fontSize": "0.75rem"})),
                html.Td(info.get("dtype",""), className="text-muted small"),
                html.Td(f"{null_pct}%",      className="small font-monospace", style=null_style),
                html.Td(f"{unique_pct:.0f}%", className="small font-monospace"),
            ]))

    return dbc.Card([
        dbc.CardHeader([html.Strong("Column Quality Matrix"),
                        html.Span(" — red >10% null · yellow 1-10%",
                                  className="text-muted small ms-2")]),
        dbc.CardBody(
            dbc.Table([
                html.Thead(html.Tr([html.Th("Column"), html.Th("Type"),
                                    html.Th("Null %"), html.Th("Unique %")])),
                html.Tbody(rows),
            ], hover=True, responsive=True, size="sm"),
            className="p-0",
        ),
    ], className="mb-3")


def _dq_verdict_hero(findings: list) -> dbc.Alert:
    critical = [f for f in findings if f.get("severity") == "critical"]
    warning  = [f for f in findings if f.get("severity") == "warning"]
    if critical:
        color, icon, verdict, items = "danger",  "bi-x-octagon-fill",              "BLOCKING",     critical[:3]
    elif warning:
        color, icon, verdict, items = "warning", "bi-exclamation-triangle-fill",   "NEEDS REVIEW", warning[:3]
    else:
        color, icon, verdict, items = "success", "bi-check-circle-fill",           "CLEAR",        []
    children = [
        html.Strong([html.I(className=f"bi {icon} me-2"), f"Data Quality: {verdict}"],
                    className="d-block mb-1"),
    ]
    if items:
        children.append(html.Ul(
            [html.Li(f.get("description", "")[:80], className="small") for f in items],
            className="mb-0 ps-3",
        ))
    return dbc.Alert(children, color=color, className="py-2 mb-3")


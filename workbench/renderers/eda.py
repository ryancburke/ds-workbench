"""Workbench renderers for EDA results (result dict → Dash components)."""
import io

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from dash import dcc, html

from skills.dtypes import is_text


# ── EDA helpers ───────────────────────────────────────────────────────────────

def _eda_dark(fig, height=260):
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#ccc", height=height,
                      margin={"t": 40, "b": 20, "l": 20, "r": 20})
    return fig


def _eda_build_chart(rec: dict, df: pd.DataFrame):
    ct, x, y, title = rec.get("chart_type"), rec.get("x_column"), rec.get("y_column"), rec.get("title", "")
    try:
        if ct == "histogram" and x and x in df.columns:
            fig = px.histogram(df, x=x, title=title)
        elif ct == "scatter" and x and y and x in df.columns and y in df.columns:
            fig = px.scatter(df, x=x, y=y, title=title, opacity=0.55)
        elif ct == "bar" and x and x in df.columns:
            vc = df[x].value_counts().head(20).reset_index()
            vc.columns = [x, "count"]
            fig = px.bar(vc, x=x, y="count", title=title)
        elif ct == "box" and y and y in df.columns:
            fig = px.box(df, y=y, title=title)
        elif ct == "heatmap":
            num = df.select_dtypes(include="number")
            if len(num.columns) < 2:
                return None
            fig = px.imshow(num.corr().round(3), title=title,
                            color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        elif ct == "violin" and y and y in df.columns:
            fig = px.violin(df, y=y, title=title, box=True)
        else:
            return None
        return dcc.Graph(figure=_eda_dark(fig), config={"displayModeBar": False})
    except Exception:
        return None


def _eda_render(result: dict, df_json: str) -> html.Div:
    try:
        df = pd.read_json(io.StringIO(df_json), orient="split")
    except Exception:
        df = pd.DataFrame()
    charts  = [c for rec in result.get("recommended_charts", [])
               if (c := _eda_build_chart(rec, df)) is not None]
    targets = result.get("potential_targets", [])

    # Pin insights that mention target column names to the top
    insights = list(result.get("key_insights", []))
    if targets and insights:
        target_cols = {t["column"].lower() for t in targets}
        insights.sort(key=lambda s: 0 if any(tc in s.lower() for tc in target_cols) else 1)

    top_target = targets[0]["column"] if targets else None
    col_health = _eda_column_stats(df) if not df.empty else None
    temporal   = _eda_temporal_charts(df) if not df.empty else None
    stratified = _eda_stratified_charts(df, target_col=top_target) if not df.empty else None

    return html.Div([
        dbc.Card([
            dbc.CardHeader(html.Strong("Overview")),
            dbc.CardBody(html.P(result.get("narrative", ""), className="mb-0 small")),
        ], className="mb-3"),
        col_health or html.Div(),
        dbc.Card([
            dbc.CardHeader(html.Strong("Key Insights")),
            dbc.CardBody(html.Ul(
                [html.Li(i, className="mb-1 small") for i in insights],
                className="mb-0",
            )),
        ], className="mb-3"),
        dbc.Card([
            dbc.CardHeader(html.Strong(f"Charts ({len(charts)})")),
            dbc.CardBody(
                dbc.Row([dbc.Col(c, width=6, className="mb-3") for c in charts])
                if charts else html.P("No charts generated.", className="text-muted mb-0"),
            ),
        ], className="mb-3") if charts else html.Div(),
        temporal   or html.Div(),
        stratified or html.Div(),
        dbc.Card([
            dbc.CardHeader(html.Strong("Potential ML Targets")),
            dbc.CardBody(
                dbc.Table(
                    [html.Thead(html.Tr([html.Th("Column"), html.Th("Task"), html.Th("Rationale")]))] +
                    [html.Tbody([html.Tr([
                        html.Td(html.Code(t["column"])),
                        html.Td(dbc.Badge(t["task_type"], color="secondary")),
                        html.Td(t["rationale"], className="small"),
                    ]) for t in targets])],
                    hover=True, responsive=True, size="sm",
                ) if targets else html.P("No obvious ML targets identified.", className="text-muted mb-0"),
                className="p-0",
            ),
        ]) if targets else html.Div(),
    ])


def _eda_column_stats(df: pd.DataFrame) -> dbc.Card | None:
    if df.empty:
        return None
    rows = []
    for col in df.columns:
        s          = df[col]
        null_pct   = round(s.isnull().mean() * 100, 1)
        unique_pct = round(s.nunique() / max(len(df), 1) * 100, 1)
        dtype_s    = str(s.dtype)
        null_style = {}
        row_style  = {}
        if null_pct > 10:
            null_style = {"backgroundColor": "#3d1515", "color": "#ff6b6b"}
            row_style  = {"borderLeft": "3px solid #dc3545"}
        elif null_pct > 1:
            null_style = {"backgroundColor": "#3d3215", "color": "#ffd93d"}
            row_style  = {"borderLeft": "3px solid #ffc107"}
        rows.append(html.Tr([
            html.Td(html.Code(col, style={"fontSize": "0.75rem"})),
            html.Td(dtype_s, className="text-muted small"),
            html.Td(f"{null_pct}%",      className="small font-monospace", style=null_style),
            html.Td(f"{unique_pct:.0f}%", className="small font-monospace"),
        ], style=row_style))
    return dbc.Card([
        dbc.CardHeader([html.Strong("Column Health"),
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


def _eda_temporal_charts(df: pd.DataFrame) -> dbc.Card | None:
    parsed = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            parsed.append((col, df[col]))
        elif is_text(df[col]) and any(k in col.lower() for k in ("date", "time", "month", "year")):
            try:
                s = pd.to_datetime(df[col], errors="raise")
                parsed.append((col, s))
            except Exception:
                pass
    if not parsed:
        return None

    num_cols = df.select_dtypes(include="number").columns.tolist()
    charts = []
    for date_col, date_series in parsed[:2]:
        tmp = df.copy()
        tmp["_month"] = date_series.dt.to_period("M").astype(str)
        monthly = tmp.groupby("_month").size().reset_index(name="count")
        if len(monthly) >= 2:
            fig = px.line(monthly, x="_month", y="count", title=f"Monthly Volume — {date_col}")
            charts.append(dcc.Graph(figure=_eda_dark(fig), config={"displayModeBar": False}))
        if num_cols:
            agg = tmp.groupby("_month")[num_cols[0]].mean().reset_index()
            fig2 = px.line(agg, x="_month", y=num_cols[0],
                           title=f"Avg {num_cols[0]} over time")
            charts.append(dcc.Graph(figure=_eda_dark(fig2), config={"displayModeBar": False}))
    if not charts:
        return None
    return dbc.Card([
        dbc.CardHeader([html.I(className="bi bi-calendar3 me-2"), html.Strong("Temporal Trends")]),
        dbc.CardBody(dbc.Row([dbc.Col(c, width=6, className="mb-3") for c in charts])),
    ], className="mb-3")


def _eda_stratified_charts(df: pd.DataFrame, target_col: str | None = None) -> dbc.Card | None:
    num_cols = [c for c in df.select_dtypes(include="number").columns
                if c != target_col]
    if not num_cols:
        return None

    # Prefer the identified ML target for stratification; fall back to best categorical
    strat_col = None
    if target_col and target_col in df.columns and 2 <= df[target_col].nunique() <= 15:
        strat_col = target_col
    else:
        cat_cols = [c for c in df.columns
                    if is_text(df[c]) and 2 <= df[c].nunique() <= 10
                    and df[c].isnull().mean() < 0.3]
        if cat_cols:
            strat_col = sorted(cat_cols, key=lambda c: df[c].isnull().mean())[0]

    if not strat_col:
        return None

    is_target = strat_col == target_col
    charts = []
    for num_col in num_cols[:4]:
        fig = px.box(df, x=strat_col, y=num_col,
                     title=f"{num_col} by {strat_col}",
                     color=strat_col)
        charts.append(dcc.Graph(figure=_eda_dark(fig, height=280),
                                config={"displayModeBar": False}))
    if not charts:
        return None
    header_label = (
        f"Target Breakdown — {strat_col}" if is_target
        else f"Stratified by {strat_col}"
    )
    header_icon = "bi-crosshair" if is_target else "bi-bar-chart-steps"
    return dbc.Card([
        dbc.CardHeader([
            html.I(className=f"bi {header_icon} me-2"),
            html.Strong(header_label),
            html.Span(" — numeric features split by ML target" if is_target else "",
                      className="text-muted small ms-2"),
        ]),
        dbc.CardBody(dbc.Row([dbc.Col(c, width=6, className="mb-3") for c in charts])),
    ], className="mb-3")


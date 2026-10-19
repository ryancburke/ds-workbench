"""
DS Workbench — single-page investigation canvas.

All five skills live as accordion sections on this page.
The session header persists so context is always visible while working.
"""
import io
import time
from datetime import datetime

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html
from dash.exceptions import PreventUpdate

from skills.dtypes import is_text
dash.register_page(__name__, path="/", name="Workbench", order=0)

# ── Constants ──────────────────────────────────────────────────────────────────

SKILL_KEYS = {"data_quality", "eda", "join_auditor", "nl_to_spark", "mlops_review",
              "model_selector", "leakage_detector", "feature_engineering"}

DATASET_OPTIONS = [
    {"label": "Orders — Dirty        (nulls, mixed-case, duplicates)",    "value": "orders_dirty"},
    {"label": "Sales — Clean         (product, pricing, discounts)",      "value": "sales_clean"},
    {"label": "HR Attrition          (salary, tenure, performance)",      "value": "hr_attrition"},
    {"label": "E-commerce            (sessions, channel, device)",        "value": "ecommerce"},
    {"label": "Transactions — Fraud  (amount outliers, mixed country)",   "value": "transactions_fraud"},
    {"label": "Sales Forecasting     (revenue, promotions, seasonal)",    "value": "sales_forecasting"},
    {"label": "E-commerce Leakage   (planted leakage — for testing)",    "value": "ecommerce_leakage"},
]

JOIN_SCENARIO_OPTIONS = [
    {"label": "Orders ⋈ Customers  (fan-out — duplicate customer IDs)",    "value": "orders_customers_fanout"},
    {"label": "Sales ⋈ Products   (clean — unique keys both sides)",       "value": "sales_products_clean"},
    {"label": "Transactions ⋈ Users  (many-to-many — dupes on both)",      "value": "transactions_users_many_to_many"},
]

ML_PLACEHOLDER = """\
from pyspark.sql import SparkSession
import pandas as pd

spark = SparkSession.builder.getOrCreate()

df = spark.read.parquet("/mnt/data/transactions/2024-01-15/")
df = df.filter(df.date >= "2023-12-15")
df = df.withColumn("revenue", df.amount * 1.082)

result = df.toPandas()
for i, row in result.iterrows():
    result.at[i, "category"] = "high" if row["revenue"] > 500 else "low"

result.to_csv("/tmp/output.csv")
print("done")
"""

# Skill-level color + icon lookup
SKILL_META = {
    "data_quality": {"label": "Data Quality",   "icon": "bi-shield-check",  "color": "primary",  "bs": "#0d6efd"},
    "eda":          {"label": "Exploration",     "icon": "bi-graph-up",      "color": "success",  "bs": "#198754"},
    "join_auditor": {"label": "Join Auditor",    "icon": "bi-link-45deg",    "color": "warning",  "bs": "#ffc107"},
    "nl_to_spark":  {"label": "NL → PySpark",   "icon": "bi-translate",     "color": "info",     "bs": "#0dcaf0"},
    "mlops_review":    {"label": "MLOps Review",    "icon": "bi-gear-fill",      "color": "danger",    "bs": "#dc3545"},
    "model_selector":      {"label": "Model Selector",      "icon": "bi-cpu",                 "color": "secondary", "bs": "#a78bfa"},
    "leakage_detector":    {"label": "Leakage Detector",    "icon": "bi-exclamation-diamond", "color": "danger",    "bs": "#dc3545"},
    "feature_engineering": {"label": "Feature Engineering", "icon": "bi-wrench-adjustable",   "color": "secondary", "bs": "#6f42c1"},
}

# MLOps constants
ML_SEV_COLOR    = {"critical": "danger", "warning": "warning", "suggestion": "info"}
ML_FEAS_COLOR   = {"ready": "success", "marginal": "warning", "insufficient": "danger"}
ML_FEAS_ICON    = {"ready": "bi-check-circle-fill text-success",
                   "marginal": "bi-exclamation-circle text-warning",
                   "insufficient": "bi-x-circle-fill text-danger"}
ML_CAT_ICON = {
    "data_leakage":       "bi-exclamation-diamond-fill",
    "hardcoded_config":   "bi-lock",
    "missing_logging":    "bi-journal-x",
    "no_error_handling":  "bi-shield-x",
    "inefficient_spark":  "bi-lightning",
    "reproducibility":    "bi-arrow-repeat",
    "missing_tests":      "bi-check2-square",
    "code_structure":     "bi-code-square",
    "security":           "bi-incognito",
    "documentation":      "bi-file-text",
}

# DQ constants
DQ_SEV_COLOR  = {"critical": "danger", "warning": "warning", "info": "info"}
DQ_SEV_RANK   = {"critical": 0, "warning": 1, "info": 2}
DQ_SEV_HEADER = {
    "critical": ("bi-exclamation-octagon-fill text-danger", "Critical Issues"),
    "warning":  ("bi-exclamation-triangle text-warning",    "Warnings"),
    "info":     ("bi-info-circle text-info",                "Info"),
}
DQ_ROW_STYLE  = {
    "critical": {"borderLeft": "3px solid #dc3545"},
    "warning":  {"borderLeft": "3px solid #ffc107"},
    "info":     {},
}

# JA constants
JA_SEV_COLOR   = {"critical": "danger", "warning": "warning", "info": "info"}
JA_VERDICT_CLR = {"safe": "success", "review_needed": "warning", "dangerous": "danger"}


# ── Layout ─────────────────────────────────────────────────────────────────────

def layout():
    return html.Div([

        # ── Session area ─────────────────────────────────────────────────────
        html.Div(id="wb-session-active", className="mb-3"),

        dbc.Card(dbc.CardBody([
            html.H5([html.I(className="bi bi-search me-2"), "What are you investigating?"],
                    className="mb-1"),
            html.P("Describe your problem — the workbench will run data quality checks on "
                   "your selected datasets and keep that context as you explore.",
                   className="text-muted small mb-4"),

            dbc.Row([
                dbc.Col([
                    dbc.Label("Problem statement", size="sm", className="fw-semibold"),
                    dbc.Textarea(
                        id="wb-problem-stmt",
                        placeholder=(
                            "e.g. Q3 revenue is down 12% vs forecast — "
                            "investigate which segments and products are driving the shortfall"
                        ),
                        rows=3,
                        style={"fontSize": "0.88rem"},
                    ),
                ], width=7),
                dbc.Col([
                    dbc.Label("Datasets to work with", size="sm", className="fw-semibold"),
                    dbc.Checklist(
                        id="wb-dataset-select",
                        options=DATASET_OPTIONS,
                        value=[],
                        className="small",
                    ),
                ], width=5),
            ], className="g-3 mb-3"),

            dbc.Row([
                dbc.Col(
                    dcc.Loading(
                        html.Div([
                            dbc.Button(
                                [html.I(className="bi bi-play-fill me-2"), "Start Session"],
                                id="wb-start-btn", color="info", className="px-4",
                            ),
                            html.Div(id="wb-start-status",
                                     className="mt-2 small text-info text-center"),
                        ]),
                        type="circle",
                        color="#0dcaf0",
                    ),
                    width="auto",
                ),
            ], align="center"),
        ]), id="wb-setup-card", className="mb-4",
           style={"border": "1px solid #3a3a5a"}),

        # ── Cross-tool synthesis panel (auto-appears when 2+ skills run) ────────
        html.Div(id="wb-synthesis", className="mb-3"),

        # ── Skills accordion ─────────────────────────────────────────────────
        dbc.Accordion(
            [
                # ── Data Quality ──────────────────────────────────────────
                dbc.AccordionItem([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Dataset", size="sm"),
                            dbc.Select(id="wb-dq-dataset", options=DATASET_OPTIONS,
                                       value="orders_dirty", size="sm"),
                        ], width=8),
                        dbc.Col(
                            dbc.Button([html.I(className="bi bi-play-fill me-1"), "Run DQ"],
                                       id="wb-dq-run-btn", color="primary", size="sm",
                                       className="w-100 mt-3"),
                            width=4, className="d-flex align-items-end",
                        ),
                    ], className="g-2 mb-3"),
                    dcc.Loading(html.Div(id="wb-dq-results"), type="circle", color="#0d6efd"),
                ],
                    title=html.Span(id="wb-dq-title"),
                    item_id="dq",
                ),

                # ── EDA ───────────────────────────────────────────────────
                dbc.AccordionItem([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Dataset", size="sm"),
                            dbc.Select(id="wb-eda-dataset", options=DATASET_OPTIONS,
                                       value="sales_clean", size="sm"),
                        ], width=8),
                        dbc.Col(
                            dbc.Button([html.I(className="bi bi-play-fill me-1"), "Run EDA"],
                                       id="wb-eda-run-btn", color="success", size="sm",
                                       className="w-100 mt-3"),
                            width=4, className="d-flex align-items-end",
                        ),
                    ], className="g-2 mb-3"),
                    dcc.Loading(html.Div(id="wb-eda-results"), type="circle", color="#198754"),
                ],
                    title=html.Span(id="wb-eda-title"),
                    item_id="eda",
                ),

                # ── Join Auditor ───────────────────────────────────────────
                dbc.AccordionItem([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Join scenario", size="sm"),
                            dbc.Select(id="wb-ja-scenario", options=JOIN_SCENARIO_OPTIONS,
                                       value="orders_customers_fanout", size="sm"),
                        ], width=8),
                        dbc.Col(
                            dbc.Button([html.I(className="bi bi-play-fill me-1"), "Run Audit"],
                                       id="wb-ja-run-btn", color="warning", size="sm",
                                       className="w-100 mt-3"),
                            width=4, className="d-flex align-items-end",
                        ),
                    ], className="g-2 mb-3"),
                    dcc.Loading(html.Div(id="wb-ja-results"), type="circle", color="#ffc107"),
                ],
                    title=html.Span(id="wb-ja-title"),
                    item_id="ja",
                ),

                # ── Leakage Detector ──────────────────────────────────────
                dbc.AccordionItem([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Dataset", size="sm"),
                            dbc.Select(id="wb-ld-dataset", options=DATASET_OPTIONS,
                                       value="ecommerce_leakage", size="sm"),
                        ], width=4),
                        dbc.Col([
                            dbc.Label("Target column", size="sm"),
                            dbc.Select(id="wb-ld-target", options=[], value=None,
                                       placeholder="Select after dataset loads…", size="sm"),
                        ], width=4),
                        dbc.Col(
                            dbc.Button(
                                [html.I(className="bi bi-exclamation-diamond me-1"), "Check Leakage"],
                                id="wb-ld-run-btn", color="danger", size="sm",
                                className="w-100 mt-3",
                            ),
                            width=4, className="d-flex align-items-end",
                        ),
                    ], className="g-2 mb-3"),
                    dcc.Loading(html.Div(id="wb-ld-results"), type="circle", color="#dc3545"),
                ],
                    title=html.Span(id="wb-ld-title"),
                    item_id="ld",
                ),

                # ── Feature Engineering ────────────────────────────────────
                dbc.AccordionItem([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Dataset", size="sm"),
                            dbc.Select(id="wb-fe-dataset", options=DATASET_OPTIONS,
                                       value="hr_attrition", size="sm"),
                        ], width=4),
                        dbc.Col([
                            dbc.Label("Target column", size="sm"),
                            dbc.Select(id="wb-fe-target", options=[], value=None,
                                       placeholder="Select after dataset loads…", size="sm"),
                        ], width=4),
                        dbc.Col(
                            dbc.Button(
                                [html.I(className="bi bi-wrench-adjustable me-1"), "Plan Features"],
                                id="wb-fe-run-btn", color="secondary", size="sm",
                                className="w-100 mt-3",
                                style={"borderColor": "#6f42c1", "color": "#6f42c1"},
                            ),
                            width=4, className="d-flex align-items-end",
                        ),
                    ], className="g-2 mb-3"),
                    dcc.Loading(html.Div(id="wb-fe-results"), type="circle", color="#6f42c1"),
                ],
                    title=html.Span(id="wb-fe-title"),
                    item_id="fe",
                ),

                # ── Model Selector ────────────────────────────────────────
                dbc.AccordionItem([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Dataset", size="sm"),
                            dbc.Select(id="wb-ms-dataset", options=DATASET_OPTIONS,
                                       value="hr_attrition", size="sm"),
                        ], width=4),
                        dbc.Col([
                            dbc.Label("Target column", size="sm"),
                            dbc.Select(id="wb-ms-target", options=[], value=None,
                                       placeholder="Select after dataset loads…", size="sm"),
                        ], width=4),
                        dbc.Col(
                            dbc.Button(
                                [html.I(className="bi bi-cpu me-1"), "Select Models"],
                                id="wb-ms-run-btn", color="secondary", size="sm",
                                className="w-100 mt-3",
                                style={"borderColor": "#a78bfa", "color": "#a78bfa"},
                            ),
                            width=4, className="d-flex align-items-end",
                        ),
                    ], className="g-2 mb-3"),
                    dcc.Loading(html.Div(id="wb-ms-results"), type="circle", color="#a78bfa"),
                ],
                    title=html.Span(id="wb-ms-title"),
                    item_id="ms",
                ),

                # ── NL → PySpark ───────────────────────────────────────────
                dbc.AccordionItem([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Dataset", size="sm"),
                            dbc.Select(id="wb-nl-dataset", options=DATASET_OPTIONS,
                                       value="sales_clean", size="sm"),
                        ], width=4),
                        dbc.Col([
                            dbc.Label("Question", size="sm"),
                            dbc.Textarea(id="wb-nl-question", rows=2,
                                         placeholder="e.g. Top 5 products by total revenue in each region",
                                         style={"fontSize": "0.85rem"}),
                        ], width=6),
                        dbc.Col(
                            dbc.Button([html.I(className="bi bi-play-fill me-1"), "Run"],
                                       id="wb-nl-run-btn", color="info", size="sm",
                                       className="w-100 mt-3"),
                            width=2, className="d-flex align-items-end",
                        ),
                    ], className="g-2 mb-3"),
                    dcc.Loading(html.Div(id="wb-nl-results"), type="circle", color="#0dcaf0"),
                    dcc.Store(id="wb-nl-tables"),
                ],
                    title=html.Span(id="wb-nl-title"),
                    item_id="nl",
                ),

                # ── MLOps Review ───────────────────────────────────────────
                dbc.AccordionItem([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Notebook code", size="sm"),
                            dbc.Textarea(
                                id="wb-ml-code",
                                placeholder=ML_PLACEHOLDER,
                                rows=10,
                                style={"fontFamily": "monospace", "fontSize": "0.78rem",
                                       "backgroundColor": "#12122a", "color": "#cdd6f4",
                                       "border": "1px solid #3a3a5a"},
                            ),
                        ], width=10),
                        dbc.Col(
                            dbc.Button([html.I(className="bi bi-play-fill me-1"), "Review"],
                                       id="wb-ml-run-btn", color="danger", size="sm",
                                       className="w-100 mt-4"),
                            width=2, className="d-flex align-items-end",
                        ),
                    ], className="g-2 mb-3"),
                    dcc.Loading(html.Div(id="wb-ml-results"), type="circle", color="#dc3545"),
                ],
                    title=html.Span(id="wb-ml-title"),
                    item_id="ml",
                ),
                dbc.AccordionItem([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Paste training metrics (JSON)", html_for="wb-et-metrics"),
                            dbc.Textarea(
                                id="wb-et-metrics",
                                placeholder='{"accuracy": 0.82, "auc": 0.75, "f1": 0.61, "train_accuracy": 0.94, "val_accuracy": 0.79}',
                                rows=5,
                                className="font-monospace",
                                style={"fontSize": "0.8rem"},
                            ),
                        ], width=12),
                    ], className="mb-3"),
                    dbc.Row([
                        dbc.Col(dbc.Button(
                            [html.I(className="bi bi-play-fill me-2"), "Interpret Results"],
                            id="wb-et-run-btn",
                            color="primary",
                        ), width="auto"),
                    ], className="mb-3"),
                    dcc.Loading(html.Div(id="wb-et-output"), type="dot"),
                ], title="Experiment Tracker", item_id="et"),
            ],
            id="wb-accordion",
            always_open=True,
            active_item=["dq"],
            className="mb-4",
        ),

        # ── Investigation log ────────────────────────────────────────────────
        html.Details([
            html.Summary(
                html.Span("INVESTIGATION LOG",
                          className="text-muted small",
                          style={"letterSpacing": "0.1em", "cursor": "pointer"}),
                className="mb-2",
            ),
            dcc.Loading(html.Div(id="wb-timeline"), type="dot", color="#0dcaf0"),
        ], open=True),
    ])


# ══════════════════════════════════════════════════════════════════════════════
# Render helpers (inlined — do NOT import from other page files)
# ══════════════════════════════════════════════════════════════════════════════

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


# ── Join Auditor helpers ───────────────────────────────────────────────────────

def _ja_findings_table(findings: list) -> list:
    blocking = [f for f in findings if f.get("severity") in ("critical", "warning")]
    info_    = [f for f in findings if f.get("severity") == "info"]
    blocking = sorted(blocking, key=lambda f: {"critical": 0, "warning": 1}.get(f.get("severity"), 1))
    rows = []
    for section_label, section_icon, section_f in [
        ("Blocking Issues", "bi-exclamation-octagon-fill text-danger", blocking),
        ("Info",            "bi-info-circle text-info",                info_),
    ]:
        if not section_f:
            continue
        rows.append(html.Tr(html.Td(
            [html.I(className=f"bi {section_icon} me-2"), html.Strong(section_label)],
            colSpan=3, className="small py-1",
            style={"background": "rgba(255,255,255,0.04)",
                   "borderTop": "1px solid rgba(255,255,255,0.1)"},
        )))
        for f in section_f:
            sev    = f.get("severity", "info")
            border = {"critical": "3px solid #dc3545", "warning": "3px solid #ffc107"}.get(sev, "")
            rows.append(html.Tr([
                html.Td(dbc.Badge(sev.upper(), color=JA_SEV_COLOR.get(sev, "secondary"))),
                html.Td(f.get("issue_type", "").replace("_", " ").title(), className="text-muted small"),
                html.Td([
                    html.Div(f.get("description", ""), className="small"),
                    html.Div(f.get("recommendation", ""), className="small text-muted mt-1"),
                ]),
            ], style={"borderLeft": border} if border else {}))
    return rows


def _ja_verdict_hero(result: dict) -> dbc.Alert:
    verdict  = result.get("overall_verdict", "review_needed")
    rc       = result.get("row_count_change", {})
    findings = result.get("findings", [])
    n_crit   = sum(1 for f in findings if f.get("severity") == "critical")
    color    = JA_VERDICT_CLR.get(verdict, "secondary")
    label_map = {"safe": "JOIN SAFE", "dangerous": "JOIN DANGEROUS", "review_needed": "REVIEW NEEDED"}
    icon_map  = {"safe": "bi-check-circle-fill", "dangerous": "bi-x-octagon-fill",
                 "review_needed": "bi-exclamation-circle-fill"}
    label = label_map.get(verdict, verdict.replace("_", " ").upper())
    icon  = icon_map.get(verdict, "bi-circle")

    left  = rc.get("left_rows",   0) or 0
    right = rc.get("right_rows",  0) or 0
    out   = rc.get("result_rows", 0) or 0
    fan   = rc.get("fan_out_factor", 1.0) or 1.0
    row_loss_pct = round((1 - out / max(left, 1)) * 100, 1) if left > 0 else 0.0

    recon_table = dbc.Table([
        html.Thead(html.Tr([html.Th("Left"), html.Th("Right"), html.Th("Joined"),
                            html.Th("Fan-out"), html.Th("Row loss")])),
        html.Tbody([html.Tr([
            html.Td(f"{left:,}",  className="font-monospace small"),
            html.Td(f"{right:,}", className="font-monospace small"),
            html.Td(f"{out:,}",   className="font-monospace small"),
            html.Td(dbc.Badge(f"×{fan:.2f}",
                              color="danger" if fan > 2 else "warning" if fan > 1.1 else "success")),
            html.Td(f"{row_loss_pct}%", className="font-monospace small",
                    style={"color": "#ff6b6b"} if row_loss_pct > 10 else {}),
        ])])
    ], size="sm", className="mb-0 mt-2", style={"maxWidth": "480px"})

    children = [
        html.Strong([html.I(className=f"bi {icon} me-2"), label], className="d-inline"),
        html.Span(f" · {n_crit} blocking issue{'s' if n_crit != 1 else ''}",
                  className="small ms-2") if n_crit else html.Span(),
        recon_table,
    ]
    return dbc.Alert(children, color=color, className="py-2 mb-3")


def _ja_render(result: dict) -> html.Div:
    findings = result.get("findings", [])
    rows     = _ja_findings_table(findings)
    return html.Div([
        _ja_verdict_hero(result),
        html.P(result.get("summary", ""), className="small mb-3"),
        dbc.Table(
            [html.Thead(html.Tr([html.Th("Sev"), html.Th("Issue"), html.Th("Detail")]))] +
            [html.Tbody(rows)],
            hover=True, responsive=True, size="sm",
        ) if rows else html.P("No issues detected.", className="text-success"),
    ])


# ── NL → PySpark helpers ──────────────────────────────────────────────────────

def _nl_render(result: dict, dq_critical: bool = False) -> html.Div:
    assumptions = result.get("assumptions", [])
    perf_notes  = result.get("performance_notes", [])
    explanation = result.get("explanation", "")

    # Extract first sentence as the one-line intent
    _intent = explanation.split(".")[0].strip() + "." if explanation else ""

    # ── Query intent hero ────────────────────────────────────────────────────
    intent_children = [
        html.Strong([html.I(className="bi bi-chat-square-text me-2"), "Query Intent: "],
                    className="d-inline"),
        html.Span(_intent, className="small"),
    ]
    if dq_critical:
        intent_children.append(html.Div(
            [html.I(className="bi bi-shield-exclamation me-1 text-danger"),
             html.Span("Critical DQ findings in session — results may be unreliable.",
                       className="small text-danger")],
            className="mt-1",
        ))
    if assumptions:
        intent_children.append(html.Ul(
            [html.Li([html.I(className="bi bi-lightbulb me-1 text-warning"), a],
                     className="small") for a in assumptions],
            className="mb-0 mt-1 ps-3",
        ))
    intent_card = dbc.Alert(intent_children,
                            color="danger" if dq_critical else "secondary",
                            className="py-2 mb-3")

    return html.Div([
        intent_card,
        dbc.Card([
            dbc.CardHeader(html.Strong("Spark SQL")),
            dbc.CardBody(html.Pre(result.get("sql_query", ""), className="mb-0",
                                  style={"fontSize": "0.8rem"})),
        ], className="mb-3"),
        dbc.Card([
            dbc.CardHeader(html.Strong("PySpark (DataFrame API)")),
            dbc.CardBody(html.Pre(result.get("pyspark_code", ""), className="mb-0",
                                  style={"fontSize": "0.8rem"})),
        ], className="mb-3"),
        dbc.Card([
            dbc.CardHeader(html.Strong("Full Explanation")),
            dbc.CardBody(html.P(explanation, className="mb-0 small")),
        ], className="mb-3"),
        dbc.Card([
            dbc.CardHeader(html.Strong("Performance Notes")),
            dbc.CardBody(html.Ul([html.Li(n, className="small") for n in perf_notes],
                                 className="mb-0")),
        ], className="mb-3") if perf_notes else html.Div(),
    ])


# ── MLOps helpers ─────────────────────────────────────────────────────────────

def _ml_gauge(score: int) -> go.Figure:
    color = "#dc3545" if score < 40 else "#ffc107" if score < 70 else "#198754"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        gauge={"axis": {"range": [0, 100], "tickcolor": "#888"}, "bar": {"color": color},
               "bgcolor": "#1e1e2e",
               "steps": [{"range": [0, 40],  "color": "#3d1515"},
                         {"range": [40, 70],  "color": "#3d3215"},
                         {"range": [70, 100], "color": "#153d1e"}]},
        title={"text": "Production Readiness", "font": {"color": "#ccc", "size": 13}},
        number={"font": {"color": color, "size": 48}},
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=180,
                      margin={"t": 40, "b": 0, "l": 10, "r": 10})
    return fig


def _ml_feasibility_card(feasibility: dict) -> dbc.Card:
    verdict  = feasibility.get("verdict", "marginal")
    color    = ML_FEAS_COLOR.get(verdict, "secondary")
    icon_cls = ML_FEAS_ICON.get(verdict, "bi-circle")
    label    = {"ready": "Ready to Model", "marginal": "Marginal", "insufficient": "Not Ready"}
    gaps     = feasibility.get("key_gaps", [])
    return dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Div("Data & Feature Feasibility", className="text-muted small mb-1"),
                    html.Div([
                        html.I(className=f"bi {icon_cls} me-2 fs-5"),
                        dbc.Badge(label.get(verdict, verdict.title()), color=color,
                                  className="p-2", style={"fontSize": "0.9rem"}),
                    ], className="d-flex align-items-center"),
                ], width="auto"),
                dbc.Col(html.P(feasibility.get("assessment", ""), className="mb-0 small mt-1")),
            ], align="center", className="mb-2"),
            html.Ul([html.Li(g, className="small mb-1") for g in gaps],
                    className="mb-0 mt-2") if gaps else html.Div(),
        ]),
    ], color=color, outline=True, className="mb-3", style={"borderWidth": "2px"})


def _ml_findings_table(findings: list) -> list:
    blocking = [f for f in findings if f.get("severity") in ("critical", "warning")]
    advisory = [f for f in findings if f.get("severity") == "suggestion"]
    blocking = sorted(blocking, key=lambda f: {"critical": 0, "warning": 1}.get(f.get("severity"), 1))
    rows = []
    for section_label, section_icon, section_f in [
        ("Blocking Issues", "bi-exclamation-octagon-fill text-danger", blocking),
        ("Advisory",        "bi-lightbulb text-info",                  advisory),
    ]:
        if not section_f:
            continue
        rows.append(html.Tr(html.Td(
            [html.I(className=f"bi {section_icon} me-2"), html.Strong(section_label)],
            colSpan=3, className="small py-1",
            style={"background": "rgba(255,255,255,0.04)",
                   "borderTop": "1px solid rgba(255,255,255,0.1)"},
        )))
        for f in section_f:
            sev    = f.get("severity", "suggestion")
            border = {"critical": "3px solid #dc3545", "warning": "3px solid #ffc107"}.get(sev, "")
            rows.append(html.Tr([
                html.Td(dbc.Badge(sev.upper(), color=ML_SEV_COLOR.get(sev, "secondary"))),
                html.Td([html.I(className=f"bi {ML_CAT_ICON.get(f.get('category',''),'bi-dot')} me-1"),
                         f.get("category", "").replace("_", " ").title()],
                        className="text-muted small"),
                html.Td([
                    html.Div(f.get("description", ""), className="small"),
                    html.Code(f.get("line_reference", ""), className="d-block mt-1",
                              style={"fontSize": "0.72rem", "color": "#aaa"})
                    if f.get("line_reference") else html.Div(),
                    html.Div(f.get("fix", ""), className="small text-muted mt-1"),
                ]),
            ], style={"borderLeft": border} if border else {}))
    return rows


def _ml_verdict_hero(result: dict) -> dbc.Alert:
    score    = result.get("readiness_score", 0)
    findings = result.get("findings", [])
    critical = [f for f in findings if f.get("severity") == "critical"]
    if critical or score < 40:
        color, label, icon = "danger",  "NOT READY",        "bi-x-octagon-fill"
    elif score < 70:
        color, label, icon = "warning", "NEEDS REVIEW",     "bi-exclamation-circle-fill"
    else:
        color, label, icon = "success", "PRODUCTION READY", "bi-check-circle-fill"
    children = [
        html.Strong([html.I(className=f"bi {icon} me-2"), f"MLOps: {label}  "],
                    className="d-inline"),
        dbc.Badge(f"{score}/100", color=color, className="ms-1", style={"fontSize": "0.75rem"}),
    ]
    if critical:
        children.append(html.Ul(
            [html.Li(f.get("description", "")[:80], className="small") for f in critical[:3]],
            className="mb-0 mt-1 ps-3",
        ))
    return dbc.Alert(children, color=color, className="py-2 mb-3")


def _ml_render(result: dict, session_store: dict | None = None) -> html.Div:
    findings    = result.get("findings", [])
    counts      = {s: sum(1 for f in findings if f.get("severity") == s)
                   for s in ("critical", "warning", "suggestion")}
    snippet     = result.get("refactored_snippet")
    feasibility = result.get("feasibility")
    rows        = _ml_findings_table(findings)

    # Cross-reference DQ columns into MLOps findings
    dq_xref = None
    if session_store:
        dq_runs = [r for r in session_store.get("data_quality", []) if r.get("status") == "complete"]
        if dq_runs:
            dq_cols: dict[str, tuple[str, str]] = {}
            for run in dq_runs:
                for f in run.get("result", {}).get("findings", []):
                    col = f.get("column", "")
                    if col and col not in dq_cols:
                        dq_cols[col] = (f.get("severity", "info"), f.get("description", ""))
            matches = []
            for f in findings:
                haystack = (f.get("description","") + " " + f.get("fix","")).lower()
                for col, (sev, dq_desc) in dq_cols.items():
                    if col.lower() in haystack:
                        matches.append((col, sev, dq_desc[:60], f.get("description","")[:70]))
            if matches:
                xref_rows = [html.Tr([
                    html.Td(html.Code(col, style={"fontSize": "0.78rem"})),
                    html.Td(dbc.Badge(sev.upper(), color=DQ_SEV_COLOR.get(sev, "secondary"),
                                      className="small")),
                    html.Td(dq_d, className="small text-muted"),
                    html.Td(ml_d, className="small"),
                ]) for col, sev, dq_d, ml_d in matches[:5]]
                dq_xref = dbc.Card([
                    dbc.CardHeader([
                        html.I(className="bi bi-link-45deg me-2 text-warning"),
                        html.Strong("DQ Cross-Reference"),
                        html.Span(" — columns flagged in Data Quality also appear in MLOps findings",
                                  className="text-muted small ms-2"),
                    ]),
                    dbc.CardBody(
                        dbc.Table([
                            html.Thead(html.Tr([html.Th("Column"), html.Th("DQ Severity"),
                                                html.Th("DQ Issue"), html.Th("MLOps Finding")])),
                            html.Tbody(xref_rows),
                        ], hover=True, responsive=True, size="sm"),
                        className="p-0",
                    ),
                ], className="mb-3", style={"borderColor": "#ffc10740"})

    return html.Div([
        _ml_verdict_hero(result),
        _ml_feasibility_card(feasibility) if feasibility else html.Div(),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=_ml_gauge(result.get("readiness_score", 0)),
                              config={"displayModeBar": False}), width=4),
            dbc.Col([
                html.P(result.get("summary", ""), className="lead mt-3 small"),
                html.Div([
                    dbc.Badge([html.I(className="bi bi-x-circle me-1"),
                               f"{counts['critical']} Critical"],
                              color="danger", className="me-2 p-2"),
                    dbc.Badge([html.I(className="bi bi-exclamation-triangle me-1"),
                               f"{counts['warning']} Warnings"],
                              color="warning", text_color="dark", className="me-2 p-2"),
                    dbc.Badge([html.I(className="bi bi-lightbulb me-1"),
                               f"{counts['suggestion']} Suggestions"],
                              color="info", text_color="dark", className="p-2"),
                ], className="mt-3"),
            ], width=8),
        ], className="mb-3"),
        dq_xref or html.Div(),
        dbc.Card([
            dbc.CardHeader(html.Strong("Findings")),
            dbc.CardBody(
                dbc.Table(
                    [html.Thead(html.Tr([html.Th("Severity"), html.Th("Category"),
                                        html.Th("Issue & Fix")]))] +
                    [html.Tbody(rows)],
                    hover=True, responsive=True, size="sm",
                ) if rows else html.P("Code looks production-ready!", className="text-success mb-0"),
                className="p-0",
            ),
        ], className="mb-3"),
        dbc.Card([
            dbc.CardHeader(html.Strong("Quick Win — Refactored Snippet")),
            dbc.CardBody(html.Pre(snippet, className="mb-0", style={"fontSize": "0.78rem"})),
        ]) if snippet else html.Div(),
    ])


# ── Experiment Tracker renderer ──────────────────────────────────────────────

_ET_VERDICT_COLOR = {"promising": "success", "needs_work": "warning", "red_flags": "danger"}
_ET_SEV_COLOR     = {"positive": "success", "note": "info", "concern": "warning", "critical": "danger"}
_ET_DIM_ICON      = {
    "performance":          "bi-graph-up",
    "overfitting":          "bi-arrows-expand",
    "class_imbalance":      "bi-distribute-vertical",
    "data_leakage":         "bi-exclamation-diamond-fill",
    "feature_quality":      "bi-table",
    "deployment_readiness": "bi-box-arrow-up-right",
}


def _et_render(result: dict) -> html.Div:
    verdict    = result.get("verdict", "needs_work")
    interp     = result.get("interpretation", "")
    findings   = result.get("findings", [])
    next_steps = result.get("next_steps", [])
    suggested  = result.get("suggested_experiments", [])

    sev_rank = {"critical": 0, "concern": 1, "note": 2, "positive": 3}
    findings = sorted(findings, key=lambda f: sev_rank.get(f.get("severity", "note"), 2))

    verdict_color = _ET_VERDICT_COLOR.get(verdict, "secondary")
    verdict_label = verdict.replace("_", " ").upper()
    v_icon = ("bi-check-circle-fill"       if verdict == "promising"  else
              "bi-exclamation-triangle-fill" if verdict == "needs_work" else
              "bi-x-octagon-fill")

    hero = dbc.Alert([
        html.Div([html.I(className=f"bi {v_icon} me-2"), html.Strong(verdict_label)],
                 className="fs-6 mb-2"),
        html.P(interp, className="mb-0 small"),
    ], color=verdict_color, className="mb-3")

    finding_rows = []
    for f in findings:
        sev      = f.get("severity", "note")
        dim      = f.get("dimension", "performance")
        icon_cls = _ET_DIM_ICON.get(dim, "bi-dot")
        row_style = ({"borderLeft": f"3px solid var(--bs-{_ET_SEV_COLOR.get(sev,'secondary')})"}
                     if sev in ("critical", "concern") else {})
        finding_rows.append(html.Tr([
            html.Td(dbc.Badge(sev, color=_ET_SEV_COLOR.get(sev, "secondary")),
                    style={"width": "90px"}),
            html.Td([html.I(className=f"bi {icon_cls} me-1 small"),
                     html.Span(dim.replace("_", " "), className="small text-muted")],
                    style={"width": "160px"}),
            html.Td([
                html.Div(f.get("observation", ""), className="small"),
                html.Div(f.get("recommendation", ""), className="small text-muted mt-1"),
            ]),
        ], style=row_style))

    findings_table = dbc.Table(
        [html.Tbody(finding_rows)], size="sm", className="mb-3", borderless=True,
    ) if finding_rows else html.Div()

    next_card = dbc.Card([
        dbc.CardHeader(html.Strong("Next Steps")),
        dbc.CardBody(html.Ol(
            [html.Li(s, className="small mb-1") for s in next_steps],
            className="mb-0 ps-3",
        )),
    ], className="mb-3") if next_steps else html.Div()

    exp_card = dbc.Card([
        dbc.CardHeader(html.Strong("Suggested Experiments")),
        dbc.CardBody(html.Ul(
            [html.Li(s, className="small mb-1") for s in suggested],
            className="mb-0",
        )),
    ]) if suggested else html.Div()

    return html.Div([hero, findings_table, next_card, exp_card])


# ── Model Selector renderer ───────────────────────────────────────────────────

_MS_STATUS_COLOR  = {"met": "success", "violated": "danger", "uncertain": "secondary"}
_MS_STATUS_ICON   = {"met": "bi-check-circle-fill text-success",
                     "violated": "bi-x-circle-fill text-danger",
                     "uncertain": "bi-question-circle text-muted"}
_MS_SUIT_COLOR    = {"strong": "success", "moderate": "warning", "weak": "secondary"}
_MS_READY_COLOR   = {"ready": "success", "needs_prep": "warning", "insufficient": "danger"}


def _ms_render(result: dict) -> html.Div:
    task        = result.get("task_type", "unknown").replace("_", " ").title()
    readiness   = result.get("data_readiness", "needs_prep")
    rdy_color   = _MS_READY_COLOR.get(readiness, "secondary")
    t_analysis  = result.get("target_analysis", {})
    assumptions = result.get("assumption_checks", [])
    recs        = result.get("recommendations", [])
    rdy_notes   = result.get("readiness_notes", [])

    # ── Verdict hero ──────────────────────────────────────────────────────────
    _top_model = recs[0].get("display_name", recs[0].get("model_family", "")) if recs else ""
    _violated  = [a for a in assumptions if a.get("status") == "violated"]
    if readiness == "insufficient" or len(_violated) >= 2:
        _v_color, _v_icon, _v_label = "danger",  "bi-x-octagon-fill",            "NOT READY"
    elif readiness == "needs_prep" or _violated:
        _v_color, _v_icon, _v_label = "warning", "bi-exclamation-triangle-fill", "PREP REQUIRED"
    else:
        _v_color, _v_icon, _v_label = "success", "bi-check-circle-fill",         "READY TO TRAIN"
    verdict_hero = dbc.Alert([
        html.Strong([html.I(className=f"bi {_v_icon} me-2"), f"Model Selector: {_v_label}"],
                    className="d-block mb-1"),
        html.Span([html.Strong("Task: "), task, "  ·  ", html.Strong("Top pick: "), _top_model or "—"],
                  className="small"),
    ], color=_v_color, className="py-2 mb-3")

    # ── Header strip ──────────────────────────────────────────────────────────
    header = dbc.Row([
        dbc.Col(html.Div([
            html.I(className="bi bi-cpu me-2", style={"color": "#a78bfa"}),
            dbc.Badge(task, color="secondary",
                      style={"border": "1px solid #a78bfa", "color": "#a78bfa",
                             "fontSize": "0.8rem"}),
        ]), width="auto"),
        dbc.Col(html.Div([
            html.Span("Data readiness: ", className="text-muted small me-1"),
            dbc.Badge(readiness.replace("_", " ").title(), color=rdy_color,
                      style={"fontSize": "0.75rem"}),
        ]), width="auto"),
    ], align="center", className="mb-3")

    # ── Target summary ────────────────────────────────────────────────────────
    target_card = dbc.Card(dbc.CardBody([
        html.Strong("Target Analysis", className="d-block mb-1 small text-muted"),
        html.P(t_analysis.get("summary", ""), className="mb-0 small"),
    ]), className="mb-3", style={"border": "1px solid #2a2a4a"})

    # ── Assumption checks ─────────────────────────────────────────────────────
    assumption_rows = []
    for a in assumptions:
        st     = a.get("status", "uncertain")
        icon   = _MS_STATUS_ICON.get(st, "bi-circle text-muted")
        badge  = dbc.Badge(st.title(), color=_MS_STATUS_COLOR.get(st, "secondary"),
                           style={"fontSize": "0.62rem"}, className="me-2")
        assumption_rows.append(html.Tr([
            html.Td(html.I(className=f"bi {icon}"), style={"width": "24px"}),
            html.Td([badge, html.Strong(a.get("assumption", ""), className="small")]),
            html.Td(a.get("evidence", ""), className="small text-muted"),
            html.Td(a.get("affected_models", ""), className="small text-muted fst-italic"),
        ]))

    assumption_card = dbc.Card([
        dbc.CardHeader(html.Strong("Assumption Checks")),
        dbc.CardBody(
            dbc.Table(
                [html.Thead(html.Tr([html.Th(""), html.Th("Assumption"),
                                     html.Th("Evidence"), html.Th("Affects")]))] +
                [html.Tbody(assumption_rows)],
                hover=True, responsive=True, size="sm",
            ) if assumption_rows else html.P("No assumptions checked.", className="text-muted mb-0"),
            className="p-0",
        ),
    ], className="mb-3")

    # ── Model recommendations ─────────────────────────────────────────────────
    rec_cards = []
    for i, rec in enumerate(recs):
        suit      = rec.get("suitability", "moderate")
        suit_clr  = _MS_SUIT_COLOR.get(suit, "secondary")
        caveats   = rec.get("caveats", [])
        code      = rec.get("starter_code", "")
        collapse_id = f"ms-code-{i}"

        rec_cards.append(dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(html.Div([
                        html.Span(f"#{rec.get('rank',i+1)} ", className="text-muted me-1"),
                        html.Strong(rec.get("display_name", rec.get("model_family", "")),
                                    style={"fontSize": "0.95rem"}),
                    ]), width=8),
                    dbc.Col(
                        dbc.Badge(suit.title(), color=suit_clr,
                                  style={"fontSize": "0.7rem"}),
                        width=4, className="text-end",
                    ),
                ], align="center", className="mb-2"),
                html.P(rec.get("rationale", ""), className="small mb-2"),
                html.Div([
                    html.Ul([html.Li(c, className="small text-warning") for c in caveats],
                            className="mb-1"),
                ]) if caveats else html.Div(),
                html.Div([
                    dbc.Button("Show starter code", id={"type": "ms-code-toggle", "index": i},
                               size="sm", color="link",
                               className="p-0 small", style={"color": "#a78bfa"}),
                    dbc.Collapse(
                        html.Pre(code, className="mt-2 mb-0",
                                 style={"fontSize": "0.72rem", "background": "#12122a",
                                        "padding": "0.75rem", "borderRadius": "4px"}),
                        id={"type": "ms-code-collapse", "index": i},
                        is_open=False,
                    ),
                ]) if code else html.Div(),
            ]),
        ], className="mb-2", style={"border": "1px solid #2a2a4a" if suit != "strong"
                                    else "1px solid #a78bfa40"}))

    recs_section = html.Div([
        html.H6("Recommended Models", className="text-muted small mb-2 fw-semibold",
                style={"letterSpacing": "0.08em"}),
        html.Div(rec_cards),
    ], className="mb-3")

    # ── Readiness notes ───────────────────────────────────────────────────────
    notes_section = dbc.Alert([
        html.Strong("Before training: "),
        html.Ul([html.Li(n, className="small") for n in rdy_notes], className="mb-0 mt-1"),
    ], color="warning", className="py-2") if rdy_notes else html.Div()

    return html.Div([verdict_hero, header, target_card, assumption_card, recs_section, notes_section])


# ── Leakage Detector helpers ──────────────────────────────────────────────────

_LD_VERDICT_COLOR = {"leakage_detected": "danger", "suspicious": "warning", "clean": "success"}
_LD_RISK_COLOR    = {"high": "danger", "medium": "warning", "low": "info", "none": "success"}
_LD_SEV_COLOR     = {"critical": "danger", "warning": "warning", "info": "info"}
_LD_ROW_STYLE     = {"critical": {"borderLeft": "3px solid #dc3545"},
                     "warning":  {"borderLeft": "3px solid #ffc107"}}


def _ld_verdict_hero(result: dict) -> dbc.Alert:
    verdict  = result.get("verdict", "clean")
    risk     = result.get("risk_level", "none")
    findings = result.get("findings", [])
    critical = [f for f in findings if f.get("severity") == "critical"]
    color    = _LD_VERDICT_COLOR.get(verdict, "secondary")
    label_map = {"leakage_detected": "LEAKAGE DETECTED", "suspicious": "SUSPICIOUS", "clean": "CLEAN"}
    icon_map  = {"leakage_detected": "bi-x-octagon-fill", "suspicious": "bi-exclamation-circle-fill",
                 "clean": "bi-check-circle-fill"}
    label = label_map.get(verdict, verdict.replace("_", " ").upper())
    icon  = icon_map.get(verdict, "bi-circle")
    children = [
        html.Strong([html.I(className=f"bi {icon} me-2"), f"Leakage: {label}  "],
                    className="d-inline"),
        dbc.Badge(f"risk: {risk}", color=_LD_RISK_COLOR.get(risk, "secondary"),
                  className="ms-1", style={"fontSize": "0.75rem"}),
    ]
    if critical:
        children.append(html.Ul(
            [html.Li(f.get("description", "")[:80], className="small") for f in critical[:3]],
            className="mb-0 mt-1 ps-3",
        ))
    return dbc.Alert(children, color=color, className="py-2 mb-3")


def _ld_top_blockers(findings: list) -> dbc.Card | None:
    critical = [f for f in findings if f.get("severity") == "critical"]
    if not critical:
        return None
    rows = []
    for f in critical[:5]:
        corr = f.get("correlation_with_target")
        rows.append(html.Tr([
            html.Td(html.Code(f.get("column", "—"), style={"fontSize": "0.78rem"})),
            html.Td(f.get("leakage_type", "").replace("_", " ").title(),
                    className="text-muted small"),
            html.Td(f"{corr:.3f}" if corr is not None else "—",
                    className="small font-monospace"),
            html.Td(f.get("description", ""), className="small"),
            html.Td(f.get("recommendation", ""), className="small text-muted"),
        ], style={"borderLeft": "3px solid #dc3545"}))
    return dbc.Card([
        dbc.CardHeader([
            html.I(className="bi bi-exclamation-octagon-fill text-danger me-2"),
            html.Strong("Critical Leakage — Remove Before Training"),
            html.Span(f" ({len(critical)} feature{'s' if len(critical) != 1 else ''})",
                      className="text-muted small ms-2"),
        ]),
        dbc.CardBody(
            dbc.Table(
                [html.Thead(html.Tr([html.Th("Column"), html.Th("Type"),
                                     html.Th("Corr"), html.Th("Evidence"), html.Th("Action")]))] +
                [html.Tbody(rows)],
                hover=True, responsive=True, size="sm",
            ),
            className="p-0",
        ),
    ], className="mb-3", style={"borderColor": "#dc354540"})


def _ld_dq_xref(ld_findings: list, dq_findings: list) -> dbc.Card | None:
    if not ld_findings or not dq_findings:
        return None
    dq_map = {f.get("column", ""): f for f in dq_findings if f.get("column")}
    matches = [
        (f, dq_map[f["column"]])
        for f in ld_findings
        if f.get("column") in dq_map
    ]
    if not matches:
        return None
    rows = []
    for ld_f, dq_f in matches[:6]:
        ld_sev = ld_f.get("severity", "info")
        dq_sev = dq_f.get("severity", "info")
        rows.append(html.Tr([
            html.Td(html.Code(ld_f.get("column", ""), style={"fontSize": "0.78rem"})),
            html.Td(dbc.Badge(f"LD: {ld_sev}", color=_LD_SEV_COLOR.get(ld_sev, "secondary"),
                              className="me-1 small")),
            html.Td(dbc.Badge(f"DQ: {dq_sev}", color=DQ_SEV_COLOR.get(dq_sev, "secondary"),
                              className="small")),
            html.Td(ld_f.get("description", "")[:60], className="small"),
            html.Td(dq_f.get("description", "")[:60], className="small text-muted"),
        ]))
    return dbc.Card([
        dbc.CardHeader([
            html.I(className="bi bi-link-45deg me-2 text-warning"),
            html.Strong("DQ ↔ Leakage Cross-Reference"),
            html.Span(" — these columns appear in both scans", className="text-muted small ms-2"),
        ]),
        dbc.CardBody(
            dbc.Table(
                [html.Thead(html.Tr([html.Th("Column"), html.Th("Leakage"), html.Th("DQ"),
                                     html.Th("Leakage Finding"), html.Th("DQ Finding")]))] +
                [html.Tbody(rows)],
                hover=True, responsive=True, size="sm",
            ),
            className="p-0",
        ),
    ], className="mb-3", style={"borderColor": "#ffc10740"})


def _ld_render(result: dict, dq_findings: list | None = None) -> html.Div:
    findings = result.get("findings", [])
    summary  = result.get("summary", "")

    sorted_f = sorted(findings,
                      key=lambda f: {"critical": 0, "warning": 1, "info": 2}.get(
                          f.get("severity", "info"), 2))
    rows = []
    for f in sorted_f:
        sev  = f.get("severity", "info")
        corr = f.get("correlation_with_target")
        rows.append(html.Tr([
            html.Td(dbc.Badge(sev.upper(), color=_LD_SEV_COLOR.get(sev, "secondary"))),
            html.Td(html.Code(f.get("column", "—"), style={"fontSize": "0.78rem"})),
            html.Td(f.get("leakage_type", "").replace("_", " ").title(),
                    className="text-muted small"),
            html.Td(f"{corr:.3f}" if corr is not None else "—",
                    className="small font-monospace"),
            html.Td([
                html.Div(f.get("description", ""), className="small"),
                html.Div(f.get("recommendation", ""), className="small text-muted mt-1"),
            ]),
        ], style=_LD_ROW_STYLE.get(sev, {})))

    top_blockers = _ld_top_blockers(findings)
    dq_xref      = _ld_dq_xref(findings, dq_findings or [])

    return html.Div([
        _ld_verdict_hero(result),
        top_blockers or html.Div(),
        dq_xref      or html.Div(),
        html.P(summary, className="small mb-3") if summary else html.Div(),
        dbc.Card([
            dbc.CardHeader(html.Strong(f"All Findings ({len(findings)})")),
            dbc.CardBody(
                dbc.Table(
                    [html.Thead(html.Tr([html.Th("Sev"), html.Th("Column"), html.Th("Type"),
                                         html.Th("Corr"), html.Th("Detail")]))] +
                    [html.Tbody(rows)],
                    hover=True, responsive=True, size="sm",
                ) if rows else html.P("No leakage detected — dataset looks clean.",
                                      className="text-success mb-0"),
                className="p-0",
            ),
        ]),
    ])


# ── Feature Engineering helpers ───────────────────────────────────────────────

_FE_ACTION_COLOR = {
    "scale":            "info",
    "encode_ohe":       "primary",
    "encode_index":     "primary",
    "extract_datetime": "success",
    "drop":             "secondary",
    "cast_to_int":      "warning",
    "keep_as_is":       "success",
    "log_then_scale":   "info",
    "encode_frequency": "info",
}


def _fe_render(result: dict, leakage_warning: str | None = None,
               sequence_warning: bool = False) -> html.Div:
    features     = result.get("features", [])
    dropped      = result.get("dropped_cols", [])
    pyspark_code = result.get("pyspark_code", "")
    warnings     = result.get("warnings", [])

    if leakage_warning:
        leakage_alert = dbc.Alert(
            [html.I(className="bi bi-exclamation-octagon-fill me-2"),
             html.Strong("Leakage Risk: "), leakage_warning],
            color="danger", className="py-2 mb-3",
        )
    elif sequence_warning:
        leakage_alert = dbc.Alert(
            [html.I(className="bi bi-exclamation-triangle-fill me-2"),
             html.Strong("Sequence Warning: "),
             "Leakage Detector has not been run on this dataset. "
             "Run leakage detection before engineering features to avoid encoding target leakage."],
            color="warning", className="py-2 mb-3",
        )
    else:
        leakage_alert = html.Div()

    feat_rows = []
    for f in features:
        action = f.get("action", "")
        impute = f.get("imputation", "none")
        feat_rows.append(html.Tr([
            html.Td(html.Code(f.get("column", ""), style={"fontSize": "0.78rem"})),
            html.Td(f.get("feature_type", "").replace("_", " ").title(), className="small text-muted"),
            html.Td(dbc.Badge(action.replace("_", " "),
                              color=_FE_ACTION_COLOR.get(action, "secondary"),
                              style={"fontSize": "0.65rem"})),
            html.Td(impute.replace("_", " ") if impute != "none" else "—",
                    className="small text-muted"),
            html.Td(html.Code(f.get("output_col", ""), style={"fontSize": "0.72rem", "color": "#aaa"})),
            html.Td(f.get("rationale", ""), className="small text-muted"),
        ]))

    model_family = result.get("model_family", "")
    imbalance = result.get("imbalance_handling", "")
    design_line = html.Div([
        dbc.Badge(f"Designed for: {model_family.replace('_', '-')} models", color="info",
                  className="me-2", style={"fontSize": "0.7rem"}) if model_family else None,
        html.Span([html.Strong("Imbalance: "), imbalance], className="small text-muted")
        if imbalance else None,
    ], className="mb-2") if (model_family or imbalance) else html.Div()

    return html.Div([
        leakage_alert,
        design_line,
        dbc.Alert([
            html.Strong("Warnings: "),
            html.Ul([html.Li(w, className="small") for w in warnings], className="mb-0 mt-1"),
        ], color="warning", className="py-2") if warnings else html.Div(),

        dbc.Card([
            dbc.CardHeader(html.Strong(f"Feature Plan ({len(features)} features)")),
            dbc.CardBody(
                dbc.Table(
                    [html.Thead(html.Tr([html.Th("Column"), html.Th("Type"), html.Th("Action"),
                                         html.Th("Impute"), html.Th("Output col"), html.Th("Rationale")]))] +
                    [html.Tbody(feat_rows)],
                    hover=True, responsive=True, size="sm",
                ) if feat_rows else html.P("No features planned.", className="text-muted mb-0"),
                className="p-0",
            ),
        ], className="mb-3"),

        html.Div([
            html.Span("Dropped: ", className="small text-muted me-2"),
            *[dbc.Badge(c, color="secondary", className="me-1", style={"fontSize": "0.65rem"})
              for c in dropped],
        ], className="mb-3") if dropped else html.Div(),

        dbc.Card([
            dbc.CardHeader(html.Strong("PySpark MLlib Pipeline")),
            dbc.CardBody(
                html.Pre(pyspark_code, className="mb-0",
                         style={"fontSize": "0.72rem", "maxHeight": "420px", "overflowY": "auto",
                                "backgroundColor": "#12122a", "padding": "0.75rem"}),
            ),
        ]) if pyspark_code else html.Div(),
    ])


# ── Timeline helper ───────────────────────────────────────────────────────────

def _timeline_row(run: dict, skill: str) -> html.Tr:
    m  = SKILL_META.get(skill, {})
    s  = run.get("summary", {})
    st = run.get("status", "complete")

    skill_cell = html.Td(dbc.Badge(
        [html.I(className=f"bi {m.get('icon','')} me-1"), m.get("label", skill)],
        color=m.get("color", "secondary"), className="px-2 py-1",
    ))
    label_cell = html.Td(html.Strong(run.get("label", "—")), style={"minWidth": "140px"})

    if st == "error":
        summary_cell = html.Td(html.Span(run.get("error", "")[:80], className="text-danger small"))
    elif skill == "data_quality":
        score = s.get("score", "?")
        sc = "success" if isinstance(score, int) and score >= 75 else \
             "warning" if isinstance(score, int) and score >= 50 else "danger"
        summary_cell = html.Td(html.Div([
            dbc.Badge(f"{score}/100", color=sc, className="me-1"),
            html.Span(f"{s.get('critical',0)} crit · {s.get('warning',0)} warn",
                      className="text-muted small"),
        ], className="d-flex align-items-center gap-1"))
    elif skill == "eda":
        summary_cell = html.Td(html.Span(
            f"{s.get('charts',0)} charts · {s.get('insights',0)} insights · "
            f"{s.get('targets',0)} ML targets", className="text-muted small"))
    elif skill == "join_auditor":
        verdict = s.get("verdict", "review_needed")
        vc = JA_VERDICT_CLR.get(verdict, "secondary")
        summary_cell = html.Td(html.Div([
            dbc.Badge(verdict.replace("_", " ").title(), color=vc, className="me-1"),
            html.Span(f"×{s.get('fan_out_factor',1):.2f}", className="text-muted small"),
        ], className="d-flex align-items-center gap-1"))
    elif skill == "nl_to_spark":
        preview = s.get("sql_preview", "")[:60]
        summary_cell = html.Td(
            html.Code(preview, style={"fontSize": "0.72rem", "color": "#7dd3fc"}))
    elif skill == "mlops_review":
        score = s.get("score", "?")
        sc = "success" if isinstance(score, int) and score >= 70 else \
             "warning" if isinstance(score, int) and score >= 40 else "danger"
        summary_cell = html.Td(html.Div([
            dbc.Badge(f"{score}/100", color=sc, className="me-1"),
            html.Span(f"{s.get('critical',0)} crit", className="text-muted small"),
        ], className="d-flex align-items-center gap-1"))
    elif skill == "model_selector":
        task = s.get("task_type", "").replace("_", " ").title()
        top  = s.get("top_model", "")
        summary_cell = html.Td(html.Div([
            dbc.Badge(task, color="secondary",
                      style={"border": "1px solid #a78bfa", "color": "#a78bfa",
                             "fontSize": "0.65rem"}, className="me-1"),
            html.Span(top[:40], className="text-muted small"),
        ], className="d-flex align-items-center gap-1"))
    elif skill == "leakage_detector":
        verdict = s.get("verdict", "clean")
        risk    = s.get("risk_level", "none")
        vc = _LD_VERDICT_COLOR.get(verdict, "secondary")
        summary_cell = html.Td(html.Div([
            dbc.Badge(verdict.replace("_", " ").title(), color=vc, className="me-1"),
            html.Span(f"risk: {risk}", className="text-muted small"),
        ], className="d-flex align-items-center gap-1"))
    elif skill == "feature_engineering":
        n_f = s.get("n_features", 0)
        n_d = s.get("n_dropped", 0)
        summary_cell = html.Td(html.Span(
            f"{n_f} features · {n_d} dropped", className="text-muted small"))
    else:
        summary_cell = html.Td()

    ts_cell = html.Td(html.Span(run.get("timestamp", ""), className="text-muted small"),
                      style={"width": "55px"})
    return html.Tr([ts_cell, skill_cell, label_cell, summary_cell])


# ── Cross-tool helpers ────────────────────────────────────────────────────────

def _readiness_strip(store: dict) -> html.Div:
    """Compact always-visible training-readiness bar for the session card."""
    dq_runs = [r for r in store.get("data_quality",     []) if r.get("status") == "complete"]
    ld_runs = [r for r in store.get("leakage_detector", []) if r.get("status") == "complete"]
    ms_runs = [r for r in store.get("model_selector",   []) if r.get("status") == "complete"]

    def _chip(text: str, color: str) -> html.Span:
        ico = {
            "success":   "bi-check-circle-fill text-success",
            "warning":   "bi-exclamation-triangle-fill text-warning",
            "danger":    "bi-x-circle-fill text-danger",
            "secondary": "bi-circle text-muted",
        }.get(color, "bi-circle text-muted")
        return html.Span(
            [html.I(className=f"bi {ico} me-1"), text],
            className=f"small {'text-muted' if color == 'secondary' else ''}",
        )

    # DQ chip
    if dq_runs:
        s = dq_runs[-1].get("summary", {})
        score, crit = s.get("score", 0), s.get("critical", 0)
        if crit:          dq_color, dq_txt = "danger",    f"DQ {score}/100 · {crit} crit"
        elif score < 75:  dq_color, dq_txt = "warning",   f"DQ {score}/100"
        else:             dq_color, dq_txt = "success",   f"DQ {score}/100"
    else:
        dq_color, dq_txt = "secondary", "DQ — not run"

    # Leakage chip
    if ld_runs:
        v = ld_runs[-1].get("result", {}).get("verdict", "clean")
        if v == "leakage_detected": ld_color, ld_txt = "danger",  "Leakage detected"
        elif v == "suspicious":     ld_color, ld_txt = "warning", "Leakage suspicious"
        else:                       ld_color, ld_txt = "success", "Leakage clean"
    else:
        ld_color, ld_txt = "secondary", "Leakage — not run"

    # Model chip
    if ms_runs:
        rd   = ms_runs[-1].get("result", {}).get("data_readiness", "needs_prep")
        recs = ms_runs[-1].get("result", {}).get("recommendations", [])
        top  = recs[0].get("display_name", recs[0].get("model_family", "")) if recs else ""
        if rd == "ready":        ms_color, ms_txt = "success", ("Model ready" + (f" · {top}" if top else ""))
        elif rd == "needs_prep": ms_color, ms_txt = "warning", "Model needs prep"
        else:                    ms_color, ms_txt = "danger",  "Model insufficient"
    else:
        ms_color, ms_txt = "secondary", "Model — not run"

    # Overall verdict
    all_run  = bool(dq_runs and ld_runs and ms_runs)
    colors   = [c for c in [dq_color, ld_color, ms_color] if c != "secondary"]
    if not any([dq_runs, ld_runs, ms_runs]):
        v_color, v_label = "secondary", "PENDING"
    elif all_run:
        if all(c == "success" for c in colors):
            v_color, v_label = "success", "READY TO TRAIN"
        elif "danger" in colors:
            v_color, v_label = "danger",  "NOT READY"
        else:
            v_color, v_label = "warning", "CAUTION"
    else:
        v_color = "danger" if "danger" in colors else "warning"
        v_label = "BLOCKED" if v_color == "danger" else "IN PROGRESS"

    return html.Div([
        html.Span("Training readiness", className="text-muted small fw-semibold me-3",
                  style={"whiteSpace": "nowrap"}),
        _chip(dq_txt, dq_color),
        html.Span("·", className="text-muted small mx-1"),
        _chip(ld_txt, ld_color),
        html.Span("·", className="text-muted small mx-1"),
        _chip(ms_txt, ms_color),
        dbc.Badge(v_label, color=v_color, pill=True,
                  className="ms-auto", style={"fontSize": "0.65rem"}),
    ], className="d-flex align-items-center flex-wrap gap-1 py-2 mt-2 mb-1",
       style={"borderTop": "1px solid #252545", "borderBottom": "1px solid #252545"})


def _pipeline_health_card(store: dict) -> html.Div | None:
    dq_runs = [r for r in store.get("data_quality",        []) if r.get("status") == "complete"]
    ja_runs = [r for r in store.get("join_auditor",        []) if r.get("status") == "complete"]
    ld_runs = [r for r in store.get("leakage_detector",    []) if r.get("status") == "complete"]
    ms_runs = [r for r in store.get("model_selector",      []) if r.get("status") == "complete"]
    ml_runs = [r for r in store.get("mlops_review",        []) if r.get("status") == "complete"]

    if not any([dq_runs, ja_runs, ld_runs, ms_runs, ml_runs]):
        return None

    items   = []
    all_ok  = True
    blockers = []

    if dq_runs:
        s     = dq_runs[-1].get("summary", {})
        score = s.get("score", 0)
        crit  = s.get("critical", 0)
        color = "danger" if crit or score < 50 else "warning" if score < 75 else "success"
        icon  = "bi-shield-x" if color == "danger" else "bi-shield-exclamation" if color == "warning" else "bi-shield-check"
        if color != "success":
            all_ok = False
            if color == "danger":
                blockers.append(f"DQ: {crit} critical issue{'s' if crit != 1 else ''}")
        items.append(dbc.Col([
            html.Div("Data Quality", className="text-muted small"),
            html.Div([html.I(className=f"bi {icon} text-{color} me-1"),
                      dbc.Badge(f"{score}/100", color=color)]),
        ], width="auto"))

    if ja_runs:
        verdict = ja_runs[-1].get("summary", {}).get("verdict", "review_needed")
        color   = JA_VERDICT_CLR.get(verdict, "secondary")
        if verdict != "safe":
            all_ok = False
            if verdict == "dangerous":
                blockers.append("Join: dangerous fan-out or data loss")
        items.append(dbc.Col([
            html.Div("Join Health", className="text-muted small"),
            dbc.Badge(verdict.replace("_", " ").title(), color=color),
        ], width="auto"))

    if ld_runs:
        ld_res  = ld_runs[-1].get("result", {})
        verdict = ld_res.get("verdict", "clean")
        risk    = ld_res.get("risk_level", "none")
        color   = {"leakage_detected": "danger", "suspicious": "warning", "clean": "success"}.get(verdict, "secondary")
        if verdict != "clean":
            all_ok = False
            if verdict == "leakage_detected":
                blockers.append("Leakage: critical features must be removed")
        items.append(dbc.Col([
            html.Div("Leakage Risk", className="text-muted small"),
            dbc.Badge(verdict.replace("_", " ").title(), color=color),
        ], width="auto"))

    if ms_runs:
        ms_res    = ms_runs[-1].get("result", {})
        readiness = ms_res.get("data_readiness", "needs_prep")
        top_model = ""
        recs = ms_res.get("recommendations", [])
        if recs:
            top_model = recs[0].get("display_name", recs[0].get("model_family", ""))
        color = {"ready": "success", "needs_prep": "warning", "insufficient": "danger"}.get(readiness, "secondary")
        if readiness != "ready":
            all_ok = False
        items.append(dbc.Col([
            html.Div("Model Readiness", className="text-muted small"),
            html.Div([
                dbc.Badge(readiness.replace("_", " ").title(), color=color),
                html.Span(f"  {top_model}", className="text-muted small ms-1") if top_model else html.Span(),
            ]),
        ], width="auto"))

    if ml_runs:
        ml_res = ml_runs[-1].get("result", {})
        score  = ml_res.get("readiness_score", 0)
        ml_crits = [f for f in ml_res.get("findings", []) if f.get("severity") == "critical"]
        color  = "danger" if ml_crits or score < 40 else "warning" if score < 70 else "success"
        if color != "success":
            all_ok = False
            if color == "danger":
                blockers.append(f"MLOps: {len(ml_crits)} blocker{'s' if len(ml_crits) != 1 else ''}")
        items.append(dbc.Col([
            html.Div("MLOps", className="text-muted small"),
            dbc.Badge(f"{score}/100", color=color),
        ], width="auto"))

    # ── Training-readiness verdict (DQ + leakage + model selector only) ─────────
    dq_s      = dq_runs[-1].get("summary", {}) if dq_runs else {}
    dq_ok     = bool(dq_runs) and dq_s.get("score", 0) >= 75 and dq_s.get("critical", 0) == 0
    ld_ok     = bool(ld_runs) and ld_runs[-1].get("result", {}).get("verdict") == "clean"
    ms_ok     = bool(ms_runs) and ms_runs[-1].get("result", {}).get("data_readiness") == "ready"
    prereqs_run = bool(dq_runs and ld_runs and ms_runs)
    training_ready = dq_ok and ld_ok and ms_ok

    if prereqs_run:
        if training_ready:
            t_color, t_icon, t_label = "success", "bi-check-circle-fill", "READY TO TRAIN"
            t_reasons: list[str] = []
        else:
            t_color, t_icon, t_label = "danger", "bi-x-octagon-fill", "NOT READY TO TRAIN"
            t_reasons = []
            if not dq_ok:
                s = dq_s.get("score", 0)
                c = dq_s.get("critical", 0)
                t_reasons.append(f"Data Quality: {s}/100 score" + (f", {c} critical issue{'s' if c != 1 else ''}" if c else ""))
            if not ld_ok:
                v = ld_runs[-1].get("result", {}).get("verdict", "?")
                t_reasons.append(f"Leakage: {v.replace('_', ' ')}")
            if not ms_ok:
                rd = ms_runs[-1].get("result", {}).get("data_readiness", "?")
                t_reasons.append(f"Model readiness: {rd.replace('_', ' ')}")
    else:
        missing = [n for ran, n in [(dq_runs, "Data Quality"), (ld_runs, "Leakage Detector"),
                                     (ms_runs, "Model Selector")] if not ran]
        t_color, t_icon, t_label = "secondary", "bi-hourglass-split", "TRAINING READINESS PENDING"
        t_reasons = [f"Still needed: {', '.join(missing)}"] if missing else []

    verdict_hero = dbc.Alert(
        [
            html.Strong([html.I(className=f"bi {t_icon} me-2"), t_label],
                        className="d-block mb-1" if t_reasons else ""),
            html.Ul([html.Li(r, className="small") for r in t_reasons],
                    className="mb-0 ps-3") if t_reasons else html.Span(),
        ],
        color=t_color, className="py-2 mb-3",
    )

    blocker_row = html.Div(
        [html.Span(f"⚠ {b}", className="text-warning small me-3") for b in blockers],
        className="mt-2",
    ) if blockers else html.Div()

    return dbc.Card([
        dbc.CardBody([
            verdict_hero,
            dbc.Row(items, align="center", className="g-3"),
            blocker_row,
        ]),
    ], style={
        "border": f"1px solid {'#19875440' if training_ready else '#ffffff15'}",
        "background": f"{'rgba(25,135,84,0.05)' if training_ready else 'rgba(255,255,255,0.02)'}",
    }, className="mb-2")


def _guided_workflow(store: dict) -> html.Div:
    pipeline = [
        ("data_quality",        "bi-shield-check",          "Data Quality",       "Assess data before modeling"),
        ("eda",                 "bi-graph-up",               "EDA",                "Understand distributions"),
        ("join_auditor",        "bi-link-45deg",             "Join Audit",         "Validate join keys"),
        ("leakage_detector",    "bi-exclamation-diamond",    "Leakage Check",      "Detect target leakage"),
        ("feature_engineering", "bi-wrench-adjustable",      "Feature Engineering","Build the feature pipeline"),
        ("model_selector",      "bi-cpu",                    "Model Selection",    "Choose the right algorithm"),
        ("mlops_review",        "bi-gear-fill",              "MLOps Review",       "Production readiness check"),
    ]
    ld_done = any(r.get("status") == "complete"
                  for r in store.get("leakage_detector", []))

    items = []
    for skill, icon, label, desc in pipeline:
        runs = store.get(skill, [])
        done = any(r.get("status") == "complete" for r in (runs if isinstance(runs, list) else []))
        order_warn = skill == "feature_engineering" and done and not ld_done
        items.append(html.Div([
            html.I(className=f"bi {'bi-check-circle-fill text-success' if done else 'bi-circle text-muted'} me-2",
                   style={"fontSize": "0.75rem"}),
            html.Span(label, className=f"small {'text-muted' if done else 'fw-semibold'}"),
            html.Span(f" — {desc}", className="text-muted small") if not done else html.Span(),
            dbc.Badge("⚠ run leakage first", color="warning", text_color="dark",
                      className="ms-2", style={"fontSize": "0.6rem"}) if order_warn else html.Span(),
        ], className="mb-1"))
    return html.Div([
        html.Div("Workflow Progress", className="text-muted small mb-2 fw-semibold",
                 style={"letterSpacing": "0.06em"}),
        html.Div(items),
    ])


def _synthesis_card(store: dict) -> html.Div | None:
    """Session digest — appears when 2+ skills have run."""
    from skills.session import build_digest

    SKILL_KEYS_ORDERED = [
        "data_quality", "leakage_detector", "join_auditor", "model_selector",
        "mlops_review", "eda", "feature_engineering",
    ]
    results = {}
    n_run = 0
    for skill in SKILL_KEYS_ORDERED:
        runs = [r for r in store.get(skill, []) if r.get("status") == "complete"]
        if runs:
            results[skill] = runs[-1].get("result", {})
            n_run += 1

    if n_run < 2:
        return None

    d = build_digest(results)
    verdict  = d["overall_verdict"]
    blockers = d["blockers"]
    cautions = d["cautions"]
    cleared  = d["cleared"]
    next_step = d["next_step"]

    vc = {"blocked": "danger", "caution": "warning",
          "ready": "success", "in_progress": "secondary"}.get(verdict, "secondary")
    vl = {"blocked": "BLOCKED", "caution": "CAUTION",
          "ready": "READY TO TRAIN", "in_progress": "IN PROGRESS"}.get(verdict, "PENDING")

    # ── Blockers ─────────────────────────────────────────────────────────────
    blocker_rows = []
    for b in blockers:
        src   = b["source"].replace("_", " ")
        color = "danger" if b["source"] in ("leakage", "leakage_detector") else "warning"
        blocker_rows.append(html.Tr([
            html.Td(dbc.Badge(src, color=color, style={"fontSize":"0.65rem"})),
            html.Td(html.Code(b["column"], style={"fontSize":"0.75rem"})),
            html.Td([
                html.Span(b["issue"], className="small"),
                html.Br() if b.get("action") else html.Span(),
                html.Span(f"→ {b['action']}", className="text-muted small") if b.get("action") else html.Span(),
            ]),
        ], style={"borderLeft": "3px solid #dc3545"}))

    blockers_section = html.Div([
        html.Div([
            html.I(className="bi bi-x-octagon-fill text-danger me-1"),
            html.Strong(f"Fix before training ({len(blockers)})", className="small text-danger"),
        ], className="mb-1"),
        dbc.Table([html.Tbody(blocker_rows)], size="sm", bordered=False,
                  style={"backgroundColor": "transparent", "fontSize": "0.8rem"}),
    ], className="mb-3") if blockers else html.Div()

    # ── Cautions ─────────────────────────────────────────────────────────────
    caution_rows = []
    for c in cautions:
        src = c["source"].replace("_", " ")
        caution_rows.append(html.Tr([
            html.Td(dbc.Badge(src, color="warning", text_color="dark",
                              style={"fontSize": "0.65rem"})),
            html.Td(html.Code(c["column"], style={"fontSize": "0.75rem"})),
            html.Td(html.Span(c["issue"], className="small")),
        ], style={"borderLeft": "3px solid #ffc107"}))

    cautions_section = html.Div([
        html.Div([
            html.I(className="bi bi-exclamation-triangle text-warning me-1"),
            html.Strong(f"Review ({len(cautions)})", className="small text-warning"),
        ], className="mb-1"),
        dbc.Table([html.Tbody(caution_rows)], size="sm", bordered=False,
                  style={"backgroundColor": "transparent", "fontSize": "0.8rem"}),
    ], className="mb-3") if cautions else html.Div()

    # ── Cleared ───────────────────────────────────────────────────────────────
    cleared_items = [
        html.Span([
            dbc.Badge(c["skill"].replace("_", " "), color="success",
                      style={"fontSize": "0.6rem"}, className="me-1"),
            html.Span(c["summary"], className="text-muted", style={"fontSize": "0.75rem"}),
        ], className="me-3")
        for c in cleared
    ]
    cleared_section = html.Div([
        html.Div([
            html.I(className="bi bi-check-circle-fill text-success me-1"),
            html.Strong("Cleared", className="small text-success me-2"),
            *cleared_items,
        ], className="d-flex align-items-center flex-wrap gap-1"),
    ], className="mb-3") if cleared else html.Div()

    # ── Next step ─────────────────────────────────────────────────────────────
    next_section = dbc.Alert([
        html.I(className="bi bi-arrow-right-circle me-2"),
        html.Strong("Next: "),
        next_step,
    ], color="success" if verdict == "ready" else "warning" if verdict == "caution" else "secondary",
       className="py-2 mb-0 small") if next_step else html.Div()

    accent = {"danger": "#dc3545", "warning": "#ffc107", "success": "#198754"}.get(vc, "#6c757d")
    body_children = (
        [dbc.Row([dbc.Col(blockers_section, md=6), dbc.Col(cautions_section, md=6)]),
         cleared_section, next_section]
        if blocker_rows else
        [cautions_section, cleared_section, next_section]
    )

    return dbc.Card([
        dbc.CardHeader([
            html.I(className="bi bi-journal-text me-2"),
            html.Strong("Session Digest"),
            html.Span(f" · {n_run} skill{'s' if n_run != 1 else ''} run",
                      className="text-muted small ms-1"),
            dbc.Badge(vl, color=vc, pill=True, className="ms-2",
                      style={"fontSize": "0.65rem"}),
        ], className="d-flex align-items-center"),
        dbc.CardBody(body_children),
    ], className="mb-3", style={"border": "1px solid #3a3a5a",
                                 "borderLeft": f"4px solid {accent}"})


def _cross_skill_conflicts(store: dict) -> html.Div | None:
    conflicts = []
    dq_runs = [r for r in store.get("data_quality",       []) if r.get("status") == "complete"]
    ml_runs = [r for r in store.get("mlops_review",       []) if r.get("status") == "complete"]
    ld_runs = [r for r in store.get("leakage_detector",   []) if r.get("status") == "complete"]
    fe_runs = [r for r in store.get("feature_engineering",[]) if r.get("status") == "complete"]
    ja_runs = [r for r in store.get("join_auditor",       []) if r.get("status") == "complete"]
    nl_runs = [r for r in store.get("nl_to_spark",        []) if r.get("status") == "complete"]

    if dq_runs and ld_runs:
        dq_findings = dq_runs[-1].get("result", {}).get("findings", [])
        ld_findings = ld_runs[-1].get("result", {}).get("findings", [])
        dq_cols = {f.get("column") for f in dq_findings
                   if f.get("column") and f.get("severity") in ("critical", "warning")}
        ld_cols = {f.get("column") for f in ld_findings
                   if f.get("column") and f.get("severity") in ("critical", "warning")}
        overlap = sorted(dq_cols & ld_cols)
        if overlap:
            col_list = ", ".join(f"`{c}`" for c in overlap[:5])
            more = f" (+{len(overlap)-5} more)" if len(overlap) > 5 else ""
            conflicts.append({
                "icon": "bi-exclamation-diamond-fill", "color": "danger",
                "title": "Columns flagged by both DQ and Leakage",
                "detail": f"{col_list}{more} — these have data quality AND leakage issues. Highest priority to resolve before training.",
            })

    if dq_runs and ml_runs:
        dq_crit  = dq_runs[-1].get("summary", {}).get("critical", 0)
        ml_score = ml_runs[-1].get("summary", {}).get("score", 0)
        if dq_crit and ml_score < 70:
            conflicts.append({
                "icon": "bi-shield-x", "color": "danger",
                "title": "DQ issues may block MLOps",
                "detail": f"{dq_crit} critical DQ finding(s) + MLOps readiness {ml_score}/100 — resolve data quality before deployment.",
            })

    if fe_runs and not ld_runs:
        conflicts.append({
            "icon": "bi-exclamation-diamond", "color": "warning",
            "title": "Feature engineering without leakage check",
            "detail": "Feature engineering was run without prior leakage detection. Features may encode target leakage.",
        })
    elif fe_runs and ld_runs and ld_runs[-1].get("summary", {}).get("verdict") == "leakage_detected":
        conflicts.append({
            "icon": "bi-exclamation-octagon-fill", "color": "danger",
            "title": "Features engineered despite detected leakage",
            "detail": "Leakage was detected but feature engineering proceeded. Review features for leaked signals.",
        })

    if ja_runs and nl_runs:
        ja_verdict = ja_runs[-1].get("summary", {}).get("verdict", "safe")
        if ja_verdict == "dangerous":
            conflicts.append({
                "icon": "bi-link-45deg", "color": "warning",
                "title": "Queries run on a dangerous join",
                "detail": "NL queries were generated after the Join Auditor flagged this join as dangerous. Results may be inflated by fan-out.",
            })

    if not conflicts:
        return None

    items = [dbc.Alert(
        [html.I(className=f"bi {c['icon']} me-2"),
         html.Strong(c["title"] + ": "),
         c["detail"]],
        color=c["color"], className="py-2 mb-2",
    ) for c in conflicts]

    return dbc.Card([
        dbc.CardHeader([html.I(className="bi bi-lightning-charge-fill me-2 text-warning"),
                        html.Strong("Cross-Skill Conflicts")]),
        dbc.CardBody(items),
    ], className="mb-3", style={"borderColor": "#dc354540"})


# ══════════════════════════════════════════════════════════════════════════════
# Callbacks
# ══════════════════════════════════════════════════════════════════════════════

# ── Session area ───────────────────────────────────────────────────────────────

@callback(
    Output("wb-session-active", "children"),
    Output("wb-setup-card",     "style"),
    Input("runs-store", "data"),
)
def render_session(store):
    from components.dq_gate import find_dq_run

    session = (store or {}).get("session")
    if not session or not session.get("active"):
        return html.Div(), {"border": "1px solid #3a3a5a"}

    stmt     = session.get("problem_statement", "")
    datasets = session.get("datasets", [])

    chips = []
    for ds in datasets:
        run = find_dq_run(ds, store)
        if run:
            s        = run.get("summary", {})
            score    = s.get("score", 0)
            critical = s.get("critical", 0)
            clr      = "danger" if critical else "warning" if score < 75 else "success"
            lbl      = f"{score}/100" + (f" · {critical} crit" if critical else "")
        else:
            clr, lbl = "secondary", "DQ not run"
        chips.append(html.Div([
            html.Span(ds.replace("_", " "), className="small fw-semibold me-2"),
            dbc.Badge(lbl, color=clr, style={"fontSize": "0.65rem"}),
        ], className="mb-1"))

    pipeline_health = _pipeline_health_card(store)
    readiness_strip = _readiness_strip(store)
    guided_wf       = _guided_workflow(store)
    conflicts       = _cross_skill_conflicts(store)

    card = dbc.Card(dbc.CardBody([
        dbc.Row([
            dbc.Col([
                html.Div([html.I(className="bi bi-bookmark-fill text-info me-2"),
                          html.Strong("Active Session", className="text-info"),
                          html.Span(f" · started {session.get('started_at','')}",
                                    className="text-muted small ms-2")],
                         className="mb-2"),
                html.P(f'"{stmt}"', className="mb-3 fst-italic small"),
                html.Div(chips),
            ], width=8),
            dbc.Col([
                html.Div("Session actions", className="text-muted small mb-2 fw-semibold"),
                dcc.Loading(
                    dbc.Button(
                        [html.I(className="bi bi-journal-code me-1"), "Export Notebook"],
                        id="wb-export-nb-btn", color="info", size="sm",
                        className="w-100 mb-2", outline=True,
                    ),
                    type="circle", color="#0dcaf0",
                ),
                html.Div(id="wb-export-nb-status", className="small text-muted mb-2"),
                dcc.Download(id="wb-export-nb-download"),
                dbc.Button([html.I(className="bi bi-arrow-counterclockwise me-1"), "New Session"],
                           id="home-clear-btn", color="outline-secondary", size="sm",
                           className="w-100"),
            ], width=4),
        ], align="center"),
        readiness_strip,
        dbc.Row([
            dbc.Col(pipeline_health, width=6) if pipeline_health else html.Div(),
            dbc.Col(guided_wf, width=6),
        ], className="mt-2") if pipeline_health else dbc.Row([
            dbc.Col(guided_wf, width=12),
        ], className="mt-2"),
        dbc.Row([dbc.Col(conflicts)], className="mt-2") if conflicts else html.Div(),
    ]),
        style={"border": "1px solid #0dcaf030", "background": "rgba(13,202,240,0.04)"},
        className="mb-3",
    )
    return card, {"display": "none"}


@callback(
    Output("runs-store",    "data", allow_duplicate=True),
    Output("wb-start-status", "children"),
    Output("wb-dq-dataset", "value"),
    Output("wb-eda-dataset","value"),
    Output("wb-nl-dataset", "value"),
    Input("wb-start-btn", "n_clicks"),
    State("wb-problem-stmt",   "value"),
    State("wb-dataset-select", "value"),
    State("runs-store",        "data"),
    prevent_initial_call=True,
)
def start_session(_, problem, datasets, store):
    from databricks_client import load_sample_dataset
    from skills.data_quality import assess

    if not problem or not problem.strip() or not datasets:
        raise PreventUpdate

    updated  = dict(store)
    dq_runs  = list(store.get("data_quality", []))
    first_ds = datasets[0]

    for ds in datasets:
        try:
            df       = load_sample_dataset(ds)
            result   = assess(df, table_name=ds)
            findings = result.get("findings", [])
            dq_runs.append({
                "id":        f"dq_{int(time.time() * 1000)}_{ds}",
                "label":     ds,
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
            })
        except Exception:
            pass

    updated["data_quality"] = dq_runs
    updated["session"] = {
        "active":            True,
        "problem_statement": problem.strip(),
        "datasets":          datasets,
        "started_at":        datetime.now().strftime("%H:%M"),
    }
    return updated, html.Div(), first_ds, first_ds, first_ds


# ── Skill accordion titles ─────────────────────────────────────────────────────

@callback(
    Output("wb-dq-title",  "children"),
    Output("wb-eda-title", "children"),
    Output("wb-ms-title",  "children"),
    Output("wb-ld-title",  "children"),
    Output("wb-fe-title",  "children"),
    Output("wb-ja-title",  "children"),
    Output("wb-nl-title",  "children"),
    Output("wb-ml-title",  "children"),
    Input("runs-store", "data"),
)
def update_skill_titles(store):
    def _title(skill, icon_cls, color_cls, runs, badge_fn):
        n = len(runs)
        base = html.Span([
            html.I(className=f"bi {icon_cls} me-2", style={"color": f"var(--bs-{color_cls})"}),
            html.Strong(SKILL_META[skill]["label"]),
        ])
        if not runs:
            return html.Span([base, html.Span(" — ", className="text-muted"),
                              html.Span("no runs yet", className="text-muted small fst-italic")],
                             className="d-flex align-items-center gap-2")
        count_badge = dbc.Badge(f"{n} run{'s' if n != 1 else ''}",
                                color="secondary", className="ms-2",
                                style={"fontSize": "0.65rem"})
        status_badge = badge_fn(runs[-1])
        return html.Span([base, count_badge, status_badge],
                         className="d-flex align-items-center gap-1")

    def dq_badge(run):
        s     = run.get("summary", {})
        score = s.get("score", 0)
        crit  = s.get("critical", 0)
        clr   = "danger" if crit else "warning" if score < 75 else "success"
        lbl   = f"{score}/100" + (f" · {crit} crit" if crit else "")
        return dbc.Badge(lbl, color=clr, style={"fontSize": "0.65rem"})

    def eda_badge(run):
        s = run.get("summary", {})
        return dbc.Badge(f"{s.get('insights',0)} insights",
                         color="success", style={"fontSize": "0.65rem"})

    def ja_badge(run):
        v   = run.get("summary", {}).get("verdict", "review_needed")
        clr = JA_VERDICT_CLR.get(v, "secondary")
        return dbc.Badge(v.replace("_", " ").title(), color=clr, style={"fontSize": "0.65rem"})

    def nl_badge(run):
        n = len(store.get("nl_to_spark", []))
        return dbc.Badge(f"{n} quer{'ies' if n != 1 else 'y'}",
                         color="info", text_color="dark", style={"fontSize": "0.65rem"})

    def ml_badge(run):
        score = run.get("summary", {}).get("score", 0)
        clr   = "success" if score >= 70 else "warning" if score >= 40 else "danger"
        return dbc.Badge(f"{score}/100", color=clr, style={"fontSize": "0.65rem"})

    def ms_badge(run):
        s    = run.get("summary", {})
        task = s.get("task_type", "").replace("_", " ").title()
        top  = s.get("top_model", "")
        lbl  = f"{task} · {top}" if top else task
        return dbc.Badge(lbl, color="secondary",
                         style={"fontSize": "0.65rem", "border": "1px solid #a78bfa",
                                "color": "#a78bfa"})

    def ld_badge(run):
        v   = run.get("summary", {}).get("verdict", "clean")
        clr = _LD_VERDICT_COLOR.get(v, "secondary")
        return dbc.Badge(v.replace("_", " ").title(), color=clr, style={"fontSize": "0.65rem"})

    def fe_badge(run):
        n = run.get("summary", {}).get("n_features", 0)
        return dbc.Badge(f"{n} features", color="secondary",
                         style={"fontSize": "0.65rem", "border": "1px solid #6f42c1",
                                "color": "#6f42c1"})

    dq  = store.get("data_quality",       [])
    eda = store.get("eda",                [])
    ms  = store.get("model_selector",     [])
    ld  = store.get("leakage_detector",   [])
    fe  = store.get("feature_engineering",[])
    ja  = store.get("join_auditor",       [])
    nl  = store.get("nl_to_spark",        [])
    ml  = store.get("mlops_review",       [])

    return (
        _title("data_quality",       "bi-shield-check",        "primary",   dq,  dq_badge),
        _title("eda",                "bi-graph-up",             "success",   eda, eda_badge),
        _title("model_selector",     "bi-cpu",                  "secondary", ms,  ms_badge),
        _title("leakage_detector",   "bi-exclamation-diamond",  "danger",    ld,  ld_badge),
        _title("feature_engineering","bi-wrench-adjustable",    "secondary", fe,  fe_badge),
        _title("join_auditor",       "bi-link-45deg",           "warning",   ja,  ja_badge),
        _title("nl_to_spark",        "bi-translate",            "info",      nl,  nl_badge),
        _title("mlops_review",       "bi-gear-fill",            "danger",    ml,  ml_badge),
    )


# ── Data Quality ───────────────────────────────────────────────────────────────

@callback(
    Output("runs-store", "data", allow_duplicate=True),
    Input("wb-dq-run-btn", "n_clicks"),
    State("wb-dq-dataset", "value"),
    State("runs-store",    "data"),
    prevent_initial_call=True,
)
def wb_run_dq(_, dataset, store):
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


# ── EDA ────────────────────────────────────────────────────────────────────────

@callback(
    Output("runs-store", "data", allow_duplicate=True),
    Input("wb-eda-run-btn", "n_clicks"),
    State("wb-eda-dataset", "value"),
    State("runs-store",     "data"),
    prevent_initial_call=True,
)
def wb_run_eda(_, dataset, store):
    from databricks_client import load_sample_dataset
    from skills.eda import analyze

    if not dataset:
        raise PreventUpdate

    df     = load_sample_dataset(dataset)
    result = analyze(df, table_name=dataset)
    run = {
        "id":        f"eda_{int(time.time() * 1000)}",
        "label":     dataset,
        "timestamp": datetime.now().strftime("%H:%M"),
        "status":    "complete",
        "result":    result,
        "df_json":   df.to_json(date_format="iso", orient="split"),
        "summary": {
            "charts":   len(result.get("recommended_charts", [])),
            "insights": len(result.get("key_insights", [])),
            "targets":  len(result.get("potential_targets", [])),
        },
    }
    updated = dict(store)
    updated["eda"] = store.get("eda", []) + [run]
    return updated


@callback(
    Output("wb-eda-results", "children"),
    Input("runs-store", "data"),
)
def wb_render_eda(store):
    from components.run_card import make_run_card

    runs = store.get("eda", [])
    if not runs:
        return html.P("No runs yet — select a dataset and click Run EDA.",
                      className="text-muted small py-3")
    cards = []
    for run in reversed(runs):
        content = _eda_render(run["result"], run.get("df_json", "{}")) \
            if run["status"] == "complete" \
            else dbc.Alert(run.get("error", "Unknown error"), color="danger")
        cards.append(make_run_card(run, "eda", content))
    return cards


# ── Model Selector ─────────────────────────────────────────────────────────────

@callback(
    Output("wb-ms-target", "options"),
    Output("wb-ms-target", "value"),
    Input("wb-ms-dataset", "value"),
    prevent_initial_call=True,
)
def wb_ms_load_cols(dataset):
    from databricks_client import load_sample_dataset
    if not dataset:
        raise PreventUpdate
    df   = load_sample_dataset(dataset)
    opts = [{"label": c, "value": c} for c in df.columns]
    # Suggest a sensible default target: last column, or first binary/categorical column
    default = df.columns[-1]
    for col in df.columns:
        if df[col].nunique() <= 10 and df[col].dtype not in ("float64", "float32"):
            default = col
            break
    return opts, default


@callback(
    Output("runs-store", "data", allow_duplicate=True),
    Input("wb-ms-run-btn",  "n_clicks"),
    State("wb-ms-dataset",  "value"),
    State("wb-ms-target",   "value"),
    State("runs-store",     "data"),
    prevent_initial_call=True,
)
def wb_run_ms(_, dataset, target_col, store):
    from databricks_client import load_sample_dataset
    from skills.model_selector import select

    if not dataset or not target_col:
        raise PreventUpdate

    session = store.get("session") or {}
    problem = session.get("problem_statement", "")

    dq_runs     = store.get("data_quality", [])
    latest_dq   = next((r for r in reversed(dq_runs) if r.get("status") == "complete"), None)
    dq_findings = (latest_dq or {}).get("result", {}).get("findings") or None

    ld_runs          = store.get("leakage_detector", [])
    latest_ld        = next((r for r in reversed(ld_runs) if r.get("status") == "complete"), None)
    leakage_findings = (latest_ld or {}).get("result", {}).get("findings") or None

    label = f"{dataset} → {target_col}"
    try:
        df     = load_sample_dataset(dataset)
        result = select(df, target_col, problem_context=problem,
                        dq_findings=dq_findings, leakage_findings=leakage_findings)
        recs         = result.get("recommendations", [])
        top          = recs[0].get("display_name", "") if recs else ""
        t_analysis   = result.get("target_analysis", {}) or {}
        top_rationale = recs[0].get("rationale", "") if recs else ""
        run = {
            "id":        f"ms_{int(time.time() * 1000)}",
            "label":     label,
            "timestamp": datetime.now().strftime("%H:%M"),
            "status":    "complete",
            "result":    result,
            "summary": {
                "task_type":        result.get("task_type", ""),
                "top_model":        top,
                "top_rationale":    top_rationale[:120] if top_rationale else "",
                "readiness":        result.get("data_readiness", ""),
                "n_recs":           len(recs),
                "class_balance":    t_analysis.get("class_balance_issue", False),
                "imbalance_ratio":  t_analysis.get("imbalance_ratio"),
            },
        }
    except PreventUpdate:
        raise
    except Exception as exc:
        run = {"id": f"ms_{int(time.time()*1000)}", "label": label,
               "timestamp": datetime.now().strftime("%H:%M"),
               "status": "error", "error": str(exc), "summary": {}}

    updated = dict(store)
    updated["model_selector"] = store.get("model_selector", []) + [run]
    return updated


@callback(
    Output("wb-ms-results", "children"),
    Input("runs-store", "data"),
)
def wb_render_ms(store):
    from components.run_card import make_run_card
    runs = store.get("model_selector", [])
    if not runs:
        return html.P("Select a dataset and target column, then click Select Models.",
                      className="text-muted small py-3")
    cards = []
    for run in reversed(runs):
        content = _ms_render(run["result"]) if run["status"] == "complete" \
            else dbc.Alert(run.get("error", "Unknown error"), color="danger")
        cards.append(make_run_card(run, "model_selector", content))
    return cards


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


# ── NL → PySpark ──────────────────────────────────────────────────────────────

@callback(
    Output("wb-nl-tables", "data"),
    Input("wb-nl-dataset", "value"),
    prevent_initial_call=True,
)
def wb_nl_load_tables(dataset):
    from databricks_client import load_sample_dataset

    if not dataset:
        raise PreventUpdate
    df = load_sample_dataset(dataset)
    return {dataset: df.to_json(date_format="iso", orient="split")}


@callback(
    Output("runs-store", "data", allow_duplicate=True),
    Input("wb-nl-run-btn",  "n_clicks"),
    State("wb-nl-question", "value"),
    State("wb-nl-tables",   "data"),
    State("wb-nl-dataset",  "value"),
    State("runs-store",     "data"),
    prevent_initial_call=True,
)
def wb_run_nl(_, question, tables, dataset, store):
    from skills.nl_to_spark import generate

    if not question or not question.strip():
        raise PreventUpdate

    session = store.get("session") or {}
    context = session.get("problem_statement", "")

    # Load table if store is empty
    if not tables and dataset:
        from databricks_client import load_sample_dataset
        df     = load_sample_dataset(dataset)
        tables = {dataset: df.to_json(date_format="iso", orient="split")}

    if not tables:
        raise PreventUpdate

    label  = (question[:40] + "…") if len(question) > 40 else question
    tables_df = {name: pd.read_json(io.StringIO(js), orient="split")
                 for name, js in tables.items()}
    result = generate(question, tables_df, context=context)
    sql    = result.get("sql_query", "")
    run = {
        "id":        f"nl_{int(time.time() * 1000)}",
        "label":     label,
        "timestamp": datetime.now().strftime("%H:%M"),
        "status":    "complete",
        "result":    result,
        "summary":   {"sql_preview": sql.split("\n")[0] if sql else ""},
    }
    updated = dict(store)
    updated["nl_to_spark"] = store.get("nl_to_spark", []) + [run]
    return updated


@callback(
    Output("wb-nl-results", "children"),
    Input("runs-store", "data"),
)
def wb_render_nl(store):
    from components.run_card import make_run_card

    runs = store.get("nl_to_spark", [])
    if not runs:
        return html.P("No runs yet — enter a question and click Run.",
                      className="text-muted small py-3")

    completed     = [r for r in runs if r.get("status") == "complete"]
    compare_panel = html.Div()
    if len(completed) >= 2:
        r1, r2 = completed[-2], completed[-1]
        compare_panel = dbc.Card([
            dbc.CardHeader([html.I(className="bi bi-columns-gap me-2"),
                            html.Strong("Last 2 Queries — Side by Side")]),
            dbc.CardBody(dbc.Row([
                dbc.Col([
                    html.Div(html.Code(r1.get("label",""), className="small text-info"),
                             className="mb-1"),
                    html.Pre(r1["result"].get("sql_query",""), className="mb-0",
                             style={"fontSize": "0.72rem", "maxHeight": "180px",
                                    "overflow": "auto", "background": "#12122a",
                                    "padding": "0.5rem", "borderRadius": "4px"}),
                ], width=6),
                dbc.Col([
                    html.Div(html.Code(r2.get("label",""), className="small text-info"),
                             className="mb-1"),
                    html.Pre(r2["result"].get("sql_query",""), className="mb-0",
                             style={"fontSize": "0.72rem", "maxHeight": "180px",
                                    "overflow": "auto", "background": "#12122a",
                                    "padding": "0.5rem", "borderRadius": "4px"}),
                ], width=6),
            ])),
        ], className="mb-3", style={"borderColor": "#0dcaf040"})

    dq_critical = any(
        r.get("summary", {}).get("critical", 0) > 0
        for r in store.get("data_quality", [])
        if r.get("status") == "complete"
    )

    cards = [compare_panel]
    for run in reversed(runs):
        if run["status"] == "complete":
            send_btn = dbc.Button(
                [html.I(className="bi bi-gear-fill me-1"), "Send to MLOps Review"],
                id={"type": "nl-to-mlops-btn", "index": run["id"]},
                size="sm", color="danger", outline=True,
                className="mb-3",
                style={"fontSize": "0.75rem"},
            )
            content = html.Div([send_btn, _nl_render(run["result"], dq_critical=dq_critical)])
        else:
            content = dbc.Alert(run.get("error", "Unknown error"), color="danger")
        cards.append(make_run_card(run, "nl_to_spark", content))
    return cards


@callback(
    Output("wb-ml-code",  "value"),
    Output("wb-accordion", "active_item"),
    Input({"type": "nl-to-mlops-btn", "index": dash.ALL}, "n_clicks"),
    State("runs-store", "data"),
    prevent_initial_call=True,
)
def nl_send_to_mlops(n_clicks_list, store):
    if not any(n for n in n_clicks_list if n):
        raise PreventUpdate
    ctx = dash.callback_context
    if not ctx.triggered_id:
        raise PreventUpdate
    run_id = ctx.triggered_id["index"]
    runs   = store.get("nl_to_spark", [])
    run    = next((r for r in runs if r["id"] == run_id), None)
    if not run or run.get("status") != "complete":
        raise PreventUpdate
    code = run["result"].get("pyspark_code", "")
    # Expand the accordion to MLOps Review
    return code, ["dq", "eda", "ms", "ld", "fe", "ja", "nl", "ml"]


# ── MLOps Review ───────────────────────────────────────────────────────────────

@callback(
    Output("runs-store", "data", allow_duplicate=True),
    Input("wb-ml-run-btn", "n_clicks"),
    State("wb-ml-code",    "value"),
    State("runs-store",    "data"),
    prevent_initial_call=True,
)
def wb_run_ml(_, code, store):
    from skills.mlops_reviewer import review

    if not code or not code.strip():
        raise PreventUpdate

    label    = (code.strip().splitlines()[0] or "notebook")[:40]
    result   = review(code)
    findings = result.get("findings", [])
    run = {
        "id":        f"mlops_{int(time.time() * 1000)}",
        "label":     label,
        "timestamp": datetime.now().strftime("%H:%M"),
        "status":    "complete",
        "result":    result,
        "summary": {
            "score":    result.get("readiness_score", 0),
            "critical": sum(1 for f in findings if f.get("severity") == "critical"),
            "warning":  sum(1 for f in findings if f.get("severity") == "warning"),
        },
    }
    updated = dict(store)
    updated["mlops_review"] = store.get("mlops_review", []) + [run]
    return updated


@callback(
    Output("wb-ml-results", "children"),
    Input("runs-store", "data"),
)
def wb_render_ml(store):
    from components.run_card import make_run_card

    runs = store.get("mlops_review", [])
    if not runs:
        return html.P("No runs yet — paste notebook code and click Review.",
                      className="text-muted small py-3")
    cards = []
    for run in reversed(runs):
        content = _ml_render(run["result"], session_store=store) \
            if run["status"] == "complete" \
            else dbc.Alert(run.get("error", "Unknown error"), color="danger")
        cards.append(make_run_card(run, "mlops_review", content))
    return cards


# ── Leakage Detector ──────────────────────────────────────────────────────────

@callback(
    Output("wb-ld-target", "options"),
    Output("wb-ld-target", "value"),
    Input("wb-ld-dataset", "value"),
    prevent_initial_call=True,
)
def wb_ld_load_cols(dataset):
    from databricks_client import load_sample_dataset
    if not dataset:
        raise PreventUpdate
    df   = load_sample_dataset(dataset)
    opts = [{"label": c, "value": c} for c in df.columns]
    default = df.columns[-1]
    for col in df.columns:
        if df[col].nunique() <= 10 and df[col].dtype not in ("float64", "float32"):
            default = col
            break
    return opts, default


@callback(
    Output("runs-store", "data", allow_duplicate=True),
    Input("wb-ld-run-btn", "n_clicks"),
    State("wb-ld-dataset",  "value"),
    State("wb-ld-target",   "value"),
    State("runs-store",     "data"),
    prevent_initial_call=True,
)
def wb_run_ld(_, dataset, target_col, store):
    from databricks_client import load_sample_dataset
    from skills.leakage_detector import check

    if not dataset or not target_col:
        raise PreventUpdate

    session = store.get("session") or {}
    context = session.get("problem_statement", "")
    label   = f"{dataset} → {target_col}"
    try:
        df     = load_sample_dataset(dataset)
        result = check(df, target_col, context=context)
        run = {
            "id":        f"ld_{int(time.time() * 1000)}",
            "label":     label,
            "timestamp": datetime.now().strftime("%H:%M"),
            "status":    "complete",
            "result":    result,
            "summary": {
                "verdict":     result.get("verdict", "clean"),
                "risk_level":  result.get("risk_level", "none"),
                "n_findings":  len(result.get("findings", [])),
            },
        }
    except Exception as exc:
        run = {"id": f"ld_{int(time.time()*1000)}", "label": label,
               "timestamp": datetime.now().strftime("%H:%M"),
               "status": "error", "error": str(exc), "summary": {}}

    updated = dict(store)
    updated["leakage_detector"] = store.get("leakage_detector", []) + [run]
    return updated


@callback(
    Output("wb-ld-results", "children"),
    Input("runs-store", "data"),
)
def wb_render_ld(store):
    from components.run_card import make_run_card
    runs = store.get("leakage_detector", [])
    if not runs:
        return html.P("Select a dataset and target column, then click Check Leakage.",
                      className="text-muted small py-3")
    dq_runs   = [r for r in store.get("data_quality", []) if r.get("status") == "complete"]
    dq_findings = dq_runs[-1]["result"].get("findings", []) if dq_runs else []
    cards = []
    for run in reversed(runs):
        content = _ld_render(run["result"], dq_findings=dq_findings) \
            if run["status"] == "complete" \
            else dbc.Alert(run.get("error", "Unknown error"), color="danger")
        cards.append(make_run_card(run, "leakage_detector", content))
    return cards


# ── Feature Engineering ────────────────────────────────────────────────────────

@callback(
    Output("wb-fe-target", "options"),
    Output("wb-fe-target", "value"),
    Input("wb-fe-dataset", "value"),
    prevent_initial_call=True,
)
def wb_fe_load_cols(dataset):
    from databricks_client import load_sample_dataset
    if not dataset:
        raise PreventUpdate
    df   = load_sample_dataset(dataset)
    opts = [{"label": c, "value": c} for c in df.columns]
    default = df.columns[-1]
    for col in df.columns:
        if df[col].nunique() <= 10 and df[col].dtype not in ("float64", "float32"):
            default = col
            break
    return opts, default


@callback(
    Output("runs-store", "data", allow_duplicate=True),
    Input("wb-fe-run-btn", "n_clicks"),
    State("wb-fe-dataset",  "value"),
    State("wb-fe-target",   "value"),
    State("runs-store",     "data"),
    prevent_initial_call=True,
)
def wb_run_fe(_, dataset, target_col, store):
    from databricks_client import load_sample_dataset
    from skills.feature_engineering import suggest

    if not dataset or not target_col:
        raise PreventUpdate

    session = store.get("session") or {}
    context = session.get("problem_statement", "")
    label   = f"{dataset} → {target_col}"

    ld_same = [r for r in store.get("leakage_detector", [])
               if r.get("status") == "complete"
               and dataset.lower() in r.get("label", "").lower()]
    leakage_warning  = None
    sequence_warning = False
    if ld_same:
        ld_res = ld_same[-1]["result"]
        if ld_res.get("verdict") == "leakage_detected":
            crits = [f for f in ld_res.get("findings", []) if f.get("severity") == "critical"]
            leakage_warning = crits[0].get("description", "Leakage detected in dataset.") if crits \
                              else "Leakage detected in dataset."
    else:
        sequence_warning = True

    try:
        df     = load_sample_dataset(dataset)
        result = suggest(df, target_col, context=context)
        run = {
            "id":               f"fe_{int(time.time() * 1000)}",
            "label":            label,
            "timestamp":        datetime.now().strftime("%H:%M"),
            "status":           "complete",
            "result":           result,
            "leakage_warning":  leakage_warning,
            "sequence_warning": sequence_warning,
            "summary": {
                "n_features": len(result.get("features", [])),
                "n_dropped":  len(result.get("dropped_cols", [])),
            },
        }
    except Exception as exc:
        run = {"id": f"fe_{int(time.time()*1000)}", "label": label,
               "timestamp": datetime.now().strftime("%H:%M"),
               "status": "error", "error": str(exc), "summary": {}}

    updated = dict(store)
    updated["feature_engineering"] = store.get("feature_engineering", []) + [run]
    return updated


@callback(
    Output("wb-fe-results", "children"),
    Input("runs-store", "data"),
)
def wb_render_fe(store):
    from components.run_card import make_run_card
    runs = store.get("feature_engineering", [])
    if not runs:
        return html.P("Select a dataset and target column, then click Plan Features.",
                      className="text-muted small py-3")
    cards = []
    for run in reversed(runs):
        if run["status"] == "complete":
            content = _fe_render(
                run["result"],
                leakage_warning=run.get("leakage_warning"),
                sequence_warning=run.get("sequence_warning", False),
            )
        else:
            content = dbc.Alert(run.get("error", "Unknown error"), color="danger")
        cards.append(make_run_card(run, "feature_engineering", content))
    return cards


# ── Cross-tool synthesis ───────────────────────────────────────────────────────

@callback(
    Output("wb-synthesis", "children"),
    Input("runs-store", "data"),
)
def wb_render_synthesis(store):
    return _synthesis_card(store) or html.Div()


# ── Investigation log ─────────────────────────────────────────────────────────

@callback(
    Output("wb-timeline", "children"),
    Input("runs-store", "data"),
)
def wb_render_timeline(store):
    all_runs = []
    for skill, runs in store.items():
        if skill not in SKILL_KEYS or not isinstance(runs, list):
            continue
        for r in runs:
            all_runs.append((skill, r))

    if not all_runs:
        return html.P("No runs yet.", className="text-muted small py-2")

    all_runs.sort(key=lambda x: x[1].get("id", ""), reverse=True)
    return dbc.Table(
        [html.Thead(html.Tr([html.Th("Time"), html.Th("Skill"),
                             html.Th("Dataset / Query"), html.Th("Result")]))] +
        [html.Tbody([_timeline_row(r, skill) for skill, r in all_runs])],
        hover=True, responsive=True, size="sm",
    )


# ── Experiment Tracker ────────────────────────────────────────────────────────

@callback(
    Output("wb-et-output", "children"),
    Input("wb-et-run-btn", "n_clicks"),
    State("wb-et-metrics", "value"),
    State("runs-store", "data"),
    prevent_initial_call=True,
)
def wb_run_et(n_clicks, metrics_json, store):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate

    if not metrics_json or not metrics_json.strip():
        return dbc.Alert("Paste your training metrics JSON above first.", color="warning")

    try:
        metrics = json.loads(metrics_json)
    except json.JSONDecodeError as exc:
        return dbc.Alert(f"Invalid JSON: {exc}", color="danger")

    session = (store or {}).get("session") or {}
    context = session.get("problem_statement", "")

    try:
        from skills.experiment_tracker import interpret
        result = interpret(metrics, context=context)
        return _et_render(result)
    except Exception as exc:
        return dbc.Alert(f"Error interpreting metrics: {exc}", color="danger")


# ── Export Notebook ────────────────────────────────────────────────────────────

@callback(
    Output("wb-export-nb-download", "data"),
    Output("wb-export-nb-status",   "children"),
    Input("wb-export-nb-btn", "n_clicks"),
    State("runs-store", "data"),
    prevent_initial_call=True,
)
def wb_export_notebook(_, store):
    from skills.notebook_generator import generate

    session = (store or {}).get("session") or {}
    if not session.get("active"):
        return None, "No active session to export."

    problem  = session.get("problem_statement", "session")
    filename = (problem[:30].strip().replace(" ", "_").lower() + ".ipynb") \
               if problem else "workbench_session.ipynb"

    result = generate(store, filename=filename)
    return (
        dcc.send_string(result["notebook_json"], filename),
        f"Exported {result['cells_generated']} cells ({', '.join(result['skills_included']) or 'overview'})",
    )

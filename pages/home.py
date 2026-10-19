"""
DS Workbench — single-page investigation canvas.

Every skill lives as an accordion section on this page. This module keeps the layout
and the session-level callbacks; per-skill pieces live in the workbench package:
  workbench/renderers/<skill>.py   result → Dash components
  workbench/callbacks/<skill>.py   run/render callbacks for that skill's section
  workbench/session_panels.py      readiness strip, pipeline health, synthesis, timeline
  workbench/constants.py           dataset options, skill metadata, colour maps
"""
import time
from datetime import datetime

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dcc, html
from dash.exceptions import PreventUpdate

from workbench.constants import (
    CAUSAL_DATASET_OPTIONS, DATASET_OPTIONS, JA_VERDICT_CLR, JOIN_SCENARIO_OPTIONS, ML_PLACEHOLDER,
    SKILL_KEYS, SKILL_META,
)
from workbench.renderers.ld import _LD_VERDICT_COLOR
from workbench.session_panels import (
    _cross_skill_conflicts, _guided_workflow, _pipeline_health_card, _readiness_strip,
    _synthesis_card, _timeline_row,
)
import workbench.callbacks  # noqa: F401  (registers per-skill callbacks)

dash.register_page(__name__, path="/", name="Workbench", order=0)

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

                # ── Causal Inference ──────────────────────────────────────
                dbc.AccordionItem([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Dataset", size="sm"),
                            dbc.Select(id="wb-ci-dataset", options=CAUSAL_DATASET_OPTIONS,
                                       value="ab_test_clean", size="sm"),
                        ], md=4),
                        dbc.Col([
                            dbc.Label("Treatment", size="sm"),
                            dbc.Select(id="wb-ci-treatment", options=[], value=None, size="sm"),
                        ], md=2),
                        dbc.Col([
                            dbc.Label("Outcome", size="sm"),
                            dbc.Select(id="wb-ci-outcome", options=[], value=None, size="sm"),
                        ], md=2),
                        dbc.Col([
                            dbc.Label("Pre-period outcome", size="sm"),
                            dbc.Select(id="wb-ci-pre", options=[], value=None, size="sm"),
                        ], md=2),
                        dbc.Col([
                            dbc.Label("Design", size="sm"),
                            dbc.Select(id="wb-ci-design", size="sm", value="randomized", options=[
                                {"label": "Randomised (A/B)", "value": "randomized"},
                                {"label": "Observational", "value": "observational"},
                                {"label": "Before/after vs control (DiD)", "value": "did"},
                                {"label": "Not sure", "value": "unsure"},
                            ]),
                        ], md=2),
                    ], className="g-2 mb-2"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Time column (DiD)", size="sm"),
                            dbc.Select(id="wb-ci-time", options=[], value=None, size="sm"),
                        ], md=3),
                        dbc.Col([
                            dbc.Label("Unit column (DiD)", size="sm"),
                            dbc.Select(id="wb-ci-unit", options=[], value=None, size="sm"),
                        ], md=3),
                        dbc.Col([
                            dbc.Label("Change went live (DiD)", size="sm"),
                            dbc.Input(id="wb-ci-start", placeholder="e.g. 2026-05-25", size="sm"),
                        ], md=3),
                    ], id="wb-ci-did-row", className="g-2 mb-2"),
                    dbc.Button([html.I(className="bi bi-diagram-2 me-1"), "Analyze Effect"],
                               id="wb-ci-run-btn", color="secondary", size="sm", className="mb-3",
                               style={"borderColor": "#20c997", "color": "#20c997"}),
                    dcc.Loading(html.Div(id="wb-ci-results"), type="circle", color="#20c997"),
                ],
                    title=html.Span(id="wb-ci-title"),
                    item_id="ci",
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
    Output("wb-ci-title",  "children"),
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
    ci  = store.get("causal_inference",   [])

    def ci_badge(run):
        verdict = run.get("summary", {}).get("verdict", "")
        color = {"valid": "success", "caution": "warning", "not_identified": "danger"}.get(verdict, "secondary")
        return dbc.Badge(verdict.replace("_", " "), color=color, style={"fontSize": "0.65rem"})

    return (
        _title("data_quality",       "bi-shield-check",        "primary",   dq,  dq_badge),
        _title("eda",                "bi-graph-up",             "success",   eda, eda_badge),
        _title("model_selector",     "bi-cpu",                  "secondary", ms,  ms_badge),
        _title("leakage_detector",   "bi-exclamation-diamond",  "danger",    ld,  ld_badge),
        _title("feature_engineering","bi-wrench-adjustable",    "secondary", fe,  fe_badge),
        _title("join_auditor",       "bi-link-45deg",           "warning",   ja,  ja_badge),
        _title("nl_to_spark",        "bi-translate",            "info",      nl,  nl_badge),
        _title("mlops_review",       "bi-gear-fill",            "danger",    ml,  ml_badge),
        _title("causal_inference",   "bi-diagram-2",            "info",      ci,  ci_badge),
    )


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

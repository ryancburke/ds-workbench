import dash_bootstrap_components as dbc
from dash import html

# SKILLS is consumed by app.py to build SKILL_KEYS — do not remove entries
SKILLS = [
    {"name": "Data Quality",   "path": "/data-quality",   "icon": "bi-shield-check",  "key": "data_quality"},
    {"name": "EDA",            "path": "/eda",             "icon": "bi-graph-up",      "key": "eda"},
    {"name": "Join Auditor",   "path": "/join-auditor",    "icon": "bi-link-45deg",    "key": "join_auditor"},
    {"name": "NL → PySpark",   "path": "/nl-to-spark",     "icon": "bi-translate",     "key": "nl_to_spark"},
    {"name": "MLOps Review",   "path": "/mlops-review",    "icon": "bi-gear-fill",     "key": "mlops_review"},
    {"name": "Model Selector",      "path": "/model-selector",      "icon": "bi-cpu",                 "key": "model_selector"},
    {"name": "Leakage Detector",    "path": "/leakage-detector",    "icon": "bi-exclamation-diamond", "key": "leakage_detector"},
    {"name": "Feature Engineering", "path": "/feature-engineering", "icon": "bi-wrench-adjustable",   "key": "feature_engineering"},
]

_NAV = [
    {"label": "Workbench",  "href": "/",          "icon": "bi-grid-1x2-fill"},
    {"label": "Sessions",   "href": "/sessions",  "icon": "bi-people"},
    {"label": "Feedback",   "href": "/feedback",  "icon": "bi-chat-left-dots",
     "badge_id": "sidebar-feedback-count"},
    {"label": "Benchmarks", "href": "/benchmarks","icon": "bi-speedometer2"},
    {"label": "Agent Sim",  "href": "/agent-sim", "icon": "bi-robot"},
    {"label": "Costs",      "href": "/costs",     "icon": "bi-cash-coin"},
]


def create_sidebar() -> html.Div:
    links = []
    for item in _NAV:
        inner = [
            html.I(className=f"bi {item['icon']} me-2"),
            html.Span(item["label"], className="flex-grow-1"),
        ]
        if "badge_id" in item:
            inner.append(html.Span(id=item["badge_id"]))
        # Keep sidebar-count spans for SKILLS so app.py callbacks don't error
        links.append(
            dbc.NavLink(
                html.Div(inner, className="d-flex align-items-center w-100"),
                href=item["href"],
                active="exact",
                class_name="sidebar-link px-3 py-2 d-flex align-items-center",
            )
        )

    # Hidden spans so app.py sidebar-count-{key} callbacks don't raise
    hidden_counts = html.Div(
        [html.Span(id=f"sidebar-count-{s['key']}", style={"display": "none"})
         for s in SKILLS],
        style={"display": "none"},
    )

    return html.Div(
        [
            html.Div(
                [html.I(className="bi bi-cpu-fill me-2"), html.Strong("DS Workbench")],
                className="text-white px-3 py-4",
                style={"fontSize": "1.05rem", "letterSpacing": "0.02em"},
            ),
            html.Div(links),
            hidden_counts,
        ],
        style={
            "backgroundColor": "#12122a",
            "minHeight": "100vh",
            "borderRight": "1px solid #2a2a4a",
        },
    )

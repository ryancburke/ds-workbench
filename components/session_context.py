import dash_bootstrap_components as dbc
from dash import html


def session_banner(store: dict) -> html.Div:
    """Small context pill shown at the top of skill pages when a session is active."""
    session = store.get("session") if store else None
    if not session or not session.get("active"):
        return html.Div()

    stmt     = session.get("problem_statement", "")
    datasets = session.get("datasets", [])
    preview  = (stmt[:72] + "…") if len(stmt) > 72 else stmt

    return html.Div(
        dbc.Row([
            dbc.Col(
                html.Div([
                    html.I(className="bi bi-bookmark-fill text-info me-2"),
                    html.Strong("Session: ", className="text-muted small"),
                    html.Span(preview, className="small fst-italic text-muted"),
                ]),
                className="d-flex align-items-center",
            ),
            dbc.Col(
                html.Div([
                    dbc.Badge(ds, color="dark", className="me-1",
                              style={"fontSize": "0.65rem", "border": "1px solid #3a3a5a"})
                    for ds in datasets
                ]),
                className="col-auto d-flex align-items-center",
            ),
        ], align="center"),
        className="mb-3 px-1",
        style={"borderLeft": "3px solid #0dcaf0", "paddingLeft": "0.6rem"},
    )

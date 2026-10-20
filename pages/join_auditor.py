import base64
import io
import time
from datetime import datetime

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, State, callback, dcc, html
from dash.exceptions import PreventUpdate

dash.register_page(__name__, path="/join-auditor", name="Join Auditor", order=3)

VERDICT_COLOR   = {"safe": "success", "review_needed": "warning", "dangerous": "danger"}
SEVERITY_COLOR  = {"critical": "danger", "warning": "warning", "info": "info"}

_SEV_RANK = {"critical": 0, "warning": 1, "info": 2}
_SEV_HEADER = {
    "critical": ("bi-exclamation-octagon-fill text-danger", "Blocking Issues"),
    "warning":  ("bi-exclamation-triangle text-warning",    "Warnings"),
    "info":     ("bi-info-circle text-info",                "Info"),
}
_ROW_STYLE = {
    "critical": {"borderLeft": "3px solid #dc3545"},
    "warning":  {"borderLeft": "3px solid #ffc107"},
    "info":     {},
}


def layout():
    return html.Div([
        html.H4([html.I(className="bi bi-link-45deg me-2"), "Join Auditor"],
                className="mb-1"),
        html.P("Detect fan-out joins, unmatched keys, and row inflation before metrics go wrong.",
               className="text-muted mb-4"),

        dbc.Card(dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Load Scenario"),
                    dbc.InputGroup([
                        dbc.Select(id="ja-scenario-select", value="orders_customers_fanout",
                            options=[
                                {"label": "Orders ⋈ Customers  (fan-out — duplicate IDs)", "value": "orders_customers_fanout"},
                                {"label": "Sales ⋈ Products  (clean join)",                "value": "sales_products_clean"},
                            ]),
                        dbc.Button("Load", id="ja-load-btn", color="secondary", outline=True),
                    ]),
                ], width=5),
                dbc.Col([
                    html.Label("Or upload CSVs"),
                    dbc.Row([
                        dbc.Col([
                            dcc.Upload(id="ja-left-upload",
                                children=html.Div(["Left table", html.Br(),
                                                   html.Small(id="ja-left-filename", className="text-muted")]),
                                className="upload-zone",
                                style={"padding": "8px", "textAlign": "center", "fontSize": "0.8rem"}),
                        ], width=6),
                        dbc.Col([
                            dcc.Upload(id="ja-right-upload",
                                children=html.Div(["Right table", html.Br(),
                                                   html.Small(id="ja-right-filename", className="text-muted")]),
                                className="upload-zone",
                                style={"padding": "8px", "textAlign": "center", "fontSize": "0.8rem"}),
                        ], width=6),
                    ]),
                ], width=4),
                dbc.Col([
                    html.Label("Join Key(s)"),
                    dbc.Input(id="ja-keys", placeholder="customer_id", className="mb-2"),
                    html.Label("Join Type"),
                    dbc.RadioItems(id="ja-join-type", value="inner", inline=True,
                        options=[{"label": t.title(), "value": t}
                                 for t in ("inner", "left", "right", "outer")],
                        className="small"),
                ], width=3),
            ], className="g-3 mb-3"),
            dbc.Button([html.I(className="bi bi-play-fill me-1"), "Audit Join"],
                       id="ja-run-btn", color="warning", className="w-100"),
        ]), className="mb-4"),

        dcc.Loading(html.Div(id="ja-runs-container"), type="circle", color="#ffc107"),
        dcc.Store(id="ja-left-store"),
        dcc.Store(id="ja-right-store"),
    ])


# ── Load scenario / upload callbacks ──────────────────────────────────────────

@callback(
    Output("ja-left-store", "data"),
    Output("ja-right-store", "data"),
    Output("ja-keys", "value"),
    Output("ja-left-filename", "children"),
    Output("ja-right-filename", "children"),
    Input("ja-load-btn", "n_clicks"),
    Input("ja-left-upload", "contents"),
    Input("ja-right-upload", "contents"),
    State("ja-scenario-select", "value"),
    State("ja-left-upload", "filename"),
    State("ja-right-upload", "filename"),
    prevent_initial_call=True,
)
def populate_tables(load_clicks, left_contents, right_contents,
                    scenario, left_fname, right_fname):
    if not (load_clicks or left_contents or right_contents):  # a click or an upload, not a page load
        raise PreventUpdate
    from dash import ctx
    from databricks_client import load_join_scenario, SAMPLE_JOIN_SCENARIOS

    if ctx.triggered_id == "ja-load-btn" and scenario:
        left_df, right_df, key = load_join_scenario(scenario)
        meta = SAMPLE_JOIN_SCENARIOS[scenario]
        return (
            left_df.to_json(date_format="iso", orient="split"),
            right_df.to_json(date_format="iso", orient="split"),
            key,
            f"{meta['left_name']} ({len(left_df):,} rows)",
            f"{meta['right_name']} ({len(right_df):,} rows)",
        )

    def _parse(contents, fname):
        if not contents:
            return None, ""
        _, cs = contents.split(",")
        df = pd.read_csv(io.StringIO(base64.b64decode(cs).decode("utf-8")))
        return df.to_json(date_format="iso", orient="split"), f"{fname} ({len(df):,} rows)"

    lj, ll = _parse(left_contents, left_fname)
    rj, rl = _parse(right_contents, right_fname)
    return lj, rj, "", ll, rl


# ── Audit callback ─────────────────────────────────────────────────────────────

@callback(
    Output("runs-store", "data", allow_duplicate=True),
    Input("ja-run-btn", "n_clicks"),
    State("ja-left-store", "data"),
    State("ja-right-store", "data"),
    State("ja-keys", "value"),
    State("ja-join-type", "value"),
    State("runs-store", "data"),
    prevent_initial_call=True,
)
def run_join_audit(_, left_json, right_json, keys_str, join_type, store):
    if not _:  # a click, not a page load: Dash Pages can fire this when the page mounts
        raise PreventUpdate
    from skills.join_auditor import audit

    if not left_json or not right_json:
        raise PreventUpdate
    if not keys_str:
        raise PreventUpdate

    label = f"{keys_str} ({join_type})"
    try:
        left_df  = pd.read_json(io.StringIO(left_json),  orient="split")
        right_df = pd.read_json(io.StringIO(right_json), orient="split")
        join_keys = [k.strip() for k in keys_str.split(",") if k.strip()]

        result = audit(left_df, right_df, join_keys, join_type)
        rc = result.get("row_count_change", {})
        run = {
            "id": f"ja_{int(time.time() * 1000)}",
            "label": label,
            "timestamp": datetime.now().strftime("%H:%M"),
            "status": "complete",
            "result": result,
            "summary": {
                "verdict":         result.get("overall_verdict", "review_needed"),
                "fan_out_factor":  rc.get("fan_out_factor", 1.0),
            },
        }
    except PreventUpdate:
        raise
    except Exception as exc:
        run = {"id": f"ja_{int(time.time()*1000)}", "label": label,
               "timestamp": datetime.now().strftime("%H:%M"),
               "status": "error", "error": str(exc), "summary": {}}

    updated = dict(store)
    updated["join_auditor"] = store.get("join_auditor", []) + [run]
    return updated


@callback(
    Output("ja-runs-container", "children"),
    Input("runs-store", "data"),
)
def render_ja_runs(store):
    from components.run_card import make_run_card
    runs = store.get("join_auditor", [])
    if not runs:
        return html.P("No runs yet — load a scenario or upload two CSVs and click Audit.",
                      className="text-muted text-center py-4")
    cards = []
    for run in reversed(runs):
        content = _render_results(run["result"]) if run["status"] == "complete" \
            else dbc.Alert(run.get("error", "Unknown error"), color="danger")
        cards.append(make_run_card(run, "join_auditor", content))
    return cards


def _findings_table(findings: list) -> list:
    """Return grouped, severity-sorted table rows with section headers."""
    sorted_f = sorted(findings, key=lambda f: _SEV_RANK.get(f.get("severity", "info"), 2))
    rows, last_sev = [], None
    for f in sorted_f:
        sev = f.get("severity", "info")
        if sev != last_sev:
            icon_cls, label = _SEV_HEADER.get(sev, ("bi-circle", sev.title()))
            rows.append(html.Tr(
                html.Td(
                    [html.I(className=f"bi {icon_cls} me-2"), html.Strong(label)],
                    colSpan=3,
                    className="small py-1",
                    style={"background": "rgba(255,255,255,0.04)", "borderTop": "1px solid rgba(255,255,255,0.1)"},
                )
            ))
            last_sev = sev
        rows.append(html.Tr([
            html.Td(dbc.Badge(sev.upper(), color=SEVERITY_COLOR.get(sev, "secondary"))),
            html.Td(f["issue_type"].replace("_", " ").title(), className="text-muted small"),
            html.Td([
                html.Div(f["description"], className="small"),
                html.Div(f["recommendation"], className="small text-muted mt-1"),
            ]),
        ], style=_ROW_STYLE.get(sev, {})))
    return rows


def _render_results(result: dict) -> html.Div:
    verdict = result.get("overall_verdict", "review_needed")
    rc = result.get("row_count_change", {})
    findings = result.get("findings", [])
    n_critical = sum(1 for f in findings if f.get("severity") == "critical")
    n_warning  = sum(1 for f in findings if f.get("severity") == "warning")

    rows = _findings_table(findings)
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Div("Verdict", className="text-muted small"),
                dbc.Badge(verdict.replace("_", " ").title(),
                          color=VERDICT_COLOR.get(verdict, "secondary"),
                          className="p-2", style={"fontSize": "1rem"}),
            ], width="auto"),
            dbc.Col([
                html.Div("Row change", className="text-muted small"),
                html.Span([
                    html.Strong(f"{rc.get('left_rows','?'):,}"), " left  ×  ",
                    html.Strong(f"{rc.get('right_rows','?'):,}"), " right  →  ",
                    html.Strong(f"{rc.get('result_rows','?'):,}"), " result  ",
                    dbc.Badge(f"×{rc.get('fan_out_factor',1):.2f}",
                              color="danger" if rc.get("fan_out_factor",1) > 2
                              else "warning" if rc.get("fan_out_factor",1) > 1.1 else "success"),
                ], className="small"),
            ]),
            dbc.Col([
                html.Div("Issues", className="text-muted small"),
                html.Span([
                    dbc.Badge(f"{n_critical} blocking", color="danger", className="me-1"),
                    dbc.Badge(f"{n_warning} warnings",  color="warning", text_color="dark"),
                ]),
            ], width="auto") if findings else html.Div(),
        ], align="center", className="mb-3"),
        html.P(result.get("summary", ""), className="mb-3"),
        dbc.Table(
            [html.Thead(html.Tr([html.Th("Severity"), html.Th("Issue"), html.Th("Detail")]))] +
            [html.Tbody(rows)],
            hover=True, responsive=True, size="sm",
        ) if rows else html.P("No issues detected.", className="text-success"),
    ])

import base64
import io
import re
import time
from datetime import datetime

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, State, callback, dcc, html
from dash.exceptions import PreventUpdate

dash.register_page(__name__, path="/nl-to-spark", name="NL → PySpark", order=4)


def layout():
    return html.Div([
        html.H4([html.I(className="bi bi-translate me-2"), "NL → PySpark / SQL"],
                className="mb-1"),
        html.P("Ask a question in plain English — get Spark SQL and PySpark code back.",
               className="text-muted mb-4"),

        dbc.Card(dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Tables"),
                    dbc.Tabs([
                        dbc.Tab([
                            dbc.Select(id="nl-sample-select", className="mt-2", value="sales_clean",
                                options=[
                                    {"label": "Sales — Clean   (product, category, pricing)", "value": "sales_clean"},
                                    {"label": "HR Attrition    (salary, tenure, department)", "value": "hr_attrition"},
                                    {"label": "E-commerce      (sessions, channel, device)",  "value": "ecommerce"},
                                    {"label": "Orders — Dirty  (customer_id, amount, status)","value": "orders_dirty"},
                                ]),
                        ], label="Sample", tab_id="tab-sample"),
                        dbc.Tab([
                            dcc.Upload(id="nl-upload",
                                children=html.Div(["Drop CSVs or ", html.A("browse")]),
                                multiple=True,
                                className="upload-zone mt-2",
                                style={"padding": "10px", "textAlign": "center", "fontSize": "0.85rem"}),
                        ], label="Upload CSVs", tab_id="tab-upload"),
                    ], id="nl-source-tabs", active_tab="tab-sample"),
                    html.Div(id="nl-table-badges", className="mt-2"),
                ], width=4),
                dbc.Col([
                    html.Label("Question"),
                    dbc.Textarea(
                        id="nl-question",
                        placeholder="e.g. Top 5 products by total revenue in each region",
                        rows=4,
                    ),
                ], width=6),
                dbc.Col([
                    dbc.Button([html.I(className="bi bi-play-fill me-1"), "Run"],
                               id="nl-run-btn", color="info", className="w-100 mt-3"),
                ], width=2, className="d-flex align-items-end"),
            ], className="g-3"),
        ]), className="mb-4"),

        html.Div(id="nl-session-ctx"),
        html.Div(id="nl-dq-gate"),
        dcc.Loading(html.Div(id="nl-runs-container"), type="circle", color="#0dcaf0"),
        dcc.Store(id="nl-tables-store"),
    ])


# ── Session context ────────────────────────────────────────────────────────────

@callback(
    Output("nl-session-ctx", "children"),
    Input("runs-store", "data"),
    prevent_initial_call=True,
)
def nl_session_ctx(store):
    from components.session_context import session_banner
    return session_banner(store)


# ── DQ gate ───────────────────────────────────────────────────────────────────

@callback(
    Output("nl-dq-gate", "children"),
    Input("nl-sample-select", "value"),
    Input("runs-store",       "data"),
    State("nl-source-tabs",   "active_tab"),
    prevent_initial_call=True,
)
def update_nl_dq_gate(sample_name, store, active_tab):
    from dash import html
    from components.dq_gate import dq_gate_banner
    if active_tab != "tab-sample" or not sample_name:
        return html.Div()
    return dq_gate_banner(sample_name, store, check_btn_id="nl-dq-check-btn")


@callback(
    Output("runs-store", "data", allow_duplicate=True),
    Input("nl-dq-check-btn", "n_clicks"),
    State("nl-sample-select", "value"),
    State("runs-store",       "data"),
    prevent_initial_call=True,
)
def nl_run_quick_dq(_, sample_name, store):
    import time as _t
    from datetime import datetime as _dt
    from databricks_client import load_sample_dataset
    from skills.data_quality import assess
    from dash.exceptions import PreventUpdate

    if not sample_name:
        raise PreventUpdate

    df = load_sample_dataset(sample_name)
    result = assess(df, table_name=sample_name)
    findings = result.get("findings", [])
    run = {
        "id":        f"dq_{int(_t.time() * 1000)}",
        "label":     sample_name,
        "timestamp": _dt.now().strftime("%H:%M"),
        "status":    "complete",
        "result":    result,
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


# ── Table store callback ───────────────────────────────────────────────────────

@callback(
    Output("nl-tables-store", "data"),
    Output("nl-table-badges", "children"),
    Input("nl-upload", "contents"),
    Input("nl-sample-select", "value"),
    State("nl-upload", "filename"),
    State("nl-source-tabs", "active_tab"),
    prevent_initial_call=True,
)
def populate_tables(contents_list, sample_name, filenames, active_tab):
    from databricks_client import load_sample_dataset

    if active_tab == "tab-sample" and sample_name:
        df = load_sample_dataset(sample_name)
        badge = dbc.Badge([html.I(className="bi bi-table me-1"),
                           f"{sample_name} ({len(df):,} rows)"],
                          color="info", className="me-1")
        return {sample_name: df.to_json(date_format="iso", orient="split")}, badge

    if not contents_list:
        return {}, html.Div()

    tables, badges = {}, []
    for contents, filename in zip(contents_list, filenames):
        _, cs = contents.split(",")
        df = pd.read_csv(io.StringIO(base64.b64decode(cs).decode("utf-8")))
        name = filename.replace(".csv", "")
        tables[name] = df.to_json(date_format="iso", orient="split")
        badges.append(dbc.Badge([html.I(className="bi bi-table me-1"),
                                  f"{name} ({len(df):,} rows)"],
                                 color="secondary", className="me-1"))
    return tables, html.Div(badges)


# ── Join diagnostics helper ────────────────────────────────────────────────────

def _extract_join_info(sql: str) -> dict | None:
    """
    Pattern-match a simple two-table JOIN from the generated SQL/Spark query.
    Returns a dict with table names and join keys, or None if no JOIN detected.

    Handles patterns like:
        FROM left_table [AS l] [INNER|LEFT|RIGHT|FULL] JOIN right_table [AS r] ON l.key = r.key
    """
    if not sql or not re.search(r"\bJOIN\b", sql, re.IGNORECASE):
        return None

    # Extract FROM table
    from_match = re.search(
        r"\bFROM\s+([`\w]+)(?:\s+(?:AS\s+)?(\w+))?",
        sql, re.IGNORECASE
    )
    # Extract JOIN table
    join_match = re.search(
        r"\b(?:INNER\s+|LEFT\s+(?:OUTER\s+)?|RIGHT\s+(?:OUTER\s+)?|FULL\s+(?:OUTER\s+)?|CROSS\s+)?JOIN\s+([`\w]+)(?:\s+(?:AS\s+)?(\w+))?",
        sql, re.IGNORECASE
    )
    # Extract ON condition  e.g.  ON l.customer_id = r.id
    on_match = re.search(
        r"\bON\s+([\w.`]+)\s*=\s*([\w.`]+)",
        sql, re.IGNORECASE
    )

    if not from_match or not join_match:
        return None

    left_table  = from_match.group(1).strip("`")
    right_table = join_match.group(1).strip("`")

    left_key = right_key = None
    if on_match:
        left_key  = on_match.group(1).split(".")[-1].strip("`")
        right_key = on_match.group(2).split(".")[-1].strip("`")

    return {
        "left_table":  left_table,
        "right_table": right_table,
        "left_key":    left_key,
        "right_key":   right_key,
    }


def _compute_join_diagnostics(tables: dict[str, pd.DataFrame], join_info: dict) -> dict:
    """
    Given the available DataFrames and extracted join metadata, compute:
      - left / right row counts
      - null counts on join keys
      - result row count (inner join simulation)
      - unmatched key counts (anti-join simulation)
      - fan-out flag
    """
    left_name  = join_info["left_table"]
    right_name = join_info["right_table"]
    left_key   = join_info["left_key"]
    right_key  = join_info["right_key"]

    # Fuzzy-match table names (case-insensitive, ignore leading/trailing spaces)
    available = {k.lower(): k for k in tables}

    left_df  = tables.get(left_name)  or tables.get(available.get(left_name.lower(), ""), None)
    right_df = tables.get(right_name) or tables.get(available.get(right_name.lower(), ""), None)

    diag: dict = {
        "left_table":       left_name,
        "right_table":      right_name,
        "left_key":         left_key,
        "right_key":        right_key,
        "left_row_count":   None,
        "right_row_count":  None,
        "result_row_count": None,
        "left_null_keys":   None,
        "right_null_keys":  None,
        "left_unmatched":   None,
        "right_unmatched":  None,
        "fan_out":          False,
        "error":            None,
    }

    if left_df is None and right_df is None:
        diag["error"] = (
            f"Neither '{left_name}' nor '{right_name}' were found in the loaded tables. "
            "Diagnostics require the source DataFrames to be available."
        )
        return diag

    if left_df is not None:
        diag["left_row_count"] = len(left_df)
    if right_df is not None:
        diag["right_row_count"] = len(right_df)

    # Key-level diagnostics only when both tables and keys are available
    if left_df is not None and right_df is not None and left_key and right_key:
        lk_col = _find_column(left_df,  left_key)
        rk_col = _find_column(right_df, right_key)

        if lk_col:
            diag["left_null_keys"] = int(left_df[lk_col].isna().sum())
        if rk_col:
            diag["right_null_keys"] = int(right_df[rk_col].isna().sum())

        if lk_col and rk_col:
            try:
                merged = left_df.merge(right_df, left_on=lk_col, right_on=rk_col, how="inner")
                diag["result_row_count"] = len(merged)
                diag["fan_out"] = len(merged) > len(left_df)

                # Anti-join: left keys not in right
                right_keys_set = set(right_df[rk_col].dropna().unique())
                left_keys_set  = set(left_df[lk_col].dropna().unique())
                diag["left_unmatched"]  = int(left_df[lk_col].isin(
                    left_keys_set - right_keys_set).sum())
                diag["right_unmatched"] = int(right_df[rk_col].isin(
                    right_keys_set - left_keys_set).sum())
            except Exception as exc:
                diag["error"] = f"Merge simulation failed: {exc}"
    elif not left_key or not right_key:
        diag["error"] = "Could not extract join keys from the ON clause — key-level diagnostics skipped."

    return diag


def _find_column(df: pd.DataFrame, name: str) -> str | None:
    """Case-insensitive column lookup."""
    for col in df.columns:
        if col.lower() == name.lower():
            return col
    return None


# ── Run callback ───────────────────────────────────────────────────────────────

@callback(
    Output("runs-store", "data", allow_duplicate=True),
    Input("nl-run-btn", "n_clicks"),
    State("nl-question", "value"),
    State("nl-tables-store", "data"),
    State("runs-store", "data"),
    prevent_initial_call=True,
)
def run_nl(_, question, tables_store, store):
    from skills.nl_to_spark import generate

    if not question or not tables_store:
        raise PreventUpdate

    label = f"{question[:40]}…" if len(question) > 40 else question
    try:
        tables = {name: pd.read_json(io.StringIO(js), orient="split")
                  for name, js in tables_store.items()}
        result = generate(question, tables)
        sql = result.get("sql_query", "")

        # ── Join diagnostics post-execution hook ──────────────────────────────
        join_diagnostics = None
        join_info = _extract_join_info(sql)
        if join_info:
            join_diagnostics = _compute_join_diagnostics(tables, join_info)
        result["join_diagnostics"] = join_diagnostics
        # ─────────────────────────────────────────────────────────────────────

        run = {
            "id": f"nl_{int(time.time() * 1000)}",
            "label": label,
            "timestamp": datetime.now().strftime("%H:%M"),
            "status": "complete",
            "result": result,
            "summary": {"sql_preview": sql.split("\n")[0] if sql else ""},
        }
    except PreventUpdate:
        raise
    except Exception as exc:
        run = {"id": f"nl_{int(time.time()*1000)}", "label": label,
               "timestamp": datetime.now().strftime("%H:%M"),
               "status": "error", "error": str(exc), "summary": {}}

    updated = dict(store)
    updated["nl_to_spark"] = store.get("nl_to_spark", []) + [run]
    return updated


@callback(
    Output("nl-runs-container", "children"),
    Input("runs-store", "data"),
)
def render_nl_runs(store):
    from components.run_card import make_run_card
    runs = store.get("nl_to_spark", [])
    if not runs:
        return html.P("No runs yet — load a table and enter a question.",
                      className="text-muted text-center py-4")
    cards = []
    for run in reversed(runs):
        content = _render_results(run["result"]) if run["status"] == "complete" \
            else dbc.Alert(run.get("error", "Unknown error"), color="danger")
        cards.append(make_run_card(run, "nl_to_spark", content))
    return cards


# ── Join diagnostics panel renderer ───────────────────────────────────────────

def _render_join_diagnostics(diag: dict) -> html.Div:
    """Render the inline join diagnostic summary card."""

    left_name  = diag.get("left_table",  "left")
    right_name = diag.get("right_table", "right")
    left_key   = diag.get("left_key")
    right_key  = diag.get("right_key")
    error      = diag.get("error")

    # ── Header row ────────────────────────────────────────────────────────────
    header_items = [
        html.I(className="bi bi-diagram-3 me-2 text-info"),
        html.Strong("Join Diagnostics"),
        dbc.Badge("AUTO", color="info", className="ms-2", pill=True,
                  style={"fontSize": "0.65rem", "verticalAlign": "middle"}),
    ]
    if left_key and right_key:
        header_items.append(
            html.Span(
                f"  {left_name}.{left_key} = {right_name}.{right_key}",
                className="text-muted ms-2",
                style={"fontSize": "0.8rem"},
            )
        )

    # ── Error notice (soft — still show partial info) ─────────────────────────
    error_alert = (
        dbc.Alert([html.I(className="bi bi-exclamation-triangle me-1"), error],
                  color="warning", className="py-2 mb-3 small")
        if error else html.Div()
    )

    # ── Row-count metric cards ────────────────────────────────────────────────
    def _metric(label, value, badge_color="secondary", icon="bi-hash"):
        display = f"{value:,}" if isinstance(value, int) else "—"
        return dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    html.Div(html.I(className=f"bi {icon} fs-4"), className="text-center mb-1"),
                    html.Div(display,
                             className="text-center fw-bold",
                             style={"fontSize": "1.25rem"}),
                    html.Div(label,
                             className="text-center text-muted",
                             style={"fontSize": "0.75rem"}),
                ], className="py-2 px-1"),
            ], color=badge_color, outline=True),
            width=3,
        )

    fan_out = diag.get("fan_out", False)
    result_color = "danger" if fan_out else "success"
    result_icon  = "bi-exclamation-circle" if fan_out else "bi-check-circle"

    metric_row = dbc.Row([
        _metric(f"{left_name} rows",  diag.get("left_row_count"),   "info",    "bi-table"),
        _metric(f"{right_name} rows", diag.get("right_row_count"),  "info",    "bi-table"),
        _metric("Result rows",        diag.get("result_row_count"),  result_color, result_icon),
        _metric("Match rate",
                _match_rate(diag.get("left_row_count"), diag.get("result_row_count")),
                result_color, "bi-percent"),
    ], className="g-2 mb-3")

    # ── Fan-out warning ───────────────────────────────────────────────────────
    fan_out_alert = html.Div()
    if fan_out:
        lrc = diag.get("left_row_count")
        rrc = diag.get("result_row_count")
        ratio = f"{rrc / lrc:.2f}×" if lrc and rrc else ""
        fan_out_alert = dbc.Alert([
            html.I(className="bi bi-exclamation-triangle-fill me-2"),
            html.Strong("Fan-out detected: "),
            f"Result ({rrc:,} rows) exceeds left table ({lrc:,} rows) — "
            f"multiplier {ratio}. "
            "This usually means duplicate keys on the right side of the join. "
            "Consider deduplicating or switching to a left-semi join.",
        ], color="danger", className="py-2 mb-3 small")

    # ── Key quality table ─────────────────────────────────────────────────────
    key_rows = []
    if diag.get("left_null_keys") is not None:
        key_rows.append(
            html.Tr([
                html.Td(html.Code(f"{left_name}.{left_key}")),
                html.Td(_fmt(diag["left_null_keys"]), className="text-center"),
                html.Td(_fmt(diag.get("left_unmatched")), className="text-center"),
                html.Td(_null_badge(diag["left_null_keys"]),  className="text-center"),
            ])
        )
    if diag.get("right_null_keys") is not None:
        key_rows.append(
            html.Tr([
                html.Td(html.Code(f"{right_name}.{right_key}")),
                html.Td(_fmt(diag["right_null_keys"]), className="text-center"),
                html.Td(_fmt(diag.get("right_unmatched")), className="text-center"),
                html.Td(_null_badge(diag["right_null_keys"]), className="text-center"),
            ])
        )

    key_section = html.Div()
    if key_rows:
        key_section = html.Div([
            html.H6("Key Diagnostics", className="mb-2 mt-1 text-muted",
                    style={"fontSize": "0.8rem", "textTransform": "uppercase", "letterSpacing": "0.05em"}),
            dbc.Table([
                html.Thead(html.Tr([
                    html.Th("Join Key"),
                    html.Th("Null Keys",      className="text-center"),
                    html.Th("Unmatched Keys", className="text-center"),
                    html.Th("Status",         className="text-center"),
                ])),
                html.Tbody(key_rows),
            ], size="sm", bordered=True, hover=True, className="mb-0"),
        ])

    return dbc.Card([
        dbc.CardHeader(html.Div(header_items, className="d-flex align-items-center")),
        dbc.CardBody([
            error_alert,
            metric_row,
            fan_out_alert,
            key_section,
        ]),
    ], className="mb-3 border-info",
       style={"borderLeft": "4px solid var(--bs-info) !important"})


def _match_rate(left_count, result_count) -> str:
    """Return a formatted match-rate string, or '—' if data unavailable."""
    if left_count and result_count and left_count > 0:
        rate = min(result_count / left_count * 100, 100)
        return f"{rate:.1f} %"
    return "—"


def _fmt(value) -> str:
    if isinstance(value, int):
        return f"{value:,}"
    return "—"


def _null_badge(null_count: int) -> dbc.Badge:
    if null_count == 0:
        return dbc.Badge("✓ Clean", color="success", pill=True)
    elif null_count < 10:
        return dbc.Badge(f"⚠ {null_count:,} nulls", color="warning", pill=True)
    else:
        return dbc.Badge(f"✗ {null_count:,} nulls", color="danger", pill=True)


# ── Results renderer ───────────────────────────────────────────────────────────

def _render_results(result: dict) -> html.Div:
    assumptions = result.get("assumptions", [])
    perf_notes  = result.get("performance_notes", [])
    join_diag   = result.get("join_diagnostics")

    children = [
        dbc.Card([
            dbc.CardHeader(html.Strong("Explanation")),
            dbc.CardBody(html.P(result.get("explanation", ""), className="mb-0")),
        ], className="mb-3"),
        dbc.Card([
            dbc.CardHeader(html.Strong("Spark SQL")),
            dbc.CardBody(html.Pre(result.get("sql_query", ""), className="mb-0")),
        ], className="mb-3"),
        dbc.Card([
            dbc.CardHeader(html.Strong("PySpark (DataFrame API)")),
            dbc.CardBody(html.Pre(result.get("pyspark_code", ""), className="mb-0")),
        ], className="mb-3"),
    ]

    # ── Insert join diagnostics panel immediately after the SQL cards ─────────
    if join_diag is not None:
        children.append(_render_join_diagnostics(join_diag))

    children.append(
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader(html.Strong("Assumptions")),
                dbc.CardBody(html.Ul([html.Li(a, className="small") for a in assumptions])),
            ]), width=6) if assumptions else html.Div(),
            dbc.Col(dbc.Card([
                dbc.CardHeader(html.Strong("Performance Notes")),
                dbc.CardBody(html.Ul([html.Li(n, className="small") for n in perf_notes])),
            ]), width=6) if perf_notes else html.Div(),
        ])
    )

    return html.Div(children)
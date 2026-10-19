"""Workbench renderers for Feature Engineering results (result dict → Dash components)."""
import dash_bootstrap_components as dbc
from dash import html


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


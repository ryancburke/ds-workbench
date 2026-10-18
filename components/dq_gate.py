"""
Data Quality gate banner — shared by EDA and NL→PySpark pages.

Shows a compact data health summary for the selected dataset using any
existing DQ result from the runs-store. If none exists, prompts the user
to run a quick inline check before proceeding.
"""

import dash_bootstrap_components as dbc
from dash import html


def find_dq_run(dataset: str, store: dict) -> dict | None:
    """Return the most recent completed DQ run for the given dataset, or None."""
    for run in reversed(store.get("data_quality", [])):
        if run.get("status") != "complete":
            continue
        label = run.get("label", "")
        if label == dataset or label == f"[Agent] {dataset}" or dataset in label:
            return run
    return None


def dq_gate_banner(dataset: str, store: dict, check_btn_id: str) -> html.Div:
    """
    Render the data health banner for the given dataset.

    check_btn_id: the id of the "Check Now" button on the hosting page,
                  used so the button can trigger that page's DQ callback.
    """
    run = find_dq_run(dataset, store)

    if run is None:
        return dbc.Alert(
            [
                html.I(className="bi bi-question-circle me-2"),
                html.Strong("No data quality check found"),
                html.Span(f" for '{dataset}'.", className="me-2"),
                html.Span("Running one first is recommended — dirty data produces "
                          "misleading analysis.", className="text-muted me-3"),
                dbc.Button(
                    [html.I(className="bi bi-shield-check me-1"), "Check Now"],
                    id=check_btn_id, size="sm", color="secondary", outline=True,
                ),
            ],
            color="secondary",
            className="py-2 mb-3 d-flex align-items-center flex-wrap gap-1",
            style={"border": "1px solid #3a3a5a"},
        )

    summary  = run.get("summary", {})
    score    = summary.get("score",    0)
    critical = summary.get("critical", 0)
    warnings = summary.get("warning",  0)
    findings = run.get("result", {}).get("findings", [])

    if critical > 0:
        color, icon, headline = "danger", "bi-exclamation-triangle-fill", "Critical issues found"
    elif warnings > 0:
        color, icon, headline = "warning", "bi-exclamation-circle", "Warnings found"
    else:
        color, icon, headline = "success", "bi-shield-check", "Data quality OK"

    # Top critical/warning findings for the collapsed detail
    top_issues = [
        f for f in findings
        if f.get("severity") in ("critical", "warning")
    ][:4]

    issue_rows = [
        html.Div([
            dbc.Badge(
                f["severity"].upper(),
                color="danger" if f["severity"] == "critical" else "warning",
                className="me-2",
                style={"fontSize": "0.6rem"},
            ),
            html.Span(f.get("column", ""), className="fw-semibold me-1 small"),
            html.Span(f"({f.get('issue_type','').replace('_',' ')})", className="text-muted small me-2"),
            html.Span(f.get("description", "")[:80], className="small text-muted"),
        ], className="mb-1")
        for f in top_issues
    ]

    return dbc.Alert(
        [
            # ── Summary row ──────────────────────────────────────────────────
            html.Div([
                html.I(className=f"bi {icon} me-2"),
                html.Strong(f"Data quality: {headline}"),
                html.Span(" — ", className="mx-1 text-muted"),
                dbc.Badge(f"{score}/100", color=color, className="me-2",
                          style={"fontSize": "0.7rem"}),
                dbc.Badge(f"{critical} critical", color="danger" if critical else "secondary",
                          className="me-1", style={"fontSize": "0.7rem"}),
                dbc.Badge(f"{warnings} warnings", color="warning" if warnings else "secondary",
                          className="me-1", style={"fontSize": "0.7rem"}),
                html.Span(f" · {run.get('timestamp','')}", className="small text-muted ms-1"),
                dbc.Button(
                    [html.I(className="bi bi-arrow-repeat me-1"), "Refresh"],
                    id=check_btn_id, size="sm", color="link",
                    className="ms-2 p-0 text-muted",
                    style={"fontSize": "0.75rem"},
                ),
            ], className="d-flex align-items-center flex-wrap gap-1 mb-0"),

            # ── Collapsed issue detail (only shown if there are issues) ─────
            html.Div(issue_rows, className="mt-2 border-top pt-2",
                     style={"borderColor": "rgba(255,255,255,0.1)"}
                     ) if issue_rows else html.Div(),

            # ── Guidance note for critical issues ────────────────────────────
            html.Div(
                "These issues may skew analysis results. Address them in Data Quality "
                "before drawing conclusions.",
                className="small mt-2 text-muted",
            ) if critical > 0 else html.Div(),
        ],
        color=color,
        className="py-2 mb-3",
        style={"border": f"1px solid {'#dc3545' if critical else '#ffc107' if warnings else '#198754'}44"},
    )

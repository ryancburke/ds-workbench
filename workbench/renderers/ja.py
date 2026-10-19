"""Workbench renderers for Join Auditor results (result dict → Dash components)."""
import dash_bootstrap_components as dbc
from dash import html

from workbench.constants import JA_SEV_COLOR, JA_VERDICT_CLR


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


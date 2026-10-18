import dash_bootstrap_components as dbc
from dash import html


def _score_badge(score: int) -> dbc.Badge:
    color = "danger" if score < 50 else "warning" if score < 75 else "success"
    return dbc.Badge(f"{score}/100", color=color, className="me-1 px-2 py-1")


def _summary_badges(run: dict, skill: str) -> list:
    s = run.get("summary", {})

    if skill == "data_quality":
        score = s.get("score")
        return [
            _score_badge(score) if score is not None else html.Span(),
            dbc.Badge(f"{s.get('critical', 0)} crit",  color="danger",  className="me-1"),
            dbc.Badge(f"{s.get('warning',  0)} warn",  color="warning", text_color="dark", className="me-1"),
            dbc.Badge(f"{s.get('info',     0)} info",  color="info",    text_color="dark"),
        ]

    if skill == "eda":
        return [
            dbc.Badge(f"{s.get('charts',   0)} charts",     color="success",   className="me-1"),
            dbc.Badge(f"{s.get('insights', 0)} insights",   color="secondary", className="me-1"),
            dbc.Badge(f"{s.get('targets',  0)} ML targets", color="info", text_color="dark"),
        ]

    if skill == "join_auditor":
        verdict = s.get("verdict", "review_needed")
        vc = {"safe": "success", "review_needed": "warning", "dangerous": "danger"}.get(verdict, "secondary")
        ff  = s.get("fan_out_factor", 1.0)
        return [
            dbc.Badge(verdict.replace("_", " ").title(), color=vc, className="me-1"),
            dbc.Badge(f"×{ff:.2f} fan-out", color="secondary"),
        ]

    if skill == "nl_to_spark":
        preview = s.get("sql_preview", "")
        return [
            html.Code(
                preview[:55] + ("…" if len(preview) > 55 else ""),
                style={"fontSize": "0.72rem", "color": "#7dd3fc"},
            )
        ]

    if skill == "mlops_review":
        score = s.get("score")
        return [
            _score_badge(score) if score is not None else html.Span(),
            dbc.Badge(f"{s.get('critical', 0)} crit", color="danger",  className="me-1"),
            dbc.Badge(f"{s.get('warning',  0)} warn", color="warning", text_color="dark"),
        ]

    if skill == "preflight":
        verdict  = s.get("verdict", "ready")
        n_block  = len(s.get("blockers", []))
        n_caut   = len(s.get("cautions", []))
        vc = {"blocked": "danger", "caution": "warning", "ready": "success"}.get(verdict, "secondary")
        label = {"blocked": "BLOCKED", "caution": "CAUTION", "ready": "READY TO TRAIN"}.get(verdict, verdict)
        badges = [dbc.Badge(label, color=vc, className="me-1")]
        if n_block:
            badges.append(dbc.Badge(f"{n_block} blocker{'s' if n_block != 1 else ''}",
                                    color="danger", className="me-1"))
        if n_caut:
            badges.append(dbc.Badge(f"{n_caut} caution{'s' if n_caut != 1 else ''}",
                                    color="warning", text_color="dark", className="me-1"))
        return badges

    if skill == "model_selector":
        task      = s.get("task_type", "").replace("_", " ").title()
        top_model = s.get("top_model", "")
        readiness = s.get("readiness", "")
        ratio     = s.get("imbalance_ratio")
        rationale = s.get("top_rationale", "")
        rc = {"ready": "success", "needs_prep": "warning", "insufficient": "danger"}.get(
            readiness, "secondary"
        )
        badges = []
        if task:
            badges.append(dbc.Badge(task, color="info", text_color="dark", className="me-1"))
        if top_model:
            badges.append(html.Strong(top_model, className="me-2 small",
                                      style={"color": "#a78bfa"}))
        if ratio and ratio > 2:
            badges.append(dbc.Badge(f"{ratio:.0f}:1 imbalance", color="warning",
                                    text_color="dark", className="me-1"))
        if readiness:
            badges.append(dbc.Badge(readiness.replace("_", " ").title(), color=rc,
                                    className="me-1"))
        if rationale:
            badges.append(html.Span(rationale, className="text-muted small d-block mt-1"))
        return badges

    return []


def make_run_card(run: dict, skill: str, result_content) -> dbc.Card:
    run_id = run["id"]
    status = run.get("status", "complete")

    if status == "complete":
        status_icon = html.I(className="bi bi-check-circle-fill text-success me-2 flex-shrink-0")
    else:
        status_icon = html.I(className="bi bi-x-circle-fill text-danger me-2 flex-shrink-0")

    badges = _summary_badges(run, skill) if status == "complete" else [
        dbc.Badge("error", color="danger"),
        html.Small(run.get("error", "")[:80], className="text-muted ms-2"),
    ]

    return dbc.Card([
        html.Div(
            dbc.Row([
                dbc.Col(
                    html.Div([status_icon, html.Strong(run.get("label", ""), className="me-3"), *badges],
                             className="d-flex align-items-center flex-wrap gap-1"),
                    className="col",
                ),
                dbc.Col(
                    html.Div([
                        html.Span(run.get("timestamp", ""), className="text-muted small me-2"),
                        dbc.Button(
                            html.I(className="bi bi-chevron-expand"),
                            id={"type": "run-toggle", "index": run_id},
                            color="link", size="sm", className="p-0 me-2 text-muted",
                        ) if status == "complete" else html.Span(),
                        dbc.Button(
                            html.I(className="bi bi-x"),
                            id={"type": "run-dismiss", "index": run_id},
                            color="link", size="sm", className="p-0 text-muted",
                        ),
                    ], className="d-flex align-items-center"),
                    className="col-auto",
                ),
            ], align="center", className="g-2"),
            className="px-3 py-2",
        ),
        dbc.Collapse(
            dbc.CardBody(result_content, className="pt-2 pb-3"),
            id={"type": "run-collapse", "index": run_id},
            is_open=False,
        ) if status == "complete" else html.Div(),
    ], className="mb-2")

"""
SkillResult — a dict subclass that renders as styled HTML in Jupyter / Databricks notebooks.

Every skill entry point wraps its return value in SkillResult(data, skill="<name>").
Existing code that treats the result as a plain dict is unaffected.
"""
from __future__ import annotations

_CSS = """
<style>
.wr-card{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  font-size:13px;border:1px solid #e0e0e0;border-radius:6px;overflow:hidden;
  margin:4px 0;background:#fff;color:#1a1a2e}
.wr-header{padding:8px 12px;display:flex;align-items:center;gap:8px;
  background:#f8f9fa;border-bottom:1px solid #e0e0e0;font-weight:600}
.wr-body{padding:12px}
.wr-badge{display:inline-block;padding:2px 8px;border-radius:12px;
  font-size:11px;font-weight:600;white-space:nowrap}
.wr-danger{background:#fde8e8;color:#c62828}
.wr-warning{background:#fff3cd;color:#856404}
.wr-success{background:#e8f5e9;color:#1b5e20}
.wr-info{background:#e3f2fd;color:#0d47a1}
.wr-secondary{background:#f0f0f0;color:#555}
.wr-purple{background:#ede7f6;color:#4527a0}
.wr-tbl{width:100%;border-collapse:collapse;font-size:12px;margin-top:8px}
.wr-tbl th{text-align:left;padding:4px 8px;background:#f8f9fa;
  border-bottom:2px solid #dee2e6;color:#555;font-weight:600;font-size:11px;
  text-transform:uppercase;letter-spacing:.04em}
.wr-tbl td{padding:4px 8px;border-bottom:1px solid #f0f0f0;vertical-align:top}
.wr-tbl tr:last-child td{border-bottom:none}
.wr-crit td{border-left:3px solid #ef5350}
.wr-warn td{border-left:3px solid #ffb300}
.wr-code{font-family:'SFMono-Regular',Consolas,monospace;font-size:11px;
  background:#f6f8fa;border:1px solid #e0e0e0;border-radius:4px;
  padding:8px 10px;overflow-x:auto;white-space:pre;max-height:260px;
  overflow-y:auto;margin-top:8px}
.wr-mono{font-family:'SFMono-Regular',Consolas,monospace;font-size:11px;
  background:#f6f8fa;padding:1px 5px;border-radius:3px}
.wr-muted{color:#888;font-size:11px}
.wr-sum{color:#444;line-height:1.5;margin:6px 0}
</style>
"""


def _badge(text: str, style: str = "secondary") -> str:
    return f'<span class="wr-badge wr-{style}">{text}</span>'


def _sev_style(sev: str) -> str:
    return {"critical": "danger", "warning": "warning", "info": "info"}.get(sev, "secondary")


def _sev_row_cls(sev: str) -> str:
    return {"critical": "wr-crit", "warning": "wr-warn"}.get(sev, "")


_SEV_ORDER = {"critical": 0, "warning": 1, "info": 2}

# Within a severity tier, findings on these issue types affect training most directly
_TRAINING_IMPACT_ORDER = {
    "null_rate": 0, "duplicate_rows": 1, "type_mismatch": 2,
    "referential_integrity": 3, "inconsistent_format": 4,
    "outlier": 5, "high_cardinality": 6, "low_cardinality": 7, "schema_issue": 8,
}


def _sort_findings(findings: list) -> list:
    return sorted(
        findings,
        key=lambda f: (
            _SEV_ORDER.get(f.get("severity", "info"), 2),
            _TRAINING_IMPACT_ORDER.get(f.get("issue_type", ""), 9),
        ),
    )


def _critical_callout(
    findings: list,
    action: str,
    col_key: str | None = "column",
    code_snippet: str | None = None,
) -> str:
    """Red stop-block listing only critical findings, rendered before the full table."""
    crits = [f for f in findings if f.get("severity") == "critical"]
    if not crits:
        return ""
    items = ""
    for f in crits:
        col  = f.get(col_key, "") if col_key else ""
        desc = f.get("description", "")
        desc = desc[:100] + "…" if len(desc) > 100 else desc
        col_part = f'<span class="wr-mono">{col}</span> — ' if col else ""
        items += f"<li>{col_part}{desc}</li>"
    snippet_html = (
        f'<div class="wr-code" style="margin-top:6px;max-height:80px">{code_snippet}</div>'
        if code_snippet else ""
    )
    return (
        '<div style="background:#fde8e8;border-left:4px solid #c62828;border-radius:4px;'
        'padding:8px 12px;margin:6px 0;font-size:12px">'
        f'<strong style="color:#c62828">⛔ Critical — {action}</strong>'
        f'<ul style="margin:4px 0 0;padding-left:16px;color:#1a1a2e">{items}</ul>'
        f'{snippet_html}'
        "</div>"
    )


def _decision_table(decision: dict) -> str:
    """The four choices for the next cell, shown before any reasoning."""
    rows = [("Model", "model"), ("Imbalance", "imbalance_strategy"),
            ("Metric", "primary_metric"), ("Validation", "validation")]
    cells = "".join(
        f'<tr><td style="font-weight:600;width:90px">{label}</td><td>{decision.get(key, "—")}</td></tr>'
        for label, key in rows if decision.get(key))
    return (f'<table class="wr-tbl" style="margin:4px 0 10px"><tbody>{cells}</tbody></table>'
            if cells else "")


def _imbalance_callout(target_analysis: dict) -> str:
    """Amber callout when the target has a class balance issue."""
    if not target_analysis.get("class_balance_issue"):
        return ""
    ratio = target_analysis.get("imbalance_ratio")
    ratio_str = f" — {ratio:.1f}:1 imbalance" if ratio else ""
    return (
        '<div style="background:#fff3cd;border-left:4px solid #f57c00;border-radius:4px;'
        'padding:8px 12px;margin:6px 0;font-size:12px">'
        f'<strong style="color:#856404">⚠ Class Imbalance{ratio_str}</strong><br>'
        '<span style="color:#666;font-size:11px">'
        "Factor into model choice and evaluation metric — see recommendations below."
        "</span></div>"
    )


_HERO_COLORS = {
    "danger":  {"bg": "#fde8e8", "border": "#c62828", "text": "#c62828", "icon": "⛔"},
    "warning": {"bg": "#fff3cd", "border": "#f57c00", "text": "#856404", "icon": "⚠"},
    "success": {"bg": "#e8f5e9", "border": "#2e7d32", "text": "#1b5e20", "icon": "✓"},
}


def _verdict_hero(
    title: str,
    verdict: str,
    style: str,
    items: list[str] | None = None,
    subtitle: str | None = None,
) -> str:
    """Full-width verdict banner — the first element every DS sees in a skill output."""
    c = _HERO_COLORS.get(style, _HERO_COLORS["warning"])
    subtitle_html = (
        f'<span style="font-size:11px;font-weight:400;opacity:0.85"> · {subtitle}</span>'
        if subtitle else ""
    )
    items_html = ""
    if items:
        bullets = "".join(f"<li>{i}</li>" for i in items[:3])
        items_html = (
            f'<ul style="margin:5px 0 0;padding-left:18px;color:#1a1a2e;font-size:12px">'
            f'{bullets}</ul>'
        )
    return (
        f'<div style="background:{c["bg"]};border-left:4px solid {c["border"]};'
        f'border-radius:4px;padding:10px 14px;margin:0 0 0 0">'
        f'<strong style="color:{c["text"]};font-size:13px">'
        f'{c["icon"]} {title}: {verdict}</strong>'
        f'{subtitle_html}'
        f'{items_html}'
        f'</div>'
    )


# ── Per-skill renderers ────────────────────────────────────────────────────────

def _render_data_quality(r: dict) -> str:
    score    = r.get("overall_score", 0)
    summary  = r.get("summary", "")
    findings = _sort_findings(r.get("findings", []))
    sc = "success" if score >= 75 else "warning" if score >= 50 else "danger"
    counts = {s: sum(1 for f in findings if f.get("severity") == s)
              for s in ("critical", "warning", "info")}

    critical = [f for f in findings if f.get("severity") == "critical"]
    warning  = [f for f in findings if f.get("severity") == "warning"]
    if critical:
        hero = _verdict_hero("Data Quality", "BLOCKING", "danger",
                             items=[f.get("description", "")[:80] for f in critical[:3]])
    elif warning:
        hero = _verdict_hero("Data Quality", "NEEDS REVIEW", "warning",
                             items=[f.get("description", "")[:80] for f in warning[:3]])
    else:
        hero = _verdict_hero("Data Quality", "CLEAR", "success",
                             subtitle=f"{score}/100")

    rows = ""
    for f in findings:
        sev = f.get("severity", "info")
        rows += (
            f'<tr class="{_sev_row_cls(sev)}">'
            f'<td>{_badge(sev.upper(), _sev_style(sev))}</td>'
            f'<td><span class="wr-mono">{f.get("column","—")}</span></td>'
            f'<td class="wr-muted">{f.get("issue_type","").replace("_"," ").title()}</td>'
            f'<td>{f.get("description","")}<br>'
            f'<span class="wr-muted">{f.get("recommendation","")}</span></td>'
            f'</tr>'
        )

    findings_tbl = (
        f'<table class="wr-tbl"><thead><tr>'
        f'<th>Sev</th><th>Column</th><th>Issue</th><th>Detail</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>'
        if rows else '<p style="color:#2e7d32;margin:6px 0">✓ No issues found.</p>'
    )

    n_crit = counts["critical"]
    n_warn = counts["warning"]
    n_info = counts["info"]
    return (
        hero
        + f'<div class="wr-header">'
        f'<span>🛡 Data Quality</span>'
        f'{_badge(f"{score}/100", sc)}'
        f'{_badge(f"{n_crit} critical", "danger")}'
        f'{_badge(f"{n_warn} warnings", "warning")}'
        f'{_badge(f"{n_info} info", "info")}'
        f'</div>'
        f'<div class="wr-body">'
        f'<p class="wr-sum">{summary}</p>'
        + _critical_callout(findings, "Fix before modeling")
        + f'{findings_tbl}'
        f'</div>'
    )


def findings_of(prior) -> list:
    """Findings from an earlier skill run, given either the result itself (e.g. the object
    leakage_detector.check() returned) or its findings list. None/empty → []."""
    if not prior:
        return []
    if isinstance(prior, dict):
        return list(prior.get("findings") or [])
    return list(prior)


def _leakage_drop_snippet(findings: list) -> str | None:
    cols = [f.get("column") for f in findings
            if f.get("severity") == "critical" and f.get("column")]
    if not cols:
        return None
    col_list = ", ".join(f'"{c}"' for c in cols)
    return (
        f"columns_to_drop = [{col_list}]   # also: result['drop_columns']\n"
        f"df = df.drop(columns=columns_to_drop)         # pandas\n"
        f"# df = df.drop(*columns_to_drop)              # PySpark"
    )


def _render_leakage_detector(r: dict) -> str:
    verdict  = r.get("verdict", "clean")
    risk     = r.get("risk_level", "none")
    findings = _sort_findings(r.get("findings", []))
    summary  = r.get("summary", "")
    xref     = r.get("cross_referenced_cols", [])
    vc = {"leakage_detected":"danger","suspicious":"warning","clean":"success"}.get(verdict,"secondary")
    rc = {"high":"danger","medium":"warning","low":"info","none":"success"}.get(risk,"secondary")

    critical = [f for f in findings if f.get("severity") == "critical"]
    warning_f = [f for f in findings if f.get("severity") == "warning"]
    pp = r.get("prediction_point")
    timing = f"prediction point: {pp}" if pp else "statistics only — no prediction point"

    def _detail(f):
        return f.get("evidence") or f.get("description", "")
    if verdict == "leakage_detected":
        hero = _verdict_hero("Leakage", "LEAKAGE DETECTED", "danger",
                             items=[f"{f.get('column', '')}: {_detail(f)}"[:80] for f in critical[:3]],
                             subtitle=f"risk: {risk} · {timing}")
    elif verdict == "suspicious":
        hero = _verdict_hero("Leakage", "SUSPICIOUS", "warning",
                             items=[f"{f.get('column', '')}: {_detail(f)}"[:80] for f in warning_f[:3]],
                             subtitle=f"risk: {risk} · {timing}")
    else:
        hero = _verdict_hero("Leakage", "CLEAN", "success", subtitle="no leakage detected")

    rows = ""
    _last_sev = None
    _sev_labels = {"critical": "⛔ Blockers — must remove before training",
                   "warning":  "⚠ Review — investigate before training",
                   "info":     "ℹ Notes — low risk, document for awareness"}
    for f in findings:
        sev  = f.get("severity", "info")
        if sev != _last_sev:
            label = _sev_labels.get(sev, sev.title())
            rows += (
                f'<tr><td colspan="5" style="background:#f8f9fa;font-size:11px;'
                f'font-weight:600;color:#555;padding:4px 8px;border-top:1px solid #dee2e6">'
                f'{label}</td></tr>'
            )
            _last_sev = sev
        corr = f.get("correlation_with_target")
        corr_str = f"{corr:.3f}" if corr is not None else "—"
        rec  = f.get("recommendation", "")
        rows += (
            f'<tr class="{_sev_row_cls(sev)}">'
            f'<td>{_badge(sev.upper(), _sev_style(sev))}</td>'
            f'<td><span class="wr-mono">{f.get("column","—")}</span></td>'
            f'<td class="wr-muted">{f.get("leakage_type","").replace("_"," ").title()}'
            f'<br><span style="font-size:10px">{f.get("evidence_type", "statistical")} evidence</span></td>'
            f'<td style="font-family:monospace;font-size:11px">{corr_str}</td>'
            f'<td>{_detail(f)}'
            + (f'<br><span class="wr-muted">{rec}</span>' if rec else "")
            + f'</td>'
            f'</tr>'
        )

    findings_tbl = (
        f'<table class="wr-tbl"><thead><tr>'
        f'<th>Sev</th><th>Column</th><th>Type</th><th>Correlation</th><th>Detail / Fix</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>'
        if rows else '<p style="color:#2e7d32;margin:6px 0">✓ No leakage detected.</p>'
    )

    return (
        hero
        + f'<div class="wr-header">'
        f'<span>⚠ Leakage Detector</span>'
        f'{_badge(verdict.replace("_"," ").title(), vc)}'
        f'{_badge(f"risk: {risk}", rc)}'
        f'</div>'
        f'<div class="wr-body">'
        + (f'<div style="background:#fff3cd;border-left:4px solid #e0a800;border-radius:4px;'
           f'padding:6px 12px;margin:6px 0;font-size:12px">{r["caveat"]}</div>' if r.get("caveat") else "")
        + _critical_callout(
            findings,
            "Exclude from training set before fitting any model",
            code_snippet=_leakage_drop_snippet(findings),
        )
        + (
            '<div style="background:#ede7f6;border-left:4px solid #6a1b9a;border-radius:4px;'
            'padding:8px 12px;margin:6px 0;font-size:12px">'
            '<strong style="color:#4527a0">⚡ Also flagged by Data Quality: </strong>'
            + " ".join(f'<span class="wr-mono">{c}</span>' for c in xref)
            + '<br><span style="color:#666;font-size:11px">'
            'These columns have both data quality issues and leakage risk — '
            'highest priority to resolve.</span></div>'
            if xref else ""
        )
        + f'<p class="wr-sum">{summary}</p>'
        + f'{findings_tbl}'
        f'</div>'
    )


def _render_feature_engineering(r: dict) -> str:
    features = r.get("features", [])
    dropped  = r.get("dropped_cols", [])
    code     = r.get("pyspark_code", "")
    warnings = r.get("warnings", [])

    action_style = {
        "scale":"info","encode_ohe":"info","encode_index":"info",
        "extract_datetime":"success","drop":"secondary",
        "cast_to_int":"warning","keep_as_is":"success",
        "log_then_scale":"info","encode_frequency":"info",
    }

    rows = ""
    for f in features:
        action = f.get("action","")
        impute = f.get("imputation","none")
        impute_str = impute.replace("_"," ") if impute not in ("none", None) else "—"
        rows += (
            f'<tr>'
            f'<td><span class="wr-mono">{f.get("column","")}</span></td>'
            f'<td class="wr-muted">{f.get("feature_type","").replace("_"," ").title()}</td>'
            f'<td>{_badge(action.replace("_"," "), action_style.get(action,"secondary"))}</td>'
            f'<td class="wr-muted">{impute_str}</td>'
            f'<td><span class="wr-mono">{f.get("output_col","")}</span></td>'
            f'</tr>'
        )

    feat_tbl = (
        f'<table class="wr-tbl"><thead><tr>'
        f'<th>Column</th><th>Type</th><th>Action</th><th>Impute</th><th>Output</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>'
        if rows else '<p class="wr-muted">No features planned.</p>'
    )

    dropped_str = (
        " ".join(_badge(c, "secondary") for c in dropped)
        if dropped else '<span class="wr-muted">none</span>'
    )

    warn_html = ""
    if warnings:
        warn_html = '<div style="background:#fff3cd;border-radius:4px;padding:6px 10px;margin-bottom:8px;font-size:12px">'
        warn_html += "".join(f"⚠ {w}<br>" for w in warnings)
        warn_html += "</div>"

    design_bits = []
    if r.get("model_family"):
        design_bits.append(f'<b>Designed for:</b> {r["model_family"].replace("_", "-")} models')
    if r.get("imbalance_handling"):
        design_bits.append(f'<b>Imbalance:</b> {r["imbalance_handling"]}')
    design_html = (f'<div class="wr-muted" style="margin-bottom:8px">{" · ".join(design_bits)}</div>'
                   if design_bits else "")

    code_html = ""
    if code:
        truncated = "\n".join(code.splitlines()[:40])
        ellipsis  = "\n... (truncated)" if len(code.splitlines()) > 40 else ""
        code_html = f'<div class="wr-code">{truncated}{ellipsis}</div>'

    return (
        f'<div class="wr-header">'
        f'<span>🔧 Feature Engineering</span>'
        f'{_badge(f"{len(features)} features", "info")}'
        f'{_badge(f"{len(dropped)} dropped", "secondary")}'
        f'</div>'
        f'<div class="wr-body">'
        f'{design_html}{warn_html}'
        f'{feat_tbl}'
        f'<div style="margin-top:8px"><span class="wr-muted">Dropped: </span>{dropped_str}</div>'
        f'{code_html}'
        f'</div>'
    )


def _render_model_selector(r: dict) -> str:
    task           = r.get("task_type","").replace("_"," ").title()
    readiness      = r.get("data_readiness","needs_prep")
    recs           = r.get("recommendations",[])
    notes          = r.get("readiness_notes",[])
    target_analysis = r.get("target_analysis", {})
    rc = {"ready":"success","needs_prep":"warning","insufficient":"danger"}.get(readiness,"secondary")

    top_model = recs[0].get("display_name", recs[0].get("model_family","")) if recs else ""
    subtitle  = f"top pick: {top_model}" if top_model else None
    if readiness == "insufficient":
        hero = _verdict_hero("Model Selector", "INSUFFICIENT DATA", "danger", subtitle=subtitle)
    elif readiness == "needs_prep":
        hero = _verdict_hero("Model Selector", "PREP REQUIRED", "warning", subtitle=subtitle)
    else:
        hero = _verdict_hero("Model Selector", "DATA SUFFICIENT", "success", subtitle=subtitle)

    rec_rows = ""
    for rec in recs[:3]:
        suit = rec.get("suitability","moderate")
        sc = {"strong":"success","moderate":"warning","weak":"secondary"}.get(suit,"secondary")
        rec_rows += (
            f'<tr>'
            f'<td style="font-weight:600">#{rec.get("rank","?")} {rec.get("display_name","")}</td>'
            f'<td>{_badge(suit.title(), sc)}</td>'
            f'<td style="color:#444;font-size:12px">{rec.get("rationale","")}</td>'
            f'</tr>'
        )

    recs_tbl = (
        f'<table class="wr-tbl"><thead><tr>'
        f'<th>Model</th><th>Fit</th><th>Rationale</th>'
        f'</tr></thead><tbody>{rec_rows}</tbody></table>'
        if rec_rows else ""
    )

    notes_html = ""
    if notes:
        notes_html = (
            '<div style="background:#fff3cd;border-radius:4px;padding:6px 10px;'
            'margin-top:8px;font-size:12px"><strong>Before training:</strong><br>'
            + "".join(f"• {n}<br>" for n in notes)
            + "</div>"
        )

    insufficient_html = ""
    if readiness == "insufficient":
        note_items = "".join(f"<li>{n}</li>" for n in notes)
        insufficient_html = (
            '<div style="background:#fde8e8;border-left:4px solid #c62828;border-radius:4px;'
            'padding:8px 12px;margin:6px 0;font-size:12px">'
            '<strong style="color:#c62828">⛔ Insufficient Data — Training not recommended</strong>'
            + (f'<ul style="margin:4px 0 0;padding-left:16px;color:#1a1a2e">{note_items}</ul>' if note_items else "")
            + "</div>"
        )
        notes_html = ""  # absorbed into the callout above

    return (
        hero
        + f'<div class="wr-header">'
        f'<span>🤖 Model Selector</span>'
        f'{_badge(task, "info")}'
        f'{_badge(readiness.replace("_"," ").title(), rc)}'
        f'</div>'
        f'<div class="wr-body">'
        + _decision_table(r.get("decision") or {})
        + insufficient_html
        + _imbalance_callout(target_analysis)
        + f'{recs_tbl}'
        f'{notes_html}'
        f'</div>'
    )


def _render_eda(r: dict) -> str:
    narrative = r.get("narrative","")
    insights  = r.get("key_insights",[])
    targets   = r.get("potential_targets",[])

    top_insight_html = ""
    if insights:
        top_insight_html = (
            '<div style="background:#e3f2fd;border-left:4px solid #1565c0;border-radius:4px;'
            'padding:8px 12px;margin:6px 0;font-size:12px">'
            f'<strong style="color:#0d47a1">📌 Key Finding</strong><br>{insights[0]}'
            "</div>"
        )

    rest_ins_html = "".join(f"<li>{i}</li>" for i in insights[1:])
    tgt_rows = ""
    for t in targets:
        tgt_rows += (
            f'<tr><td><span class="wr-mono">{t.get("column","")}</span></td>'
            f'<td>{_badge(t.get("task_type",""), "info")}</td>'
            f'<td class="wr-muted">{t.get("rationale","")}</td></tr>'
        )

    return (
        f'<div class="wr-header"><span>📊 EDA</span>'
        f'{_badge(f"{len(insights)} insights", "info")}'
        f'{_badge(f"{len(targets)} ML targets", "success")}'
        f'</div>'
        f'<div class="wr-body">'
        f'<p class="wr-sum">{narrative}</p>'
        + top_insight_html
        + (f'<ul style="font-size:12px;padding-left:18px;margin:6px 0">{rest_ins_html}</ul>' if rest_ins_html else "")
        + (
            f'<table class="wr-tbl" style="margin-top:8px"><thead><tr>'
            f'<th>Column</th><th>Task</th><th>Rationale</th>'
            f'</tr></thead><tbody>{tgt_rows}</tbody></table>'
            if tgt_rows else ""
        )
        + "</div>"
    )


def _render_join_auditor(r: dict) -> str:
    verdict  = r.get("overall_verdict","review_needed")
    rc_data  = r.get("row_count_change",{})
    findings = _sort_findings(r.get("findings", []))
    summary  = r.get("summary","")
    vc = {"safe":"success","review_needed":"warning","dangerous":"danger"}.get(verdict,"secondary")

    fo = rc_data.get("fan_out_factor",1)
    fo_badge = _badge(f"×{fo:.2f}", "danger" if fo > 2 else "warning" if fo > 1.1 else "success")

    critical = [f for f in findings if f.get("severity") == "critical"]
    warning_f = [f for f in findings if f.get("severity") == "warning"]
    if verdict == "dangerous":
        hero = _verdict_hero("Join Auditor", "DANGEROUS", "danger",
                             items=[f.get("description", "")[:80] for f in critical[:3]],
                             subtitle=f"×{fo:.2f} fan-out")
    elif verdict == "review_needed":
        hero = _verdict_hero("Join Auditor", "REVIEW NEEDED", "warning",
                             items=[f.get("description", "")[:80] for f in (critical + warning_f)[:3]],
                             subtitle=f"×{fo:.2f} fan-out")
    else:
        hero = _verdict_hero("Join Auditor", "SAFE", "success",
                             subtitle=f"×{fo:.2f} fan-out")

    rows = ""
    for f in findings[:8]:
        sev = f.get("severity","info")
        rec = f.get("recommendation", "")
        rows += (
            f'<tr class="{_sev_row_cls(sev)}">'
            f'<td>{_badge(sev.upper(), _sev_style(sev))}</td>'
            f'<td class="wr-muted">{f.get("issue_type","").replace("_"," ").title()}</td>'
            f'<td>{f.get("description","")}'
            + (f'<br><span class="wr-muted">{rec}</span>' if rec else "")
            + f'</td>'
            f'</tr>'
        )

    left_rows   = rc_data.get("left_rows", 0)
    right_rows  = rc_data.get("right_rows", 0)
    result_rows = rc_data.get("result_rows", 0)
    drop_pct    = (1 - result_rows / left_rows) * 100 if left_rows else 0
    drop_color  = "#c62828" if drop_pct > 20 else "#856404" if drop_pct > 5 else "#2e7d32"
    delta_html  = (
        '<div style="display:flex;align-items:center;gap:10px;margin:6px 0;'
        'font-size:12px;font-family:monospace;background:#f8f9fa;'
        'border:1px solid #e0e0e0;border-radius:4px;padding:6px 12px">'
        f'<span><strong>{left_rows:,}</strong> <span style="color:#888">left rows</span></span>'
        '<span style="color:#888">→ join →</span>'
        f'<span><strong>{result_rows:,}</strong> <span style="color:#888">result rows</span></span>'
        f'<span style="color:{drop_color};font-weight:600">({drop_pct:.1f}% dropped)</span>'
        f'<span style="color:#888">·</span>'
        f'<span><strong>{right_rows:,}</strong> <span style="color:#888">right rows</span></span>'
        f'<span>·</span>'
        f'<span><span style="color:#888">fan-out</span> <strong>×{fo:.2f}</strong></span>'
        '</div>'
    )

    return (
        hero
        + f'<div class="wr-header"><span>🔗 Join Auditor</span>'
        f'{_badge(verdict.replace("_"," ").title(), vc)}'
        f'{fo_badge}'
        f'</div>'
        f'<div class="wr-body">'
        + delta_html
        + f'<p class="wr-sum">{summary}</p>'
        + _critical_callout(findings, "Resolve before using join results", col_key=None)
        + (
            f'<table class="wr-tbl"><thead><tr>'
            f'<th>Sev</th><th>Issue</th><th>Detail / Fix</th>'
            f'</tr></thead><tbody>{rows}</tbody></table>'
            if rows else ""
        )
        + "</div>"
    )


def _render_nl_to_spark(r: dict) -> str:
    sql         = r.get("sql_query","")
    pyspark     = r.get("pyspark_code","")
    expl        = r.get("explanation","")
    assumptions = r.get("assumptions",[])

    assumptions_html = ""
    if assumptions:
        items = "".join(f"<li>{a}</li>" for a in assumptions)
        assumptions_html = (
            '<div style="background:#fff3cd;border-left:4px solid #f57c00;border-radius:4px;'
            'padding:8px 12px;margin:6px 0;font-size:12px">'
            '<strong style="color:#856404">⚠ Assumptions &amp; Caveats</strong>'
            f'<ul style="margin:4px 0 0;padding-left:16px;color:#1a1a2e">{items}</ul>'
            "</div>"
        )

    sql_preview = "\n".join(sql.splitlines()[:30])
    sql_ellipsis = "\n..." if len(sql.splitlines()) > 30 else ""

    pyspark_html = ""
    if pyspark:
        py_preview = "\n".join(pyspark.splitlines()[:30])
        py_ellipsis = "\n..." if len(pyspark.splitlines()) > 30 else ""
        pyspark_html = (
            '<div style="font-size:11px;color:#555;font-weight:600;margin-top:10px;margin-bottom:2px">'
            "PySpark</div>"
            f'<div class="wr-code">{py_preview}{py_ellipsis}</div>'
        )

    return (
        f'<div class="wr-header"><span>💬 NL → PySpark</span></div>'
        f'<div class="wr-body">'
        + assumptions_html
        + f'<p class="wr-sum">{expl}</p>'
        + '<div style="font-size:11px;color:#555;font-weight:600;margin-bottom:2px">SQL</div>'
        + f'<div class="wr-code">{sql_preview}{sql_ellipsis}</div>'
        + pyspark_html
        + "</div>"
    )


_MLOPS_CAT_ORDER = {
    "data_leakage": 0, "reproducibility": 1, "security": 2,
    "no_error_handling": 3, "hardcoded_config": 4, "missing_logging": 5,
    "inefficient_spark": 6, "missing_tests": 7, "code_structure": 8, "documentation": 9,
}
_MLOPS_SEV_ORDER = {"critical": 0, "warning": 1, "suggestion": 2}


def _render_mlops_reviewer(r: dict) -> str:
    score    = r.get("readiness_score",0)
    summary  = r.get("summary","")
    findings = sorted(
        r.get("findings", []),
        key=lambda f: (
            _MLOPS_SEV_ORDER.get(f.get("severity", "suggestion"), 2),
            _MLOPS_CAT_ORDER.get(f.get("category", ""), 9),
        ),
    )
    sc = "success" if score >= 70 else "warning" if score >= 40 else "danger"

    critical = [f for f in findings if f.get("severity") == "critical"]
    if critical or score < 40:
        hero = _verdict_hero("MLOps", "NOT READY", "danger",
                             items=[f.get("description", "")[:80] for f in critical[:3]],
                             subtitle=f"{score}/100")
    elif score < 70:
        hero = _verdict_hero("MLOps", "NEEDS REVIEW", "warning", subtitle=f"{score}/100")
    else:
        hero = _verdict_hero("MLOps", "PRODUCTION READY", "success", subtitle=f"{score}/100")

    rows = ""
    for f in findings[:6]:
        sev = f.get("severity","suggestion")
        sev_style = {"critical":"danger","warning":"warning","suggestion":"info"}.get(sev,"secondary")
        fix = f.get("fix", "")
        rows += (
            f'<tr class="{_sev_row_cls(sev)}">'
            f'<td>{_badge(sev.upper(), sev_style)}</td>'
            f'<td class="wr-muted">{f.get("category","").replace("_"," ").title()}</td>'
            f'<td>{f.get("description","")}'
            + (f'<br><span class="wr-muted">{fix}</span>' if fix else "")
            + f'</td>'
            f'</tr>'
        )

    return (
        hero
        + f'<div class="wr-header"><span>⚙ MLOps Review</span>'
        f'{_badge(f"{score}/100", sc)}'
        f'</div>'
        f'<div class="wr-body">'
        f'<p class="wr-sum">{summary}</p>'
        + (
            f'<table class="wr-tbl"><thead><tr>'
            f'<th>Sev</th><th>Category</th><th>Issue</th>'
            f'</tr></thead><tbody>{rows}</tbody></table>'
            if rows else ""
        )
        + "</div>"
    )


def _render_preflight(r: dict) -> str:
    verdict  = r.get("verdict", "incomplete")   # an unknown state is never shown as ready
    blockers = r.get("blockers", [])
    cautions = r.get("cautions", [])
    cleared  = r.get("cleared", [])
    summary  = r.get("summary", "")
    dq_ok    = r.get("dq_available", False)
    ld_ok    = r.get("leakage_available", False)
    failed   = {f["skill"]: f["error"] for f in r.get("failed", [])}

    if verdict == "blocked":
        hero = _verdict_hero(
            "Pre-flight Check", "BLOCKED",
            "danger",
            items=[f'<span class="wr-mono">{b["column"]}</span> [{b["source"]}] — {b["reason"]}'
                   for b in blockers[:3]],
        )
    elif verdict == "caution":
        hero = _verdict_hero(
            "Pre-flight Check", "CAUTION",
            "warning",
            items=[f'<span class="wr-mono">{c["column"]}</span> [{c["source"]}] — {c["reason"]}'
                   for c in cautions[:3]],
        )
    elif verdict == "ready_for_review":
        hero = _verdict_hero("Pre-flight Check", "READY FOR REVIEW", "success")
    else:
        hero = _verdict_hero(
            "Pre-flight Check", "INCOMPLETE", "warning",
            items=[f"{s.replace('_', ' ')} not run" for s in r.get("missing", [])]
                  + [f"{s.replace('_', ' ')} failed: {e[:80]}" for s, e in failed.items()],
        )

    def _src(skill, ok, label):
        if skill in failed:
            return _badge(f"{label} — failed", "danger")
        return _badge(f"{label} ✓", "success") if ok else _badge(f"{label} — not run", "secondary")
    source_badges = _src("data_quality", dq_ok, "DQ") + " " + _src("leakage_detector", ld_ok, "Leakage")
    vc = {"blocked": "danger", "caution": "warning", "ready_for_review": "success"}.get(verdict, "warning")

    blockers_html = ""
    if blockers:
        rows = ""
        for b in blockers:
            src_style = "danger" if b["source"] == "leakage" else "warning"
            rows += (
                f'<tr class="wr-crit">'
                f'<td>{_badge(b["source"].replace("_", " "), src_style)}</td>'
                f'<td><span class="wr-mono">{b["column"]}</span></td>'
                f'<td>{b["reason"]}<br><span class="wr-muted">→ {b["action"]}</span></td>'
                f'</tr>'
            )
        blockers_html = (
            '<div style="background:#fde8e8;border-left:4px solid #c62828;border-radius:4px;'
            'padding:8px 12px;margin:8px 0 4px">'
            '<strong style="color:#c62828;font-size:12px">⛔ Must fix before training</strong>'
            f'<table class="wr-tbl" style="margin-top:6px"><thead><tr>'
            f'<th>Source</th><th>Column</th><th>Issue &amp; Fix</th>'
            f'</tr></thead><tbody>{rows}</tbody></table>'
            '</div>'
        )

    cautions_html = ""
    if cautions:
        rows = ""
        for c in cautions:
            src_style = "danger" if c["source"] == "leakage" else "warning"
            rows += (
                f'<tr class="wr-warn">'
                f'<td>{_badge(c["source"].replace("_", " "), src_style)}</td>'
                f'<td><span class="wr-mono">{c["column"]}</span></td>'
                f'<td>{c["reason"]}</td>'
                f'</tr>'
            )
        cautions_html = (
            '<div style="background:#fff3cd;border-left:4px solid #f57c00;border-radius:4px;'
            'padding:8px 12px;margin:8px 0 4px">'
            '<strong style="color:#856404;font-size:12px">⚠ Review before training</strong>'
            f'<table class="wr-tbl" style="margin-top:6px"><thead><tr>'
            f'<th>Source</th><th>Column</th><th>Issue</th>'
            f'</tr></thead><tbody>{rows}</tbody></table>'
            '</div>'
        )

    cleared_html = ""
    if cleared:
        items = "".join(
            f'<li style="color:#2e7d32">✓ {c}</li>' for c in cleared
            if not c.startswith("Leakage check not yet run") and not c.startswith("Data quality check not yet run")
        )
        pending = "".join(
            f'<li style="color:#888">○ {c}</li>' for c in cleared
            if c.startswith("Leakage check not yet run") or c.startswith("Data quality check not yet run")
        )
        all_items = items + pending
        if all_items:
            cleared_html = (
                '<ul style="list-style:none;padding-left:0;margin:8px 0 0;font-size:12px">'
                + all_items
                + '</ul>'
            )

    return (
        hero
        + f'<div class="wr-header">'
        f'<span>✈ Pre-flight</span>'
        f'{_badge(verdict.upper(), vc)}'
        f'<span style="margin-left:auto">{source_badges}</span>'
        f'</div>'
        f'<div class="wr-body">'
        f'<p class="wr-sum">{summary}</p>'
        + blockers_html
        + cautions_html
        + _confirmed_html(r)
        + cleared_html
        + '</div>'
    )


def _gaps_html(r: dict) -> str:
    """Required checks that didn't run or failed — shown before anything else."""
    items = [f"{m.replace('_', ' ')} not run" for m in r.get("missing", [])]
    items += [f"{f['skill'].replace('_', ' ')} failed: {f['error'][:100]}" for f in r.get("failed", [])]
    if not items:
        return ""
    return ('<div style="background:#fff3cd;border-left:4px solid #e0a800;border-radius:4px;'
            'padding:6px 12px;margin:6px 0;font-size:12px"><strong>Incomplete — required checks:</strong> '
            + "; ".join(items) + "</div>")


def _confirmed_html(r: dict) -> str:
    """Leakage caveat, and columns a person confirmed available (info, with who/when/why)."""
    out = ""
    if r.get("leakage_caveat"):
        out += ('<div style="background:#fff3cd;border-left:4px solid #e0a800;border-radius:4px;'
                f'padding:6px 12px;margin:6px 0;font-size:12px">{r["leakage_caveat"]}</div>')
    items = [f'<li><span class="wr-mono">{c["column"]}</span> — confirmed available by {c["confirmed_by"]} '
             f'on {c["confirmed_at"][:10]}: “{c["note"]}” <span class="wr-muted">(was {c["was"]})</span></li>'
             for c in r.get("confirmed", [])]
    if items:
        out += ('<div style="margin:6px 0;font-size:12px">'
                f'{_badge("INFO", "info")} <strong>Confirmed by a person</strong>'
                f'<ul style="margin:4px 0">{"".join(items)}</ul></div>')
    return out


def _render_session_digest(r: dict) -> str:
    verdict  = r.get("overall_verdict", "incomplete")
    blockers = r.get("blockers", [])
    cautions = r.get("cautions", [])
    cleared  = r.get("cleared", [])
    next_step = r.get("next_step", "")
    skills   = r.get("skills_run", [])

    vc = {"blocked": "danger", "caution": "warning", "incomplete": "warning",
          "ready_for_review": "success"}.get(verdict, "warning")
    vl = {"blocked": "BLOCKED", "caution": "CAUTION", "incomplete": "INCOMPLETE",
          "ready_for_review": "READY FOR REVIEW"}.get(verdict, "INCOMPLETE")

    # Blockers table
    blockers_html = ""
    if blockers:
        rows = ""
        for b in blockers:
            src = b["source"].replace("_", " ")
            sc  = "danger" if b["source"] in ("leakage", "leakage_detector") else "warning"
            rows += (
                f'<tr class="wr-crit">'
                f'<td>{_badge(src, sc)}</td>'
                f'<td><span class="wr-mono">{b["column"]}</span></td>'
                f'<td>{b["issue"]}'
                + (f'<br><span class="wr-muted">→ {b["action"]}</span>' if b.get("action") else "")
                + f'</td></tr>'
            )
        n = len(blockers)
        blockers_html = (
            '<div style="margin-bottom:10px">'
            f'<div style="font-size:11px;font-weight:600;color:#c62828;margin-bottom:4px">'
            f'⛔ FIX BEFORE TRAINING ({n})</div>'
            f'<table class="wr-tbl"><thead><tr>'
            f'<th>Source</th><th>Column</th><th>Issue &amp; Fix</th>'
            f'</tr></thead><tbody>{rows}</tbody></table></div>'
        )

    # Cautions table
    cautions_html = ""
    if cautions:
        rows = ""
        for c in cautions:
            src = c["source"].replace("_", " ")
            rows += (
                f'<tr class="wr-warn">'
                f'<td>{_badge(src, "warning")}</td>'
                f'<td><span class="wr-mono">{c["column"]}</span></td>'
                f'<td>{c["issue"]}</td></tr>'
            )
        n = len(cautions)
        cautions_html = (
            '<div style="margin-bottom:10px">'
            f'<div style="font-size:11px;font-weight:600;color:#856404;margin-bottom:4px">'
            f'⚠ REVIEW ({n})</div>'
            f'<table class="wr-tbl"><thead><tr>'
            f'<th>Source</th><th>Column</th><th>Issue</th>'
            f'</tr></thead><tbody>{rows}</tbody></table></div>'
        )

    # Cleared row
    cleared_html = ""
    if cleared:
        items = "&nbsp; · &nbsp;".join(
            f'{_badge(c["skill"].replace("_"," "), "success")}'
            f' <span class="wr-muted" style="font-size:11px">{c["summary"]}</span>'
            for c in cleared
        )
        cleared_html = (
            '<div style="margin-bottom:10px;font-size:12px">'
            f'<span style="font-size:11px;font-weight:600;color:#1b5e20;margin-right:6px">✓ CLEARED</span>'
            f'{items}</div>'
        )

    # Next step
    next_html = ""
    if next_step:
        next_html = (
            '<div style="background:#e8f5e9;border-radius:4px;padding:7px 12px;font-size:12px">'
            f'<strong style="color:#1b5e20">→ Next: </strong>{next_step}</div>'
        )

    n_skills = len(skills)
    s = "s" if n_skills != 1 else ""
    return (
        f'<div class="wr-header">'
        f'<span>📋 Session Digest</span>'
        f'{_badge(f"{n_skills} skill{s} run", "secondary")}'
        f'{_badge(vl, vc)}'
        f'</div>'
        f'<div class="wr-body">'
        + _gaps_html(r)
        + blockers_html
        + _confirmed_html(r)
        + cautions_html
        + cleared_html
        + next_html
        + '</div>'
    )


def _render_error(r: dict) -> str:
    err   = r.get("error", "Unknown error")
    skill = r.get("skill_name", "Skill")
    return (
        '<div style="background:#fde8e8;border-left:4px solid #c62828;border-radius:4px;'
        'padding:10px 14px;margin:0">'
        f'<strong style="color:#c62828">⛔ {skill} — Skill Error</strong><br>'
        f'<span style="color:#444;font-size:12px;font-family:monospace">{err}</span><br><br>'
        '<span style="color:#888;font-size:11px">Try running the skill again. '
        'If the error persists, check that your DataFrame is valid and the target column exists.'
        '</span>'
        '</div>'
    )


def _render_causal_inference(r: dict) -> str:
    verdict = r.get("verdict", "caution")
    est = r.get("estimate", {})
    style, label = {"valid": ("success", "VALID"), "caution": ("warning", "CAUTION"),
                    "not_identified": ("danger", "NOT IDENTIFIED")}.get(verdict, ("secondary", verdict.upper()))
    blockers = [f.get("evidence", "")[:90] for f in r.get("findings", []) if f.get("severity") == "critical"]
    hero = _verdict_hero("Causal effect", label, style, items=blockers[:3] or None,
                         subtitle=r.get("design", "").replace("_", " "))

    effect_html = ""
    if est.get("effect") is not None:
        excluded = est.get("excluded_covariates") or []
        # With no identified effect, the number is context, not an answer
        lead = ("<b>Not a causal estimate</b> (for reference only): " if verdict == "not_identified"
                else "<b>Effect: ")
        effect_html = (
            '<div style="font-size:14px;margin:6px 0 10px">'
            + lead + f'{est["effect"]:+.4g}' + ("" if verdict == "not_identified" else "</b>") + ' '
            f'<span class="wr-muted">(95% CI {est["ci_low"]:.4g} to {est["ci_high"]:.4g}, '
            f'p = {est["p_value"]:.3g}) · {est.get("method", "").replace("_", " ")}'
            + (f' · not adjusting for {", ".join(excluded)}' if excluded else "")
            + "</span></div>")

    rows = ""
    for f in _sort_findings(r.get("findings", [])):
        sev = f.get("severity", "info")
        rows += (f'<tr class="{_sev_row_cls(sev)}"><td>{_badge(sev.upper(), _sev_style(sev))}</td>'
                 f'<td>{f.get("check", "").replace("_", " ")}</td>'
                 f'<td><span class="wr-mono">{f.get("column") or "—"}</span></td>'
                 f'<td>{f.get("evidence", "")}<br><span class="wr-muted">{f.get("recommendation", "")}</span></td></tr>')
    findings_tbl = (f'<table class="wr-tbl"><thead><tr><th>Sev</th><th>Check</th><th>Column</th>'
                    f'<th>Evidence / what to do</th></tr></thead><tbody>{rows}</tbody></table>') if rows else ""

    assumptions = "".join(f"<li>{a}</li>" for a in r.get("assumptions", []))
    steps = "".join(f"<li>{s}</li>" for s in r.get("next_steps", []))
    return (
        hero
        + '<div class="wr-header"><span>⚖ Causal Inference</span>'
        + _badge(r.get("identification_strategy", "").replace("_", " "), "info") + '</div>'
        + '<div class="wr-body">'
        + effect_html
        + f'<p style="margin:0 0 8px">{r.get("summary", "")}</p>'
        + findings_tbl
        + (f'<div style="margin-top:8px"><b>Relies on:</b><ul style="margin:2px 0">{assumptions}</ul></div>' if assumptions else "")
        + (f'<div><b>Next steps:</b><ul style="margin:2px 0">{steps}</ul></div>' if steps else "")
        + '</div>'
    )


def _render_problem_framer(r: dict) -> str:
    from html import escape as e
    from ds_workbench.core.problem_spec import primary_question
    ans = r.get("answerability", {}) or {}
    status = ans.get("status", "needs_clarification")
    style = {"answerable": "success", "answerable_with_gaps": "warning"}.get(status, "danger")
    pq = primary_question(r)
    blocking = [m for m in r.get("missing", []) if m.get("blocking")]
    hero = _verdict_hero("Problem framing", status.replace("_", " ").upper(), style,
                         items=[f"Blocking: {e(m['item'])}" for m in blocking] or None,
                         subtitle=f"{pq.get('type', '?')} question · {e(ans.get('reason', ''))[:120]}")
    dec = r.get("decision", {}) or {}
    rows = "".join(
        f'<tr><td>{_badge(q.get("type", "?"), "info")}{" ★" if q.get("primary") else ""}</td>'
        f'<td>{e(q.get("statement", ""))}<br><span class="wr-muted">'
        + " · ".join(f"{k.replace('_concept', '').replace('_', ' ')}: {e(q[k])}" for k in
                     ("outcome_concept", "exposure_concept", "comparison", "unit_of_analysis",
                      "population", "time_window", "prediction_point") if q.get(k))
        + "</span></td></tr>"
        for q in r.get("questions", []))
    gaps = "".join(f'<li>{"⛔ " if m.get("blocking") else ""}{e(m["item"])} '
                   f'<span class="wr-muted">— {e(m.get("why_it_matters", ""))}</span></li>'
                   for m in r.get("missing", []))
    assumptions = "".join(f'<li>{e(a["statement"])} <span class="wr-muted">({a["source"]}'
                          f'{", confirm" if a.get("must_confirm") else ""})</span></li>'
                          for a in r.get("assumptions", []))
    asks = "".join(f"<li>{e(c)}</li>" for c in r.get("clarifying_questions", []))
    confirmed = (f'{_badge("CONFIRMED", "success")} by {e(r["confirmed_by"])} on {r.get("confirmed_at", "")[:10]}'
                 if r.get("status") == "confirmed" else
                 f'{_badge("DRAFT", "warning")} <span class="wr-muted">review it, then '
                 f'<code>session.confirm_spec(note=...)</code></span>')
    return (
        hero
        + f'<div class="wr-header"><span>🧭 Problem Framer</span>{confirmed}</div><div class="wr-body">'
        + (f'<p><strong>Decision:</strong> {e(dec.get("statement", ""))}'
           + (f' <span class="wr-muted">options: {e("; ".join(dec.get("options", [])))}</span>' if dec.get("options") else "")
           + (f'<br><span class="wr-muted">criteria: {e(dec["criteria"])}</span>' if dec.get("criteria") else "")
           + (f'<br><span class="wr-muted">decision inputs (optional — costs, ROI): '
              f'{e("; ".join(dec["inputs_needed"]))}</span>' if dec.get("inputs_needed") else "")
           + "</p>" if dec.get("statement") else "")
        + f'<table class="wr-tbl"><thead><tr><th>Type</th><th>Question</th></tr></thead><tbody>{rows}</tbody></table>'
        + (f"<p><strong>Missing information</strong></p><ul>{gaps}</ul>" if gaps else "")
        + (f"<p><strong>Assumptions</strong></p><ul>{assumptions}</ul>" if assumptions else "")
        + (f"<p><strong>Questions to answer first</strong></p><ol>{asks}</ol>" if asks else "")
        + f'<p class="wr-muted">Required checks: {", ".join(r.get("required_checks", []))}</p>'
        + "</div>"
    )


def _render_data_scout(r: dict) -> str:
    from html import escape as e
    feas = r.get("feasibility", "not_covered")
    style = {"covered": "success", "partially_covered": "warning"}.get(feas, "danger")
    gaps = [g for g in r.get("coverage", []) if not g["covered"]]
    hero = _verdict_hero("Data Scout", feas.replace("_", " ").upper(), style,
                         items=[f"No data found for {g['kind'].replace('_', ' ')}: {e(g['text'])}"
                                for g in gaps if g["required"]] or None,
                         subtitle=f"{sum(c['recommendation'] == 'investigate' for c in r.get('candidates', []))} "
                                  f"to investigate · {e(r.get('catalog', ''))} catalog · metadata only")
    rec_style = {"investigate": "success", "maybe": "warning", "reject": "danger"}
    rows = ""
    for c in r.get("candidates", []):
        roles = "<br>".join(f'<span class="wr-mono">{e(x["column"])}</span> → {e(x["concept"])}'
                            for x in c["role_matches"])
        joins = "<br>".join(f'{e(j["left_table"].split(".")[-1])}.{e(j["left_column"])} = '
                            f'{e(j["right_table"].split(".")[-1])}.{e(j["right_column"])} '
                            f'{_badge(j["status"].replace("_", " "), "secondary")}' for j in c["join_keys"])
        notes = "".join(f"<li>{e(x)}</li>" for x in c["concerns"]) + \
            "".join(f"<li>🧑‍🔬 {e(x)}</li>" for x in c["expert_assumptions"])
        rows += (f'<tr><td>{_badge(c["recommendation"].upper(), rec_style[c["recommendation"]])}</td>'
                 f'<td><span class="wr-mono">{e(c["table"])}</span><br><span class="wr-muted">'
                 f'{e(c["grain"]["value"])} · updated {e(str(c.get("freshness") or "?"))}</span></td>'
                 f'<td>{e(c["why"])}' + (f'<ul style="margin:2px 0">{notes}</ul>' if notes else "") + "</td>"
                 f"<td>{roles}</td><td>{joins}</td></tr>")
    cov = "".join(f'<li>{"✓" if g["covered"] else "✗"} {e(g["kind"].replace("_", " "))}: {e(g["text"])}'
                  + (f' <span class="wr-muted">({e(", ".join(g["by"]))})</span>' if g["by"] else "") + "</li>"
                  for g in r.get("coverage", []))
    dropped = r.get("validation", {}).get("dropped", [])
    return (
        hero
        + f'<div class="wr-header"><span>🔭 Data Scout</span>{_badge("metadata only", "secondary")}</div>'
        + '<div class="wr-body">'
        + "".join(f'<div class="wr-muted">⚠ {e(w)}</div>' for w in r.get("warnings", []))
        + (f"<p>{e(r['summary'])}</p>" if r.get("summary") else "")
        + '<table class="wr-tbl"><thead><tr><th></th><th>Table</th><th>Why / concerns</th>'
          f'<th>Measures</th><th>Join keys</th></tr></thead><tbody>{rows}</tbody></table>'
        + f"<p><strong>Coverage</strong></p><ul>{cov}</ul>"
        + (f'<p class="wr-muted">Removed {len(dropped)} claim(s) not grounded in the catalog: '
           + e("; ".join(f"{d['claim']} ({d['reason']})" for d in dropped[:5])) + "</p>" if dropped else "")
        + '<p class="wr-muted">Evidence is metadata (names, types, comments). Join keys are '
          "unverified until audited; pick the shortlist, then verify it against the data.</p>"
        + "</div>"
    )


def _render_data_verification(r: dict) -> str:
    from html import escape as e
    verdict = r.get("verdict", "failed")
    style = {"verified": "success", "verified_with_issues": "warning"}.get(verdict, "danger")
    findings = r.get("findings", [])
    crit = [f for f in findings if f["severity"] == "critical"]
    hero = _verdict_hero("Data verification", verdict.replace("_", " ").upper(), style,
                         items=[f"{e(f['table'].split('.')[-1])}: {e(f['description'])}"[:160] for f in crit] or None,
                         subtitle=f"{len(r.get('tables', {}))} tables · {len(r.get('joins', []))} joins · aggregates only")
    trs = "".join(
        f'<tr><td><span class="wr-mono">{e(t)}</span></td><td>{v["row_count"]:,}</td>'
        f'<td>{e(", ".join(v["roles"]) or "—")}</td>'
        f'<td>{e(v["time_column"] or "—")}: {e(" → ".join(x[:10] for x in v["time_range"]) if v["time_range"] else "—")}</td>'
        f'<td>{e(str(v["updated_at"] or "?"))}</td></tr>' for t, v in r.get("tables", {}).items())
    js = "".join(
        f'<tr><td>{e(j["left_table"].split(".")[-1])}.{e(j["left_column"])} = '
        f'{e(j["right_table"].split(".")[-1])}.{e(j["right_column"])}</td><td>{e(j["cardinality"])}</td>'
        f'<td>{j["child_match_rate"]:.0%} of {e(j["child"].split(".")[-1])}</td>'
        f'<td>{_badge(j["status"].replace("_", " "), {"verified": "success", "verified_with_issues": "warning"}.get(j["status"], "danger"))}'
        + (f'<br><span class="wr-muted">{e(j["note"])}</span>' if j["note"] else "") + "</td></tr>"
        for j in r.get("joins", []))
    fs = "".join(f'<tr class="{_sev_row_cls(f["severity"])}"><td>{_badge(f["severity"].upper(), _sev_style(f["severity"]))}</td>'
                 f'<td>{e(f["table"].split(".")[-1])}</td><td>{e(f["description"])}'
                 + (f'<br><span class="wr-muted">{e(f["recommendation"])}</span>' if f["recommendation"] else "")
                 + "</td></tr>" for f in findings)
    dq = "".join(f'<li><span class="wr-mono">{e(t)}</span>: '
                 + (f'failed — {e(d["error"][:120])}' if d.get("error") else
                    f'{d["score"]}/100 · {d["critical"]} critical · {d["warnings"]} warnings')
                 + "</li>" for t, d in (r.get("dq") or {}).items())
    outside = r.get("not_from_shortlist") or []
    return (
        hero
        + f'<div class="wr-header"><span>🔬 Data verification</span>{_badge("aggregates only", "secondary")}</div>'
        + '<div class="wr-body">'
        + (f'<div class="wr-muted">⚠ Not from the scout\'s shortlist: {e(", ".join(outside))}</div>' if outside else "")
        + (f'<table class="wr-tbl"><thead><tr><th></th><th>Table</th><th>Finding</th></tr></thead><tbody>{fs}</tbody></table>'
           if fs else '<p style="color:#2e7d32">✓ No problems found.</p>')
        + '<p><strong>Joins</strong></p>'
        + (f'<table class="wr-tbl"><thead><tr><th>Join</th><th>Cardinality</th><th>Match</th><th>Status</th></tr></thead>'
           f'<tbody>{js}</tbody></table>' if js else '<p class="wr-muted">No joins between the selected tables were proposed.</p>')
        + '<p><strong>Tables</strong></p><table class="wr-tbl"><thead><tr><th>Table</th><th>Rows</th><th>Roles</th>'
          f'<th>Time range</th><th>Updated</th></tr></thead><tbody>{trs}</tbody></table>'
        + (f"<p><strong>Data quality (sampled)</strong></p><ul>{dq}</ul>" if dq else "")
        + "</div>"
    )


def _render_investigation_plan(r: dict) -> str:
    from html import escape as e
    nxt = r.get("next", {})
    style = {"complete": "success", "fix": "danger", "stop": "danger"}.get(nxt.get("kind"), "info")
    hero = _verdict_hero("Investigation plan", f"{r.get('done', 0)}/{r.get('total', 0)} STEPS", style,
                         items=[f"Next: {e(nxt.get('text', ''))}"],
                         subtitle=f"{r.get('question_type') or '?'} question")
    icon = {"done": "✅", "ready": "▶️", "waiting": "⏳", "failed": "❌", "blocked": "⛔", "unavailable": "🚫"}
    rows = "".join(
        f'<tr><td>{icon.get(st["status"], "")} {e(st["status"])}</td>'
        f'<td><strong>{e(st["id"])}</strong>{" 🧑" if st.get("human") else ""}'
        f'{"" if st["mandatory"] else " <span class=wr-muted>(optional)</span>"}</td>'
        f'<td>{e(st["skill"])}{" · " + e(st["method"]) if st.get("method") else ""}</td>'
        f'<td class="wr-muted">{e(st["why"])}</td></tr>' for st in r.get("steps", []))
    recipe = "".join(f"<li>{e(x)}</li>" for x in r.get("recipe", []))
    contract = "".join(f'<li><span class="wr-mono">{e(k)}</span>: {e(v)}</li>' for k, v in (r.get("contract") or {}).items())
    method = r.get("method") or {}
    return (
        hero
        + '<div class="wr-header"><span>🗺 Investigation plan</span></div><div class="wr-body">'
        + f'<table class="wr-tbl"><thead><tr><th>Status</th><th>Step</th><th>Skill</th><th>Why</th></tr></thead>'
          f"<tbody>{rows}</tbody></table>"
        + (f'<p><strong>Method: {e(method["method"])}</strong> — {e(method["why"])}</p>' if method else "")
        + (f"<p><strong>Build the analysis table</strong> (you run this)</p><ol>{recipe}</ol>" if recipe else "")
        + (f"<p><strong>Columns it must have</strong></p><ul>{contract}</ul>" if contract else "")
        + '<p class="wr-muted">Required checks: ' + e(", ".join(r.get("required_checks", []))) + "</p></div>"
    )


_RENDERERS = {
    "investigation_plan":  _render_investigation_plan,
    "data_verification":   _render_data_verification,
    "data_scout":          _render_data_scout,
    "problem_framer":      _render_problem_framer,
    "data_quality":        _render_data_quality,
    "leakage_detector":    _render_leakage_detector,
    "feature_engineering": _render_feature_engineering,
    "model_selector":      _render_model_selector,
    "eda":                 _render_eda,
    "join_auditor":        _render_join_auditor,
    "nl_to_spark":         _render_nl_to_spark,
    "mlops_reviewer":      _render_mlops_reviewer,
    "preflight":           _render_preflight,
    "session_digest":      _render_session_digest,
    "causal_inference":    _render_causal_inference,
}


# ── Public class ───────────────────────────────────────────────────────────────

class SkillResult(dict):
    """
    A dict subclass returned by all ds_workbench skills.
    Renders as styled HTML in Jupyter / Databricks notebook cells.
    Behaves identically to a plain dict in all other contexts.
    """

    def __init__(self, data: dict, skill: str, session_warnings: list[str] | None = None):
        super().__init__(data)
        self._skill            = skill
        self._session_warnings = session_warnings or []

    def _repr_html_(self) -> str:
        if "error" in self:
            body = _render_error(dict(self) | {"skill_name": self._skill.replace("_", " ").title()})
            return f'{_CSS}<div class="wr-card">{body}</div>'
        renderer = _RENDERERS.get(self._skill)
        if renderer is None:
            return f"<pre>{dict(self)}</pre>"
        try:
            body  = renderer(self)
            warns = "".join(self._session_warnings)
            return f'{_CSS}<div class="wr-card">{warns}{body}</div>'
        except Exception as exc:
            return f'<pre style="color:red">Render error ({self._skill}): {exc}</pre>'

    def __repr__(self) -> str:
        skill = self._skill
        keys  = list(self.keys())
        return f"SkillResult({skill}) {keys}"

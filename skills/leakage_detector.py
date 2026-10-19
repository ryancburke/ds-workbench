"""
Leakage Detector skill.

Detects target leakage in a DataFrame before model training.
Combines statistical pre-screening (correlation, conditional null rates) with
LLM analysis to flag features that would inflate training accuracy but fail in production.
"""
import json
import pandas as pd
import numpy as np
from skills.result import SkillResult, findings_of
from skills.api_retry import api_call, get_client, skill_entry, tool_input
from skills.models import THINKING_ADAPTIVE, role


_TOOL = {
    "name": "report_leakage",
    "description": "Report target leakage findings for features in a dataset",
    "input_schema": {
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["leakage_detected", "suspicious", "clean"],
            },
            "risk_level": {
                "type": "string",
                "enum": ["high", "medium", "low", "none"],
            },
            "summary": {"type": "string"},
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "column": {"type": "string"},
                        "leakage_type": {
                            "type": "string",
                            "enum": ["direct", "proxy", "temporal", "high_correlation", "identifier"],
                        },
                        "severity": {"type": "string", "enum": ["critical", "warning", "info"]},
                        "evidence": {"type": "string"},
                        "recommendation": {"type": "string"},
                    },
                    "required": ["column", "leakage_type", "severity", "evidence", "recommendation"],
                },
            },
        },
        "required": ["verdict", "risk_level", "summary", "findings"],
    },
}


def _single_feature_auc(values: pd.Series, y: pd.Series) -> float | None:
    """Rank-based AUC of one feature against a 0/1 target, folded to >= 0.5.
    Unlike Pearson r it sees any monotone separation, linear or not."""
    valid = values.notna() & y.notna()
    v, t = values[valid].astype(float), y[valid].astype(float)
    n1 = float(t.sum()); n0 = float(len(t) - n1)
    if n1 < 5 or n0 < 5:
        return None
    ranks = v.rank()
    auc = (ranks[t == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return round(float(max(auc, 1 - auc)), 4)


def build_leakage_profile(df: pd.DataFrame, target_col: str) -> dict:
    """Compute statistical signals that reveal target leakage."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")

    target = df[target_col]
    is_binary = target.dtype == bool or (
        target.nunique() == 2 and pd.api.types.is_numeric_dtype(target)
    )

    if target.dtype == bool:
        target_numeric = target.astype(float)
    elif is_binary:
        uvals = sorted(target.dropna().unique())
        target_numeric = target.map({uvals[0]: 0.0, uvals[1]: 1.0})
    elif pd.api.types.is_numeric_dtype(target) and target.dtype != bool:
        target_numeric = target.astype(float)
    else:
        target_numeric = None

    profile: dict = {
        "target_col": target_col,
        "target_dtype": str(target.dtype),
        "target_value_counts": {str(k): int(v) for k, v in target.value_counts().head(5).items()},
        "n_rows": len(df),
        "features": {},
    }

    for col in df.columns:
        if col == target_col:
            continue
        feat = df[col]
        info: dict = {
            "dtype": str(feat.dtype),
            "null_pct": round(float(feat.isnull().mean() * 100), 2),
            "unique_count": int(feat.nunique()),
        }

        # Point-biserial / Pearson correlation with target
        if (
            target_numeric is not None
            and pd.api.types.is_numeric_dtype(feat)
            and feat.dtype != bool
        ):
            valid = feat.notna() & target_numeric.notna()
            if valid.sum() > 10:
                try:
                    corr = float(
                        pd.Series(feat[valid].values, dtype=float).corr(
                            pd.Series(target_numeric[valid].values, dtype=float)
                        )
                    )
                    if not np.isnan(corr):
                        info["correlation_with_target"] = round(abs(corr), 4)
                except Exception:
                    pass

        # Single-feature AUC — catches non-linear separation that Pearson r misses
        if is_binary and target_numeric is not None and pd.api.types.is_numeric_dtype(feat) \
                and feat.dtype != bool:
            auc = _single_feature_auc(feat, target_numeric)
            if auc is not None:
                info["single_feature_auc"] = auc
            # Values that are zero for one class only (e.g. a refund credit after the event)
            zeros = feat.fillna(0) == 0
            if 0.1 <= zeros.mean() <= 0.95:
                info["zero_pct_by_class"] = {
                    str(val): round(float(zeros[target == val].mean() * 100), 2)
                    for val in target.dropna().unique() if (target == val).sum() >= 5}

        # Categorical: target rate per value, and values seen in only one class
        if is_binary and target_numeric is not None and not pd.api.types.is_numeric_dtype(feat) \
                and 1 < feat.nunique() <= 50:
            rates = target_numeric.groupby(feat.astype(str)).agg(["mean", "size"])
            rates = rates[rates["size"] >= max(5, int(0.01 * len(df)))]
            if len(rates):
                info["target_rate_by_value"] = {
                    str(k): round(float(r["mean"]), 3)
                    for k, r in rates.sort_values("size", ascending=False).head(8).iterrows()}
                pure = rates[(rates["mean"] <= 0.0) | (rates["mean"] >= 1.0)]
                if len(pure):
                    info["single_class_values"] = {
                        str(k): {"rows": int(r["size"]), "target_rate": round(float(r["mean"]), 3)}
                        for k, r in pure.iterrows()}
                encoded = feat.astype(str).map(rates["mean"]).astype(float)
                auc = _single_feature_auc(encoded, target_numeric)
                if auc is not None:
                    info["single_feature_auc"] = auc

        # Conditional null rate — exposes post-event proxy features
        if feat.isnull().any() and is_binary:
            cond: dict = {}
            for val in target.dropna().unique():
                mask = target == val
                if mask.sum() >= 5:
                    cond[str(val)] = round(float(feat[mask].isnull().mean() * 100), 2)
            if cond:
                info["conditional_null_pct"] = cond

        if pd.api.types.is_numeric_dtype(feat) and feat.dtype != bool:
            non_null = feat.dropna().astype(float)
            if len(non_null):
                info["stats"] = {
                    "min": round(float(non_null.min()), 4),
                    "max": round(float(non_null.max()), 4),
                    "mean": round(float(non_null.mean()), 4),
                }
        else:
            vc = feat.value_counts(dropna=False).head(5)
            info["sample_values"] = {str(k): int(v) for k, v in vc.items()}

        profile["features"][col] = info

    return profile


def _maybe_sample(
    df: pd.DataFrame, target_col: str, n: int = 50_000
) -> tuple[pd.DataFrame, str | None]:
    if len(df) <= n:
        return df, None
    try:
        sampled = (
            df.groupby(target_col, group_keys=False)
            .apply(lambda x: x.sample(min(len(x), max(1, int(n * len(x) / len(df))))))
            .reset_index(drop=True)
        )
    except Exception:
        sampled = df.sample(n, random_state=42).reset_index(drop=True)
    return sampled, f"Sampled {len(sampled):,} of {len(df):,} rows (stratified) for analysis"


@skill_entry("leakage_detector")
def check(
    df: pd.DataFrame,
    target_col: str,
    context: str = "",
    dq_findings: list | None = None,
    session=None,
) -> dict:
    """Check a DataFrame for target leakage before model training.

    Args:
        df: Full training DataFrame including the target column.
        target_col: Name of the column being predicted.
        context: Optional description of the prediction task.
        dq_findings: Optional findings from a prior data_quality.assess() run.
            Populated automatically when a WorkflowSession is provided.
        session: Optional WorkflowSession. Enables prerequisite warnings and
            automatic cross-skill context.

    Returns:
        Dict with verdict, risk_level, findings, and summary.
    """
    try:
        return _check(df, target_col, context, dq_findings, session)
    except Exception as exc:
        return SkillResult({"error": str(exc)}, skill="leakage_detector")


def _check(
    df: pd.DataFrame,
    target_col: str,
    context: str = "",
    dq_findings: list | None = None,
    session=None,
) -> dict:
    if session is not None:
        if dq_findings is None:
            dq_findings = session.get_findings("data_quality")
        if not context:
            context = getattr(session, "context", "")

    dq_findings = findings_of(dq_findings)  # accepts a prior result or its findings list
    session_warnings = session.prerequisite_warnings("leakage_detector") if session else []

    if session is not None and session.has_run("feature_engineering"):
        session_warnings.append(
            '<div style="background:#fde8e8;border-left:4px solid #c62828;border-radius:4px;'
            'padding:8px 12px;margin-bottom:6px;font-size:12px">'
            '<strong style="color:#c62828">⚠ Feature pipeline ran before leakage check</strong><br>'
            '<span style="color:#666;font-size:11px">'
            'Your feature_engineering plan may include leaking columns. '
            'Review the findings below and re-run feature_engineering after resolving any leakage.'
            '</span></div>'
        )

    df, sample_note = _maybe_sample(df, target_col)
    profile = build_leakage_profile(df, target_col)
    context_line = f"\nTask context: {context}\n" if context else ""

    dq_block = ""
    if dq_findings:
        crit = [f for f in dq_findings if f.get("severity") == "critical"]
        warn = [f for f in dq_findings if f.get("severity") == "warning"]
        dq_lines = [
            f"  - {f.get('column','?')} [{f.get('issue_type','?')}]: {f.get('description','')}"
            for f in (crit + warn)[:5]
        ]
        dq_block = (
            f"\nData quality context ({len(crit)} critical, {len(warn)} warnings from earlier DQ scan):\n"
            + "\n".join(dq_lines)
            + "\nFactor these into severity judgments — a null pattern already flagged by DQ "
            "may not be leakage, just a data quality issue.\n"
        )

    _prompt = f"""You are a senior ML engineer performing a pre-training leakage audit on '{target_col}'.
{context_line}{dq_block}
Leakage profile (computed from the dataset):
{json.dumps(profile, indent=2)}

Identify any features that constitute target leakage — information about the target that would
not be available at prediction time in production.

Leakage types:
- direct: feature IS the target or a trivial transformation of it
- proxy: feature only exists as a downstream consequence of the target
  (e.g. a refund_id only assigned after a return; a claim_id only exists after filing a claim)
- high_correlation: the feature separates the target far better than any pre-event signal
  plausibly could — correlation >0.8, or single_feature_auc ≥ 0.97 even when correlation is
  low (a non-linear leak) — with no plausible causal explanation
  (a legitimate feature can have 0.5-0.7 correlation — that's predictive, not leakage)
- temporal: feature uses data from after the prediction event
- identifier: ID whose values encode the target (e.g. fraud_case_id → only fraudulent rows)

Profile signals:
- single_feature_auc: how well this one column alone separates the classes (0.5 = not at all,
  1.0 = perfectly). Legitimate strong predictors usually sit at 0.6–0.9; ≥ 0.97 from a single
  column is rare without leakage. Pearson r can be low for a near-perfect non-linear leak.
- single_class_values: category values (≥1% of rows) that occur in only one target class, e.g. a
  status written by the event itself.
- zero_pct_by_class: a column that is zero for every row of one class but not the other, e.g. a
  credit or fee issued after the event (the zero-valued twin of conditional_null_pct).
A strong signal says where to look; whether it is leakage depends on when the value exists.
An attribute of the thing being scored (a transaction's amount, an applicant's income) is known
at prediction time, so near-perfect separation there is the signal the model is for: at most a
warning to confirm how it was recorded. Call it critical when something says the value is made
by or after the outcome: a derived score or index, a status or ID written by the outcome flow, a
value present or non-zero for one class only.

Severity rules:
- critical: almost certainly leakage (correlation >0.85, conditional null gap >90%, direct proxy).
  Must be removed before training.
- warning: strong signal but worth manual review (correlation 0.7-0.85, gap 60-90%)
- info: weak signal, possibly legitimate — document for awareness only

Verdict:
- leakage_detected: one or more critical findings
- suspicious: only warning-level findings
- clean: no findings or only info-level notes

Do NOT flag:
- Legitimate predictors with moderate correlation (0.3-0.65) — that is predictive signal, not leakage
- Null values uniformly distributed across target classes (equal conditional null rate)
- Sequential or random ID columns (e.g. account_number, ticket_uuid, row_key) that exist
  for EVERY row regardless of the target value. These are not target leakage — they're identifiers
  that should be excluded from feature sets as good practice. Only flag an ID as leakage if the
  conditional_null_pct shows the ID is nearly 100% null for one target class (post-event proxy).
  If you mention an ID column, use info severity at most — never critical or warning.
- Event timestamps that represent WHEN the event was recorded (not post-event data)

CALIBRATION EXAMPLES — illustrative cases from a hospital-readmission dataset (target
'readmitted'); judge this dataset on its own profile:

CLEAN — legitimate predictor, do NOT flag:
  length_of_stay_days, correlation_with_target: 0.38
  → Moderate correlation is predictive signal. Not leakage.

CRITICAL LEAKAGE — post-event proxy, must flag:
  readmission_case_id, conditional_null_pct: {{"True": 3.0, "False": 98.9}}
  → Almost always null when readmitted=False, present when readmitted=True.
    Created by the readmission itself. leakage_type=proxy, severity=critical.

IDENTIFIER — not leakage, info only:
  patient_mrn, unique_count=5000, n_rows=5000,
  conditional_null_pct: {{"True": 0.2, "False": 0.2}}
  → Equal null rates across classes. Row identifier, not leakage.
    leakage_type=identifier, severity=info at most.

SUSPICIOUS HIGH CORRELATION — flag critical:
  care_team_risk_score, correlation_with_target: 0.91
  → r=0.91 with 'readmitted'. Assigned by the care team after the patient came back, so it
    encodes the outcome. leakage_type=high_correlation, severity=critical.

STRONG LEGITIMATE FEATURE — do NOT flag:
  prior_admissions_12m, correlation_with_target: 0.55
  → r=0.55. Prior admissions plausibly drive readmission. Predictive, not leaky.

Reference the specific statistics from the profile in your evidence fields."""

    # Pass 1: reason through leakage patterns with extended thinking
    _r1 = api_call(
        get_client().messages.create,
        **role("leakage"),
        max_tokens=12000,
        thinking=THINKING_ADAPTIVE,
        messages=[{"role": "user", "content": _prompt}],
    )
    _analysis = next((b.text for b in _r1.content if b.type == "text"), "")
    if not _analysis.strip():
        raise RuntimeError("leakage_detector Pass 1 returned no text — aborting to avoid empty structured output")

    # Pass 2: format findings into structured output
    response = api_call(
        get_client().messages.create,
        **role("structured"),
        max_tokens=4096,
        tools=[_TOOL],
        tool_choice={"type": "tool", "name": "report_leakage"},
        messages=[
            {"role": "user", "content": _prompt},
            {"role": "assistant", "content": _analysis},
            {"role": "user", "content": "Now report your findings using the report_leakage tool."},
        ],
    )

    tool_args = tool_input(response, _TOOL["name"])
    result = SkillResult(tool_args, skill="leakage_detector",
                         session_warnings=session_warnings)
    # Ready for the next cell: df.drop(columns=result["drop_columns"])
    result["drop_columns"] = list(dict.fromkeys(
        f["column"] for f in result.get("findings", [])
        if f.get("severity") == "critical" and f.get("column")))
    if sample_note:
        result["sample_note"] = sample_note

    # Cross-reference: columns flagged by both DQ and leakage
    if dq_findings:
        dq_cols = {f.get("column") for f in dq_findings
                   if f.get("column") and f.get("severity") in ("critical", "warning")}
        ld_cols = {f.get("column") for f in result.get("findings", [])
                   if f.get("column") and f.get("severity") in ("critical", "warning")}
        overlap = sorted(dq_cols & ld_cols)
        if overlap:
            result["cross_referenced_cols"] = overlap

    if session is not None:
        session.register("leakage_detector", result)
    return result

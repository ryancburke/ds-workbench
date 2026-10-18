"""
Leakage Detector skill.

Detects target leakage in a DataFrame before model training.
Combines statistical pre-screening (correlation, conditional null rates) with
LLM analysis to flag features that would inflate training accuracy but fail in production.
"""
import json
import anthropic
import pandas as pd
import numpy as np
from config import config
from skills.result import SkillResult
from skills.api_retry import api_call
from skills.models import LEAKAGE_MODEL, STRUCTURED_MODEL, THINKING_ADAPTIVE, LEAKAGE_EXTRA

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=config.anthropic_api_key)
    return _client


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
  (e.g. order_id only assigned after conversion; claim_id only exists after filing a claim)
- high_correlation: absolute correlation with target >0.8 with no plausible causal explanation
  (a legitimate feature can have 0.5-0.7 correlation — that's predictive, not leakage)
- temporal: feature uses data from after the prediction event
- identifier: ID whose values encode the target (e.g. fraud_case_id → only fraudulent rows)

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
- Sequential or random ID columns (employee_id, session_id, customer_id, transaction_id) that exist
  for EVERY row regardless of the target value. These are not target leakage — they're identifiers
  that should be excluded from feature sets as good practice. Only flag an ID as leakage if the
  conditional_null_pct shows the ID is nearly 100% null for one target class (post-event proxy).
  If you mention an ID column, use info severity at most — never critical or warning.
- Event timestamps that represent WHEN the event was recorded (not post-event data)

CALIBRATION EXAMPLES — anchor your judgments against these:

CLEAN — legitimate predictor, do NOT flag:
  monthly_income, correlation_with_target: 0.42
  → Moderate correlation is predictive signal. Not leakage.

CRITICAL LEAKAGE — post-event proxy, must flag:
  order_id, conditional_null_pct: {{"True": 2.1, "False": 99.8}}
  → 99.8% null when target=False, 2.1% when target=True.
    Exists AFTER the purchase event. leakage_type=proxy, severity=critical.

IDENTIFIER — not leakage, info only:
  customer_id, unique_count=1000, n_rows=1000,
  conditional_null_pct: {{"True": 0.5, "False": 0.5}}
  → Equal null rates across classes. Row identifier, not leakage.
    leakage_type=identifier, severity=info at most.

SUSPICIOUS HIGH CORRELATION — flag critical:
  conversion_score, correlation_with_target: 0.89
  → r=0.89 with target 'converted'. No causal mechanism for a score to
    exist before the conversion event. leakage_type=high_correlation, severity=critical.

STRONG LEGITIMATE FEATURE — do NOT flag:
  cart_value, correlation_with_target: 0.61
  → r=0.61. High cart value plausibly causes conversion. Predictive, not leaky.

Reference the specific statistics from the profile in your evidence fields."""

    # Pass 1: reason through leakage patterns with extended thinking
    _r1 = api_call(
        _get_client().messages.create,
        model=LEAKAGE_MODEL,
        max_tokens=12000,
        thinking=THINKING_ADAPTIVE,
        **LEAKAGE_EXTRA,
        messages=[{"role": "user", "content": _prompt}],
    )
    _analysis = next((b.text for b in _r1.content if b.type == "text"), "")
    if not _analysis.strip():
        raise RuntimeError("leakage_detector Pass 1 returned no text — aborting to avoid empty structured output")

    # Pass 2: format findings into structured output
    response = api_call(
        _get_client().messages.create,
        model=STRUCTURED_MODEL,
        max_tokens=4096,
        tools=[_TOOL],
        tool_choice={"type": "tool", "name": "report_leakage"},
        messages=[
            {"role": "user", "content": _prompt},
            {"role": "assistant", "content": _analysis},
            {"role": "user", "content": "Now report your findings using the report_leakage tool."},
        ],
    )

    tool_block = next(b for b in response.content if b.type == "tool_use")
    result = SkillResult(tool_block.input, skill="leakage_detector",
                         session_warnings=session_warnings)
    if sample_note:
        result["sample_note"] = sample_note
    if session is not None:
        session.register("leakage_detector", result)
    return result

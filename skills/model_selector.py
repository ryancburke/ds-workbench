import anthropic
import pandas as pd
from config import config
from skills.result import SkillResult
from skills.api_retry import api_call
from skills.models import ANALYSIS_MODEL, STRUCTURED_MODEL, THINKING_ADAPTIVE, ANALYSIS_EXTRA

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=config.anthropic_api_key)
    return _client


_TOOL = {
    "name": "report_model_selection",
    "description": "Recommend ML model families and check statistical assumptions for a dataset",
    "input_schema": {
        "type": "object",
        "properties": {
            "task_type": {
                "type": "string",
                "enum": [
                    "binary_classification", "multiclass_classification",
                    "regression", "time_series_forecasting",
                    "clustering", "anomaly_detection",
                ],
            },
            "target_analysis": {
                "type": "object",
                "properties": {
                    "summary":              {"type": "string"},
                    "class_balance_issue":  {"type": "boolean"},
                    "imbalance_ratio":      {"type": "number"},
                },
                "required": ["summary", "class_balance_issue"],
            },
            "assumption_checks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "assumption":      {"type": "string"},
                        "status":          {"type": "string", "enum": ["met", "violated", "uncertain"]},
                        "evidence":        {"type": "string"},
                        "affected_models": {"type": "string"},
                    },
                    "required": ["assumption", "status", "evidence", "affected_models"],
                },
            },
            "recommendations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "rank":         {"type": "integer"},
                        "model_family": {"type": "string"},
                        "display_name": {"type": "string"},
                        "suitability":  {"type": "string", "enum": ["strong", "moderate", "weak"]},
                        "rationale":    {"type": "string"},
                        "caveats":      {"type": "array", "items": {"type": "string"}},
                        "starter_code": {"type": "string"},
                    },
                    "required": ["rank", "model_family", "display_name", "suitability", "rationale"],
                },
            },
            "data_readiness":  {"type": "string", "enum": ["ready", "needs_prep", "insufficient"]},
            "readiness_notes": {"type": "array",  "items": {"type": "string"}},
            "summary":         {"type": "string"},
        },
        "required": [
            "task_type", "target_analysis", "assumption_checks",
            "recommendations", "data_readiness", "summary",
        ],
    },
}


def _build_profile(df: pd.DataFrame, target_col: str) -> str:
    """Derive a rich statistical profile so Claude reasons from real numbers."""
    lines = []
    n_rows, n_cols = df.shape
    lines.append(f"Shape: {n_rows:,} rows × {n_cols} columns")
    lines.append(f"Target: '{target_col}'")
    lines.append("")

    # ── Target analysis ───────────────────────────────────────────────────────
    target = df[target_col].dropna()
    n_unique = target.nunique()

    if df[target_col].dtype in ("object", "bool", "category") or n_unique <= 15:
        vc = target.value_counts()
        lines.append("Target (categorical):")
        for val, cnt in vc.head(10).items():
            lines.append(f"  {val!r}: {cnt:,} ({cnt / n_rows:.1%})")
        if len(vc) == 2:
            ratio = vc.iloc[0] / vc.iloc[1]
            lines.append(f"  Binary imbalance ratio: {ratio:.1f}:1")
        elif len(vc) > 2:
            lines.append(f"  {len(vc)} classes")
    else:
        skew = target.skew()
        lines.append("Target (numeric):")
        lines.append(f"  Mean={target.mean():.3f}, Std={target.std():.3f}")
        lines.append(f"  Min={target.min():.3f}, Max={target.max():.3f}")
        lines.append(f"  Skewness={skew:.2f}" +
                     (" ← consider log-transform" if abs(skew) > 1.5 else ""))
        lines.append(f"  Null rate: {df[target_col].isnull().mean():.1%}")

    lines.append("")

    # ── Feature profile ───────────────────────────────────────────────────────
    numeric_features = [
        c for c in df.select_dtypes(include="number").columns
        if c != target_col
    ]
    cat_features = [
        c for c in df.select_dtypes(include=["object", "category", "bool"]).columns
        if c != target_col
    ]

    lines.append(f"Features: {len(numeric_features)} numeric, {len(cat_features)} categorical")
    lines.append("")
    lines.append("Per-feature summary (null%, cardinality):")

    # Compute target correlation for numeric features
    corr_with_target: dict[str, float] = {}
    if numeric_features:
        target_num = None
        if df[target_col].dtype in ("float64", "int64", "int32", "float32"):
            target_num = df[target_col]
        elif df[target_col].nunique() == 2:
            try:
                target_num = pd.get_dummies(df[target_col], drop_first=True).iloc[:, 0].astype(float)
            except Exception:
                pass
        if target_num is not None:
            for col in numeric_features:
                try:
                    r = df[col].corr(target_num)
                    if not pd.isna(r):
                        corr_with_target[col] = round(float(r), 3)
                except Exception:
                    pass

    for col in df.columns:
        if col == target_col:
            continue
        null_pct = df[col].isnull().mean()
        unique   = df[col].nunique()
        dtype    = str(df[col].dtype)
        corr_str = f", r={corr_with_target[col]:+.2f}" if col in corr_with_target else ""
        lines.append(f"  {col} ({dtype}): {unique} unique, {null_pct:.1%} null{corr_str}")

    # ── High correlations between features ────────────────────────────────────
    if len(numeric_features) >= 2:
        lines.append("")
        lines.append("Feature–feature correlations |r| > 0.70:")
        corr_matrix = df[numeric_features].corr()
        found = False
        for i, a in enumerate(numeric_features):
            for b in numeric_features[i + 1:]:
                r = corr_matrix.loc[a, b]
                if abs(r) > 0.70:
                    lines.append(f"  {a} ↔ {b}: r={r:.2f}")
                    found = True
        if not found:
            lines.append("  None above 0.70 threshold")

    # ── Sample size note ─────────────────────────────────────────────────────
    lines.append("")
    if n_rows < 500:
        lines.append(f"⚠ Small dataset ({n_rows:,} rows) — favour simpler models to avoid overfitting")
    elif n_rows < 5000:
        lines.append(f"Moderate dataset ({n_rows:,} rows) — most algorithms viable; watch overfitting")
    else:
        lines.append(f"Large dataset ({n_rows:,} rows) — complex models and deep learning are options")

    return "\n".join(lines)


def select(
    df: pd.DataFrame,
    target_col: str,
    problem_context: str = "",
    dq_findings: list | None = None,
    leakage_findings: list | None = None,
    session=None,
) -> dict:
    try:
        return _select(df, target_col, problem_context, dq_findings, leakage_findings, session)
    except Exception as exc:
        return SkillResult({"error": str(exc)}, skill="model_selector")


def _select(
    df: pd.DataFrame,
    target_col: str,
    problem_context: str = "",
    dq_findings: list | None = None,
    leakage_findings: list | None = None,
    session=None,
) -> dict:
    """
    Recommend model families and check statistical assumptions.
    Returns structured output suitable for the workbench renderer.
    """
    if session is not None:
        if dq_findings is None:
            dq_findings = session.get_findings("data_quality")
        if leakage_findings is None:
            leakage_findings = session.get_findings("leakage_detector")
        if not problem_context:
            problem_context = getattr(session, "context", "")

    session_warnings = session.prerequisite_warnings("model_selector") if session else []

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")

    profile = _build_profile(df, target_col)

    context_block = (
        f"\nBusiness context: {problem_context}\n"
        if problem_context else ""
    )

    dq_block = ""
    if dq_findings:
        critical = [f for f in dq_findings if f.get("severity") == "critical"]
        warnings_ = [f for f in dq_findings if f.get("severity") == "warning"]
        dq_lines = [
            f"  - {f.get('column','?')} [{f.get('issue_type','?')}]: {f.get('description','')}"
            for f in (critical + warnings_)[:6]
        ]
        dq_block = (
            f"\nData quality pre-check ({len(critical)} critical, {len(warnings_)} warnings):\n"
            + "\n".join(dq_lines)
            + "\nFactor these issues into data_readiness and readiness_notes.\n"
        )

    leakage_block = ""
    if leakage_findings:
        critical_ld = [f for f in leakage_findings if f.get("severity") == "critical"]
        warning_ld  = [f for f in leakage_findings if f.get("severity") == "warning"]
        ld_lines = [
            f"  - {f.get('column','?')} [{f.get('leakage_type','?')}]: {f.get('description','')}"
            for f in (critical_ld + warning_ld)[:6]
        ]
        leakage_block = (
            f"\nLeakage scan ({len(critical_ld)} critical, {len(warning_ld)} warnings):\n"
            + "\n".join(ld_lines)
            + "\nExclude critical leakage columns from your recommendations and flag them "
            "explicitly in readiness_notes. Suspicious columns should be noted as caveats.\n"
        )

    _prompt = f"""You are a senior ML engineer advising a data scientist on model selection.
{context_block}{dq_block}{leakage_block}
Data profile:
{profile}

## Your task

1. **Identify the task type** from the target column's distribution (binary_classification,
   multiclass_classification, regression, time_series_forecasting, clustering, anomaly_detection).

2. **Check statistical assumptions** relevant to that task type. For each, cite actual numbers
   from the profile. Cover:
   - Target distribution / class balance (imbalance ratio, skewness, outliers)
   - Feature–target relationship strength (linear vs. non-linear signal)
   - Multicollinearity (any high feature–feature correlations)
   - Data sufficiency (row count vs. model complexity)
   - Missing data patterns (any features with >10% nulls that constrain model choice)

3. **Recommend 3–4 model families**, ranked. For rank 1, include a PySpark MLlib starter
   code snippet (with Databricks in mind). Be specific — cite actual numbers in rationale
   (e.g. "the 5.3:1 imbalance makes gradient boosting preferable to logistic regression
   without resampling").

4. **Assess data readiness**: ready (train immediately), needs_prep (specific transforms
   required first), or insufficient (fundamental gaps).

Ground all reasoning in the actual numbers above, not generic ML advice."""

    # Pass 1: reason with extended thinking
    _r1 = api_call(
        _get_client().messages.create,
        model=ANALYSIS_MODEL,
        max_tokens=12000,
        thinking=THINKING_ADAPTIVE,
        **ANALYSIS_EXTRA,
        messages=[{"role": "user", "content": _prompt}],
    )
    _analysis = next((b.text for b in _r1.content if b.type == "text"), "")
    if not _analysis.strip():
        raise RuntimeError("model_selector Pass 1 returned no text — aborting to avoid empty structured output")

    # Pass 2: format into structured output
    response = api_call(
        _get_client().messages.create,
        model=STRUCTURED_MODEL,
        max_tokens=8192,
        tools=[_TOOL],
        tool_choice={"type": "tool", "name": "report_model_selection"},
        messages=[
            {"role": "user", "content": _prompt},
            {"role": "assistant", "content": _analysis},
            {"role": "user", "content": "Now report your recommendations using the report_model_selection tool."},
        ],
    )

    tool_block = next(b for b in response.content if b.type == "tool_use")
    # Validate Pass 2 didn't return empty — retry once with an explicit prompt if so
    if not tool_block.input.get("recommendations"):
        response = api_call(
            _get_client().messages.create,
            model=STRUCTURED_MODEL,
            max_tokens=8192,
            tools=[_TOOL],
            tool_choice={"type": "tool", "name": "report_model_selection"},
            messages=[
                {"role": "user", "content": _prompt},
                {"role": "assistant", "content": _analysis},
                {"role": "user", "content": (
                    "Your previous response had empty recommendations. "
                    "Please populate all required fields — task_type, target_analysis, "
                    "assumption_checks, recommendations (at least 2–3 model families), "
                    "data_readiness, and summary. Use the report_model_selection tool."
                )},
            ],
        )
        tool_block = next(b for b in response.content if b.type == "tool_use")

    result = SkillResult(tool_block.input, skill="model_selector",
                         session_warnings=session_warnings)
    if session is not None:
        session.register("model_selector", result)
    return result

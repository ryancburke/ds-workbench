"""
Feature Engineering Assistant skill.

Analyses a DataFrame and produces a complete PySpark MLlib preprocessing pipeline:
per-column transformation decisions (imputation, scaling, encoding) plus runnable
PySpark code ready to paste into a Databricks notebook.
"""
import json
import pandas as pd
import numpy as np
from skills.dtypes import is_text
from skills.result import SkillResult, findings_of
from skills.api_retry import api_call, get_client, skill_entry, strict_tool, tool_input
from skills.models import role


_TOOL = {
    "name": "suggest_feature_engineering",
    "description": "Produce a feature engineering plan and PySpark MLlib pipeline code",
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "features": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "column": {"type": "string"},
                        "feature_type": {
                            "type": "string",
                            "enum": ["numeric", "categorical", "high_cardinality", "datetime",
                                     "boolean", "identifier", "numeric_as_object"],
                        },
                        "action": {
                            "type": "string",
                            "description": (
                                "What the pipeline does to this column. encode_frequency: replace each "
                                "value with its frequency count. encode_index: StringIndexer only. "
                                "keep_as_is: the raw value goes into the feature vector unchanged."
                            ),
                            "enum": ["scale", "log_then_scale", "encode_ohe", "encode_index",
                                     "encode_frequency", "extract_datetime", "drop",
                                     "cast_to_int", "keep_as_is"],
                        },
                        "imputation": {
                            "type": "string",
                            "enum": ["mean", "median", "mode", "constant_zero",
                                     "constant_unknown", "none"],
                        },
                        "output_col": {"type": "string"},
                        "rationale": {"type": "string"},
                    },
                    "required": ["column", "feature_type", "action", "imputation",
                                 "output_col", "rationale"],
                },
            },
            "target_handling": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "transform": {
                        "type": "string",
                        "enum": ["cast_to_int", "string_index", "none"],
                    },
                    "output_col": {
                        "type": "string",
                        "description": "Use 'label' for MLlib compatibility",
                    },
                },
                "required": ["column", "transform", "output_col"],
            },
            "dropped_cols": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Every column the plan drops, including identifiers and leakage columns",
            },
            "model_family": {
                "type": "string",
                "enum": ["tree_based", "linear", "either"],
                "description": "Model family the encoding and scaling choices are designed for",
            },
            "imbalance_handling": {
                "type": "string",
                "description": "How the pipeline handles class imbalance (e.g. a weightCol), or why none is needed",
            },
            "pyspark_code": {
                "type": "string",
                "description": "Complete, runnable PySpark MLlib pipeline code",
            },
            "warnings": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["summary", "features", "target_handling", "dropped_cols",
                     "model_family", "imbalance_handling", "pyspark_code", "warnings"],
    },
}

_ID_SUFFIXES = ("_id", "id", "_key", "_ref", "_uuid", "_code")
HIGH_CARDINALITY = 50


def _infer_col_type(col: str, s: pd.Series, n_rows: int) -> str:
    """Classify a column into a feature type."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if s.dtype == bool:
        return "boolean"

    unique_count = s.nunique()
    unique_pct = unique_count / max(n_rows, 1) * 100

    if is_text(s):
        # Check if it's actually numeric stored as text
        numeric_try = pd.to_numeric(s.dropna(), errors="coerce")
        if len(numeric_try) > 0 and numeric_try.notna().mean() > 0.9:
            return "numeric_as_object"
        # High-cardinality string → probably an identifier
        is_id_name = any(col.lower().endswith(sfx) for sfx in _ID_SUFFIXES)
        if unique_pct > 70 or (is_id_name and unique_pct > 30):
            return "identifier"
        # Too many levels to one-hot; a StringIndexer index here has no meaningful order
        if unique_count > HIGH_CARDINALITY:
            return "high_cardinality"
        return "categorical"

    if pd.api.types.is_numeric_dtype(s):
        is_id_name = any(col.lower().endswith(sfx) for sfx in _ID_SUFFIXES)
        if unique_count == n_rows or (is_id_name and unique_pct > 90):
            return "identifier"
        return "numeric"

    return "categorical"


def build_feature_profile(df: pd.DataFrame, target_col: str) -> dict:
    """Compute per-column signals for feature engineering decisions."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")

    target = df[target_col]
    if target.dtype == bool:
        target_numeric = target.astype(float)
    elif target.nunique() == 2 and pd.api.types.is_numeric_dtype(target):
        target_numeric = target.astype(float)
    elif pd.api.types.is_numeric_dtype(target) and target.dtype != bool:
        target_numeric = target.astype(float)
    else:
        target_numeric = None

    profile: dict = {
        "target_col": target_col,
        "target_dtype": str(target.dtype),
        "n_rows": len(df),
        "features": {},
    }
    if target.nunique() <= 20:
        counts = target.value_counts()
        profile["target_classes"] = {str(k): int(v) for k, v in counts.items()}
        if len(counts) >= 2:
            profile["imbalance_ratio"] = round(float(counts.iloc[0] / counts.iloc[-1]), 2)

    for col in df.columns:
        if col == target_col:
            continue
        s = df[col]
        col_type = _infer_col_type(col, s, len(df))
        null_pct = round(float(s.isnull().mean() * 100), 2)
        unique_count = int(s.nunique())
        unique_pct = round(float(unique_count / max(len(df), 1) * 100), 2)

        info: dict = {
            "col_type": col_type,
            "dtype": str(s.dtype),
            "null_pct": null_pct,
            "unique_count": unique_count,
            "unique_pct": unique_pct,
        }

        # Correlation with target for numeric/numeric_as_object columns
        if target_numeric is not None and col_type in ("numeric", "numeric_as_object"):
            s_num = pd.to_numeric(s, errors="coerce") if col_type == "numeric_as_object" else s
            valid = s_num.notna() & target_numeric.notna()
            if valid.sum() > 10:
                try:
                    corr = float(
                        pd.Series(s_num[valid].values, dtype=float).corr(
                            pd.Series(target_numeric[valid].values, dtype=float)
                        )
                    )
                    if not np.isnan(corr):
                        info["correlation_with_target"] = round(abs(corr), 4)
                except Exception:
                    pass

        if col_type in ("numeric", "numeric_as_object"):
            s_num = pd.to_numeric(s, errors="coerce") if col_type == "numeric_as_object" else s
            non_null = s_num.dropna().astype(float)
            if len(non_null):
                info["stats"] = {
                    "min": round(float(non_null.min()), 4),
                    "max": round(float(non_null.max()), 4),
                    "mean": round(float(non_null.mean()), 4),
                    "std": round(float(non_null.std()), 4),
                    "skew": round(float(non_null.skew()), 2) if len(non_null) > 2 else 0.0,
                }
        elif col_type in ("categorical", "high_cardinality", "identifier"):
            vc = s.value_counts(dropna=False).head(8)
            info["sample_values"] = {str(k): int(v) for k, v in vc.items()}

        # Suggested action
        if col_type == "identifier":
            info["suggested_action"] = "drop"
        elif col_type == "boolean":
            info["suggested_action"] = "cast_to_int"
        elif col_type == "datetime":
            info["suggested_action"] = "extract_datetime_or_drop"
        elif col_type in ("numeric", "numeric_as_object"):
            info["suggested_action"] = "scale"
            if null_pct > 0:
                info["suggested_action"] = "impute_then_scale"
        elif col_type == "categorical":
            info["suggested_action"] = "encode_ohe" if unique_count <= 15 else "encode_index_trees_only"
        elif col_type == "high_cardinality":
            info["suggested_action"] = "encode_frequency_or_drop"

        profile["features"][col] = info

    return profile


def _reconcile_drops(plan: dict, leakage_findings: list | None) -> dict:
    """Make dropped_cols agree with the per-column plan, and drop critical leakage columns
    even if the model kept them."""
    dropped = list(dict.fromkeys(plan.get("dropped_cols") or []))
    for f in plan.get("features", []):
        if f.get("action") == "drop" and f.get("column") not in dropped:
            dropped.append(f["column"])
    critical = {f.get("column") for f in (leakage_findings or []) if f.get("severity") == "critical"}
    kept_leaks = [f["column"] for f in plan.get("features", [])
                  if f.get("column") in critical and f.get("action") != "drop"]
    for f in plan.get("features", []):
        if f.get("column") in critical:
            f["action"] = "drop"
            f["rationale"] = "Critical target leakage (from leakage detector)"
    for col in critical:
        if col and col not in dropped:
            dropped.append(col)
    if kept_leaks:
        plan.setdefault("warnings", []).insert(0, (
            f"Leakage columns {kept_leaks} were dropped from the plan; remove them from the "
            "PySpark code before training."))
    plan["dropped_cols"] = dropped
    return plan


@skill_entry("feature_engineering")
def suggest(
    df: pd.DataFrame,
    target_col: str,
    context: str = "",
    leakage_findings: list | None = None,
    session=None,
) -> dict:
    try:
        return _suggest(df, target_col, context, leakage_findings, session)
    except Exception as exc:
        return SkillResult({"error": str(exc)}, skill="feature_engineering")


def _suggest(
    df: pd.DataFrame,
    target_col: str,
    context: str = "",
    leakage_findings: list | None = None,
    session=None,
) -> dict:
    """Generate a feature engineering plan and PySpark pipeline for ML training.

    Args:
        df: Full DataFrame including the target column.
        target_col: Column to predict.
        context: Optional task description (e.g. 'Predicting customer churn').
        leakage_findings: Optional findings from a prior leakage_detector.check() run.
            Populated automatically when a WorkflowSession is provided.
        session: Optional WorkflowSession. Enables prerequisite warnings and
            automatic cross-skill context.

    Returns:
        Dict with per-column plan, PySpark code, dropped columns, and warnings.
    """
    if session is not None:
        if leakage_findings is None:
            leakage_findings = session.get_findings("leakage_detector")
        if not context:
            context = getattr(session, "context", "")

    leakage_findings = findings_of(leakage_findings)  # accepts a prior result or its findings list
    session_warnings = session.prerequisite_warnings("feature_engineering") if session else []

    profile = build_feature_profile(df, target_col)
    context_line = f"\nTask context: {context}\n" if context else ""

    leakage_block = ""
    if leakage_findings:
        critical_ld = [f for f in leakage_findings if f.get("severity") == "critical"]
        warning_ld  = [f for f in leakage_findings if f.get("severity") == "warning"]
        if critical_ld or warning_ld:
            ld_lines = [
                f"  - {f.get('column','?')} [{f.get('leakage_type','?')}]: {f.get('description','')}"
                for f in (critical_ld + warning_ld)[:6]
            ]
            leakage_block = (
                f"\nLEAKAGE PRE-SCAN RESULTS ({len(critical_ld)} critical, {len(warning_ld)} warnings):\n"
                + "\n".join(ld_lines)
                + "\nCritical leakage columns are dropped automatically and must not feed derived "
                "features. Drop warning-level columns too unless you can justify keeping one.\n"
            )

    response = api_call(
        get_client().messages.create,
        **role("general"),
        # thinking + a full pipeline script; 8192 truncated the tool input on fraud data.
        # Kept under ~21k so the SDK still allows a non-streaming request.
        max_tokens=20000,
        tools=[strict_tool(_TOOL)],
        tool_choice={"type": "auto"},
        messages=[{
            "role": "user",
            "content": f"""You are a senior ML engineer building a PySpark feature engineering pipeline
for a Databricks notebook. The target column is '{target_col}'.
{context_line}{leakage_block}
Feature profile (auto-computed from dataset; suggested_action is a starting heuristic, not a rule):
{json.dumps(profile, indent=2)}

Produce a feature plan and runnable PySpark MLlib code that a data scientist could train on today.
Decide which model family the pipeline is for (tree-based, linear, or either) and make the
encoding and scaling choices consistent with it, because they differ:
- A StringIndexer index has no real order. Linear models read it as a magnitude, so for them use
  one-hot for low-cardinality columns and frequency encoding (or drop) for high-cardinality ones.
  MLlib tree models can treat an indexed column as categorical only if maxBins is at least its
  cardinality and no scaler or OneHotEncoder sits between them.
- Identifier columns carry row identity, not signal: drop them. High-cardinality keys such as
  user or merchant IDs can carry real behaviour; if you keep one, use a frequency count rather
  than a raw index, and say why.
- Heavily skewed positive numerics (|skew| > 2) are usually better as log1p before scaling.
  Trees don't need scaling at all.
- Hour and day-of-week extracted from timestamps wrap around; encode them as sin/cos pairs for
  linear models.
- If the target is imbalanced (see imbalance_ratio), handle it in the pipeline, e.g. a class-weight
  column passed as weightCol, and say how in imbalance_handling.
- Impute before encoding or scaling, and fit every stage inside the Pipeline so nothing is learned
  from validation data.

dropped_cols must list every column the plan or code drops. Cast a boolean target to an integer
column named "label" (MLlib convention).

The pyspark_code must be complete and copy-paste ready: imports from pyspark.ml, pyspark.ml.feature
and pyspark.sql.functions, a Pipeline whose final VectorAssembler writes outputCol="features", and
one-line comments for each stage group.

Report your results by calling the suggest_feature_engineering tool.""",
        }],
    )

    tool_args = _reconcile_drops(tool_input(response, _TOOL["name"]), leakage_findings)
    result = SkillResult(tool_args, skill="feature_engineering",
                         session_warnings=session_warnings)
    if session is not None:
        session.register("feature_engineering", result)
    return result

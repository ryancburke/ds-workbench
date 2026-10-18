"""
Feature Engineering Assistant skill.

Analyses a DataFrame and produces a complete PySpark MLlib preprocessing pipeline:
per-column transformation decisions (imputation, scaling, encoding) plus runnable
PySpark code ready to paste into a Databricks notebook.
"""
import json
import anthropic
import pandas as pd
import numpy as np
from config import config
from skills.result import SkillResult
from skills.api_retry import api_call
from skills.models import GENERAL_MODEL, GENERAL_EXTRA

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=config.anthropic_api_key)
    return _client


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
                            "enum": ["numeric", "categorical", "datetime", "boolean",
                                     "identifier", "numeric_as_object"],
                        },
                        "action": {
                            "type": "string",
                            "enum": ["scale", "encode_ohe", "encode_index",
                                     "extract_datetime", "drop", "cast_to_int", "keep_as_is"],
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
        "required": ["summary", "features", "target_handling",
                     "dropped_cols", "pyspark_code", "warnings"],
    },
}

_ID_SUFFIXES = ("_id", "id", "_key", "_ref", "_uuid", "_code")


def _infer_col_type(col: str, s: pd.Series, n_rows: int) -> str:
    """Classify a column into a feature type."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if s.dtype == bool:
        return "boolean"

    unique_count = s.nunique()
    unique_pct = unique_count / max(n_rows, 1) * 100

    if s.dtype == object:
        # Check if it's actually numeric stored as object
        numeric_try = pd.to_numeric(s.dropna(), errors="coerce")
        if len(numeric_try) > 0 and numeric_try.notna().mean() > 0.9:
            return "numeric_as_object"
        # High-cardinality string → probably an identifier
        is_id_name = any(col.lower().endswith(sfx) for sfx in _ID_SUFFIXES)
        if unique_pct > 70 or (is_id_name and unique_pct > 30):
            return "identifier"
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
                }
        elif col_type in ("categorical", "identifier"):
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
            info["suggested_action"] = "encode_ohe" if unique_count <= 15 else "encode_index"

        profile["features"][col] = info

    return profile


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
                + "\nCRITICAL leakage columns MUST appear in dropped_cols. "
                "Do NOT create derived features from them. "
                "Warning-level columns should also be in dropped_cols unless you have a specific justification.\n"
            )

    response = api_call(
        _get_client().messages.create,
        model=GENERAL_MODEL,
        **GENERAL_EXTRA,
        max_tokens=8192,
        tools=[_TOOL],
        tool_choice={"type": "auto"},
        messages=[{
            "role": "user",
            "content": f"""You are a senior ML engineer building a PySpark feature engineering pipeline
for a Databricks notebook. The target column is '{target_col}'.
{context_line}{leakage_block}
Feature profile (auto-computed from dataset):
{json.dumps(profile, indent=2)}

Produce a complete feature engineering plan AND runnable PySpark MLlib code.

TRANSFORMATION RULES:
- identifier columns (unique IDs, sequential keys): drop — they leak row identity, not signal
- numeric columns: StandardScaler (impute median first if null_pct > 0)
- numeric_as_object columns (dtype=object but values are numbers): cast to float, then impute + scale
- categorical columns with unique_count ≤ 15: StringIndexer → OneHotEncoder
- categorical columns with unique_count > 15: StringIndexer only (OHE would be too sparse)
- categorical columns with nulls: impute with constant "unknown" or most-frequent before encoding
- datetime columns: extract day_of_week and hour features, or drop if sessions are independent events
- boolean non-target columns: cast to integer
- boolean target: cast to integer as "label" (MLlib convention)

PYSPARK CODE REQUIREMENTS:
The pyspark_code field must be a complete, copy-paste-ready script using these exact APIs:
  from pyspark.ml import Pipeline
  from pyspark.ml.feature import Imputer, StandardScaler, VectorAssembler, StringIndexer, OneHotEncoder
  from pyspark.sql import functions as F

Structure the code in this order:
  1. Drop identifier columns (df.drop(...))
  2. Handle target: df.withColumn("label", F.col("{target_col}").cast("integer")).drop("{target_col}")
  3. Handle object-dtype numeric columns: F.col(...).cast("double")
  4. Imputer stage (if any nulls exist)
  5. StringIndexer stages (one per categorical column)
  6. OneHotEncoder stage (for low-cardinality categoricals)
  7. VectorAssembler for numeric features → StandardScaler
  8. Final VectorAssembler combining scaled numeric + encoded categoricals → outputCol="features"
  9. pipeline = Pipeline(stages=[...])

Use clear variable names. Add one-line comments for each stage group.""",
        }],
    )

    tool_block = next(b for b in response.content if b.type == "tool_use")
    result = SkillResult(tool_block.input, skill="feature_engineering",
                         session_warnings=session_warnings)
    if session is not None:
        session.register("feature_engineering", result)
    return result

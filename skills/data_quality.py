import json
import pandas as pd
import numpy as np
from skills.result import SkillResult
from skills.api_retry import api_call, get_client, skill_entry, tool_input
from skills.models import REASONING_MODEL, STRUCTURED_MODEL, THINKING_ADAPTIVE, REASONING_EXTRA


_TOOL = {
    "name": "report_data_quality",
    "description": "Report structured data quality findings for a dataset",
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "overall_score": {"type": "integer", "minimum": 0, "maximum": 100},
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "column": {"type": "string"},
                        "issue_type": {
                            "type": "string",
                            "enum": [
                                "null_rate", "duplicate_rows", "inconsistent_format",
                                "outlier", "low_cardinality", "high_cardinality",
                                "type_mismatch", "referential_integrity", "schema_issue",
                            ],
                        },
                        "severity": {"type": "string", "enum": ["critical", "warning", "info"]},
                        "description": {"type": "string"},
                        "recommendation": {"type": "string"},
                        "affected_rows_pct": {"type": "number"},
                    },
                    "required": ["column", "issue_type", "severity", "description", "recommendation"],
                },
            },
        },
        "required": ["summary", "overall_score", "findings"],
    },
}


def build_profile(df: pd.DataFrame) -> dict:
    profile: dict = {
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "duplicate_rows": int(df.duplicated().sum()),
        "columns": {},
    }
    for col in df.columns:
        s = df[col]
        info: dict = {
            "dtype": str(s.dtype),
            "null_count": int(s.isnull().sum()),
            "null_pct": round(float(s.isnull().mean() * 100), 2),
            "unique_count": int(s.nunique()),
            "unique_pct": round(float(s.nunique() / max(len(df), 1) * 100), 2),
        }
        if pd.api.types.is_numeric_dtype(s) and s.dtype != bool:
            non_null = s.dropna().astype(float)
            if len(non_null):
                q1, q3 = float(non_null.quantile(0.25)), float(non_null.quantile(0.75))
                iqr = q3 - q1
                info["stats"] = {
                    "min": round(float(non_null.min()), 4),
                    "max": round(float(non_null.max()), 4),
                    "mean": round(float(non_null.mean()), 4),
                    "std": round(float(non_null.std()), 4),
                    "p25": round(q1, 4),
                    "p75": round(q3, 4),
                    "outlier_count": int(
                        ((non_null < q1 - 3 * iqr) | (non_null > q3 + 3 * iqr)).sum()
                    ),
                }
        else:
            vc = s.value_counts(dropna=False).head(10)
            info["top_values"] = {str(k): int(v) for k, v in vc.items()}
        profile["columns"][col] = info
    return profile


def _maybe_sample(df: pd.DataFrame, n: int = 50_000) -> tuple[pd.DataFrame, str | None]:
    if len(df) <= n:
        return df, None
    sampled = df.sample(n, random_state=42).reset_index(drop=True)
    return sampled, f"Sampled {n:,} of {len(df):,} rows for analysis"


@skill_entry("data_quality")
def assess(df: pd.DataFrame, table_name: str = "dataset", session=None) -> dict:
    try:
        return _assess(df, table_name, session)
    except Exception as exc:
        return SkillResult({"error": str(exc)}, skill="data_quality")


def _assess(df: pd.DataFrame, table_name: str = "dataset", session=None) -> dict:
    df, sample_note = _maybe_sample(df)
    profile = build_profile(df)
    n_rows = profile["shape"]["rows"]
    col_names = list(profile["columns"].keys())

    _prompt = f"""You are a senior data engineer reviewing dataset '{table_name}' before ML training.

Profile:
{json.dumps(profile, indent=2)}

Work through each step below. For every candidate you find, reason about whether it is a genuine
quality problem — don't just list it. Think through the domain implications.

STEP 1 — NULL RATES
Check null_pct for every column. For each column where null_pct > 5.0:
  • State the exact rate.
  • Reason: is this on a critical column (identifier, join key, widely-used feature) or peripheral?
  • Severity: >20% on a key column → critical; 5–20% → usually warning;
    very peripheral features at 5–8% may warrant only info.

STEP 2 — DUPLICATE ROWS
Read the top-level "duplicate_rows" field. State the count and its percentage of {n_rows} rows.
If > 0, assess the training impact: exact duplicates bias model weights toward over-represented patterns.

STEP 3 — FORMAT CONSISTENCY
For every string/object column, inspect top_values.
Flag columns where:
  (a) The same concept appears with different casing in that column's own values
      (e.g. "completed" AND "Completed" AND "COMPLETED" all in the same column), OR
  (b) Different values each use a different casing convention, meaning the column has no consistent
      style at all (e.g. "NORTH"=ALL CAPS, "South"=Title Case, "West"=Title Case, "east"=lowercase —
      four different conventions in one column).
Do NOT flag: all-lowercase different words ("online", "retail", "wholesale"),
all-Title-Case sets, datetime/numeric/boolean columns.
For each flagged column, reason: will this cause downstream encoding inconsistency?

STEP 4 — NUMERIC OUTLIERS
For every numeric column where outlier_count > 0 (pre-computed via 3×IQR fence),
work through all three questions:
  Q1 Rate: outlier_count / {n_rows} rows — is this above 1%?
  Q2 Volume: is outlier_count ≥ 10?
  Q3 Plausibility: are the extreme min/max values realistic for what this column measures?
     Think about the column's domain. A salary of 750000 when mean=85000 (≈30σ) is impossible in
     a normal HR dataset. A holiday revenue spike of 3× the mean is plausible.
     An age of -5 or 150 is impossible. A price of $0 in a paid-only service is suspicious.
Severity: warning if any of Q1, Q2, Q3 triggers; info if all three say "probably fine".

STEP 5 — IDENTIFIER INTEGRITY
Find columns whose name suggests an identifier (contains 'id', 'key', 'code', 'ref', 'uuid').
Flag any where unique_pct < 95% — duplicate identifiers indicate referential integrity problems.

STEP 6 — TYPE MISMATCHES
Find object/string columns where ALL top_values consist only of digits (no letters).
These may be numeric columns stored as strings. Reason: identifier stored as text (acceptable)
or numeric feature that will cause errors unless cast?

CALIBRATION — anchor your severity reasoning against these known cases:

FORMAT:
  channel: {{"online": 412, "retail": 301, "wholesale": 287}}
    → All lowercase, different words. Consistent style. No finding.
  status: {{"completed": 180, "Completed": 95, "COMPLETED": 23}}
    → Same concept, three casing variants. inconsistent_format, severity=warning.
  region: {{"NORTH": 215, "South": 197, "West": 203, "east": 185}}
    → No consistent convention across values. inconsistent_format, severity=warning.
  region: {{"North": 250, "South": 250, "East": 250, "West": 250}}
    → All Title Case. Consistent. No finding.

OUTLIERS:
  revenue: 3 outliers / 1000 rows (0.3%), max=12000, mean=4200
    → Low rate, low volume, plausible business extreme. info at most.
  salary: 7 outliers / 500 rows (1.4%), min=5000, max=750000, mean=85000, std=22000
    → Q1 triggers (1.4% > 1%). Q3 triggers (750000 ≈ 30σ above mean — impossible). warning.
  price: 47 outliers, min=0.0, mean=45.2
    → Q2 triggers (47 ≥ 10). Q3 triggers ($0 in a paid service is suspicious). warning.

FINAL ASSESSMENT
After all six steps:
1. Write a 1–2 sentence overall verdict.
2. List confirmed findings in this format:
   column | issue_type | severity | brief justification
3. State an overall_score out of 100.
   Scoring: start at 100; deduct 15–25 per critical finding, 5–10 per warning, 1–3 per info.
   Multiple criticals can push the score below 50."""

    # Pass 1: systematic sweep + domain reasoning with extended thinking
    pass1 = api_call(
        get_client().messages.create,
        model=REASONING_MODEL,
        max_tokens=14000,
        thinking=THINKING_ADAPTIVE,
        **REASONING_EXTRA,
        messages=[{"role": "user", "content": _prompt}],
    )
    analysis = next((b.text for b in pass1.content if b.type == "text"), "")
    if not analysis.strip():
        raise RuntimeError("data_quality Pass 1 returned no text — aborting to avoid empty structured output")

    # Pass 2: extract and format the reasoned analysis into structured output
    response = api_call(
        get_client().messages.create,
        model=STRUCTURED_MODEL,
        max_tokens=4096,
        tools=[_TOOL],
        tool_choice={"type": "tool", "name": "report_data_quality"},
        messages=[{
            "role": "user",
            "content": (
                f"Format this data quality analysis for '{table_name}' into structured output.\n\n"
                f"Analysis:\n{analysis}\n\n"
                "Formatting rules:\n"
                f"- Use ONLY column names from this list: {col_names}\n"
                f"- For duplicate_rows findings, use '{table_name}' as the column field\n"
                "- Match severity exactly to what the analysis concluded\n"
                "- Include affected_rows_pct wherever the analysis provides a percentage\n"
                "- The overall_score must match what the analysis calculated\n"
                "- Format only what the analysis identified — do not add or remove findings"
            ),
        }],
    )
    tool_args = tool_input(response, _TOOL["name"])
    result = SkillResult(tool_args, skill="data_quality")
    result["profile"] = profile
    if sample_note:
        result["sample_note"] = sample_note
    if session is not None:
        session.register("data_quality", result)
    return result

import json
import anthropic
import pandas as pd
from config import config
from skills.api_retry import api_call, skill_entry
from skills.models import ANALYSIS_MODEL, STRUCTURED_MODEL, THINKING_ADAPTIVE, ANALYSIS_EXTRA

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=config.anthropic_api_key)
    return _client


_TOOL = {
    "name": "report_join_audit",
    "description": "Report join quality findings",
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "overall_verdict": {
                "type": "string",
                "enum": ["safe", "review_needed", "dangerous"],
            },
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "issue_type": {
                            "type": "string",
                            "enum": [
                                "fan_out", "unmatched_keys", "many_to_many",
                                "null_keys", "duplicate_keys", "type_mismatch", "safe",
                            ],
                        },
                        "severity": {"type": "string", "enum": ["critical", "warning", "info"]},
                        "description": {"type": "string"},
                        "recommendation": {"type": "string"},
                    },
                    "required": ["issue_type", "severity", "description", "recommendation"],
                },
            },
        },
        "required": ["summary", "overall_verdict", "findings"],
    },
}


@skill_entry("join_auditor")
def audit(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    join_keys: list[str],
    join_type: str = "inner",
    left_name: str = "left",
    right_name: str = "right",
    session=None,
) -> dict:
    try:
        return _audit(left_df, right_df, join_keys, join_type, left_name, right_name, session)
    except Exception as exc:
        from skills.result import SkillResult
        return SkillResult({"error": str(exc)}, skill="join_auditor")


def _audit(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    join_keys: list[str],
    join_type: str = "inner",
    left_name: str = "left",
    right_name: str = "right",
    session=None,
) -> dict:
    try:
        result_df = left_df.merge(right_df, on=join_keys, how=join_type)
        result_rows = len(result_df)
    except Exception:
        result_rows = -1

    left_keys_set = set(map(tuple, left_df[join_keys].dropna().values.tolist()))
    right_keys_set = set(map(tuple, right_df[join_keys].dropna().values.tolist()))

    stats = {
        "left_name": left_name,
        "right_name": right_name,
        "join_type": join_type,
        "join_keys": join_keys,
        "left_rows": len(left_df),
        "right_rows": len(right_df),
        "result_rows": result_rows,
        "left_key_null_rows": int(left_df[join_keys].isnull().any(axis=1).sum()),
        "right_key_null_rows": int(right_df[join_keys].isnull().any(axis=1).sum()),
        "left_key_duplicates": int(left_df.duplicated(subset=join_keys).sum()),
        "right_key_duplicates": int(right_df.duplicated(subset=join_keys).sum()),
        "unmatched_keys_in_left": len(left_keys_set - right_keys_set),
        "unmatched_keys_in_right": len(right_keys_set - left_keys_set),
    }

    _prompt = f"""You are a senior data engineer auditing a join operation.

Join statistics:
{json.dumps(stats, indent=2)}

Assess for:
1. Fan-out (result rows >> left rows — indicates many-to-many or duplicate keys on right)
2. Silent data loss (many unmatched keys, especially in left/right joins)
3. Null keys that will be silently dropped
4. Many-to-many relationships that inflate aggregations

Be precise. A fan-out factor above 1.1 is a warning; above 2.0 is critical.

For every finding, the `recommendation` field must contain a concrete, copy-paste-ready fix —
not generic advice. Examples of good recommendations:
- fan_out: "Deduplicate {right_name} on the join key before joining: `{right_name}.drop_duplicates(subset={join_keys})`"
- unmatched_keys: "Inspect dropped rows before the join: `{left_name}[~{left_name}[key].isin({right_name}[key])]` — decide whether to LEFT JOIN or investigate source"
- null_keys: "Drop null join keys before joining: `df = df.dropna(subset={join_keys})`"
- many_to_many: "Aggregate one side to unique keys first: `{right_name}.groupby(key).agg(...).reset_index()`"
Reference the actual table names ({left_name}, {right_name}) and key columns ({join_keys}) in every recommendation."""

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
        raise RuntimeError("join_auditor Pass 1 returned no text — aborting to avoid empty structured output")

    # Pass 2: format into structured output
    response = api_call(
        _get_client().messages.create,
        model=STRUCTURED_MODEL,
        max_tokens=2048,
        tools=[_TOOL],
        tool_choice={"type": "tool", "name": "report_join_audit"},
        messages=[
            {"role": "user", "content": _prompt},
            {"role": "assistant", "content": _analysis},
            {"role": "user", "content": "Now report your findings using the report_join_audit tool."},
        ],
    )
    tool_block = next(b for b in response.content if b.type == "tool_use")
    result = tool_block.input
    result["row_count_change"] = {
        "left_rows": len(left_df),
        "right_rows": len(right_df),
        "result_rows": result_rows,
        "fan_out_factor": round(result_rows / len(left_df), 2) if len(left_df) > 0 and result_rows > 0 else 1.0,
    }
    from skills.result import SkillResult
    sr = SkillResult(result, skill="join_auditor")
    if session is not None:
        session.register("join_auditor", sr)
    return sr

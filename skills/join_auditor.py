import json
import pandas as pd
from ds_workbench.core.joins import join_stats
from skills.api_retry import api_call, get_client, skill_entry, tool_input
from skills.models import THINKING_ADAPTIVE, role


_TOOL = {
    "name": "report_join_audit",
    "description": "Report join quality findings",
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
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
        "required": ["summary", "findings"],
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
    # Exact counts from per-key counts; the join itself is never built (a many-to-many join
    # can be far larger than either table)
    stats = join_stats(left_df, right_df, join_keys, join_type, left_name, right_name)
    verdict, reasons = stats.pop("verdict"), stats.pop("verdict_reasons")

    _prompt = f"""You are a senior data engineer auditing a join operation.

Join statistics (exact; null keys never match, as in SQL/Spark):
{json.dumps(stats, indent=2)}

Verdict (set from the statistics): {verdict}{" — " + "; ".join(reasons) if reasons else ""}

The verdict is fixed by the numbers above. Your job is to explain what is wrong and how to
fix it. Report each problem the statistics show:
1. Fan-out: fan_out_factor is rows produced per matched left row; above 1 means duplicate keys
   on the right multiply rows (max_rows_from_one_key shows the worst key)
2. Silent data loss: left_rows_dropped / unmatched keys, especially in inner and right joins
3. Null keys, which never match and are dropped by inner joins
4. Many-to-many keys, which inflate any aggregation done after the join
Make severities consistent with the verdict: critical for what makes it dangerous, warning for
what makes it need review.

For every finding, the `recommendation` field must contain a concrete, copy-paste-ready fix —
not generic advice. Examples of good recommendations:
- fan_out: "Deduplicate {right_name} on the join key before joining: `{right_name}.drop_duplicates(subset={join_keys})`"
- unmatched_keys: "Inspect dropped rows before the join: `{left_name}[~{left_name}[key].isin({right_name}[key])]` — decide whether to LEFT JOIN or investigate source"
- null_keys: "Drop null join keys before joining: `df = df.dropna(subset={join_keys})`"
- many_to_many: "Aggregate one side to unique keys first: `{right_name}.groupby(key).agg(...).reset_index()`"
Reference the actual table names ({left_name}, {right_name}) and key columns ({join_keys}) in every recommendation."""

    # Pass 1: reason with extended thinking
    _r1 = api_call(
        get_client().messages.create,
        **role("analysis"),
        max_tokens=12000,
        thinking=THINKING_ADAPTIVE,
        messages=[{"role": "user", "content": _prompt}],
    )
    _analysis = next((b.text for b in _r1.content if b.type == "text"), "")
    if not _analysis.strip():
        raise RuntimeError("join_auditor Pass 1 returned no text — aborting to avoid empty structured output")

    # Pass 2: format into structured output
    response = api_call(
        get_client().messages.create,
        **role("structured"),
        max_tokens=2048,
        tools=[_TOOL],
        tool_choice={"type": "tool", "name": "report_join_audit"},
        messages=[
            {"role": "user", "content": _prompt},
            {"role": "assistant", "content": _analysis},
            {"role": "user", "content": "Now report your findings using the report_join_audit tool."},
        ],
    )
    result = tool_input(response, _TOOL["name"])
    result["overall_verdict"] = verdict
    result["verdict_reasons"] = reasons
    result["row_count_change"] = {k: stats[k] for k in (
        "left_rows", "right_rows", "result_rows", "fan_out_factor", "row_ratio",
        "left_rows_dropped", "left_rows_dropped_pct", "many_to_many_keys")}
    from skills.result import SkillResult
    sr = SkillResult(result, skill="join_auditor")
    if session is not None:
        session.register("join_auditor", sr)
    return sr

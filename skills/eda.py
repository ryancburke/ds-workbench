import json
import pandas as pd
from skills.api_retry import api_call, get_client, skill_entry, tool_input
from skills.models import role


_TOOL = {
    "name": "report_eda",
    "description": "Report structured EDA findings and chart recommendations",
    "input_schema": {
        "type": "object",
        "properties": {
            "narrative": {"type": "string"},
            "key_insights": {"type": "array", "items": {"type": "string"}},
            "recommended_charts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "chart_type": {
                            "type": "string",
                            "enum": ["histogram", "scatter", "bar", "box", "heatmap", "violin"],
                        },
                        "x_column": {"type": "string"},
                        "y_column": {"type": "string"},
                        "title": {"type": "string"},
                        "rationale": {"type": "string"},
                    },
                    "required": ["chart_type", "title", "rationale"],
                },
            },
            "potential_targets": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "column": {"type": "string"},
                        "task_type": {"type": "string", "enum": ["classification", "regression", "clustering"]},
                        "rationale": {"type": "string"},
                    },
                    "required": ["column", "task_type", "rationale"],
                },
            },
        },
        "required": ["narrative", "key_insights", "recommended_charts"],
    },
}


_CHART_TYPES = set(_TOOL["input_schema"]["properties"]["recommended_charts"]["items"]
                   ["properties"]["chart_type"]["enum"])


def _valid_charts(charts) -> list[dict]:
    """Keep well-formed chart recommendations (the tool isn't strict), at most 8."""
    if not isinstance(charts, list):
        return []
    return [c for c in charts if isinstance(c, dict) and c.get("chart_type") in _CHART_TYPES][:8]


@skill_entry("eda")
def analyze(
    df: pd.DataFrame,
    table_name: str = "dataset",
    context: str = "",
    session=None,
) -> dict:
    try:
        return _analyze(df, table_name, context, session)
    except Exception as exc:
        from skills.result import SkillResult
        return SkillResult({"error": str(exc)}, skill="eda")


def _analyze(
    df: pd.DataFrame,
    table_name: str = "dataset",
    context: str = "",
    session=None,
) -> dict:
    """Run EDA on a DataFrame and return structured findings.

    Args:
        df: The DataFrame to analyse.
        table_name: Human-readable name shown in output headers.
        context: Plain-English description of what you're investigating
            (e.g. "Q3 revenue shortfall by region", "customer churn drivers").
            When provided, findings are prioritised around this question rather
            than reporting generic statistical patterns.
            Populated automatically from WorkflowSession.context when a
            session is passed and no explicit value is given.
        session: Optional WorkflowSession for automatic context and registration.
    """
    if session is not None and not context:
        context = getattr(session, "context", "")

    from skills.data_quality import build_profile
    profile = build_profile(df)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    high_correlations: dict = {}
    if len(numeric_cols) > 1:
        corr_df = df[numeric_cols].corr().round(3)
        for i, c1 in enumerate(numeric_cols):
            for c2 in numeric_cols[i + 1:]:
                val = float(corr_df.loc[c1, c2])
                if abs(val) > 0.5:
                    high_correlations[f"{c1} vs {c2}"] = val

    context_block = ""
    if context:
        context_block = (
            f"\nBusiness question / investigation focus: {context}\n\n"
            "Prioritise this question above generic statistical observations. "
            "Lead key_insights with findings that directly explain, confirm, or contradict it. "
            "Recommend charts that illuminate this specific question first.\n"
        )

    response = api_call(
        get_client().messages.create,
        **role("general"),
        # thinking on messy tables can be long (12000 cut off on orders_dirty); kept under
        # ~21k so the SDK allows a non-streaming request
        max_tokens=20000,
        # Not strict: with strict decoding this schema (a list of objects with optional fields)
        # ran away on orders_dirty in 2 of 4 runs — to max_tokens, or a 1,305-item list.
        # Non-strict was 4/4 clean; the charts are validated below instead.
        tools=[_TOOL],
        tool_choice={"type": "auto"},
        messages=[{
            "role": "user",
            "content": f"""You are an expert data scientist performing EDA on '{table_name}'.
{context_block}
Data profile:
{json.dumps(profile, indent=2)}

High correlations (|r| > 0.5):
{json.dumps(high_correlations, indent=2) if high_correlations else "None detected"}

Identify patterns, anomalies, and relationships. Be precise about which columns to use in chart recommendations.
Recommend 3-6 charts that would show these patterns.

Report your results by calling the report_eda tool.""",
        }],
    )
    tool_args = tool_input(response, _TOOL["name"])
    tool_args["recommended_charts"] = _valid_charts(tool_args.get("recommended_charts"))
    from skills.result import SkillResult
    result = SkillResult(tool_args, skill="eda")
    if session is not None:
        session.register("eda", result)
    return result

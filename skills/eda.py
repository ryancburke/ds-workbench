import json
import anthropic
import pandas as pd
from config import config
from skills.models import GENERAL_MODEL

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=config.anthropic_api_key)
    return _client


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

    response = _get_client().messages.create(
        model=GENERAL_MODEL,
        max_tokens=4096,
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

Identify patterns, anomalies, and relationships. Be precise about which columns to use in chart recommendations.""",
        }],
    )
    tool_block = next(b for b in response.content if b.type == "tool_use")
    from skills.result import SkillResult
    result = SkillResult(tool_block.input, skill="eda")
    if session is not None:
        session.register("eda", result)
    return result

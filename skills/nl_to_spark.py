import pandas as pd
from skills.api_retry import api_call, get_client, skill_entry, strict_tool, tool_input
from skills.models import GENERAL_MODEL, GENERAL_EXTRA


_TOOL = {
    "name": "generate_query",
    "description": "Generate SQL and PySpark from a natural language question",
    "input_schema": {
        "type": "object",
        "properties": {
            "sql_query": {"type": "string"},
            "pyspark_code": {"type": "string"},
            "explanation": {"type": "string"},
            "assumptions": {"type": "array", "items": {"type": "string"}},
            "performance_notes": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["sql_query", "pyspark_code", "explanation"],
    },
}


@skill_entry("nl_to_spark")
def generate(question: str, tables: dict[str, pd.DataFrame], context: str = "") -> dict:
    try:
        return _generate(question, tables, context)
    except Exception as exc:
        from skills.result import SkillResult
        return SkillResult({"error": str(exc)}, skill="nl_to_spark")


def _generate(question: str, tables: dict[str, pd.DataFrame], context: str = "") -> dict:
    schema_lines = []
    for name, df in tables.items():
        cols = "\n".join(f"    {c}  {dtype}" for c, dtype in df.dtypes.items())
        schema_lines.append(f"Table: {name}\n{cols}")

    context_block = f"\nBusiness context: {context}\n" if context else ""

    response = api_call(
        get_client().messages.create,
        model=GENERAL_MODEL,
        **GENERAL_EXTRA,
        max_tokens=12000,  # headroom for thinking; a cut-off reply is an error
        tools=[strict_tool(_TOOL)],
        tool_choice={"type": "auto"},
        messages=[{
            "role": "user",
            "content": f"""You are an expert in Databricks SQL and PySpark.

Available tables:
{chr(10).join(schema_lines)}
{context_block}
User question: {question}

Generate both Spark SQL and equivalent PySpark DataFrame API code.
Use Delta Lake features where applicable. Avoid unnecessary shuffles.
Prefer predicate pushdown. Flag any ambiguities as assumptions.
When the query returns metric columns (rates, counts, scores, revenue, amounts), include ORDER BY <metric> DESC. Note the sort rationale in performance_notes.

Report your results by calling the generate_query tool.""",
        }],
    )
    tool_args = tool_input(response, _TOOL["name"])
    from skills.result import SkillResult
    return SkillResult(tool_args, skill="nl_to_spark")

import anthropic
from config import config
from skills.result import SkillResult
from skills.models import GENERAL_MODEL

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=config.anthropic_api_key)
    return _client


_TOOL = {
    "name": "report_mlops_review",
    "description": "Report production readiness findings for notebook code",
    "input_schema": {
        "type": "object",
        "properties": {
            "feasibility": {
                "type": "object",
                "description": "Data and feature sufficiency assessment — answer this first",
                "properties": {
                    "verdict": {
                        "type": "string",
                        "enum": ["ready", "marginal", "insufficient"],
                        "description": "ready=data+features sufficient for production model; marginal=buildable but gaps exist; insufficient=missing critical data or features",
                    },
                    "assessment": {
                        "type": "string",
                        "description": "1-2 sentence direct answer to: can we build this model with the data described?",
                    },
                    "key_gaps": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific data or feature gaps blocking or limiting model quality. Empty if verdict is ready.",
                    },
                },
                "required": ["verdict", "assessment", "key_gaps"],
            },
            "summary": {"type": "string"},
            "readiness_score": {"type": "integer", "minimum": 0, "maximum": 100},
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "enum": [
                                "data_leakage", "hardcoded_config", "missing_logging",
                                "no_error_handling", "inefficient_spark", "reproducibility",
                                "missing_tests", "code_structure", "security", "documentation",
                            ],
                        },
                        "severity": {"type": "string", "enum": ["critical", "warning", "suggestion"]},
                        "line_reference": {"type": "string"},
                        "description": {"type": "string"},
                        "fix": {"type": "string"},
                    },
                    "required": ["category", "severity", "description", "fix"],
                },
            },
            "refactored_snippet": {"type": "string"},
        },
        "required": ["feasibility", "summary", "readiness_score", "findings"],
    },
}


def review(code: str) -> dict:
    try:
        return _review(code)
    except Exception as exc:
        return SkillResult({"error": str(exc)}, skill="mlops_reviewer")


def _review(code: str) -> dict:
    response = _get_client().messages.create(
        model=GENERAL_MODEL,
        max_tokens=4096,
        tools=[_TOOL],
        tool_choice={"type": "auto"},
        messages=[{
            "role": "user",
            "content": f"""You are an MLOps engineer reviewing notebook code before it goes to production.

Code to review:
```python
{code}
```

## Step 1 — Feasibility (answer this first)

Before anything else: assess whether the data and features described in this code are sufficient \
to build a reliable production model. Specifically:
- Does the code suggest adequate labeled training data exists (volume, recency, class balance)?
- Are the features described complete enough for the stated prediction task?
- Is there anything about the data pipeline that would prevent this model from being built or trusted?

Set verdict to "ready", "marginal", or "insufficient" and state the single most critical gap if any.

## Step 2 — Code review

Check for the following. Classify each finding as critical (must fix before prod), \
warning (should fix), or suggestion (nice to have):

1. Hardcoded paths, credentials, dates, thresholds (should be params/configs) — usually critical
2. Missing logging — should log data shapes, key metrics, start/end — usually warning
3. No error handling or brittle assumptions — severity depends on blast radius
4. Inefficient Spark: .collect() on large data, row-wise iterations, missing partitioning hints
5. Reproducibility: missing random seeds, timestamp dependencies
6. Monolithic code that should be extracted into testable functions
7. Security: credentials in code, overly broad file permissions

Reference actual code snippets. Provide concrete fixes, not vague advice.
If there is one clear quick win, include a refactored_snippet showing it.""",
        }],
    )
    tool_block = next(b for b in response.content if b.type == "tool_use")
    return SkillResult(tool_block.input, skill="mlops_reviewer")

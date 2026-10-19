import json
from skills.result import SkillResult
from skills.api_retry import api_call, get_client, skill_entry, strict_tool, tool_input
from skills.models import role


_TOOL = {
    "name": "interpret_experiment",
    "description": "Interpret model training results and recommend next steps",
    "input_schema": {
        "type": "object",
        "required": ["verdict", "interpretation", "findings", "next_steps"],
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["promising", "needs_work", "red_flags"],
                "description": (
                    "Overall assessment: promising=good results worth continuing, "
                    "needs_work=clear issues to fix, red_flags=results suggest fundamental problems"
                ),
            },
            "interpretation": {
                "type": "string",
                "description": "2-4 sentence plain-English interpretation of these results for a data scientist",
            },
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["dimension", "observation", "severity", "recommendation"],
                    "properties": {
                        "dimension": {
                            "type": "string",
                            "enum": [
                                "performance", "overfitting", "class_imbalance",
                                "data_leakage", "feature_quality", "deployment_readiness",
                            ],
                        },
                        "observation":     {"type": "string"},
                        "severity":        {"type": "string", "enum": ["positive", "note", "concern", "critical"]},
                        "recommendation":  {"type": "string"},
                    },
                },
            },
            "next_steps": {
                "type": "array",
                "items": {"type": "string"},
                "description": "3-5 concrete next experiments or actions, in priority order",
            },
            "suggested_experiments": {
                "type": "array",
                "items": {"type": "string"},
                "description": "1-3 specific experiment ideas with expected impact (optional)",
            },
        },
    },
}


@skill_entry("experiment_tracker")
def interpret(metrics: dict, context: str = "") -> dict:
    """
    Interpret model training metrics and return structured findings + next steps.

    metrics: free-form dict of training results — accuracy, auc, f1, precision, recall,
             train_accuracy, val_accuracy, test_accuracy, feature_importances, cv_scores, etc.
    context: optional problem statement from the session for grounding the interpretation.
    """
    try:
        return _interpret(metrics, context)
    except Exception as exc:
        return SkillResult({"error": str(exc)}, skill="experiment_tracker")


def _interpret(metrics: dict, context: str = "") -> dict:
    metrics_block = json.dumps(metrics, indent=2)
    context_block = f"\nBusiness context: {context}\n" if context else ""

    response = api_call(
        get_client().messages.create,
        **role("general"),
        max_tokens=8000,  # headroom for thinking; a cut-off reply is an error
        tools=[strict_tool(_TOOL)],
        tool_choice={"type": "auto"},
        messages=[{
            "role": "user",
            "content": f"""You are an experienced ML engineer reviewing training results submitted by a data scientist.
{context_block}
Training metrics:
{metrics_block}

Review these results carefully. Look for:
- Overfitting: large gap between train and val/test performance
- Suspiciously perfect scores (AUC > 0.99, accuracy > 0.99) which suggest target leakage
- Class imbalance effects: very high accuracy but low recall/F1 on the minority class
- Underfitting: uniformly poor metrics across train and val
- Missing critical metrics (no val/test split is a red flag)
- Whether the reported metrics make sense together (e.g. high accuracy + low F1 on imbalanced data)

Be direct and specific. Flag red flags clearly — data scientists need to know before they ship.
If the results look genuinely good, say so. Don't manufacture concerns where there are none.

Report your results by calling the interpret_experiment tool.""",
        }],
    )

    tool_args = tool_input(response, _TOOL["name"])
    result = SkillResult(tool_args, skill="experiment_tracker")
    return result

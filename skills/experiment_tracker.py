import json
import anthropic
from config import config
from skills.result import SkillResult
from skills.api_retry import api_call
from skills.models import GENERAL_MODEL, GENERAL_EXTRA

_client_instance: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client_instance
    if _client_instance is None:
        _client_instance = anthropic.Anthropic(api_key=config.anthropic_api_key)
    return _client_instance


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


def interpret(metrics: dict, context: str = "") -> dict:
    """
    Interpret model training metrics and return structured findings + next steps.

    metrics: free-form dict of training results — accuracy, auc, f1, precision, recall,
             train_accuracy, val_accuracy, test_accuracy, feature_importances, cv_scores, etc.
    context: optional problem statement from the session for grounding the interpretation.
    """
    metrics_block = json.dumps(metrics, indent=2)
    context_block = f"\nBusiness context: {context}\n" if context else ""

    response = api_call(
        _get_client().messages.create,
        model=GENERAL_MODEL,
        **GENERAL_EXTRA,
        max_tokens=2048,
        tools=[_TOOL],
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
If the results look genuinely good, say so. Don't manufacture concerns where there are none.""",
        }],
    )

    tool_block = next(b for b in response.content if b.type == "tool_use")
    result = SkillResult(tool_block.input, skill="experiment_tracker")
    return result

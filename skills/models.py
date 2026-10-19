"""
Model configuration for DS Workbench skills and agents.

Model IDs and effort levels are settings in config.py (overridable via .env). Skills ask
for a *role* at call time — `**role("reasoning")` expands to `model=...` and
`output_config={"effort": ...}` — so a benchmark sweep can swap the model or effort for
one role without touching any prompt:

    with override_roles(leakage=(config.reasoning_model, "medium")):
        leakage_detector.check(df, "converted")

Never hardcode a model ID.
"""
import contextvars
from contextlib import contextmanager

from config import config

# role -> (config attribute for the model, config attribute for effort or None)
ROLES = {
    "reasoning":  ("reasoning_model", "reasoning_effort"),    # DQ Pass 1
    "analysis":   ("analysis_model", "analysis_effort"),      # model_selector, join_auditor Pass 1
    "general":    ("general_model", "general_effort"),        # EDA, MLOps, NL→Spark, FE, tracker
    "leakage":    ("leakage_model", "leakage_effort"),        # leakage Pass 1
    # All Pass 2 extraction. Must accept forced tool_choice: Haiku 5.5 does;
    # Opus 5.5 / Sonnet 5.5 / Fable 5.1 return a 400. Runs at the model's default effort.
    "structured": ("structured_model", None),
    "sim":        ("sim_model", "sim_effort"),                # persona think/interpret
    "sim_judge":  ("sim_judge_model", "sim_judge_effort"),    # debrief, rating, reflection
    "code":       ("code_model", "code_effort"),              # feedback-loop drafting
}

# Which role each benchmarked skill's main call uses
SKILL_ROLE = {
    "data_quality": "reasoning", "leakage_detector": "leakage", "model_selector": "analysis",
    "join_auditor": "analysis", "feature_engineering": "general", "eda": "general",
    "nl_to_spark": "general", "mlops_review": "general", "experiment_tracker": "general",
}

_overrides: contextvars.ContextVar[dict] = contextvars.ContextVar("role_overrides", default={})


def role(name: str) -> dict:
    """Request kwargs for a role: model, plus output_config.effort where the role sets one."""
    model_attr, effort_attr = ROLES[name]
    model = getattr(config, model_attr)
    effort = getattr(config, effort_attr) if effort_attr else None
    model, effort = _overrides.get().get(name, (model, effort))
    kwargs = {"model": model}
    if effort:
        kwargs["output_config"] = {"effort": effort}
    return kwargs


def role_model(name: str) -> str:
    return role(name)["model"]


@contextmanager
def override_roles(**roles: tuple[str, str | None]):
    """Temporarily run roles on another (model, effort), e.g. for a benchmark sweep."""
    token = _overrides.set({**_overrides.get(), **roles})
    try:
        yield
    finally:
        _overrides.reset(token)


# All 5.x models use adaptive thinking (on by default); budget_tokens returns a 400
THINKING_ADAPTIVE = {"type": "adaptive"}

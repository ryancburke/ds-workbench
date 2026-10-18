"""
Model configuration for DS Workbench skills and agents.
Model IDs and effort levels are settings in config.py (overridable via .env) — this
module only re-exports them under the names the skills import. Never hardcode a model ID.
"""
from config import config

# ── Model IDs ────────────────────────────────────────────────────────────────

# Heavy reasoning: large profile analysis, DQ Pass 1
REASONING_MODEL = config.reasoning_model

# Extended analysis: model_selector, join_auditor Pass 1
ANALYSIS_MODEL = config.analysis_model

# General generation: EDA, MLOps, NL-to-Spark, Feature Engineering, Experiment Tracker
GENERAL_MODEL = config.general_model

# Leakage detection: Fable family has better recall on proxy leakage than Opus
LEAKAGE_MODEL = config.leakage_model

# Structured output extraction: all Pass 2. Must accept forced tool_choice —
# Haiku 5.5 does; Opus 5.5 / Sonnet 5.5 / Fable 5.1 return a 400.
STRUCTURED_MODEL = config.structured_model

# Agents (agents/simulation.py, agents/scheduler.py)
SIM_MODEL = config.sim_model
SIM_JUDGE_MODEL = config.sim_judge_model
CODE_MODEL = config.code_model

# ── Thinking / effort configs ────────────────────────────────────────────────

# All 5.x models use adaptive thinking (on by default); budget_tokens returns a 400
THINKING_ADAPTIVE = {"type": "adaptive"}


def _effort(level: str) -> dict:
    return {"output_config": {"effort": level}}


REASONING_EXTRA = _effort(config.reasoning_effort)
ANALYSIS_EXTRA = _effort(config.analysis_effort)
GENERAL_EXTRA = _effort(config.general_effort)
LEAKAGE_EXTRA = _effort(config.leakage_effort)
SIM_EXTRA = _effort(config.sim_effort)
SIM_JUDGE_EXTRA = _effort(config.sim_judge_effort)
CODE_EXTRA = _effort(config.code_effort)

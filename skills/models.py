"""
Central model configuration for DS Workbench skills.
Update model IDs here when Anthropic releases new versions — no other files need to change.
"""

# ── Model IDs ────────────────────────────────────────────────────────────────

# Heavy reasoning: large profile analysis, DQ Pass 1
REASONING_MODEL = "claude-opus-5-5"

# Extended analysis: model_selector, join_auditor Pass 1 (adaptive thinking)
ANALYSIS_MODEL = "claude-sonnet-5-5"

# General generation: EDA, MLOps, NL-to-Spark, Feature Engineering (no thinking)
GENERAL_MODEL = "claude-sonnet-5-5"

# Leakage detection: Fable family has better recall on proxy leakage than Opus
LEAKAGE_MODEL = "claude-fable-5-1"

# Structured output extraction: all Pass 2 (fast, no thinking needed)
STRUCTURED_MODEL = "claude-haiku-5-5"

# ── Thinking configs ─────────────────────────────────────────────────────────

# All 5.x models use adaptive thinking; "enabled" + budget_tokens is a 4.x-only API
THINKING_ADAPTIVE = {"type": "adaptive"}

# Leakage: Fable family with medium effort
LEAKAGE_EXTRA = {"output_config": {"effort": "medium"}}

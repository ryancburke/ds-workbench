import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str):
    return field(default_factory=lambda: os.getenv(name, default))


@dataclass
class Config:
    anthropic_api_key: str = _env("ANTHROPIC_API_KEY", "")
    databricks_host: str = _env("DATABRICKS_HOST", "")
    databricks_token: str = _env("DATABRICKS_TOKEN", "")
    databricks_http_path: str = _env("DATABRICKS_HTTP_PATH", "")
    local_dev_mode: bool = field(
        default_factory=lambda: os.getenv("LOCAL_DEV_MODE", "true").lower() == "true"
    )
    # 0 = manual: simulations run only from the Feedback page or run_sim.py, so each
    # run can be tied to a change. Set e.g. 6 to run on a schedule.
    sim_interval_hours: float = field(
        default_factory=lambda: float(os.getenv("SIM_INTERVAL_HOURS", "0"))
    )

    # ── Models (override any of these in .env) ────────────────────────────────
    # Skills
    reasoning_model: str = _env("REASONING_MODEL", "claude-opus-5-5")    # DQ Pass 1
    analysis_model: str = _env("ANALYSIS_MODEL", "claude-sonnet-5-5")    # model_selector, join_auditor Pass 1
    general_model: str = _env("GENERAL_MODEL", "claude-sonnet-5-5")      # EDA, MLOps, NL→Spark, FE, tracker
    # leakage Pass 1. Opus 5.5 matched Fable 5.1 on all four leakage cases (2 trials each,
    # incl. disguised and precision cases) at ~40% of the cost — see PR for the sweep.
    leakage_model: str = _env("LEAKAGE_MODEL", "claude-opus-5-5")
    structured_model: str = _env("STRUCTURED_MODEL", "claude-haiku-5-5") # all Pass 2 (forced tool call)
    # Agents
    sim_model: str = _env("SIM_MODEL", "claude-sonnet-5-5")              # persona think/interpret
    sim_judge_model: str = _env("SIM_JUDGE_MODEL", "claude-opus-5-5")    # debrief, rating, reflection
    code_model: str = _env("CODE_MODEL", "claude-sonnet-5-5")            # feedback-loop code drafting

    # ── Effort per role: low | medium | high | xhigh | max ────────────────────
    # Skill defaults match each model's API default, so behaviour is unchanged
    # until a benchmark run justifies raising one.
    reasoning_effort: str = _env("REASONING_EFFORT", "medium")
    analysis_effort: str = _env("ANALYSIS_EFFORT", "high")
    general_effort: str = _env("GENERAL_EFFORT", "high")
    leakage_effort: str = _env("LEAKAGE_EFFORT", "medium")
    sim_effort: str = _env("SIM_EFFORT", "low")
    sim_judge_effort: str = _env("SIM_JUDGE_EFFORT", "low")
    code_effort: str = _env("CODE_EFFORT", "medium")



# List prices per million tokens (input, output), checked against the pricing page on
# 2026-10-19. Used only to report and compare spend (benchmark sweeps, live checks).
MODEL_PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5-5":   (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-fable-5-1":  (10.00, 50.00),
    "claude-haiku-5-5":  (0.10, 0.50),   # prompts ≤100K tokens
}


def call_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Approximate list-price cost of one call in USD (0 for an unknown model)."""
    price_in, price_out = MODEL_PRICES.get(model, (0.0, 0.0))
    return (input_tokens * price_in + output_tokens * price_out) / 1e6


config = Config()

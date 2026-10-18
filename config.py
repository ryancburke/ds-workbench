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
    sim_interval_hours: float = field(
        default_factory=lambda: float(os.getenv("SIM_INTERVAL_HOURS", "6"))
    )

    # ── Models (override any of these in .env) ────────────────────────────────
    # Skills
    reasoning_model: str = _env("REASONING_MODEL", "claude-opus-5-5")    # DQ Pass 1
    analysis_model: str = _env("ANALYSIS_MODEL", "claude-sonnet-5-5")    # model_selector, join_auditor Pass 1
    general_model: str = _env("GENERAL_MODEL", "claude-sonnet-5-5")      # EDA, MLOps, NL→Spark, FE, tracker
    leakage_model: str = _env("LEAKAGE_MODEL", "claude-fable-5-1")       # leakage Pass 1
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


config = Config()

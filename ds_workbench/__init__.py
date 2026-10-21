"""
ds_workbench — AI-powered data science skill library.

Designed to be imported directly from Databricks notebooks:

    from ds_workbench import data_quality, leakage_detector, feature_engineering

    dq_result = data_quality.assess(df, table_name="my_table")
    leak      = leakage_detector.check(df, target_col="churn")
    fe_plan   = feature_engineering.suggest(df, target_col="churn")
    review    = experiment_tracker.interpret({"train_auc": 0.91, "val_auc": 0.84})

Deterministic functions that need no API key live in ds_workbench.core.

Skills load on first use, so `import ds_workbench.core` stays light (no config, no SDK).
"""
import importlib

_SKILLS = {
    "data_quality": "skills.data_quality",
    "eda": "skills.eda",
    "join_auditor": "skills.join_auditor",
    "nl_to_spark": "skills.nl_to_spark",
    "mlops_reviewer": "skills.mlops_reviewer",
    "model_selector": "skills.model_selector",
    "leakage_detector": "skills.leakage_detector",
    "feature_engineering": "skills.feature_engineering",
    "preflight": "skills.preflight",
    "experiment_tracker": "skills.experiment_tracker",
    "causal_inference": "skills.causal_inference",
    "problem_framer": "skills.problem_framer",
}

__all__ = [*_SKILLS, "WorkflowSession", "core"]
__version__ = "0.1.0"


def __getattr__(name: str):
    if name in _SKILLS:
        module = importlib.import_module(_SKILLS[name])
        globals()[name] = module
        return module
    if name == "WorkflowSession":
        from skills.session import WorkflowSession
        globals()[name] = WorkflowSession
        return WorkflowSession
    if name == "core":
        return importlib.import_module("ds_workbench.core")
    raise AttributeError(f"module 'ds_workbench' has no attribute {name!r}")


def __dir__():
    return sorted(__all__)

"""
ds_workbench — AI-powered data science skill library.

Designed to be imported directly from Databricks notebooks:

    from ds_workbench import data_quality, leakage_detector, feature_engineering

    dq_result = data_quality.assess(df, table_name="my_table")
    leak      = leakage_detector.check(df, target_col="churn")
    fe_plan   = feature_engineering.suggest(df, target_col="churn")
    review    = experiment_tracker.interpret({"train_auc": 0.91, "val_auc": 0.84})
"""

from skills import data_quality
from skills import eda
from skills import join_auditor
from skills import nl_to_spark
from skills import mlops_reviewer
from skills import model_selector
from skills import leakage_detector
from skills import feature_engineering
from skills import preflight
from skills import experiment_tracker
from skills import causal_inference
from skills.session import WorkflowSession

__all__ = [
    "data_quality",
    "eda",
    "join_auditor",
    "nl_to_spark",
    "mlops_reviewer",
    "model_selector",
    "leakage_detector",
    "feature_engineering",
    "preflight",
    "experiment_tracker",
    "causal_inference",
    "WorkflowSession",
]

__version__ = "0.1.0"

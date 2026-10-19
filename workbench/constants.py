"""Workbench constants: dataset and join-scenario options, skill metadata, colour maps."""



# ── Constants ──────────────────────────────────────────────────────────────────

SKILL_KEYS = {"data_quality", "eda", "join_auditor", "nl_to_spark", "mlops_review",
              "model_selector", "leakage_detector", "feature_engineering"}

DATASET_OPTIONS = [
    {"label": "Orders — Dirty        (nulls, mixed-case, duplicates)",    "value": "orders_dirty"},
    {"label": "Sales — Clean         (product, pricing, discounts)",      "value": "sales_clean"},
    {"label": "HR Attrition          (salary, tenure, performance)",      "value": "hr_attrition"},
    {"label": "E-commerce            (sessions, channel, device)",        "value": "ecommerce"},
    {"label": "Transactions — Fraud  (amount outliers, mixed country)",   "value": "transactions_fraud"},
    {"label": "Sales Forecasting     (revenue, promotions, seasonal)",    "value": "sales_forecasting"},
    {"label": "E-commerce Leakage   (planted leakage — for testing)",    "value": "ecommerce_leakage"},
]

JOIN_SCENARIO_OPTIONS = [
    {"label": "Orders ⋈ Customers  (fan-out — duplicate customer IDs)",    "value": "orders_customers_fanout"},
    {"label": "Sales ⋈ Products   (clean — unique keys both sides)",       "value": "sales_products_clean"},
    {"label": "Transactions ⋈ Users  (many-to-many — dupes on both)",      "value": "transactions_users_many_to_many"},
]

ML_PLACEHOLDER = """\
from pyspark.sql import SparkSession
import pandas as pd

spark = SparkSession.builder.getOrCreate()

df = spark.read.parquet("/mnt/data/transactions/2024-01-15/")
df = df.filter(df.date >= "2023-12-15")
df = df.withColumn("revenue", df.amount * 1.082)

result = df.toPandas()
for i, row in result.iterrows():
    result.at[i, "category"] = "high" if row["revenue"] > 500 else "low"

result.to_csv("/tmp/output.csv")
print("done")
"""

# Skill-level color + icon lookup
SKILL_META = {
    "data_quality": {"label": "Data Quality",   "icon": "bi-shield-check",  "color": "primary",  "bs": "#0d6efd"},
    "eda":          {"label": "Exploration",     "icon": "bi-graph-up",      "color": "success",  "bs": "#198754"},
    "join_auditor": {"label": "Join Auditor",    "icon": "bi-link-45deg",    "color": "warning",  "bs": "#ffc107"},
    "nl_to_spark":  {"label": "NL → PySpark",   "icon": "bi-translate",     "color": "info",     "bs": "#0dcaf0"},
    "mlops_review":    {"label": "MLOps Review",    "icon": "bi-gear-fill",      "color": "danger",    "bs": "#dc3545"},
    "model_selector":      {"label": "Model Selector",      "icon": "bi-cpu",                 "color": "secondary", "bs": "#a78bfa"},
    "leakage_detector":    {"label": "Leakage Detector",    "icon": "bi-exclamation-diamond", "color": "danger",    "bs": "#dc3545"},
    "feature_engineering": {"label": "Feature Engineering", "icon": "bi-wrench-adjustable",   "color": "secondary", "bs": "#6f42c1"},
}

# MLOps constants
ML_SEV_COLOR    = {"critical": "danger", "warning": "warning", "suggestion": "info"}
ML_FEAS_COLOR   = {"ready": "success", "marginal": "warning", "insufficient": "danger"}
ML_FEAS_ICON    = {"ready": "bi-check-circle-fill text-success",
                   "marginal": "bi-exclamation-circle text-warning",
                   "insufficient": "bi-x-circle-fill text-danger"}
ML_CAT_ICON = {
    "data_leakage":       "bi-exclamation-diamond-fill",
    "hardcoded_config":   "bi-lock",
    "missing_logging":    "bi-journal-x",
    "no_error_handling":  "bi-shield-x",
    "inefficient_spark":  "bi-lightning",
    "reproducibility":    "bi-arrow-repeat",
    "missing_tests":      "bi-check2-square",
    "code_structure":     "bi-code-square",
    "security":           "bi-incognito",
    "documentation":      "bi-file-text",
}

# DQ constants
DQ_SEV_COLOR  = {"critical": "danger", "warning": "warning", "info": "info"}
DQ_SEV_RANK   = {"critical": 0, "warning": 1, "info": 2}
DQ_SEV_HEADER = {
    "critical": ("bi-exclamation-octagon-fill text-danger", "Critical Issues"),
    "warning":  ("bi-exclamation-triangle text-warning",    "Warnings"),
    "info":     ("bi-info-circle text-info",                "Info"),
}
DQ_ROW_STYLE  = {
    "critical": {"borderLeft": "3px solid #dc3545"},
    "warning":  {"borderLeft": "3px solid #ffc107"},
    "info":     {},
}

# JA constants
JA_SEV_COLOR   = {"critical": "danger", "warning": "warning", "info": "info"}
JA_VERDICT_CLR = {"safe": "success", "review_needed": "warning", "dangerous": "danger"}



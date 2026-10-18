"""
Ground-truth answer keys for synthetic datasets and scenarios.
Derived directly from the generation code in databricks_client.py
and the scenario MLOps code in agents/simulation.py.

Used by auto_score.py to compute quantitative recall-based benchmark scores.
"""

# ── Data Quality answer keys ──────────────────────────────────────────────────
# Each expected_finding must be caught by the DQ skill.
# column="_table" means a table-level issue (e.g. duplicate_rows).
# Tier: "required" = clear/unambiguous; "expected" = model should find it.

DQ_ANSWER_KEYS: dict[str, dict] = {
    "orders_dirty": {
        "description": "Orders table with injected nulls, mixed-case status/region, duplicate rows",
        "expected_findings": [
            # Required — clear signal, must be caught
            {"column": "status",  "issue_type": "inconsistent_format", "tier": "required"},
            {"column": "region",  "issue_type": "inconsistent_format", "tier": "required"},
            {"column": "amount",  "issue_type": "null_rate",           "tier": "required"},   # ~9%
            {"column": "_table",  "issue_type": "duplicate_rows",      "tier": "required"},   # ~3%
            # Expected — ~4-5% nulls, above info threshold
            {"column": "customer_id", "issue_type": "null_rate",       "tier": "expected"},
            {"column": "email",       "issue_type": "null_rate",       "tier": "expected"},
            {"column": "status",      "issue_type": "null_rate",       "tier": "expected"},
            {"column": "region",      "issue_type": "null_rate",       "tier": "expected"},
        ],
    },
    "hr_attrition": {
        "description": "HR data with extreme salary outliers and missing performance/satisfaction scores",
        "expected_findings": [
            {"column": "salary",             "issue_type": "outlier",   "tier": "required"},  # 8 injected extremes
            {"column": "performance_score",  "issue_type": "null_rate", "tier": "required"},  # ~12%
            {"column": "satisfaction_score", "issue_type": "null_rate", "tier": "required"},  # ~8%
        ],
    },
    "ecommerce": {
        "description": "Session data with nulls across user_id, channel, time_on_site_s",
        "expected_findings": [
            {"column": "user_id",        "issue_type": "null_rate", "tier": "required"},   # ~10%
            {"column": "channel",        "issue_type": "null_rate", "tier": "expected"},   # ~5%
            {"column": "time_on_site_s", "issue_type": "null_rate", "tier": "expected"},   # ~5%
        ],
    },
    "sales_clean": {
        "description": "Clean dataset — no injected issues; skill should report high quality (≥80/100)",
        "expected_findings": [],  # scored by absence of false critical/warning alarms
        "expected_score_min": 80,
    },
    "transactions_fraud": {
        "description": "Transactions with outlier amounts, mixed-case country, null user IDs, class imbalance",
        "expected_findings": [
            {"column": "amount",  "issue_type": "outlier",             "tier": "required"},
            {"column": "country", "issue_type": "inconsistent_format", "tier": "required"},
            {"column": "user_id", "issue_type": "null_rate",           "tier": "required"},  # ~8%
        ],
    },
    "sales_forecasting": {
        "description": "Clean sales dataset with strong promotions→revenue correlation; no quality issues",
        "expected_findings": [],
        "expected_score_min": 80,
    },
}

# ── Join Auditor answer keys ───────────────────────────────────────────────────

JOIN_ANSWER_KEYS: dict[str, dict] = {
    "orders_customers_fanout": {
        "description": "customers right-side has ~10% duplicate IDs → fan-out; orders has ~4% null customer_ids",
        "expected_issue_types": ["duplicate_keys", "fan_out", "null_keys"],
        "expected_verdict": "dangerous",
        "acceptable_verdicts": ["review_needed"],  # acceptable if model is conservative
    },
    "sales_products_clean": {
        "description": "Clean join — unique keys both sides; should report safe",
        "expected_issue_types": ["safe"],
        "expected_verdict": "safe",
        "acceptable_verdicts": [],
    },
    "transactions_users_many_to_many": {
        "description": "Many-to-many: both transactions and users tables have duplicate join keys",
        "expected_issue_types": ["many_to_many", "duplicate_keys", "fan_out"],
        "expected_verdict": "dangerous",
        "acceptable_verdicts": ["review_needed"],
    },
}

# ── MLOps answer keys ─────────────────────────────────────────────────────────
# Known issues in each scenario's mlops_code, by category enum.

MLOPS_ANSWER_KEYS: dict[str, dict] = {
    "q3_revenue": {
        "description": "Hardcoded path/tax-rate/output, no logging, no error handling, .toPandas() risk",
        "expected_categories": [
            "hardcoded_config",    # /mnt/data/orders/2024-Q3/, 1.082 tax rate, /tmp/q3_revenue.csv
            "missing_logging",     # no shape/metric logging
            "no_error_handling",   # no try/except
            "inefficient_spark",   # .toPandas() after groupBy
        ],
    },
    "churn_risk": {
        "description": "Hardcoded path/features/hyperparams/output, no train-test split, no logging, no error handling",
        "expected_categories": [
            "hardcoded_config",    # path, feature list, maxIter, model save path
            "missing_logging",
            "no_error_handling",
            "reproducibility",     # no train/test split, no random seed
        ],
    },
    "pipeline_handoff": {
        "description": "Hardcoded date path/output, .toPandas() on join, Python iterrows loop, no logging/error handling",
        "expected_categories": [
            "hardcoded_config",    # /mnt/data/orders_new/2024-01-15/, /tmp/pipeline_output.csv
            "missing_logging",
            "no_error_handling",
            "inefficient_spark",   # .toPandas() on join + iterrows loop
        ],
    },
}

# ── EDA answer keys ───────────────────────────────────────────────────────────
# expected_targets: columns the skill should identify as ML targets
# expected_correlation_pairs: (col_a, col_b) where |r| > 0.5 known from generation

EDA_ANSWER_KEYS: dict[str, dict] = {
    "hr_attrition": {
        "expected_targets": [
            {"column": "attrition", "task_type": "classification"},
        ],
        "expected_correlation_pairs": [],  # no strong correlations seeded
        "key_topics": ["attrition", "salary", "performance", "tenure"],
    },
    "ecommerce": {
        "expected_targets": [
            {"column": "converted", "task_type": "classification"},
        ],
        "expected_correlation_pairs": [],
        "key_topics": ["conversion", "channel", "session", "cart"],
    },
    "sales_forecasting": {
        "expected_targets": [
            {"column": "revenue", "task_type": "regression"},
        ],
        "expected_correlation_pairs": [("promotions_spend", "revenue")],
        "key_topics": ["revenue", "promotions", "seasonality"],
    },
    "orders_dirty": {
        "expected_targets": [],  # no clear ML target in this dataset
        "expected_correlation_pairs": [],
        "key_topics": ["amount", "status", "order"],
    },
}

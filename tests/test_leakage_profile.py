"""Leakage profile signals that don't depend on column names or linear correlation."""
import databricks_client as dc
from skills.leakage_detector import build_leakage_profile

PROFILE = build_leakage_profile(dc._gen_subscription_churn(), "churned")["features"]


def test_nonlinear_leak_is_visible_through_auc_not_correlation():
    eng = PROFILE["engagement_index_v2"]
    assert eng["correlation_with_target"] < 0.3   # Pearson misses it
    assert eng["single_feature_auc"] >= 0.99      # rank AUC doesn't


def test_category_value_seen_in_one_class_only():
    pure = PROFILE["status_code"]["single_class_values"]
    assert pure["C2"]["target_rate"] == 1.0


def test_zero_for_one_class_only():
    zeros = PROFILE["credit_balance"]["zero_pct_by_class"]
    assert zeros["False"] == 100.0 and zeros["True"] < 60


def test_legitimate_predictors_stay_well_below_leak_level():
    for col in ("tenure_months", "last_login_days", "support_tickets_90d", "monthly_spend"):
        assert PROFILE[col]["single_feature_auc"] < 0.9, col


def test_high_cardinality_categoricals_get_no_in_sample_auc():
    """In-sample target encoding of an ID would show a fake AUC of ~1.0."""
    assert "single_feature_auc" not in PROFILE["customer_ref"]

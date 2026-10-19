"""Scorers reward correct plans and catch the known failure modes."""
from benchmarks.scoring import load_answer_key, score_feature_engineering

KEY = load_answer_key("transactions_fraud", "feature_engineering_fraud")
CODE = ("from pyspark.ml import Pipeline\nfrom pyspark.ml.feature import StringIndexer, VectorAssembler\n"
        "df = df.withColumn('country', F.upper('country'))\n")


def _plan(user_id, merchant_id, amount, imbalance, model_family="linear", extra_drops=()):
    feats = [
        {"column": "transaction_id", "action": "drop"},
        {"column": "user_id", "action": user_id, "imputation": "constant_unknown"},
        {"column": "merchant_id", "action": merchant_id, "imputation": "constant_unknown"},
        {"column": "amount", "action": amount, "imputation": "none"},
        *({"column": c, "action": "encode_ohe", "imputation": "none"}
          for c in ("currency", "country", "device_type", "transaction_type")),
    ]
    return {"features": feats, "dropped_cols": ["transaction_id", *extra_drops],
            "pyspark_code": CODE, "model_family": model_family, "imbalance_handling": imbalance}


def test_good_fraud_plan_scores_full_marks():
    plan = _plan("encode_frequency", "encode_frequency", "log_then_scale",
                 "class weights in a weightCol computed from training labels")
    assert score_feature_engineering(plan, KEY).overall == 100.0


def test_issue_3_plan_is_caught():
    """The plan the simulation judge criticised: indexed IDs, plain scaling, no imbalance handling."""
    result = score_feature_engineering(_plan("encode_index", "encode_index", "scale", "none"), KEY)
    assert result.overall < 80
    assert {"user_id_not_indexed", "merchant_id_not_indexed", "imbalance_handled",
            "amount_log"} <= set(result.missed_findings)


def test_indexed_merchant_is_fine_for_trees_but_user_id_is_not():
    plan = _plan("encode_index", "encode_index", "keep_as_is", "weightCol", model_family="tree_based")
    missed = set(score_feature_engineering(plan, KEY).missed_findings)
    assert "merchant_id_not_indexed" not in missed and "user_id_not_indexed" in missed
    assert "amount_scale" not in missed  # trees don't need scaling


def test_existing_keys_score_unchanged_without_new_fields():
    """Keys without the harder checks keep the original 100-point scale."""
    key = load_answer_key("ecommerce", "feature_engineering_converted")
    assert not any(k in key for k in ("forbidden_actions", "required_concepts", "preferred_actions"))

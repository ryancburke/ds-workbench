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


def test_dq_score_range_credit_fades_instead_of_falling_off_a_cliff():
    from benchmarks.scoring import score_dq
    key = load_answer_key("transactions_fraud", "data_quality")
    findings = [
        {"column": "user_id", "issue_type": "null_rate", "severity": "warning"},
        {"column": "country", "issue_type": "inconsistent_format", "severity": "warning"},
        {"column": "amount", "issue_type": "outlier", "severity": "warning"},
    ]
    hi = key["score"]["max"]
    in_range = score_dq({"findings": findings, "overall_score": hi}, key).overall
    one_over = score_dq({"findings": findings, "overall_score": hi + 1}, key).overall
    far_over = score_dq({"findings": findings, "overall_score": hi + 15}, key).overall
    assert in_range == 100.0
    assert 95 <= one_over < 100      # 1 point of arithmetic noise costs ~2.5, not 25
    assert far_over == 75.0          # far outside the range still loses all range credit


def test_experiment_tracker_scorer_penalises_false_alarms_on_healthy_runs():
    from benchmarks.scoring import score_experiment_tracker
    key = load_answer_key("experiment_metrics", "experiment_tracker_healthy")
    calm = {"verdict": "promising", "findings": [{"dimension": "performance", "severity": "positive"}]}
    alarmed = {"verdict": "red_flags", "findings": [{"dimension": "overfitting", "severity": "critical"}]}
    assert score_experiment_tracker(calm, key).overall == 100.0
    assert score_experiment_tracker(alarmed, key).overall == 50.0


def test_experiment_tracker_scorer_requires_the_red_flag():
    from benchmarks.scoring import score_experiment_tracker
    key = load_answer_key("experiment_metrics", "experiment_tracker_imbalance")
    missed = {"verdict": "promising", "findings": []}
    assert score_experiment_tracker(missed, key).overall == 20.0


def test_split_before_learning_heuristic():
    from benchmarks.scoring import split_before_learning
    good = "train, test = df.randomSplit([0.8, 0.2], seed=42)\nfreq = train.groupBy('m').count()\nmodel = pipe.fit(train)"
    leaky = "freq = df.groupBy('m').count()\nw = df.groupBy('label').agg(F.count('*'))\ntrain, test = df.randomSplit([0.8, 0.2])\nmodel = pipe.fit(train)"
    no_holdout = "model = LogisticRegression().fit(df)"
    assert split_before_learning(good)[0]
    assert not split_before_learning(leaky)[0]
    assert "never holds data out" in split_before_learning(no_holdout)[1]
    assert split_before_learning("pipeline = Pipeline(stages=[a, b])")[0]  # nothing fitted yet


def test_hard_dq_case_separates_thorough_from_naive_reviews():
    from benchmarks.scoring import score
    f = lambda c, t, s="warning": {"column": c, "issue_type": t, "severity": s}
    thorough = {"overall_score": 45, "findings": [
        f("email", "null_rate"), f("annual_income", "type_mismatch"), f("age", "outlier"),
        f("signup_date", "schema_issue"), f("country", "inconsistent_format"), f("monthly_fee", "outlier"),
        f("region", "inconsistent_format", "info")]}          # info note on a clean column is allowed
    naive = {"overall_score": 80, "findings": [               # only what null counts and IQR fences show
        f("age", "outlier"), f("monthly_fee", "outlier"), f("country", "inconsistent_format"),
        f("tenure_days", "outlier")]}
    assert score(thorough, "customer_accounts", "data_quality").overall == 100.0
    naive_result = score(naive, "customer_accounts", "data_quality")
    assert naive_result.overall < 50
    assert {"email_placeholder_nulls", "income_text_and_na", "future_signups"} <= set(naive_result.missed_findings)

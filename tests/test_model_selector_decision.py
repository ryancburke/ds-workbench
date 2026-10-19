"""Model selection leads with the decision: model, imbalance strategy, metric, validation."""
from benchmarks.scoring import load_answer_key, score_model_selector
from skills.model_selector import _TOOL
from skills.result import SkillResult

DECISION = {"model": "GBTClassifier", "imbalance_strategy": "weightCol, positive weight ≈ 32",
            "primary_metric": "PR-AUC; recall at 90% precision", "validation": "time-based holdout"}
RESULT = {"task_type": "binary_classification", "decision": DECISION, "data_readiness": "needs_prep",
          "target_analysis": {"summary": "s", "class_balance_issue": True, "imbalance_ratio": 32.3},
          "assumption_checks": [], "summary": "s",
          "recommendations": [{"rank": 1, "model_family": "gradient_boosting", "display_name": "GBT",
                               "suitability": "strong", "rationale": "RATIONALE-TEXT"}]}


def test_decision_is_required_by_the_schema():
    assert "decision" in _TOOL["input_schema"]["required"]


def test_notebook_render_shows_the_decision_before_the_reasoning():
    html = SkillResult(RESULT, skill="model_selector")._repr_html_()
    assert html.index("weightCol, positive weight") < html.index("RATIONALE-TEXT")


def test_workbench_render_shows_the_decision_after_the_verdict():
    import app  # noqa: F401
    from workbench.renderers.ms import _ms_render
    children = _ms_render(RESULT).children
    assert "PR-AUC" in str(children[1].to_plotly_json())


def test_accuracy_as_the_primary_metric_loses_points_on_imbalanced_data():
    key = load_answer_key("transactions_fraud", "model_selector_fraud")
    good = score_model_selector(RESULT, key)
    bad = score_model_selector({**RESULT, "decision": {**DECISION, "primary_metric": "accuracy"}}, key)
    assert "decision_primary_metric" in good.matched_findings
    assert "decision_primary_metric" in bad.missed_findings and bad.overall < good.overall


def test_spark_class_names_map_to_model_families():
    from benchmarks.scoring import _normalise_family
    assert _normalise_family("GBTClassifier") == "gradient_boosting"
    assert _normalise_family("RandomForestClassifier") == "random_forest"
    assert _normalise_family("LogisticRegression") == "logistic_regression"
    assert _normalise_family("LinearSVC") == "svm"
    assert _normalise_family("logistic_regression") == "logistic_regression"  # unchanged

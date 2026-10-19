"""Causal inference skill: numbers come from code, the model's choice is recomputed."""
import databricks_client as dc
from benchmarks.scoring import load_answer_key, score_causal_inference
from skills import causal_inference
from skills.result import SkillResult
from tests.conftest import tool_reply


def _reply(method, exclude=(), verdict="valid", findings=()):
    return lambda kw: tool_reply(kw["model"], "report_causal_analysis", {
        "verdict": verdict, "design": "randomized_experiment", "identification_strategy": "randomization",
        "recommended_estimate": {"method": method, "exclude_covariates": list(exclude), "rationale": "r"},
        "findings": list(findings), "assumptions": ["a"], "summary": "s", "next_steps": ["n"]})


def test_request_is_strict_and_names_the_tool(fake_api):
    fake_api.responder = _reply("difference_in_means")
    causal_inference.analyze(dc._gen_ab_test_underpowered(), "variant", "revenue_30d", randomized=True,
                             treated_value="B")
    call = fake_api.calls[0]
    assert call["tools"][0]["strict"] is True and "report_causal_analysis" in str(call["messages"])


def test_the_reported_estimate_is_recomputed_in_code(fake_api):
    """The model only names a method and covariates to exclude; the number comes from code.
    Excluding the mediator gives ~2.0 (the truth), keeping it gives ~1.0."""
    df = dc._gen_ab_test_bad_control()
    fake_api.responder = _reply("regression_adjustment", exclude=["sessions_first_week"])
    good = causal_inference.analyze(df, "new_onboarding", "orders_30d", randomized=True)
    fake_api.responder = _reply("regression_adjustment")
    biased = causal_inference.analyze(df, "new_onboarding", "orders_30d", randomized=True)
    assert abs(good["estimate"]["effect"] - 2.0) < 0.1
    assert abs(biased["estimate"]["effect"] - 1.0) < 0.1
    assert good["estimate"]["excluded_covariates"] == ["sessions_first_week"]


def test_srm_only_for_experiments(fake_api):
    fake_api.responder = _reply("regression_adjustment", verdict="caution")
    obs = causal_inference.analyze(dc._gen_observational_coupon(), "got_coupon", "spend_next_90d",
                                   randomized=False)
    assert "not applicable" in obs["evidence"]["sample_ratio"]["note"]
    prompt = fake_api.calls[0]["messages"][0]["content"]
    assert "p_value" not in prompt.split('"sample_ratio"')[1][:200]


def test_bad_inputs_return_a_readable_error(fake_api):
    out = causal_inference.analyze(dc._gen_ab_test_clean(200), "country", "revenue_30d")
    assert "exactly 2 values" in out["error"]


def test_notebook_render_leads_with_verdict_and_effect(fake_api):
    fake_api.responder = _reply("cuped")
    r = causal_inference.analyze(dc._gen_ab_test_clean(), "variant", "revenue_30d", randomized=True,
                                 pre_outcome_col="revenue_prev_30d", treated_value="B")
    html = SkillResult(dict(r), skill="causal_inference")._repr_html_()
    assert html.index("VALID") < html.index("Effect:") < html.index("Relies on")


def test_scorer_grades_the_estimate_against_the_true_effect():
    key = load_answer_key("ab_test_bad_control", "causal_inference")
    finding = {"check": "bad_control", "severity": "warning", "column": "sessions_first_week",
               "evidence": "e", "recommendation": "r"}
    good = {"verdict": "caution", "findings": [finding], "estimate": {"effect": 1.98, "se": 0.03}}
    biased = {**good, "estimate": {"effect": 0.95, "se": 0.03}}
    assert score_causal_inference(good, key).overall == 100.0
    assert "estimate_near_truth" in score_causal_inference(biased, key).missed_findings


def test_not_identified_result_does_not_present_an_effect(fake_api):
    fake_api.responder = _reply("difference_in_means", verdict="not_identified")
    r = causal_inference.analyze(dc._gen_ab_test_srm(), "variant", "revenue_30d", randomized=True,
                                 treated_value="B")
    html = SkillResult(dict(r), skill="causal_inference")._repr_html_()
    assert "Not a causal estimate" in html and "Effect:" not in html

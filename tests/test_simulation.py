"""Simulation records failures explicitly instead of inventing outcomes."""
import pytest

from agents import scheduler, simulation
from tests.conftest import text_reply, tool_reply

PERSONA = simulation.PERSONAS[0]
SCENARIO = next(s for s in simulation.SCENARIOS if s["id"] == "fraud_detection")


def test_think_step_without_tool_call_raises(fake_api):
    fake_api.responder = lambda kw: text_reply(kw["model"], "I think we should debrief.")
    with pytest.raises(RuntimeError, match="did not call decide_action"):
        simulation.think_step(PERSONA, SCENARIO, [])


def test_run_skill_without_skill_raises(fake_api):
    fake_api.responder = lambda kw: tool_reply(kw["model"], "decide_action", {
        "reasoning": "r", "action": "run_skill", "skill": "none", "dataset": "none"})
    state = simulation.init_sim_state("fraud_detection")
    with pytest.raises(RuntimeError, match="without a skill"):
        simulation.run_sim_step(state)


def test_decide_action_schema_is_strict_with_none_option(fake_api):
    fake_api.responder = lambda kw: tool_reply(kw["model"], "decide_action", {
        "reasoning": "r", "action": "give_debrief", "skill": "none", "dataset": "none"})
    simulation.think_step(PERSONA, SCENARIO, [])
    tool = fake_api.calls[0]["tools"][0]
    schema = tool["input_schema"]
    assert tool["strict"] is True and schema["additionalProperties"] is False
    assert {"skill", "dataset"} <= set(schema["required"])
    assert "none" in schema["properties"]["skill"]["enum"]


def test_missing_rating_is_an_error_record_not_a_score(fake_api):
    fake_api.responder = lambda kw: text_reply(kw["model"], "I'd say 4/5")
    rec = simulation.evaluate_output_step(PERSONA, "eda", "ecommerce", {"key_insights": []}, SCENARIO)
    assert rec["status"] == "error"
    assert rec["overall"] is None and rec["qualitative"] is None
    assert "did not call rate_output" in rec["error"]


def test_rating_is_recorded(fake_api):
    fake_api.responder = lambda kw: tool_reply(kw["model"], "rate_output",
                                               {"usefulness": 4, "clarity": 5, "brief_note": "none"})
    rec = simulation.evaluate_output_step(PERSONA, "eda", "ecommerce", {"key_insights": []}, SCENARIO)
    assert rec["status"] == "ok" and rec["overall"] is not None


def test_reflection_tool_is_strict_and_covers_all_skills(fake_api):
    fake_api.responder = lambda kw: text_reply(kw["model"], "")
    assert simulation.reflect_step(PERSONA, SCENARIO, []) == []
    tool = fake_api.calls[0]["tools"][0]
    assert tool["strict"] is True
    enum = tool["input_schema"]["properties"]["skill_affected"]["enum"]
    assert {"experiment_tracker", "notebook_generator"} <= set(enum)


def test_scheduler_is_manual_by_default(monkeypatch):
    from config import config
    monkeypatch.setattr(config, "sim_interval_hours", 0.0)
    monkeypatch.setenv("WERKZEUG_RUN_MAIN", "true")
    assert scheduler.start_scheduler() is None


def test_judge_rates_the_full_output_not_the_summary(fake_api):
    fake_api.responder = lambda kw: tool_reply(kw["model"], "rate_output",
                                               {"usefulness": 4, "clarity": 4, "brief_note": "none"})
    long_evidence = "x" * 500 + " END-OF-EVIDENCE"
    result = {"verdict": "suspicious", "profile": {"huge": "profiling data"},
              "findings": [{"column": f"c{i}", "severity": "info", "evidence": long_evidence}
                           for i in range(6)]}
    simulation.evaluate_output_step(PERSONA, "leakage_detector", "ecommerce", result, SCENARIO)
    prompt = fake_api.calls[0]["messages"][0]["content"]
    assert prompt.count("END-OF-EVIDENCE") == 6      # every finding, untruncated
    assert "profiling data" not in prompt             # internal profile omitted


def test_personas_can_run_causal_inference_and_it_is_scored(fake_api):
    from tests.conftest import tool_reply as reply
    scenario = next(s for s in simulation.SCENARIOS if s["id"] == "launch_impact")

    def respond(kw):
        name = kw["tools"][0]["name"] if kw.get("tools") else None
        if name == "report_causal_analysis":
            return reply(kw["model"], name, {
                "verdict": "caution", "design": "randomized_experiment", "identification_strategy": "randomization",
                "recommended_estimate": {"method": "difference_in_means", "exclude_covariates": [], "rationale": "r"},
                "findings": [{"check": "bad_control", "severity": "warning", "column": "sessions_first_week",
                              "evidence": "e", "recommendation": "r"}],
                "assumptions": [], "summary": "s", "next_steps": []})
        if name == "rate_output":
            return reply(kw["model"], name, {"usefulness": 4, "clarity": 4, "brief_note": "none"})
        return text_reply(kw["model"], "interpretation")

    fake_api.responder = respond
    result, record = simulation._execute_skill("causal_inference", "ab_test_bad_control", {}, scenario)
    assert record["status"] == "complete" and record["summary"]["verdict"] == "caution"
    bench = simulation.evaluate_output_step(PERSONA, "causal_inference", "ab_test_bad_control", result, scenario)
    assert bench["quantitative"]["score"] == 5.0          # bad control flagged, estimate ≈ true 2.0
    assert "Verdict: caution" in simulation._result_summary("causal_inference", result)

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

"""Failure paths surface a clear error instead of an empty one or a made-up result."""
import json

import pytest

import databricks_client as dc
import skills.api_retry as api_retry
from skills import data_quality, eda, leakage_detector
from skills.api_retry import call_context, strict_tool, tool_input
from tests.conftest import refusal_reply, text_reply


def test_text_reply_instead_of_tool_gives_readable_error(fake_api):
    fake_api.responder = lambda kw: text_reply(kw["model"], "Here are my thoughts in prose.")
    out = eda.analyze(dc._gen_ecommerce(100), "ecommerce")
    assert "did not call report_eda" in out["error"]
    assert "prose" in out["error"]  # start of the reply is included for debugging


def test_refusal_is_raised_logged_and_tagged(fake_api):
    fake_api.responder = lambda kw: refusal_reply(kw["model"])
    with call_context(dataset="transactions_fraud"):
        out = leakage_detector.check(dc._gen_transactions_fraud(100), "is_fraud")
    assert "declined" in out["error"]

    entries = [json.loads(line) for line in api_retry.REFUSAL_LOG.read_text().splitlines()]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["skill"] == "leakage_detector"
    assert entry["dataset"] == "transactions_fraud"
    assert entry["category"] == "cyber"
    assert entry["prompt"]["user_head"]


def test_skill_entry_uses_table_name_as_dataset(fake_api):
    fake_api.responder = lambda kw: refusal_reply(kw["model"])
    data_quality.assess(dc._gen_orders_dirty(50), table_name="orders_dirty")
    entry = json.loads(api_retry.REFUSAL_LOG.read_text().splitlines()[0])
    assert (entry["skill"], entry["dataset"]) == ("data_quality", "orders_dirty")


def test_tool_input_requires_the_named_tool():
    from tests.conftest import tool_reply
    resp = tool_reply("m", "other_tool", {"a": 1})
    with pytest.raises(RuntimeError, match="did not call wanted"):
        tool_input(resp, "wanted")
    assert tool_input(tool_reply("m", "wanted", {"a": 1}), "wanted") == {"a": 1}


def test_strict_tool_moves_bounds_to_description_without_mutating():
    tool = {"name": "t", "input_schema": {"type": "object", "properties": {
        "score": {"type": "integer", "minimum": 0, "maximum": 100, "description": "Score"},
        "items": {"type": "array", "items": {"type": "object", "properties": {"x": {"type": "string"}}}},
    }}}
    strict = strict_tool(tool)
    assert strict["strict"] is True
    score = strict["input_schema"]["properties"]["score"]
    assert "minimum" not in score and "maximum" not in score
    assert "minimum=0" in score["description"] and "maximum=100" in score["description"]
    assert strict["input_schema"]["additionalProperties"] is False
    assert strict["input_schema"]["properties"]["items"]["items"]["additionalProperties"] is False
    assert "minimum" in tool["input_schema"]["properties"]["score"]  # original untouched

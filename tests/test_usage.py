"""Usage ledger: every call recorded with who/what/cost; never prompts."""
import json
from types import SimpleNamespace as NS

import databricks_client as dc
from config import call_cost
from skills import data_quality, usage
from skills.api_retry import call_context
from tests.conftest import refusal_reply, tool_reply


def _with_usage(resp, i=1000, o=200, cr=0, cw=0):
    resp.usage = NS(input_tokens=i, output_tokens=o, cache_read_input_tokens=cr, cache_creation_input_tokens=cw)
    return resp


def _rows():
    return [json.loads(line) for line in usage.LEDGER_FILE.read_text().splitlines()]


def _dq_responder(kw):
    if kw.get("tool_choice", {}).get("type") == "tool":
        return _with_usage(tool_reply(kw["model"], "report_data_quality", {"summary": "s", "findings": []}), 3000, 400)
    from tests.conftest import text_reply
    return _with_usage(text_reply(kw["model"], "analysis"), 2000, 1500)


def test_each_call_is_recorded_and_runs_are_grouped(fake_api):
    fake_api.responder = _dq_responder
    with call_context(user="ana@example.com"):
        data_quality.assess(dc._gen_orders_dirty(50), "orders")
    rows = _rows()
    assert len(rows) == 2                                    # Pass 1 + Pass 2
    assert {r["skill"] for r in rows} == {"data_quality"}
    assert len({r["run_id"] for r in rows}) == 1 and rows[0]["run_id"]
    assert all(r["user"] == "ana@example.com" and r["status"] == "ok" for r in rows)
    assert rows[0]["cost_usd"] == round(call_cost(rows[0]["model"], 2000, 1500), 6)
    assert "messages" not in json.dumps(rows) and "analysis" not in json.dumps(rows)   # no content


def test_refusals_and_truncations_are_statuses(fake_api):
    fake_api.responder = lambda kw: _with_usage(refusal_reply(kw["model"]), 500, 0)
    data_quality.assess(dc._gen_orders_dirty(50), "orders")
    assert _rows()[-1]["status"] == "refusal"
    fake_api.responder = lambda kw: _with_usage(NS(model=kw["model"], stop_reason="max_tokens", content=[]), 500, 900)
    data_quality.assess(dc._gen_orders_dirty(50), "orders")
    assert _rows()[-1]["status"] == "truncated"


def test_user_resolution_order(monkeypatch):
    monkeypatch.setenv("WORKBENCH_USER", "env-user")
    usage.set_user_resolver(lambda: "app-user")
    try:
        assert usage.current_user({"user": "explicit"}) == "explicit"
        assert usage.current_user({}) == "app-user"
        usage.set_user_resolver(lambda: None)
        assert usage.current_user({}) == "env-user"
    finally:
        usage.set_user_resolver(None)


def test_cache_tokens_are_priced():
    full = call_cost("claude-sonnet-5-5", 1_000_000, 0)
    assert call_cost("claude-sonnet-5-5", 0, 0, cache_read_tokens=1_000_000) == full * 0.05
    assert call_cost("claude-sonnet-5-5", 0, 0, cache_write_tokens=1_000_000) == full * 1.25


def test_streamed_calls_are_recorded(fake_api):
    from skills.api_retry import get_client
    fake_api.responder = lambda kw: _with_usage(NS(model=kw["model"], stop_reason="end_turn", content=[]), 10, 20)
    with call_context(skill="code_draft"):
        with get_client().messages.stream(model="claude-sonnet-5-5", max_tokens=10, messages=[]) as s:
            s.get_final_message()
    row = _rows()[-1]
    assert (row["skill"], row["output_tokens"]) == ("code_draft", 20)


def test_summary_by_skill_user_and_success(tmp_path):
    rows = [
        # a successful two-call DQ run by ana, a failed leakage run by ben, an un-run sim call
        {"ts": "2026-10-20T09:00:00+00:00", "user": "ana", "skill": "data_quality", "run_id": "r1",
         "model": "m", "status": "ok", "cost_usd": 0.02, "latency_s": 20},
        {"ts": "2026-10-20T09:00:30+00:00", "user": "ana", "skill": "data_quality", "run_id": "r1",
         "model": "h", "status": "ok", "cost_usd": 0.001, "latency_s": 5},
        {"ts": "2026-10-20T10:00:00+00:00", "user": "ben", "skill": "leakage_detector", "run_id": "r2",
         "model": "m", "status": "refusal", "cost_usd": 0.01, "latency_s": 8},
        {"ts": "2026-10-20T11:00:00+00:00", "user": "ana", "skill": "simulation", "run_id": None,
         "model": "m", "status": "ok", "cost_usd": 0.005, "latency_s": 3},
    ]
    path = tmp_path / "u.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    s = usage.summarize(usage.load(path))
    assert s["runs"] == 2 and s["calls"] == 4 and s["users"] == 2
    assert round(s["total_spend"], 3) == 0.036 and round(s["failure_spend"], 3) == 0.01
    assert round(s["cost_per_successful_run"], 3) == 0.031            # both runs' cost / 1 good run
    skills = {r["skill"]: r for r in s["by_skill"]}
    assert skills["data_quality"]["success_rate"] == 1.0 and skills["leakage_detector"]["success_rate"] == 0.0
    assert {r["user"] for r in s["by_user"]} == {"ana", "ben"}

    import app  # noqa: F401
    from pages.costs import cost_panel
    assert "leakage_detector" in str(cost_panel(s).to_plotly_json())

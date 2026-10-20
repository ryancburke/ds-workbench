"""Team sessions: the 30-second view a lead scans."""
import databricks_client as dc
from skills import team_sessions as ts
from skills.session import WorkflowSession

DQ_CLEAN = {"overall_score": 92, "findings": [{"severity": "info", "column": "x", "description": "d"}]}
DQ_BLOCKED = {"overall_score": 40, "findings": [{"severity": "critical", "column": "amount", "description": "40% null"}]}
LD_CLEAN = {"verdict": "clean", "findings": [], "drop_columns": []}
MS_READY = {"data_readiness": "ready", "decision": {"model": "GBT", "primary_metric": "PR-AUC"}}


def _save(sid, results, datasets=("orders",), updated=None):
    runs = [{"skill": k, "run_id": f"{sid}_{k}"} for k in results]
    return ts.save_session(sid, owner="ana", question=f"q {sid}", datasets=list(datasets),
                           results=results, runs=runs, source="workbench")


def test_verdicts_and_gates():
    ready = ts.summarize_session({"results": {"data_quality": DQ_CLEAN, "leakage_detector": LD_CLEAN,
                                              "model_selector": MS_READY}, "runs": []})
    assert (ready["verdict"], ready["attention"]) == ("ready_for_review", "green")
    assert ready["gates"]["feature_engineering"] == "skipped"
    blocked = ts.summarize_session({"results": {"data_quality": DQ_BLOCKED}, "runs": []})
    assert blocked["verdict"] == "blocked" and blocked["gates"]["data_quality"] == "blocked"


def test_modelling_without_prerequisite_checks_is_incomplete():
    s = ts.summarize_session({"results": {"model_selector": MS_READY}, "runs": [{"skill": "model_selector"}]})
    assert (s["verdict"], s["attention"]) == ("incomplete", "amber")
    assert s["missing"] == ["data_quality", "leakage_detector"]
    assert s["gates"]["data_quality"] == "skipped"


def test_overview_orders_by_attention_and_tracks_change_on_the_same_data():
    _save("s1", {"data_quality": DQ_BLOCKED})
    _save("s2", {"data_quality": DQ_CLEAN, "leakage_detector": LD_CLEAN})
    _save("s3", {"data_quality": DQ_CLEAN}, datasets=("other_table",))
    rows = ts.team_overview()
    assert [r["id"] for r in rows][0] == "s1"                 # blocked first
    s2 = next(r for r in rows if r["id"] == "s2")
    assert s2["blocker_delta"] == -1 and s2["previous_session"] == "s1"
    assert next(r for r in rows if r["id"] == "s3")["blocker_delta"] is None


def test_notebook_session_persists_with_run_ids(fake_api):
    from tests.conftest import text_reply, tool_reply
    fake_api.responder = lambda kw: (tool_reply(kw["model"], "report_data_quality", {"summary": "s", "findings": []})
                                     if kw.get("tool_choice", {}).get("type") == "tool" else text_reply(kw["model"]))
    from skills import data_quality
    session = WorkflowSession("Is the orders table ready?", datasets=["orders"])
    data_quality.assess(dc._gen_orders_dirty(50), "orders", session=session)
    saved = ts.load_sessions()
    assert len(saved) == 1 and saved[0]["source"] == "notebook"
    assert saved[0]["runs"][0]["run_id"]                      # cost can be joined from the ledger
    assert "profile" not in saved[0]["results"]["data_quality"]


def test_workbench_store_maps_to_a_session():
    from workbench.callbacks.session_sync import store_to_session
    store = {"session": {"id": "wb_1", "active": True, "problem_statement": "q", "datasets": ["orders"]},
             "data_quality": [{"id": "dq_1729000000001", "status": "complete", "result": DQ_BLOCKED},
                              {"id": "dq_1729000000003", "status": "complete", "result": DQ_CLEAN}],
             "leakage_detector": [{"id": "ld_1729000000002", "status": "error", "error": "boom"}]}
    p = store_to_session(store)
    assert [r["skill"] for r in p["runs"]] == ["data_quality", "leakage_detector", "data_quality"]
    assert p["results"]["data_quality"] == DQ_CLEAN           # latest run wins
    assert p["results"]["leakage_detector"] == {"error": "boom"}
    assert store_to_session({"session": None}) is None


def test_sessions_page_renders():
    import app  # noqa: F401
    from pages.sessions import sessions_panel
    _save("s1", {"data_quality": DQ_BLOCKED})
    html_json = str(sessions_panel(ts.team_overview()).to_plotly_json())
    assert "Blocked" in html_json and "amount" in html_json

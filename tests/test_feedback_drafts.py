"""Feedback-loop drafts are exact, checked edits exported as patches."""
import pytest

from agents import scheduler
from tests.conftest import text_reply, tool_reply


def test_apply_edits_requires_a_unique_match():
    src = "a = 1\nb = 2\nb = 2\n"
    assert scheduler.apply_edits(src, [{"old": "a = 1", "new": "a = 3"}]) == "a = 3\nb = 2\nb = 2\n"
    with pytest.raises(ValueError, match="matches 2 times"):
        scheduler.apply_edits(src, [{"old": "b = 2", "new": "b = 4"}])
    with pytest.raises(ValueError, match="matches 0 times"):
        scheduler.apply_edits(src, [{"old": "c = 1", "new": ""}])


@pytest.mark.parametrize("rec,expected", [
    ({"skill_affected": "feature_engineering", "issue_type": "data_insight"}, "skills/feature_engineering.py"),
    ({"skill_affected": "mlops_review", "issue_type": "missing_feature"}, "skills/mlops_reviewer.py"),
    ({"skill_affected": "eda", "issue_type": "notebook_output_quality"}, "skills/result.py"),
    ({"skill_affected": "data_quality", "issue_type": "ux_friction"}, "workbench/renderers/dq.py"),
    ({"skill_affected": "notebook_generator", "issue_type": "ux_friction"}, "pages/home.py"),
    ({"skill_affected": "agent_sim", "issue_type": "ux_friction"}, "pages/agent_sim.py"),
    ({"skill_affected": "general", "issue_type": "visualization_gap"}, "pages/home.py"),
])
def test_target_file(rec, expected):
    assert scheduler.target_file(rec) == expected


@pytest.fixture
def rec_store(monkeypatch, tmp_path):
    rec = {"id": "r1", "title": "Show imbalance", "description": "d", "user_story": "u",
           "implementation_hint": "h", "skill_affected": "experiment_tracker", "issue_type": "data_insight",
           "status": "pending"}
    store = {"recommendations": [rec]}
    monkeypatch.setattr(scheduler, "_read_recs", lambda: store)
    monkeypatch.setattr(scheduler, "_write_recs", lambda data: None)
    monkeypatch.setattr(scheduler, "_PATCH_DIR", tmp_path / "patches")
    return rec


def _first_line(path):
    return (scheduler.Path(scheduler.__file__).parent.parent / path).read_text().splitlines()[0]


def test_draft_produces_a_diff_and_export_writes_a_patch(fake_api, rec_store):
    first = _first_line("skills/experiment_tracker.py")
    fake_api.responder = lambda kw: tool_reply(kw["model"], "propose_edits", {
        "summary": "Add a comment", "edits": [{"old": first, "new": "# patched\n" + first}]})
    path, diff = scheduler.draft_code_change("r1")
    assert path == "skills/experiment_tracker.py"
    assert "+# patched" in diff
    assert fake_api.calls[0]["tools"][0]["strict"] is True
    assert rec_store["status"] == "drafted"

    patch = scheduler.export_patch("r1")
    assert patch.read_text().count("+# patched") == 1
    assert rec_store["status"] == "implemented"


def test_draft_with_a_bad_edit_is_rejected(fake_api, rec_store):
    fake_api.responder = lambda kw: tool_reply(kw["model"], "propose_edits", {
        "summary": "s", "edits": [{"old": "text that is not in the file", "new": "x"}]})
    with pytest.raises(ValueError, match="matches 0 times"):
        scheduler.draft_code_change("r1")
    assert rec_store["status"] == "pending"


def test_draft_that_breaks_syntax_is_rejected(fake_api, rec_store):
    first = _first_line("skills/experiment_tracker.py")
    fake_api.responder = lambda kw: tool_reply(kw["model"], "propose_edits", {
        "summary": "s", "edits": [{"old": first, "new": "def broken(:\n"}]})
    with pytest.raises(SyntaxError):
        scheduler.draft_code_change("r1")


def test_export_refuses_a_stale_draft(fake_api, rec_store):
    rec_store.update(drafted_file="skills/experiment_tracker.py", drafted_diff="--- a\n+++ b\n",
                     drafted_base_sha="0000000000000000")
    with pytest.raises(ValueError, match="changed since this draft"):
        scheduler.export_patch("r1")


def test_stale_recommendations_revive_when_suggested_again(monkeypatch, tmp_path):
    import json
    recs = tmp_path / "recs.json"
    recs.write_text(json.dumps({"recommendations": [
        {"id": "old", "title": "Lead with the verdict before the findings", "status": "pending",
         "votes": 3, "created_at": "2026-10-17T10:00:00", "last_seen": "2026-10-17T10:00:00"},
        {"id": "gone", "title": "Show truncated evidence in full", "status": "pending",
         "votes": 2, "created_at": "2026-10-17T11:00:00", "last_seen": "2026-10-17T11:00:00"},
        {"id": "new", "title": "Add an effect size chart", "status": "pending",
         "votes": 1, "created_at": "2026-10-20T09:00:00", "last_seen": "2026-10-20T09:00:00"},
    ]}))
    monkeypatch.setattr(scheduler, "RECS_FILE", recs)
    assert scheduler.mark_stale("2026-10-19T09:38") == 2
    assert {r["id"] for r in scheduler.get_recommendations("pending")} == {"new"}

    rid = scheduler.add_recommendation({"title": "Lead with the verdict before findings", "persona_name": "p"})
    assert rid == "old"
    old = scheduler.get_rec("old")
    assert old["status"] == "pending" and old["votes"] == 4 and "revived_at" in old

    assert scheduler.dismiss_stale() == 1
    assert scheduler.get_rec("gone")["status"] == "dismissed"

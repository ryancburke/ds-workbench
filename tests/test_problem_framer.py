"""Problem Framer: the model drafts a ProblemSpec; code enforces its rules, derives the required
checks, and only a person confirms it."""
import json
import pathlib
from types import SimpleNamespace

import pytest

from ds_workbench.core.problem_spec import confirm, normalize, primary_question, required_checks
from ds_workbench.core.readiness import assess

ROOT = pathlib.Path(__file__).parent.parent

CAUSAL_DRAFT = {
    "decision": {"statement": "Expand the campaign or not", "options": ["expand", "stop"],
                 "criteria": "incremental margin exceeds cost", "inputs_needed": ["campaign cost per customer"]},
    "questions": [{"id": "q1", "type": "causal", "statement": "Did the campaign raise sales?", "primary": True,
                   "unit_of_analysis": "customer", "population": "active customers",
                   "outcome_concept": "sales in the 90 days after", "exposure_concept": "received the campaign",
                   "comparison": "customers who didn't", "time_window": "spring 2026"}],
    "assumptions": [{"statement": "Exposure was targeted, not random", "source": "user", "must_confirm": True}],
    "missing": [],
    "clarifying_questions": ["How were customers chosen?"],
    "answerability": {"status": "answerable", "reason": "exposure and outcome are defined"},
}


# ── Spec rules (code, not prompt) ─────────────────────────────────────────────

def test_a_draft_is_always_a_draft_and_unknown_fields_are_dropped():
    spec = normalize({**CAUSAL_DRAFT, "status": "confirmed", "confirmed_by": "model", "secret": 1})
    assert spec["status"] == "draft" and "confirmed_by" not in spec and "secret" not in spec


def test_decision_inputs_are_optional_and_kept_with_the_decision():
    spec = normalize(CAUSAL_DRAFT)
    assert spec["decision"]["inputs_needed"] == ["campaign cost per customer"]
    assert not any(m["blocking"] for m in spec["missing"])
    assert spec["answerability"]["status"] == "answerable_with_gaps"     # can't claim fully answerable


def test_a_causal_question_without_an_exposure_blocks():
    draft = json.loads(json.dumps(CAUSAL_DRAFT))
    del draft["questions"][0]["exposure_concept"]
    spec = normalize(draft)
    assert any(m["blocking"] and "exposure" in m["item"] for m in spec["missing"])
    assert spec["answerability"]["status"] == "needs_clarification"


def test_an_unrecognised_question_type_blocks():
    spec = normalize({"questions": [{"type": "vibes", "statement": "?"}],
                      "answerability": {"status": "answerable", "reason": ""}})
    assert spec["questions"][0]["type"] == "unrecognised"
    assert spec["answerability"]["status"] == "needs_clarification"


def test_no_questions_or_unknown_answerability_needs_clarification():
    assert normalize({})["answerability"]["status"] == "needs_clarification"
    s = normalize({**CAUSAL_DRAFT, "answerability": {"status": "probably"}})
    assert s["answerability"]["status"] == "needs_clarification"


def test_a_predictive_question_without_a_prediction_point_records_the_gap():
    spec = normalize({"questions": [{"type": "predictive", "statement": "who cancels", "outcome_concept": "cancel"}],
                      "answerability": {"status": "answerable", "reason": ""}})
    gap = next(m for m in spec["missing"] if "prediction point" in m["item"])
    assert gap["blocking"] is False and spec["answerability"]["status"] == "answerable_with_gaps"


def test_exactly_one_primary_question():
    qs = [{"type": "causal", "statement": "a", "primary": True, "exposure_concept": "x", "outcome_concept": "y"},
          {"type": "descriptive", "statement": "b", "primary": True}]
    spec = normalize({"questions": qs})
    assert [q["primary"] for q in spec["questions"]] == [True, False]
    spec = normalize({"questions": [{"type": "descriptive", "statement": "c"}]})
    assert spec["questions"][0]["primary"] is True


@pytest.mark.parametrize("types,checks", [
    (["causal"], ["data_quality", "causal_inference"]),
    (["predictive"], ["data_quality", "leakage_detector"]),
    (["descriptive"], ["data_quality"]),
    (["diagnostic"], ["data_quality"]),
    (["causal", "predictive"], ["data_quality", "causal_inference", "leakage_detector"]),
])
def test_required_checks_follow_the_question_types(types, checks):
    spec = {"questions": [{"type": t} for t in types]}
    assert list(required_checks(spec)) == checks


def test_optimisation_can_be_framed_but_says_no_skill_exists_yet():
    spec = normalize({"questions": [{"type": "optimization", "statement": "best price", "outcome_concept": "margin"}],
                      "answerability": {"status": "answerable", "reason": ""}})
    assert any("optimisation skill" in m["item"] for m in spec["missing"])
    assert spec["answerability"]["status"] == "answerable_with_gaps"


def test_only_a_person_confirms():
    spec = normalize(CAUSAL_DRAFT)
    with pytest.raises(ValueError):
        confirm(spec, user=" ")
    c = confirm(spec, user="ana@example.com", note="exposure = got the email", ts="2026-10-21T09:00:00+00:00")
    assert (c["status"], c["confirmed_by"], c["confirmed_at"]) == ("confirmed", "ana@example.com",
                                                                   "2026-10-21T09:00:00+00:00")


# ── End to end with the model faked ──────────────────────────────────────────

def _fake_model(monkeypatch, report: dict, seen: list):
    from skills import problem_framer

    def fake_api_call(_create, **kw):
        seen.append(kw)
        if "tools" in kw:
            return SimpleNamespace(content=[SimpleNamespace(type="tool_use", name="report_problem_spec", input=report)],
                                   stop_reason="tool_use")
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="reasoning")], stop_reason="end_turn")
    monkeypatch.setattr(problem_framer, "api_call", fake_api_call)
    monkeypatch.setattr(problem_framer, "get_client", lambda: SimpleNamespace(messages=SimpleNamespace(create=None)))


def test_frame_declares_the_checks_on_the_session_and_a_person_confirms(monkeypatch):
    from skills import problem_framer, team_sessions
    from skills.session import WorkflowSession
    seen = []
    _fake_model(monkeypatch, CAUSAL_DRAFT, seen)
    monkeypatch.setenv("WORKBENCH_USER", "ana@example.com")
    s = WorkflowSession(persist=False)
    spec = problem_framer.frame("Did the spring campaign raise sales, and should we expand it?",
                                context="Marketing picked the customers", session=s)
    assert "Marketing picked the customers" in seen[0]["messages"][0]["content"]
    assert seen[1]["tool_choice"] == {"type": "tool", "name": "report_problem_spec"}
    assert spec["status"] == "draft" and s.spec["status"] == "draft"
    assert s.required_checks == ["data_quality", "causal_inference"]
    assert s.context.startswith("Did the spring campaign")

    # A causal investigation is ready without a leakage scan
    s.register("data_quality", {"overall_score": 95, "findings": []})
    s.register("causal_inference", {"verdict": "valid", "findings": []})
    assert s.digest()["overall_verdict"] == "ready_for_review"

    confirmed = s.confirm_spec(note="exposure = got the email")
    assert confirmed["confirmed_by"] == "ana@example.com" and confirmed["status"] == "confirmed"

    row = team_sessions.summarize_session({"results": dict(s._results), "runs": [],
                                           "required_checks": s.required_checks, "spec": s.spec})
    assert row["framing"]["type"] == "causal" and row["framing"]["status"] == "confirmed"
    assert row["verdict"] == "ready_for_review"


def test_confirm_spec_needs_a_spec():
    from skills.session import WorkflowSession
    with pytest.raises(ValueError, match="problem_framer.frame"):
        WorkflowSession(persist=False).confirm_spec()


def test_frame_returns_an_error_instead_of_raising(monkeypatch):
    from skills import problem_framer
    assert "error" in problem_framer.frame("   ")

    def boom(*a, **k):
        raise RuntimeError("model declined")
    monkeypatch.setattr(problem_framer, "api_call", boom)
    monkeypatch.setattr(problem_framer, "get_client", lambda: SimpleNamespace(messages=SimpleNamespace(create=None)))
    assert "declined" in problem_framer.frame("why did sales drop?")["error"]


def test_the_notebook_view_leads_with_answerability_and_says_how_to_confirm():
    from skills.result import SkillResult
    draft = json.loads(json.dumps(CAUSAL_DRAFT))
    del draft["questions"][0]["exposure_concept"]
    html = SkillResult(normalize(draft, "q"), skill="problem_framer")._repr_html_()
    assert "NEEDS CLARIFICATION" in html and "Blocking: Define the exposure" in html
    assert "DRAFT" in html and "session.confirm_spec" in html


def test_the_sessions_page_shows_the_framing():
    import app  # noqa: F401
    from pages.sessions import sessions_panel
    from skills import team_sessions
    spec = normalize(CAUSAL_DRAFT, "Did the spring campaign raise sales?")
    session = {"id": "s1", "owner": "ana", "question": "q", "datasets": [], "source": "notebook",
               "started_at": "2026-10-21T09:00:00+00:00", "updated_at": "2026-10-21T09:05:00+00:00",
               "results": {}, "runs": [], "spec": spec, "required_checks": spec["required_checks"]}
    html = str(sessions_panel(team_sessions.team_overview([session])).to_plotly_json())
    assert "Framing: " in html and "draft — not confirmed" in html and "causal" in html


# ── Benchmark integrity and scoring ───────────────────────────────────────────

KEYS = sorted((ROOT / "benchmarks" / "answer_keys" / "framing").glob("*.json"))


@pytest.mark.parametrize("path", KEYS, ids=lambda p: p.stem)
def test_no_benchmark_question_or_scenario_word_appears_in_the_prompt(path):
    source = (ROOT / "skills" / "problem_framer.py").read_text().lower()
    key = json.loads(path.read_text())
    assert key["input"]["question"].lower() not in source
    # must_mention keywords are generic ("cost", "random"); the scenario's own words must not leak
    distinctive = {"loyalty tier", "checkout", "sku", "mobile app", "subscriber", "q3", "north region",
                   "lifetime value", "retention team", "nationwide", "inventory"}
    assert not [w for w in distinctive if w in source]


def _perfect(key):
    t = key["primary_type"][0]
    q = {"type": t, "statement": "s", "primary": True,
         **{f: kws[0] for f, kws in key["concepts"].items()}}
    if t == "causal":
        q.setdefault("exposure_concept", "x"); q.setdefault("outcome_concept", "y")
    return {"questions": [q] + [{"type": rt, "statement": "x"} for rt in key["required_types"] if rt != t],
            "answerability": {"status": key["answerability"][0], "reason": ""},
            "missing": [{"item": m["any_of"][0], "why_it_matters": "", "blocking": False} for m in key["must_mention"]],
            "assumptions": [], "clarifying_questions": [], "decision": {"inputs_needed": []}}


@pytest.mark.parametrize("path", KEYS, ids=lambda p: p.stem)
def test_a_correct_answer_passes_and_a_wrong_type_or_answerability_fails(path):
    from benchmarks.scoring import score_problem_framer
    key = json.loads(path.read_text())
    good = _perfect(key)
    r = score_problem_framer(good, key)
    assert r.passed and r.overall == 100, r.notes
    wrong_type = next(t for t in ("descriptive", "causal", "predictive") if t not in key["primary_type"]) \
        if len(key["primary_type"]) < 5 else None
    if wrong_type:
        bad = json.loads(json.dumps(good)); bad["questions"][0]["type"] = wrong_type
        assert not score_problem_framer(bad, key).passed
    wrong_ans = next(a for a in ("answerable", "not_answerable_as_posed", "needs_clarification")
                     if a not in key["answerability"])
    bad = json.loads(json.dumps(good)); bad["answerability"]["status"] = wrong_ans
    assert not score_problem_framer(bad, key).passed


def test_the_runner_sends_only_the_case_input():
    from benchmarks.runner import _get_benchmarks
    cases = [b for b in _get_benchmarks() if b["skill"] == "problem_framer"]
    assert len(cases) == len(KEYS) == 12
    for c in cases:
        assert set(c["df_fn"]()) == {"question", "context"}


def test_a_secondary_causal_question_without_an_exposure_records_a_gap_but_does_not_block():
    qs = [{"type": "diagnostic", "statement": "why did revenue fall", "primary": True, "outcome_concept": "revenue"},
          {"type": "causal", "statement": "if a driver is hypothesised, did it cause the fall?", "outcome_concept": "revenue"}]
    spec = normalize({"questions": qs, "answerability": {"status": "answerable", "reason": ""}})
    gap = next(m for m in spec["missing"] if "exposure" in m["item"])
    assert gap["blocking"] is False and spec["answerability"]["status"] == "answerable_with_gaps"


def test_blocking_gaps_are_listed_first():
    draft = {"questions": [{"type": "causal", "statement": "x", "outcome_concept": "y"}],
             "missing": [{"item": "minor", "why_it_matters": "", "blocking": False}]}
    assert normalize(draft)["missing"][0]["blocking"] is True

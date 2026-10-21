"""Decision synthesis: the status, conclusion and ROI are code; the narrative must cite evidence
and can't change them."""
import json
import pathlib
from types import SimpleNamespace

import pytest

from ds_workbench.core.decision import decision_readiness, roi, validate_narrative

ROOT = pathlib.Path(__file__).parent.parent
KEYS = sorted((ROOT / "benchmarks" / "answer_keys" / "decision").glob("*.json"))


def _case(name):
    return json.loads((ROOT / f"benchmarks/answer_keys/decision/decision_synthesis_{name}.json").read_text())


# ── Status, conclusion, ROI: code ─────────────────────────────────────────────

@pytest.mark.parametrize("path", KEYS, ids=lambda p: p.stem)
def test_each_case_gets_its_status_and_conclusion_from_code(path):
    key = json.loads(path.read_text())
    d = decision_readiness(key["input"]["state"], key["input"]["inputs"])
    assert d["status"] == key["expected_status"] and d["conclusion"] in key["expected_conclusion"]


def test_insufficient_evidence_hides_the_estimates_direction_and_withholds_roi():
    k = _case("confounded_campaign")
    d = decision_readiness(k["input"]["state"], k["input"]["inputs"])
    assert d["conclusion"] == "not_established" and d["estimate_direction"] == "positive_effect"
    assert d["roi"] is None and any("ROI not computed" in c for c in d["caveats"])


def test_roi_is_computed_across_the_interval_and_against_the_threshold():
    est = {"effect": 0.021, "ci_low": 0.012, "ci_high": 0.030}
    r = roi(est, {"cost": 15000, "units": 40000, "value_per_outcome_unit": 60, "roi_threshold": 0.2})
    assert (r["roi"], r["roi_low"], r["roi_high"], r["clears_threshold"]) == (2.36, 0.92, 3.8, "yes")
    r = roi(est, {"cost": 30000, "units": 40000, "value_per_outcome_unit": 60, "roi_threshold": 0.2})
    assert r["clears_threshold"] == "uncertain"
    assert roi(est, {"units": 10}) is None and roi(est, {"cost": 100}) is None
    assert roi(est, {"cost": 100, "units": 10})["assumed_value_per_unit"] is True


def test_an_uncertain_return_downgrades_supported_to_caveats():
    k = _case("ab_clean_roi")
    d = decision_readiness(k["input"]["state"], {**k["input"]["inputs"], "cost": 30000})
    assert d["status"] == "supported_with_caveats" and d["roi"]["clears_threshold"] == "uncertain"


def test_missing_checks_or_a_failed_estimate_mean_insufficient_evidence():
    k = _case("ab_clean_roi")
    state = json.loads(json.dumps(k["input"]["state"]))
    del state["results"]["data_quality"]
    assert decision_readiness(state)["status"] == "insufficient_evidence"
    state = json.loads(json.dumps(k["input"]["state"]))
    state["results"]["causal_inference"] = {"error": "model declined"}
    assert decision_readiness(state)["status"] == "insufficient_evidence"
    state["results"]["causal_inference"] = {"verdict": "probably_fine", "findings": []}
    assert decision_readiness(state)["status"] == "insufficient_evidence"
    assert decision_readiness({})["status"] == "insufficient_evidence"


def test_an_unconfirmed_spec_is_a_caveat_not_a_pass():
    k = _case("ab_clean_roi")
    state = json.loads(json.dumps(k["input"]["state"]))
    state["spec"]["status"] = "draft"
    d = decision_readiness(state, k["input"]["inputs"])
    assert d["status"] == "supported_with_caveats" and any("never confirmed" in c for c in d["caveats"])


def test_the_ledger_numbers_the_facts():
    k = _case("confounded_campaign")
    ev = decision_readiness(k["input"]["state"], k["input"]["inputs"])["evidence"]
    assert [e["id"] for e in ev] == [f"E{i}" for i in range(1, len(ev) + 1)]
    sources = {e["source"] for e in ev}
    assert {"spec", "readiness", "verification", "data_quality", "causal"} <= sources


# ── The narrative: cited or removed, never a status ──────────────────────────

def test_uncited_claims_are_removed_and_listed():
    ev = [{"id": "E1", "source": "spec", "text": ""}, {"id": "E2", "source": "causal", "text": ""}]
    out = validate_narrative({
        "headline": {"text": "Not established", "cites": ["E2"]},
        "established": [{"text": "a", "cites": ["E1"]}, {"text": "b", "cites": []}, {"text": "c", "cites": ["E9"]}],
        "status": "supported", "not_a_section": [{"text": "x", "cites": ["E1"]}]}, ev)
    n = out["narrative"]
    assert [c["text"] for c in n["established"]] == ["a"] and "status" not in n and "not_a_section" not in n
    assert [d["text"] for d in out["dropped"]] == ["b", "c"] and "E9" in out["dropped"][1]["reason"]
    assert validate_narrative({"headline": {"text": "h", "cites": []}}, ev)["narrative"]["headline"] is None


def _fake_model(monkeypatch, report):
    from skills import decision_synthesis
    seen = []

    def fake(_create, **kw):
        seen.append(kw)
        if "tools" in kw:
            return SimpleNamespace(content=[SimpleNamespace(type="tool_use", name="report_decision_summary", input=report)],
                                   stop_reason="tool_use")
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="draft")], stop_reason="end_turn")
    monkeypatch.setattr(decision_synthesis, "api_call", fake)
    monkeypatch.setattr(decision_synthesis, "get_client", lambda: SimpleNamespace(messages=SimpleNamespace(create=None)))
    return seen


def test_the_model_cannot_change_the_status(monkeypatch):
    from skills.decision_synthesis import synthesize
    k = _case("confounded_campaign")
    seen = _fake_model(monkeypatch, {"headline": {"text": "The campaign worked", "cites": ["E1"]},
                                     "status": "supported", "conclusion": "positive_effect",
                                     **{s: [] for s in ("established", "not_established", "alternative_explanations",
                                                        "assumptions_and_limitations", "options_and_tradeoffs", "next_steps")}})
    r = synthesize(state=k["input"]["state"], inputs=k["input"]["inputs"])
    assert (r["status"], r["conclusion"]) == ("insufficient_evidence", "not_established")
    prompt = seen[0]["messages"][0]["content"]
    assert "Decision status: insufficient_evidence" in prompt and "E1 [spec]" in prompt
    assert "status" not in json.dumps(seen[1]["tools"][0]["input_schema"]["properties"])
    from benchmarks.scoring import score_decision_synthesis
    assert not score_decision_synthesis(r, k).passed          # "the campaign worked" is a forbidden claim


def test_decide_works_from_a_session_at_any_point_and_is_saved(monkeypatch, tmp_path):
    import app  # noqa: F401
    from pages.sessions import sessions_panel
    from skills import team_sessions
    from skills.session import WorkflowSession
    monkeypatch.setattr(team_sessions, "SESSIONS_FILE", tmp_path / "s.json")
    k = _case("ab_no_effect")
    _fake_model(monkeypatch, {"headline": {"text": "No detectable effect on retention", "cites": ["E4"]},
                              **{s: [] for s in ("established", "not_established", "alternative_explanations",
                                                 "assumptions_and_limitations", "options_and_tradeoffs", "next_steps")}})
    s = WorkflowSession("retention", persist=True)
    s.spec = k["input"]["state"]["spec"]
    for skill, r in k["input"]["state"]["results"].items():
        s._results[skill] = r
    assert s.next()["step"] != "decide"          # the plan isn't finished, but deciding is allowed
    r = s.decide()
    assert r["status"] == "supported" and s.decision["conclusion"] == "no_detectable_effect"
    html = r._repr_html_()
    assert "SUPPORTED" in html and "status computed in code" in html and "E4" in html
    out = str(sessions_panel(team_sessions.team_overview()).to_plotly_json())
    assert "Decision: " in out and "No detectable effect on retention" in out


def test_a_blocked_plan_offers_decide_as_a_valid_outcome():
    from ds_workbench.core.plan import build_plan
    k = _case("confounded_campaign")
    state = json.loads(json.dumps(k["input"]["state"]))
    state["results"]["data_verification"]["verdict"] = "failed"
    state["selection"], state["scout"] = {"tables": ["t"]}, {"candidates": []}
    nxt = build_plan(state)["next"]
    assert nxt["kind"] == "fix" and "session.decide()" in nxt["text"]


@pytest.mark.parametrize("path", KEYS, ids=lambda p: p.stem)
def test_scoring_passes_a_good_summary_and_fails_forbidden_claims(path):
    from benchmarks.scoring import score_decision_synthesis
    key = json.loads(path.read_text())
    d = decision_readiness(key["input"]["state"], key["input"]["inputs"])
    good = {**d, "dropped": [], "narrative": {
        "headline": {"text": "h", "cites": ["E1"]},
        "established": [{"text": m["any_of"][0], "cites": ["E1"]} for m in key["must_mention"]]}}
    assert score_decision_synthesis(good, key).passed
    if key["forbidden_phrases"]:
        bad = json.loads(json.dumps(good))
        bad["narrative"]["established"].append({"text": key["forbidden_phrases"][0], "cites": ["E1"]})
        assert not score_decision_synthesis(bad, key).passed


def test_no_case_text_leaks_into_the_synthesis_prompt():
    source = (ROOT / "skills" / "decision_synthesis.py").read_text().lower()
    for path in KEYS:
        key = json.loads(path.read_text())
        assert key["input"]["state"]["spec"]["question"].lower() not in source
        # status names are the prompt's vocabulary, not case text
        from ds_workbench.core.decision import STATUS_LABEL
        labels = {v.lower() for v in STATUS_LABEL.values()}
        assert not [p for p in key["forbidden_phrases"] if p.lower() in source and p.lower() not in labels]


@pytest.mark.parametrize("text,claims", [
    ("The campaign increased purchases by 12%.", True),
    ("We can't say whether the campaign increased purchases.", False),
    ("We cannot yet say whether the campaign increased purchases: the groups differ.", False),
    ("There is no evidence the campaign increased purchases.", False),
    ("Targeting was not random; the campaign increased purchases among loyal members.", True),
    ("It is unclear if the campaign increased purchases.", False),
])
def test_forbidden_phrases_count_only_when_asserted(text, claims):
    from benchmarks.scoring import asserted
    assert asserted("campaign increased purchases", text) is claims


def test_forbidden_claims_are_checked_where_the_narrative_asserts():
    from benchmarks.scoring import score_decision_synthesis
    key = _case("national_price_not_answerable")
    d = decision_readiness(key["input"]["state"], key["input"]["inputs"])
    base = {**d, "dropped": [], "narrative": {"headline": {"text": "No comparison group: a staggered test is needed", "cites": ["E1"]}}}
    ok = json.loads(json.dumps(base))
    ok["narrative"]["not_established"] = [{"text": "That the price increase caused the drop.", "cites": ["E1"]}]
    assert score_decision_synthesis(ok, key).passed
    bad = json.loads(json.dumps(base))
    bad["narrative"]["established"] = [{"text": "The price increase caused the drop in volume.", "cites": ["E1"]}]
    assert not score_decision_synthesis(bad, key).passed

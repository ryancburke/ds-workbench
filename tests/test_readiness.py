"""One readiness policy (ds_workbench.core.readiness), used by preflight, sessions and the
Workbench: a missing or failed check never counts as passed."""
import pytest

from ds_workbench.core.readiness import assess, mandatory_for
from skills import preflight, team_sessions
from skills.session import build_digest

DQ_CLEAN = {"overall_score": 92, "findings": [{"severity": "info", "column": "x", "description": "minor"}]}
DQ_WARN = {"overall_score": 80, "findings": [{"severity": "warning", "column": "email", "description": "6% nulls"}]}
DQ_CRIT = {"overall_score": 40, "findings": [{"severity": "critical", "column": "amount", "description": "40% null"}]}
LD_CLEAN = {"verdict": "clean", "findings": []}
LD_LEAK = {"verdict": "leakage_detected", "findings": [{"severity": "critical", "column": "score_v2",
                                                         "evidence": "AUC 1.0", "recommendation": "drop"}]}
FAILED = {"error": "claude-opus-5-5 declined the request (cyber): test"}


# ── The three cases that used to return "ready" from preflight ───────────────

def test_preflight_with_leakage_never_run_is_incomplete():
    r = preflight.check(DQ_CLEAN)
    assert r["verdict"] == "incomplete" and r["missing"] == ["leakage_detector"]


def test_preflight_with_a_failed_leakage_check_is_incomplete_and_shows_the_error():
    r = preflight.check(DQ_CLEAN, FAILED)
    assert r["verdict"] == "incomplete"
    assert r["failed"] == [{"skill": "leakage_detector", "error": FAILED["error"]}]
    assert not r["leakage_available"]


def test_preflight_with_a_failed_dq_check_is_incomplete():
    r = preflight.check(FAILED, LD_CLEAN)
    assert r["verdict"] == "incomplete" and r["failed"][0]["skill"] == "data_quality"
    assert not any("Data quality" in c for c in r["cleared"])     # never narrated as clean


# ── Each rule ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("results,expected", [
    ({"data_quality": DQ_CLEAN}, "incomplete"),                                   # mandatory missing
    ({"data_quality": DQ_CLEAN, "leakage_detector": FAILED}, "incomplete"),       # mandatory failed
    ({"data_quality": DQ_CLEAN, "leakage_detector": LD_CLEAN,
      "model_selector": FAILED}, "incomplete"),                                   # any check failed
    ({"data_quality": DQ_CRIT, "leakage_detector": LD_CLEAN}, "blocked"),         # critical findings
    ({"data_quality": DQ_CLEAN, "leakage_detector": LD_LEAK}, "blocked"),
    ({"data_quality": DQ_WARN, "leakage_detector": LD_CLEAN}, "caution"),         # warnings
    ({"data_quality": DQ_CLEAN, "leakage_detector": LD_CLEAN}, "ready_for_review"),
    ({}, "incomplete"),
])
def test_policy(results, expected):
    assert assess(results)["status"] == expected


def test_a_known_blocker_outranks_a_missing_check_but_both_are_reported():
    r = assess({"data_quality": DQ_CRIT})
    assert r["status"] == "blocked" and r["missing"] == ["leakage_detector"]


def test_skill_level_verdicts_count():
    assert assess({"data_quality": DQ_CLEAN, "leakage_detector": {"verdict": "suspicious", "findings": []}})["status"] == "caution"
    assert assess({"data_quality": DQ_CLEAN, "leakage_detector": LD_CLEAN,
                   "model_selector": {"data_readiness": "insufficient"}})["status"] == "blocked"


def test_causal_work_has_its_own_mandatory_checks():
    causal = {"verdict": "valid", "findings": []}
    assert mandatory_for({"causal_inference": causal}) == ("data_quality", "causal_inference")
    assert assess({"data_quality": DQ_CLEAN, "causal_inference": causal})["status"] == "ready_for_review"
    assert assess({"data_quality": DQ_CLEAN, "causal_inference": {"verdict": "not_identified", "findings": []}})["status"] == "blocked"


# ── Every consumer agrees ────────────────────────────────────────────────────

CASES = [
    {"data_quality": DQ_CLEAN},
    {"data_quality": DQ_CLEAN, "leakage_detector": FAILED},
    {"data_quality": FAILED, "leakage_detector": LD_CLEAN},
    {"data_quality": DQ_CRIT, "leakage_detector": LD_CLEAN},
    {"data_quality": DQ_WARN, "leakage_detector": LD_CLEAN},
    {"data_quality": DQ_CLEAN, "leakage_detector": LD_CLEAN},
]


@pytest.mark.parametrize("results", CASES)
def test_preflight_sessions_workbench_and_digest_agree(results):
    import app  # noqa: F401
    from workbench.session_panels import _readiness_strip, latest_results

    expected = assess(results)["status"]
    assert preflight.check(results.get("data_quality"), results.get("leakage_detector"))["verdict"] == expected
    assert team_sessions.summarize_session({"results": results, "runs": []})["verdict"] == expected
    assert build_digest(results)["overall_verdict"] == expected

    # Workbench: the same results as runs in the store (a failed run stored as status "error")
    store = {skill: [{"id": f"{skill}_1", "status": "error", "error": r["error"]} if r.get("error")
                     else {"id": f"{skill}_1", "status": "complete", "result": r}]
             for skill, r in results.items()}
    assert assess(latest_results(store))["status"] == expected
    label = {"blocked": "BLOCKED", "incomplete": "INCOMPLETE", "caution": "CAUTION",
             "ready_for_review": "READY FOR REVIEW"}[expected]
    assert label in str(_readiness_strip(store).to_plotly_json())


def test_digest_never_lists_a_failed_check_as_cleared():
    d = build_digest({"data_quality": FAILED, "leakage_detector": LD_CLEAN})
    assert not any(c["skill"] == "data_quality" for c in d["cleared"])
    assert d["failed"][0]["skill"] == "data_quality" and "Re-run" in d["next_step"]


def test_renderers_never_default_an_unknown_verdict_to_ready():
    from skills.result import SkillResult
    html = SkillResult({"blockers": [], "cautions": [], "cleared": []}, skill="preflight")._repr_html_()
    assert "READY" not in html and "INCOMPLETE" in html

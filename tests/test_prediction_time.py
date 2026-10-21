"""Prediction-time awareness in leakage detection: the model reasons about timing, only a person
can confirm a column is available, and a confirmation applies to exactly one
(column, dataset, prediction point)."""
from types import SimpleNamespace

import pandas as pd
import pytest

from ds_workbench.core.readiness import assess, make_confirmation
from skills import leakage_detector, team_sessions
from skills.leakage_detector import STATISTICAL_ACTION, STATISTICS_ONLY_CAVEAT, finalize
from skills.session import WorkflowSession, build_digest

PP = "1st of each month, predicting churn in the next 30 days"
DQ_CLEAN = {"overall_score": 95, "findings": []}


def _ld(column="support_tickets_mtd", evidence_type="statistical", dataset="renewals", pp=PP, **extra):
    """A leakage result as the skill returns it after finalize()."""
    model = {"verdict": "leakage_detected", "risk_level": "high", "summary": "",
             "findings": [{"column": column, "leakage_type": "high_correlation", "severity": "critical",
                           "evidence_type": evidence_type, "evidence": "AUC 0.94", "recommendation": "drop",
                           **extra}]}
    return finalize(model, prediction_point=pp, dataset=dataset)


def _confirm(column="support_tickets_mtd", dataset="renewals", pp=PP, note="snapshot taken at 00:00 on the 1st"):
    return make_confirmation(column, dataset=dataset, prediction_point=pp, note=note, user="ana@example.com",
                             ts="2026-10-20T09:00:00+00:00")


# ── Confirmations clear the block, scoped to their context ───────────────────

def test_a_confirmed_column_no_longer_blocks_and_shows_as_info_with_attribution():
    r = assess({"data_quality": DQ_CLEAN, "leakage_detector": _ld()}, confirmations=[_confirm()])
    assert r["status"] == "ready_for_review" and r["blockers"] == []
    c = r["confirmed"][0]
    assert (c["column"], c["severity"], c["was"]) == ("support_tickets_mtd", "info", "critical")
    assert (c["confirmed_by"], c["confirmed_at"], c["note"]) == (
        "ana@example.com", "2026-10-20T09:00:00+00:00", "snapshot taken at 00:00 on the 1st")


def test_a_statistical_finding_without_confirmation_still_blocks():
    r = assess({"data_quality": DQ_CLEAN, "leakage_detector": _ld()})
    assert r["status"] == "blocked" and r["blockers"][0]["column"] == "support_tickets_mtd"
    assert r["confirmed"] == []


def test_a_confirmation_for_a_different_prediction_point_does_not_apply():
    other = _confirm(pp="15th of each month, predicting churn by month end")
    r = assess({"data_quality": DQ_CLEAN, "leakage_detector": _ld()}, confirmations=[other])
    assert r["status"] == "blocked" and r["confirmed"] == []


def test_a_confirmation_for_the_same_column_in_a_different_dataset_does_not_apply():
    r = assess({"data_quality": DQ_CLEAN, "leakage_detector": _ld(dataset="renewals_eu")},
               confirmations=[_confirm(dataset="renewals")])
    assert r["status"] == "blocked" and r["confirmed"] == []


def test_a_result_without_a_dataset_or_prediction_point_cannot_be_cleared():
    for result in (_ld(dataset=""), _ld(pp="")):
        r = assess({"data_quality": DQ_CLEAN, "leakage_detector": result}, confirmations=[_confirm()])
        assert r["status"] == "blocked"


def test_only_confirmed_columns_are_cleared():
    two = _ld()
    two["findings"].append({**two["findings"][0], "column": "retention_offer_code"})
    r = assess({"data_quality": DQ_CLEAN, "leakage_detector": two}, confirmations=[_confirm()])
    assert r["status"] == "blocked" and [b["column"] for b in r["blockers"]] == ["retention_offer_code"]


@pytest.mark.parametrize("missing", ["note", "dataset", "prediction_point", "user"])
def test_a_confirmation_needs_a_note_and_its_full_context(missing):
    kwargs = {"dataset": "renewals", "prediction_point": PP, "note": "checked the DAG", "user": "ana"}
    kwargs[missing] = "  "
    with pytest.raises(ValueError):
        make_confirmation("support_tickets_mtd", **kwargs)


# ── The model can never confirm a column ─────────────────────────────────────

def test_the_tool_schema_has_no_way_to_confirm():
    finding = leakage_detector._TOOL["input_schema"]["properties"]["findings"]["items"]["properties"]
    assert set(finding) == {"column", "leakage_type", "severity", "evidence_type", "evidence", "recommendation"}
    assert finding["evidence_type"]["enum"] == ["statistical", "timing"]
    assert not any("confirm" in k for k in leakage_detector._TOOL["input_schema"]["properties"])


def test_model_written_confirmations_are_stripped_and_ignored():
    model = {"verdict": "leakage_detected", "risk_level": "high", "summary": "",
             "confirmations": [_confirm()],
             "findings": [{"column": "support_tickets_mtd", "severity": "critical", "evidence_type": "statistical",
                           "evidence": "AUC 0.94", "recommendation": "", "confirmed": True,
                           "confirmed_by": "model", "note": "available"}]}
    out = finalize(model, prediction_point=PP, dataset="renewals")
    assert "confirmations" not in out and not {"confirmed", "confirmed_by", "note"} & set(out["findings"][0])
    # even an unsanitised result can't clear itself: confirmations only come from the caller
    for result in (out, model | {"dataset": "renewals", "prediction_point": PP}):
        assert assess({"data_quality": DQ_CLEAN, "leakage_detector": result})["status"] == "blocked"


# ── Evidence types and the statistics-only caveat ────────────────────────────

def test_output_without_a_prediction_point_carries_the_caveat_and_is_statistical_only():
    out = _ld(evidence_type="timing", pp="")
    assert out["timing_basis"] == "statistics_only" and out["caveat"] == STATISTICS_ONLY_CAVEAT
    assert out["findings"][0]["evidence_type"] == "statistical"     # timing can't be established
    with_pp = _ld(evidence_type="timing")
    assert with_pp["timing_basis"] == "prediction_point" and "caveat" not in with_pp
    assert with_pp["findings"][0]["evidence_type"] == "timing"


def test_statistical_findings_stay_critical_with_the_confirm_or_drop_action():
    f = _ld()["findings"][0]
    assert f["severity"] == "critical" and f["recommendation"].startswith(STATISTICAL_ACTION)
    assert _ld(evidence_type="timing")["findings"][0]["recommendation"] == "drop"


def test_an_unknown_evidence_type_is_statistical():
    assert _ld(evidence_type="vibes")["findings"][0]["evidence_type"] == "statistical"


# ── End to end: skill + session + Sessions page (model calls faked) ─────────

def _fake_model(monkeypatch, report: dict, seen: list):
    def fake_api_call(_create, **kw):
        seen.append(kw)
        if "tools" in kw:
            return SimpleNamespace(content=[SimpleNamespace(type="tool_use", name="report_leakage", input=report)],
                                   stop_reason="tool_use")
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="analysis")], stop_reason="end_turn")
    monkeypatch.setattr(leakage_detector, "api_call", fake_api_call)
    monkeypatch.setattr(leakage_detector, "get_client", lambda: SimpleNamespace(messages=SimpleNamespace(create=None)))


def _df():
    return pd.DataFrame({"support_tickets_mtd": [0, 1, 4, 5] * 25, "tenure_months": range(100),
                         "churned": [False, False, True, True] * 25})


REPORT = {"verdict": "leakage_detected", "risk_level": "high", "summary": "s",
          "findings": [{"column": "support_tickets_mtd", "leakage_type": "high_correlation",
                        "severity": "critical", "evidence_type": "statistical", "evidence": "AUC 0.94",
                        "recommendation": "drop"}]}


def test_the_prediction_point_reaches_the_model_and_the_result(monkeypatch):
    seen = []
    _fake_model(monkeypatch, REPORT, seen)
    s = WorkflowSession("churn", datasets=["renewals"], persist=False, prediction_point=PP,
                        column_notes={"support_tickets_mtd": "opened 1st-14th"})
    r = leakage_detector.check(_df(), "churned", session=s)
    prompt = seen[0]["messages"][0]["content"]
    assert f"Prediction point: {PP}" in prompt and "opened 1st-14th" in prompt
    assert (r["prediction_point"], r["dataset"], r["timing_basis"]) == (PP, "renewals", "prediction_point")


def test_without_a_prediction_point_the_prompt_and_output_say_statistics_only(monkeypatch):
    seen = []
    _fake_model(monkeypatch, REPORT, seen)
    r = leakage_detector.check(_df(), "churned")
    assert "No prediction point was given" in seen[0]["messages"][0]["content"]
    assert r["caveat"] == STATISTICS_ONLY_CAVEAT and STATISTICS_ONLY_CAVEAT in r._repr_html_()


def test_session_confirm_available_records_the_user_and_clears_the_block_everywhere(monkeypatch):
    seen = []
    _fake_model(monkeypatch, REPORT, seen)
    monkeypatch.setenv("WORKBENCH_USER", "ana@example.com")
    s = WorkflowSession("churn", datasets=["renewals"], persist=False, prediction_point=PP)
    s.register("data_quality", DQ_CLEAN)
    leakage_detector.check(_df(), "churned", session=s)
    assert build_digest(s._results, s.confirmations)["overall_verdict"] == "blocked"

    with pytest.raises(ValueError):
        s.confirm_available("support_tickets_mtd", note="")
    [c] = s.confirm_available("support_tickets_mtd", note="counted up to the prediction date")
    assert (c["user"], c["dataset"], c["prediction_point"]) == ("ana@example.com", "renewals", PP) and c["ts"]

    digest = build_digest(s._results, s.confirmations)
    assert digest["overall_verdict"] == "ready_for_review" and digest["blockers"] == []
    assert "confirmed available by ana@example.com" in s.digest()._repr_html_()

    record = {"results": dict(s._results), "runs": [], "confirmations": s.confirmations, "prediction_point": PP}
    summary = team_sessions.summarize_session(record)
    assert summary["verdict"] == "ready_for_review" and summary["confirmed"][0]["note"] == c["note"]

    # the prediction point changes: the old confirmation no longer applies
    s.prediction_point = "15th of each month, predicting churn by month end"
    leakage_detector.check(_df(), "churned", session=s)
    assert build_digest(s._results, s.confirmations)["overall_verdict"] == "blocked"


def test_confirming_needs_a_prediction_point():
    s = WorkflowSession("churn", datasets=["renewals"], persist=False)
    with pytest.raises(ValueError, match="prediction point"):
        s.confirm_available("x", note="checked")


def test_the_sessions_page_shows_who_confirmed():
    import app  # noqa: F401
    from pages.sessions import sessions_panel
    session = {"id": "s1", "owner": "ana", "question": "churn", "datasets": ["renewals"], "source": "notebook",
               "started_at": "2026-10-20T09:00:00+00:00", "updated_at": "2026-10-20T09:05:00+00:00",
               "results": {"data_quality": DQ_CLEAN, "leakage_detector": _ld()}, "runs": [],
               "prediction_point": PP, "confirmations": [_confirm()]}
    html = str(sessions_panel(team_sessions.team_overview([session])).to_plotly_json())
    assert "confirmed available by ana@example.com on 2026-10-20" in html and PP in html


def test_benchmark_inputs_carry_no_answers():
    """Both renewal_timing cases pass the same data and notes; only the prediction point differs,
    and nothing from the answer keys reaches the skill."""
    import json
    from pathlib import Path
    from benchmarks.runner import _get_benchmarks
    cases = [b for b in _get_benchmarks() if b["dataset"] == "renewal_timing"]
    assert len(cases) == 2
    a, b = (c["run_kwargs"] for c in cases)
    assert a["column_notes"] == b["column_notes"] and a["prediction_point"] != b["prediction_point"]
    for case in cases:
        key = json.loads((Path("benchmarks/answer_keys") / case["dataset"] / f"{case['benchmark']}.json").read_text())
        sent = json.dumps(case["run_kwargs"]).lower()
        for text in [key["_meta"]["description"], *[f.get("note", "") for f in
                     key.get("required_findings", []) + key.get("forbidden_findings", [])]]:
            assert text.lower() not in sent
        assert "leak" not in sent and "legitimate" not in sent

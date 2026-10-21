"""Pilot harness: blinded sheets, and scoring from expert ratings (offline)."""
import csv
import json

from benchmarks import pilot
from ds_workbench.core.catalog import SyntheticCatalog

CAT = SyntheticCatalog.load("retail_marketing")
QUESTIONS = json.loads(open("benchmarks/pilot_questions.example.json").read())


def _spec(q):
    return {"question": q, "status": "confirmed", "missing": [], "decision": {"inputs_needed": []},
            "questions": [{"id": "q1", "type": "causal", "primary": True, "statement": q,
                           "unit_of_analysis": "customer", "outcome_concept": "purchases", "exposure_concept": "campaign"}]}


def _fake_scout(spec, catalog):
    tables = ["demo.marketing.campaign_exposure", "demo.sales.customer_orders", "demo.crm.customers",
              "demo.sales.sales_forecast"]
    recs = ["investigate", "investigate", "investigate", "reject"]
    return {"run_id": None, "coverage_gaps": [], "validation": {"dropped": []}, "candidates": [
        {"table": t, "rank": i + 1, "recommendation": r, "why": f"why {t}", "concerns": []}
        for i, (t, r) in enumerate(zip(tables, recs))]}


def _prepare(tmp_path):
    return pilot.prepare(QUESTIONS, CAT, tmp_path, scout_fn=_fake_scout, frame_fn=lambda q, context="": _spec(q))


def test_the_rating_sheet_is_blind(tmp_path):
    key = _prepare(tmp_path)
    text = (tmp_path / pilot.SHEET).read_text().lower()
    assert "scout" not in text and "baseline" not in text
    rows = list(csv.DictReader((tmp_path / pilot.SHEET).open()))
    assert {r["list"] for r in rows} == {"A", "B"} and all(r["relevance_0_1_2"] == "" for r in rows)
    assert all(set(q["arms"].values()) == {"scout", "baseline"} for q in key["questions"].values())
    assert "demo.sales.sales_forecast" not in key["questions"]["q01"]["lists"]["scout"]   # rejects aren't offered


def test_scoring_from_filled_sheets(tmp_path):
    _prepare(tmp_path)
    expert = {q["id"]: set(q["expert_tables"]) for q in QUESTIONS}
    rows = list(csv.DictReader((tmp_path / pilot.SHEET).open()))
    for r in rows:
        r["relevance_0_1_2"] = "2" if r["table"] in expert[r["question_id"]] else "0"
    pilot._write(tmp_path / pilot.SHEET, rows)
    notes = list(csv.DictReader((tmp_path / pilot.NOTES).open()))
    for i, r in enumerate(notes):
        r["note_correct_y_n"] = "y" if i % 2 == 0 else "n"
    pilot._write(tmp_path / pilot.NOTES, notes)
    out = pilot.score(tmp_path)
    assert out["questions"] == 2 and out["unrated"] == 0
    assert out["scout"]["recall@5"] == round((1 + 1 / 3) / 2, 3)          # q01 all three; q02 only customers
    assert out["scout_confident_but_wrong"] == 2      # q02: exposure and orders were 'investigate', rated 0
    assert out["scout_note_accuracy"] == 0.5
    assert (tmp_path / "results.json").exists()


def test_a_failed_question_is_reported_not_fatal(tmp_path):
    key = pilot.prepare(QUESTIONS, CAT, tmp_path, scout_fn=lambda spec, catalog: {"error": "refused"},
                        frame_fn=lambda q, context="": _spec(q))
    assert all("scout failed" in q["error"] for q in key["questions"].values())
    assert pilot.score(tmp_path)["failed"] == 2

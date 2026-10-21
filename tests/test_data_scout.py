"""Data Scout: a bounded metadata-only tool loop whose every claim is validated in code against
what the catalog returned."""
import json
import pathlib
from types import SimpleNamespace

import pytest

from ds_workbench.core.catalog import NotAvailable, SyntheticCatalog
from ds_workbench.core.problem_spec import confirm, normalize
from ds_workbench.core.scout import concepts_from_spec, validate_report

ROOT = pathlib.Path(__file__).parent.parent
EXPOSURE, ORDERS, CUSTOMERS = "demo.marketing.campaign_exposure", "demo.sales.customer_orders", "demo.crm.customers"


@pytest.fixture
def cat():
    return SyntheticCatalog.load("retail_marketing")


def _spec(confirmed=True):
    s = normalize({"questions": [{"id": "q1", "type": "causal", "primary": True, "statement": "campaign effect",
                                  "unit_of_analysis": "customer", "population": "customers",
                                  "outcome_concept": "purchases after", "exposure_concept": "received the campaign"}],
                   "answerability": {"status": "answerable_with_gaps", "reason": ""}}, "Did the campaign work?")
    return confirm(s, user="ana", ts="2026-10-21T00:00:00+00:00") if confirmed else s


# ── Catalog ───────────────────────────────────────────────────────────────────

def test_the_synthetic_catalog_answers_metadata_only(cat):
    hits = cat.search(["campaign"])
    assert hits and all(set(h) == {"name", "comment", "score", "matched"} for h in hits)
    meta = cat.describe(EXPOSURE)
    assert {c["name"] for c in meta["columns"]} >= {"email_hash", "sent_at"} and "rows" not in meta
    meta["columns"].clear()
    assert cat.describe(EXPOSURE)["columns"]                      # callers get a copy
    with pytest.raises(KeyError):
        cat.describe("demo.nope.nothing")
    assert cat.lineage("demo.sales.store_weekly_sales")["upstream"] == [ORDERS]


def test_lineage_is_optional():
    c = SyntheticCatalog({"name": "x", "tables": [{"name": "a.b.c", "columns": []}]})
    assert not c.supports_lineage
    with pytest.raises(NotAvailable):
        c.lineage("a.b.c")


@pytest.mark.parametrize("bad", ["../secrets", "retail/marketing", "", "Retail"])
def test_fixture_names_cannot_escape_the_catalog_folder(bad):
    with pytest.raises(ValueError):
        SyntheticCatalog.load(bad)


# ── Validation: no invented tables, columns or joins ──────────────────────────

def _cand(table, **kw):
    return {"table": table, "rank": 1, "recommendation": "investigate", "why": "", "grain": "", "role_matches": [],
            "join_keys": [], "concerns": [], "expert_assumptions": [], **kw}


def test_ungrounded_claims_are_dropped_and_listed(cat):
    described = {t: cat.describe(t) for t in (EXPOSURE, CUSTOMERS)}
    concepts = concepts_from_spec(_spec())
    report = {"candidates": [
        _cand("demo.sales.invented_table"),
        _cand(EXPOSURE, role_matches=[{"concept": "q1.exposure", "column": "sent_at", "basis": "send time"},
                                      {"concept": "q1.exposure", "column": "made_up_col", "basis": ""},
                                      {"concept": "q9.whatever", "column": "opened", "basis": ""}],
              join_keys=[{"left_table": EXPOSURE, "left_column": "email_hash", "right_table": CUSTOMERS,
                          "right_column": "email_hash", "note": ""},
                         {"left_table": EXPOSURE, "left_column": "opened", "right_table": CUSTOMERS,
                          "right_column": "customer_id", "note": "types differ"},
                         {"left_table": EXPOSURE, "left_column": "email_hash", "right_table": ORDERS,
                          "right_column": "customer_id", "note": "orders never described"}]),
    ]}
    out = validate_report(report, described, concepts)
    assert [c["table"] for c in out["candidates"]] == [EXPOSURE]
    [c] = out["candidates"]
    assert [r["column"] for r in c["role_matches"]] == ["sent_at"]
    assert c["role_matches"][0]["evidence"] == "metadata"
    assert [(j["left_column"], j["status"]) for j in c["join_keys"]] == [("email_hash", "name_match")]
    reasons = " | ".join(d["reason"] for d in out["validation"]["dropped"])
    for why in ("not described", "no such column", "unknown concept", "types differ", "wasn't described"):
        assert why in reasons


def test_join_status_comes_from_code_never_verified(cat):
    described = {t: cat.describe(t) for t in (CUSTOMERS, "demo.crm.subscriptions")}
    j = {"left_table": CUSTOMERS, "left_column": "email_hash", "right_table": "demo.crm.subscriptions",
         "right_column": "plan", "note": "", "status": "verified"}
    out = validate_report({"candidates": [_cand(CUSTOMERS, join_keys=[j])]}, described, [])
    assert out["candidates"][0]["join_keys"][0]["status"] == "type_compatible"


def test_coverage_is_computed_not_claimed(cat):
    described = {EXPOSURE: cat.describe(EXPOSURE)}
    concepts = concepts_from_spec(_spec())
    report = {"candidates": [_cand(EXPOSURE, role_matches=[{"concept": "q1.exposure", "column": "sent_at", "basis": ""}])],
              "coverage_notes": [{"concept": "q1.outcome", "note": "covered, trust me"}]}
    out = validate_report(report, described, concepts)
    assert "q1.outcome" in out["coverage_gaps"] and "q1.exposure" not in out["coverage_gaps"]
    assert out["feasibility"] == "not_covered"          # a required concept (outcome) has no data
    maybe = validate_report({"candidates": [_cand(EXPOSURE, recommendation="maybe", role_matches=[
        {"concept": "q1.exposure", "column": "sent_at", "basis": ""}])]}, described, concepts)
    assert "q1.exposure" in maybe["coverage_gaps"]       # only 'investigate' candidates cover a concept


# ── The loop, with the model scripted ────────────────────────────────────────

def _tool(name, inp, i=[0]):
    i[0] += 1
    return SimpleNamespace(type="tool_use", name=name, input=inp, id=f"tu_{i[0]}")


def _scripted(monkeypatch, turns):
    from skills import data_scout
    seen = []

    def fake_api_call(_create, **kw):
        seen.append({**kw, "messages": list(kw["messages"])})      # snapshot: the loop keeps appending
        content = turns.pop(0) if turns else [_tool("search_catalog", {"terms": ["again"]})]
        return SimpleNamespace(content=content, stop_reason="tool_use")
    monkeypatch.setattr(data_scout, "api_call", fake_api_call)
    monkeypatch.setattr(data_scout, "get_client", lambda: SimpleNamespace(messages=SimpleNamespace(create=None)))
    return seen


GOOD_REPORT = {"summary": "s", "coverage_notes": [], "candidates": [
    _cand(EXPOSURE, role_matches=[{"concept": "q1.exposure", "column": "sent_at", "basis": "send log"}],
          join_keys=[{"left_table": EXPOSURE, "left_column": "email_hash", "right_table": CUSTOMERS,
                      "right_column": "email_hash", "note": "bridge"}]),
    _cand(ORDERS, rank=2, role_matches=[{"concept": "q1.outcome", "column": "net_amount_eur", "basis": ""}],
          join_keys=[{"left_table": CUSTOMERS, "left_column": "customer_id", "right_table": ORDERS,
                      "right_column": "customer_id", "note": ""}]),
    _cand(CUSTOMERS, rank=3, role_matches=[{"concept": "q1.unit", "column": "customer_id", "basis": ""}]),
    _cand("demo.sales.sales_forecast", recommendation="reject", why="forecasts, not actuals"),
]}


def test_the_scout_runs_catalog_tools_and_validates_its_report(monkeypatch, cat):
    from skills import data_scout
    from skills.session import WorkflowSession
    turns = [[_tool("search_catalog", {"terms": ["campaign", "orders"]})],
             [_tool("describe_table", {"table": t}) for t in (EXPOSURE, ORDERS, CUSTOMERS)],
             [_tool("describe_table", {"table": "demo.sales.sales_forecast"}), _tool("get_lineage", {"table": ORDERS})],
             [_tool("report_candidates", GOOD_REPORT)]]
    seen = _scripted(monkeypatch, turns)
    s = WorkflowSession(persist=False)
    s.spec = _spec()
    r = data_scout.scout(session=s, catalog=cat)
    assert "error" not in r, r.get("error")
    assert [c["tool"] for c in r["catalog_calls"]] == ["search_catalog"] + ["describe_table"] * 4 + ["get_lineage"]
    # population isn't mapped by any column: reported, but not a required concept
    assert r["feasibility"] == "partially_covered" and r["coverage_gaps"] == ["q1.population"]
    assert r["validation"]["dropped"] == []
    assert [c["table"] for c in r["candidates"]][-1] == "demo.sales.sales_forecast"   # rejects last
    assert s.scout["feasibility"] == "partially_covered"
    # tool results went back to the model as tool_result blocks; the tool loop ran on the scout role
    assert seen[1]["messages"][-1]["content"][0]["type"] == "tool_result"
    assert seen[0]["tool_choice"] == {"type": "auto"} and all(t.get("strict") for t in seen[0]["tools"])
    report_tool = next(t for t in seen[0]["tools"] if t["name"] == "report_candidates")
    concept_enum = report_tool["input_schema"]["properties"]["candidates"]["items"]["properties"][
        "role_matches"]["items"]["properties"]["concept"]["enum"]
    assert "q1.outcome" in concept_enum and "q1.exposure" in concept_enum
    assert "metadata only" in r._repr_html_() and "PARTIALLY COVERED" in r._repr_html_()


def test_a_report_naming_tables_it_never_read_is_cut_back(monkeypatch, cat):
    from skills import data_scout
    _scripted(monkeypatch, [[_tool("describe_table", {"table": EXPOSURE})], [_tool("report_candidates", GOOD_REPORT)]])
    r = data_scout.scout(spec=_spec(), catalog=cat)
    assert [c["table"] for c in r["candidates"]] == [EXPOSURE]
    assert len(r["validation"]["dropped"]) >= 3 and any("ungrounded" in w for w in r["warnings"])
    assert r["feasibility"] == "not_covered"


def test_the_catalog_call_budget_is_enforced(monkeypatch, cat):
    from skills import data_scout
    monkeypatch.setattr(data_scout, "MAX_TOOL_CALLS", 3)
    monkeypatch.setattr(data_scout, "MAX_TURNS", 6)
    _scripted(monkeypatch, [])                           # a model that searches forever
    r = data_scout.scout(spec=_spec(), catalog=cat)
    assert "didn't report within 6 turns (3 catalog calls)" in r["error"]


def test_unknown_tables_come_back_as_tool_errors_not_crashes(monkeypatch, cat):
    from skills import data_scout
    seen = _scripted(monkeypatch, [[_tool("describe_table", {"table": "demo.nope.x"})],
                                   [_tool("report_candidates", {"summary": "", "candidates": [], "coverage_notes": []})]])
    r = data_scout.scout(spec=_spec(), catalog=cat)
    assert "no table named" in seen[1]["messages"][-1]["content"][0]["content"]
    assert r["feasibility"] == "not_covered"


def test_spec_gating(monkeypatch, cat):
    from config import config
    from skills import data_scout
    _scripted(monkeypatch, [[_tool("report_candidates", {"summary": "", "candidates": [], "coverage_notes": []})]])
    r = data_scout.scout(spec=_spec(confirmed=False), catalog=cat)
    assert any("isn't confirmed" in w for w in r["warnings"])          # recommended, not required
    monkeypatch.setattr(config, "require_confirmed_spec", True)
    assert "Confirm the problem spec" in data_scout.scout(spec=_spec(confirmed=False), catalog=cat)["error"]
    blocked = normalize({"questions": [{"type": "causal", "statement": "x", "outcome_concept": "y"}]}, "q")
    assert "blocking gaps" in data_scout.scout(spec=blocked, catalog=cat)["error"]
    assert "problem spec" in data_scout.scout(catalog=cat)["error"]


def test_the_sessions_page_shows_the_shortlist(monkeypatch, cat, tmp_path):
    import app  # noqa: F401
    from pages.sessions import sessions_panel
    from skills import team_sessions
    monkeypatch.setattr(team_sessions, "SESSIONS_FILE", tmp_path / "s.json")
    out = validate_report(GOOD_REPORT, {t: cat.describe(t) for t in (EXPOSURE, ORDERS, CUSTOMERS, "demo.sales.sales_forecast")},
                          concepts_from_spec(_spec()))
    out["catalog"] = "retail_marketing"
    rec = team_sessions.save_session("s1", owner="ana", question="q", datasets=[], results={}, runs=[],
                                     source="notebook", scout=out)
    html = str(sessions_panel(team_sessions.team_overview([rec])).to_plotly_json())
    assert "investigate campaign_exposure, customer_orders, customers" in html and "metadata only" in html


# ── Benchmark integrity and scoring ───────────────────────────────────────────

KEYS = sorted((ROOT / "benchmarks" / "answer_keys" / "scout").glob("*.json"))


def test_no_catalog_table_or_column_names_in_the_scout_code(cat):
    source = (ROOT / "skills" / "data_scout.py").read_text() + (ROOT / "ds_workbench" / "core" / "scout.py").read_text()
    names = {t.split(".")[-1] for t in cat.table_names()}
    names |= {c["name"] for t in cat.table_names() for c in cat.describe(t)["columns"] if "_" in c["name"]}
    assert not [n for n in names if n in source and "_" in n]


def _perfect(key):
    tables = [t if isinstance(t, str) else t["any_of"][-1] for t in key["required_tables"]]
    cands = [_cand(t, role_matches=[{"concept": f"q1.{r['concept_kind']}", "column": "x", "basis": ""}
                                    for r in key["required_roles"] if r["table"] == t]) for t in tables]
    for a, b in [j["any_of"][-1] if isinstance(j, dict) else j for j in key["required_joins"]]:
        cands[0]["join_keys"].append({"left_table": a, "left_column": "k", "right_table": b, "right_column": "k",
                                      "status": "name_match"})
    return {"candidates": cands, "coverage_gaps": [f"q1.{g}" for g in key["required_gaps"]],
            "feasibility": (key["acceptable_feasibility"] or ["covered"])[0], "validation": {"dropped": []}}


@pytest.mark.parametrize("path", KEYS, ids=lambda p: p.stem)
def test_scoring_passes_a_correct_report_and_fails_the_traps(path):
    from benchmarks.scoring import score_data_scout
    key = json.loads(path.read_text())
    good = _perfect(key)
    r = score_data_scout(good, key)
    assert r.passed and r.overall == 100, r.notes
    lookalike = json.loads(json.dumps(good))
    lookalike["candidates"].append(_cand(key["forbidden_tables"][0]))
    assert not score_data_scout(lookalike, key).passed
    if key["required_gaps"]:
        assert not score_data_scout({**good, "coverage_gaps": []}, key).passed
    for a, b in key["forbidden_joins"]:
        bad = json.loads(json.dumps(good))
        bad["candidates"][0]["join_keys"].append({"left_table": a, "left_column": "k", "right_table": b,
                                                  "right_column": "k", "status": "type_compatible"})
        assert not score_data_scout(bad, key).passed


def test_benchmark_specs_are_valid_confirmed_specs_and_tables_exist(cat):
    for path in KEYS:
        key = json.loads(path.read_text())
        assert key["input"]["spec"]["status"] == "confirmed"
        for t in key["required_tables"] + key["forbidden_tables"]:
            for name in (t["any_of"] if isinstance(t, dict) else [t]):
                assert name in cat.table_names(), name


def test_a_searched_table_can_be_rejected_without_being_described(cat):
    report = {"candidates": [_cand("demo.marketing.campaign_plan", recommendation="reject", why="plans, not delivery",
                                   role_matches=[{"concept": "q1.exposure", "column": "segment", "basis": ""}]),
                             _cand("demo.sales.sales_forecast", recommendation="investigate")]}
    out = validate_report(report, {}, concepts_from_spec(_spec()),
                          searched={"demo.marketing.campaign_plan", "demo.sales.sales_forecast"})
    [c] = out["candidates"]
    assert (c["table"], c["recommendation"], c["role_matches"], c["described"]) == \
        ("demo.marketing.campaign_plan", "reject", [], False)
    reasons = [d["reason"] for d in out["validation"]["dropped"]]
    assert "table not described: no column claims" in reasons          # its column claim is dropped
    assert any("not described in this run" in r for r in reasons)      # can't be recommended unread

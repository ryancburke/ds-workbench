"""Data verification: deterministic, aggregate-only checks of the tables a person selected."""
from datetime import date

import pytest

from ds_workbench.core.catalog import SAMPLE_CAP, SyntheticCatalog
from ds_workbench.core.problem_spec import confirm, normalize
from ds_workbench.core.readiness import assess
from ds_workbench.core.verify import verify_tables

EXPOSURE, ORDERS, CUSTOMERS = "demo.marketing.campaign_exposure", "demo.sales.customer_orders", "demo.crm.customers"
FEATURES, WEEKLY, PROMO = "demo.crm.customer_features_daily", "demo.sales.store_weekly_sales", "demo.marketing.promo_calendar"
TODAY = date(2026, 10, 21)


@pytest.fixture(scope="module")
def cat():
    return SyntheticCatalog.load("retail_marketing")


def _causal_spec():
    s = normalize({"questions": [{"id": "q1", "type": "causal", "primary": True, "statement": "campaign effect",
                                  "unit_of_analysis": "customer", "population": "customers",
                                  "outcome_concept": "purchases", "exposure_concept": "received the campaign"}],
                   "answerability": {"status": "answerable_with_gaps", "reason": ""}}, "q")
    return confirm(s, user="ana", ts="2026-10-21T00:00:00+00:00")


def _cand(table, roles=(), joins=(), rec="investigate"):
    return {"table": table, "recommendation": rec,
            "role_matches": [{"concept": c, "column": col, "basis": "", "evidence": "metadata"} for c, col in roles],
            "join_keys": [{"left_table": a, "left_column": ac, "right_table": b, "right_column": bc,
                           "status": "name_match" if ac == bc else "type_compatible", "note": ""}
                          for a, ac, b, bc in joins]}


CAMPAIGN_SCOUT = {"candidates": [
    _cand(EXPOSURE, [("q1.exposure", "sent_at")], [(EXPOSURE, "email_hash", CUSTOMERS, "email_hash")]),
    _cand(ORDERS, [("q1.outcome", "net_amount_eur"), ("time", "order_date")],
          [(CUSTOMERS, "customer_id", ORDERS, "customer_id")]),
    _cand(CUSTOMERS, [("q1.unit", "customer_id")]),
    _cand(FEATURES, [("covariate", "spend_prior_90d_eur")], [(FEATURES, "customer_id", CUSTOMERS, "customer_id")]),
]}


# ── Catalog aggregates ────────────────────────────────────────────────────────

def test_profile_is_aggregates_only(cat):
    p = cat.profile(EXPOSURE, ["sent_at", "email_hash"])
    assert set(p) == {"row_count", "columns"}
    assert set(p["columns"]["sent_at"]) == {"null_pct", "distinct", "min", "max"}
    assert p["columns"]["sent_at"]["min"].startswith("2026-01") and p["columns"]["email_hash"]["min"] is None
    with pytest.raises(KeyError):
        cat.profile(EXPOSURE, ["no_such_column"])


def test_join_profile_returns_counts_not_rows(cat):
    j = cat.join_profile(EXPOSURE, ["email_hash"], CUSTOMERS, ["email_hash"])
    assert j["join_type"] == "inner" and 10 < j["left_rows_dropped_pct"] < 20
    assert not any(hasattr(v, "shape") for v in j.values())
    with pytest.raises(ValueError):
        cat.join_profile(EXPOSURE, ["email_hash"], CUSTOMERS, [])


def test_samples_are_capped(cat):
    assert len(cat.sample(ORDERS, n=10**9)) == SAMPLE_CAP
    assert len(cat.sample(ORDERS, n=0)) == 1


def test_a_fixture_cannot_name_an_arbitrary_module():
    with pytest.raises(ValueError):
        SyntheticCatalog({"name": "x", "tables": [], "data_module": "os.path"})


# ── Planted problems the metadata hides ───────────────────────────────────────

@pytest.fixture(scope="module")
def campaign(cat):
    return verify_tables(cat, [EXPOSURE, ORDERS, CUSTOMERS, FEATURES], CAMPAIGN_SCOUT, _causal_spec(), today=TODAY)


def _find(result, check, table):
    return [f for f in result["findings"] if f["check"] == check and f["table"] == table]


def test_covariates_recorded_only_after_exposure_are_critical(campaign):
    [f] = _find(campaign, "temporal", FEATURES)
    assert f["severity"] == "critical" and "after every exposure" in f["description"]
    assert campaign["verdict"] == "failed"


def test_unregistered_recipients_and_anonymous_orders_are_reported_once_each(campaign):
    [exp] = _find(campaign, "join_match", EXPOSURE)
    assert exp["severity"] == "warning" and "14% of campaign_exposure rows have no match" in exp["description"]
    [orders] = _find(campaign, "join_match", ORDERS)
    assert "20% of customer_orders" in orders["description"]
    assert not _find(campaign, "join_key_nulls", ORDERS)           # same fact, said once


def test_parents_without_children_are_noted_not_flagged(campaign):
    j = next(j for j in campaign["joins"] if EXPOSURE in (j["left_table"], j["right_table"]))
    assert j["child"] == EXPOSURE and j["cardinality"] in ("one-to-many", "many-to-one")
    assert "customers rows have no match (units without exposure: the comparison group)" in j["note"]
    assert not _find(campaign, "join_match", CUSTOMERS)


def test_only_verification_sets_join_status(cat):
    scout = {"candidates": [_cand(FEATURES, [], [(FEATURES, "customer_id", CUSTOMERS, "customer_id")]),
                            _cand(CUSTOMERS)]}
    scout["candidates"][0]["join_keys"][0]["status"] = "verified"      # whatever a report claims
    r = verify_tables(cat, [FEATURES, CUSTOMERS], scout, None, today=TODAY)
    [j] = r["joins"]
    assert j["status"] == "verified" and j["child"] == FEATURES and j["child_match_rate"] == 1.0
    bad = verify_tables(cat, [EXPOSURE, CUSTOMERS], {"candidates": [_cand(EXPOSURE, [], [
        (EXPOSURE, "email_hash", CUSTOMERS, "email_hash")]), _cand(CUSTOMERS)]}, None, today=TODAY)
    assert bad["joins"][0]["status"] == "verified_with_issues"


def test_a_clean_store_panel_verifies(cat):
    scout = {"candidates": [_cand(WEEKLY, [("q1.outcome", "revenue_eur"), ("time", "week_start")],
                                  [(PROMO, "store_id", WEEKLY, "store_id")]),
                            _cand(PROMO, [("q1.exposure", "start_date"), ("time", "start_date")])]}
    r = verify_tables(cat, [WEEKLY, PROMO], scout, _causal_spec(), today=TODAY)
    assert r["verdict"] == "verified", r["findings"]
    [j] = r["joins"]
    assert (j["child"], j["status"]) == (PROMO, "verified")
    assert "comparison group" in j["note"]          # non-promotion store-weeks are the controls


def test_effective_dated_tables_are_many_to_many(cat):
    scout = {"candidates": [_cand("demo.finance.product_margins", [], [("demo.finance.product_margins", "sku", ORDERS, "sku")]),
                            _cand(ORDERS)]}
    r = verify_tables(cat, ["demo.finance.product_margins", ORDERS], scout, None, today=TODAY)
    assert r["joins"][0]["cardinality"] == "many-to-many"
    assert _find(r, "join_cardinality", "demo.finance.product_margins")[0]["severity"] == "warning"


def test_deprecated_stale_and_unlinked_tables(cat):
    r = verify_tables(cat, ["demo.sales.sales_daily_v1", CUSTOMERS], None, None, today=TODAY)
    assert _find(r, "deprecated", "demo.sales.sales_daily_v1")[0]["severity"] == "critical"
    assert _find(r, "freshness", "demo.sales.sales_daily_v1")[0]["severity"] == "warning"
    assert _find(r, "unlinked", CUSTOMERS)                       # no join was proposed between them


def test_an_empty_mapped_column_is_critical(cat):
    scout = {"candidates": [_cand("demo.crm.subscriptions", [("q1.outcome", "end_date")])]}
    r = verify_tables(cat, ["demo.crm.subscriptions"], scout, None, today=TODAY)
    [f] = _find(r, "role_column", "demo.crm.subscriptions")
    assert f["severity"] == "warning" and "mostly empty" in f["description"]   # ~75% still active


# ── The skill: selection, readiness, sessions ────────────────────────────────

def _session(cat):
    from skills.session import WorkflowSession
    s = WorkflowSession(persist=False)
    s.spec, s.scout = _causal_spec(), CAMPAIGN_SCOUT
    return s


def test_verification_needs_a_selection_made_by_a_person(cat, monkeypatch):
    from skills import data_verification
    monkeypatch.setenv("WORKBENCH_USER", "ana@example.com")
    s = _session(cat)
    assert "select_tables" in data_verification.verify(session=s, catalog=cat)["error"]
    sel = s.select_tables([EXPOSURE, ORDERS, CUSTOMERS, FEATURES, "demo.sales.sales_forecast"], note="core four")
    assert sel["user"] == "ana@example.com" and sel["outside_shortlist"] == ["demo.sales.sales_forecast"]
    r = data_verification.verify(session=s, catalog=cat, today=TODAY)
    assert r["not_from_shortlist"] == ["demo.sales.sales_forecast"]
    assert s.get_result("data_verification")["verdict"] == "failed"
    assert "no such" not in str(r) and "error" not in r


def test_unknown_tables_are_refused(cat):
    from skills import data_verification
    assert "Not in the retail_marketing catalog" in data_verification.verify(
        tables=["demo.nope.x"], catalog=cat)["error"]


@pytest.mark.parametrize("verdict,status", [("failed", "blocked"), ("verified_with_issues", "caution"),
                                            ("verified", "ready_for_review"), ("looks_ok", "incomplete")])
def test_verification_is_a_readiness_check(verdict, status):
    results = {"data_quality": {"overall_score": 95, "findings": []},
               "causal_inference": {"verdict": "valid", "findings": []},
               "data_verification": {"verdict": verdict, "findings": []}}
    assert assess(results, declared=["data_quality", "causal_inference"])["status"] == status


def test_dq_is_opt_in_and_its_critical_findings_count(cat, monkeypatch):
    import skills.data_quality as dq
    from skills import data_verification
    calls = []

    def fake_assess(df, table_name):
        calls.append((table_name, len(df)))
        if table_name == ORDERS:
            return {"error": "model declined"}
        return {"overall_score": 70, "findings": [{"severity": "critical", "column": "region", "description": "mixed case"}]}
    monkeypatch.setattr(dq, "assess", fake_assess)
    free = data_verification.verify(tables=[WEEKLY, PROMO], catalog=cat, today=TODAY)
    assert calls == [] and "dq" not in free
    r = data_verification.verify(tables=[CUSTOMERS, ORDERS], catalog=cat, dq=True, sample_rows=500, today=TODAY)
    assert calls == [(CUSTOMERS, 500), (ORDERS, 500)]
    assert r["dq"][CUSTOMERS]["critical"] == 1 and r["dq"][ORDERS] == {"error": "model declined"}
    assert {f["table"] for f in r["findings"] if f["check"] == "data_quality"} == {CUSTOMERS, ORDERS}
    assert r["verdict"] == "failed"
    html = r._repr_html_()
    assert "FAILED" in html and "aggregates only" in html and "Data quality (sampled)" in html


def test_the_sessions_page_headline():
    from skills.team_sessions import headline
    line = headline("data_verification", {"verdict": "verified_with_issues", "tables": {"a": {}, "b": {}},
                                          "joins": [{"status": "verified"}, {"status": "verified_with_issues"}],
                                          "findings": []})
    assert line == "verified with issues · 2 tables · 1/2 joins clean"

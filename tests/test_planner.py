"""Investigation planner: deterministic templates per question type, recomputed from session state,
with human steps kept human and failures blocking what follows."""
from datetime import date

import pandas as pd
import pytest

from ds_workbench.core.catalog import SyntheticCatalog
from ds_workbench.core.plan import build_plan, choose_causal_method
from ds_workbench.core.problem_spec import confirm, normalize
from ds_workbench.core.verify import verify_tables

EXPOSURE, ORDERS, CUSTOMERS = "demo.marketing.campaign_exposure", "demo.sales.customer_orders", "demo.crm.customers"
FEATURES, WEEKLY, PROMO = "demo.crm.customer_features_daily", "demo.sales.store_weekly_sales", "demo.marketing.promo_calendar"
TODAY = date(2026, 10, 21)


@pytest.fixture(scope="module")
def cat():
    return SyntheticCatalog.load("retail_marketing")


def _spec(qtype, **q):
    base = {"id": "q1", "type": qtype, "primary": True, "statement": "s", "unit_of_analysis": "customer",
            "population": "customers", "outcome_concept": "purchases"}
    if qtype == "causal":
        base["exposure_concept"] = "received the spring campaign"
        base["time_window"] = "spring 2026"
    s = normalize({"questions": [{**base, **q}], "answerability": {"status": "answerable_with_gaps", "reason": ""}},
                  "Did the spring campaign raise purchases?")
    return confirm(s, user="ana", ts="2026-10-21T00:00:00+00:00")


def _cand(table, roles=(), joins=()):
    return {"table": table, "recommendation": "investigate",
            "role_matches": [{"concept": c, "column": col, "basis": "", "evidence": "metadata"} for c, col in roles],
            "join_keys": [{"left_table": a, "left_column": ac, "right_table": b, "right_column": bc, "status": "name_match",
                           "note": ""} for a, ac, b, bc in joins]}


CAMPAIGN_SCOUT = {"candidates": [
    _cand(EXPOSURE, [("q1.exposure", "sent_at")], [(EXPOSURE, "email_hash", CUSTOMERS, "email_hash")]),
    _cand(ORDERS, [("q1.outcome", "net_amount_eur"), ("time", "order_date")],
          [(CUSTOMERS, "customer_id", ORDERS, "customer_id")]),
    _cand(CUSTOMERS, [("q1.unit", "customer_id"), ("covariate", "loyalty_member")]),
    _cand(FEATURES, [("covariate", "spend_prior_90d_eur")], [(FEATURES, "customer_id", CUSTOMERS, "customer_id")]),
]}
PROMO_SCOUT = {"candidates": [
    _cand(WEEKLY, [("q1.outcome", "revenue_eur"), ("time", "week_start")], [(PROMO, "store_id", WEEKLY, "store_id")]),
    _cand(PROMO, [("q1.exposure", "start_date")]),
]}


def _state(cat, spec, scout, tables, verified=True, **extra):
    st = {"spec": spec, "scout": scout, "selection": {"tables": tables}, "results": {}}
    if verified:
        st["results"]["data_verification"] = verify_tables(cat, tables, scout, spec, today=TODAY)
    return st | extra


def _skills(plan):
    return [s["skill"] for s in plan["steps"]]


# ── Templates per question type ───────────────────────────────────────────────

PRE = ["frame", "scout", "select", "data_verification", "assemble", "data_quality"]


@pytest.mark.parametrize("qtype,path", [
    ("causal", PRE + ["causal_inference"]),
    ("predictive", PRE + ["leakage_detector", "feature_engineering", "model_selector"]),
    ("descriptive", PRE + ["eda"]),
    ("diagnostic", PRE + ["eda"]),
    ("optimization", PRE + ["unavailable"]),
])
def test_each_question_type_gets_its_own_path(qtype, path):
    assert _skills(build_plan({"spec": _spec(qtype)})) == path


def test_a_causal_question_never_takes_the_modelling_path():
    p = build_plan({"spec": _spec("causal")})
    assert not {"leakage_detector", "feature_engineering", "model_selector"} & set(_skills(p))
    assert p["required_checks"] == ["data_verification", "data_quality", "causal_inference"]


def test_an_optimisation_question_stops_with_a_reason(cat):
    st = {"spec": _spec("optimization"), "scout": {"candidates": []}, "selection": {"tables": [ORDERS]},
          "results": {"data_verification": {"verdict": "verified", "findings": []},
                      "data_quality": {"overall_score": 90, "findings": []}},
          "analysis_table": {"columns": ["x"], "rows": 1}}
    p = build_plan(st)
    assert p["next"]["kind"] == "stop" and "no optimisation skill" in p["next"]["text"]


# ── Status and the next action ────────────────────────────────────────────────

def test_steps_progress_and_only_one_is_ready_at_a_time(cat):
    p = build_plan({"spec": None})
    assert [s["status"] for s in p["steps"][:3]] == ["ready", "waiting", "waiting"]
    p = build_plan({"spec": _spec("causal"), "scout": CAMPAIGN_SCOUT})
    assert p["next"] == {"step": "select", "kind": "human", "text": "Choose the tables: session.select_tables([...], note=...)"}


def test_a_spec_with_blocking_gaps_blocks_everything(cat):
    spec = normalize({"questions": [{"type": "causal", "statement": "x", "outcome_concept": "y"}]}, "q")
    p = build_plan({"spec": spec})
    assert p["steps"][0]["status"] == "failed" and all(s["status"] == "blocked" for s in p["steps"][1:])
    assert p["next"]["kind"] == "fix"


def test_failed_verification_blocks_the_analysis(cat):
    st = _state(cat, _spec("causal"), CAMPAIGN_SCOUT, [EXPOSURE, ORDERS, CUSTOMERS, FEATURES])
    assert st["results"]["data_verification"]["verdict"] == "failed"
    p = build_plan(st)
    by = {s["id"]: s["status"] for s in p["steps"]}
    assert by["verify"] == "failed" and by["assemble"] == by["dq"] == by["causal"] == "blocked"
    assert p["next"]["kind"] == "fix" and "verification findings" in p["next"]["text"]


def test_a_verified_selection_moves_to_the_human_assembly_step(cat):
    st = _state(cat, _spec("causal"), CAMPAIGN_SCOUT, [EXPOSURE, ORDERS, CUSTOMERS])
    assert st["results"]["data_verification"]["verdict"] == "verified_with_issues"
    p = build_plan(st)
    assert p["next"]["step"] == "assemble" and p["next"]["kind"] == "human"


# ── Method choice, recipe and contract ────────────────────────────────────────

def test_a_period_applied_to_groups_means_difference_in_differences(cat):
    did = choose_causal_method(_state(cat, _spec("causal", unit_of_analysis="store"), PROMO_SCOUT, [WEEKLY, PROMO]))
    assert did["method"] == "did" and "start/end" in did["why"]
    per_unit = choose_causal_method(_state(cat, _spec("causal"), CAMPAIGN_SCOUT, [EXPOSURE, ORDERS, CUSTOMERS]))
    assert per_unit["method"] == "analyze"


def test_the_recipe_follows_verified_joins_and_verification_findings(cat):
    p = build_plan(_state(cat, _spec("causal"), CAMPAIGN_SCOUT, [EXPOSURE, ORDERS, CUSTOMERS, FEATURES]))
    text = "\n".join(p["recipe"])
    assert "Start from demo.crm.customers" in text
    assert "campaign_exposure via customers.email_hash = campaign_exposure.email_hash" in text
    assert "spans 2026-01-12 to 2026-03-29" in text and "spring 2026" in text      # filter to this campaign
    assert "Don't use customer_features_daily for covariates" in text            # verification said so
    assert "Watch: " in text and "14% of campaign_exposure rows have no match" in text
    assert p["contract"] == {"treatment": "treated", "outcome": "outcome", "covariates": "<pre-exposure columns>",
                             "pre_outcome": "outcome_pre (optional)"}


def test_a_table_with_no_verified_path_is_called_out(cat):
    p = build_plan(_state(cat, _spec("causal"), {"candidates": [_cand(CUSTOMERS, [("q1.unit", "customer_id")]),
                                                               _cand(WEEKLY)]}, [CUSTOMERS, WEEKLY]))
    assert any("store_weekly_sales has no verified join path" in s for s in p["recipe"])


# ── The session runs the plan ─────────────────────────────────────────────────

@pytest.fixture
def session(cat):
    from skills.session import WorkflowSession
    s = WorkflowSession(persist=False)
    s.spec, s.scout = _spec("causal"), CAMPAIGN_SCOUT
    s.selection = {"tables": [EXPOSURE, ORDERS, CUSTOMERS]}
    s.register("data_verification", verify_tables(cat, s.selection["tables"], s.scout, s.spec, today=TODAY))
    return s


def _analysis_df(n=400):
    import numpy as np
    rng = np.random.default_rng(0)
    pre = rng.gamma(2, 50, n)
    treated = (rng.random(n) < 0.4).astype(int)
    return pd.DataFrame({"customer_id": [f"C{i}" for i in range(n)], "treated": treated,
                         "outcome": pre * 0.8 + 5 * treated + rng.normal(0, 10, n), "outcome_pre": pre,
                         "loyalty_member": rng.random(n) < 0.3})


def test_human_steps_stay_human(session):
    with pytest.raises(ValueError, match="Build the analysis table"):
        session.run_next()
    with pytest.raises(ValueError, match="missing contract column"):
        session.run_next(pd.DataFrame({"treated": [0, 1]}))


def test_run_next_calls_each_skill_with_arguments_from_the_plan(session, monkeypatch):
    import skills.causal_inference as ci
    import skills.data_quality as dq
    calls = []
    monkeypatch.setattr(dq, "assess", lambda df, table_name, session=None: (
        calls.append(("dq", table_name, len(df))), session.register("data_quality", {"overall_score": 92, "findings": []}))[1]
        or {"overall_score": 92, "findings": []})

    def fake_analyze(df, session=None, **kw):
        calls.append(("analyze", kw))
        r = {"verdict": "valid", "findings": []}
        session.register("causal_inference", r)
        return r
    monkeypatch.setattr(ci, "analyze", fake_analyze)

    session.run_next(_analysis_df())                         # the assembly step: registers the table
    assert session.next()["step"] == "dq"
    session.run_next()
    session.run_next()
    (_, name, rows), (_, kw) = calls
    assert (name, rows) == ("analysis_table", 400)
    assert (kw["treatment_col"], kw["outcome_col"], kw["pre_outcome_col"]) == ("treated", "outcome", "outcome_pre")
    # assignment unknown: not treated as an experiment, so the pre-period outcome is also adjusted for
    assert kw["covariates"] == ["loyalty_member", "outcome_pre"] and kw["randomized"] is None
    assert kw["context"] == "Did the spring campaign raise purchases?"
    assert session.next()["kind"] == "complete"
    # the plan's checks are the session's required checks: verified + DQ + causal valid → ready for review
    assert session.required_checks == ["data_quality", "causal_inference", "data_verification"]
    assert session.digest()["overall_verdict"] == "caution"     # verification had warnings


def test_overrides_win_and_did_needs_a_treatment_start(cat, monkeypatch):
    import skills.causal_inference as ci
    from skills.session import WorkflowSession
    s = WorkflowSession(persist=False)
    s.spec, s.scout, s.selection = _spec("causal", unit_of_analysis="store"), PROMO_SCOUT, {"tables": [WEEKLY, PROMO]}
    s.register("data_verification", verify_tables(cat, [WEEKLY, PROMO], PROMO_SCOUT, s.spec, today=TODAY))
    s.register("data_quality", {"overall_score": 95, "findings": []})
    s.run_next(pd.DataFrame({"unit_id": ["S1"], "period": [1], "treated_group": [1], "outcome": [3.0]}))
    seen = {}
    monkeypatch.setattr(ci, "did", lambda df, session=None, **kw: seen.update(kw) or {"verdict": "valid", "findings": []})
    with pytest.raises(ValueError, match="treatment_start"):
        s.run_next()
    s.run_next(treatment_start="2026-04-06", outcome_col="outcome")
    assert (seen["group_col"], seen["time_col"], seen["unit_col"], seen["treatment_start"]) == \
        ("treated_group", "period", "unit_id", "2026-04-06")


def test_a_person_can_override_the_method(session):
    session.method_override = "did"
    p = session.plan()
    assert p["method"]["method"] == "did" and "chosen by you" in p["method"]["why"]


def test_the_plan_renders_and_shows_on_the_sessions_page(session, cat, tmp_path, monkeypatch):
    import app  # noqa: F401
    from pages.sessions import sessions_panel
    from skills import team_sessions
    html = session.plan()._repr_html_()
    assert "Investigation plan" in html and "Build the analysis table" in html and "Method: analyze" in html
    monkeypatch.setattr(team_sessions, "SESSIONS_FILE", tmp_path / "s.json")
    # a DiD session saved and reloaded: the page recomputes the plan from the saved record
    spec = _spec("causal", unit_of_analysis="store")
    ver = verify_tables(cat, [WEEKLY, PROMO], PROMO_SCOUT, spec, today=TODAY)
    rec = team_sessions.save_session("s1", owner="ana", question="q", datasets=[], source="notebook",
                                     results={"data_verification": ver}, runs=[], spec=spec, scout=PROMO_SCOUT,
                                     selection={"tables": [WEEKLY, PROMO]})
    out = str(sessions_panel(team_sessions.team_overview([rec])).to_plotly_json())
    assert "Plan: " in out and "4/7 steps" in out and "method did" in out


@pytest.mark.parametrize("assignment,randomized,adjust_pre", [
    ("randomized", True, False), ("not_randomized", False, True), (None, None, True)])
def test_assignment_decides_how_the_effect_is_estimated(cat, monkeypatch, assignment, randomized, adjust_pre):
    import skills.causal_inference as ci
    from skills.session import WorkflowSession
    s = WorkflowSession(persist=False)
    s.spec = _spec("causal", **({"assignment": assignment} if assignment else {}))
    s.scout, s.selection = CAMPAIGN_SCOUT, {"tables": [EXPOSURE, ORDERS, CUSTOMERS]}
    s.register("data_verification", verify_tables(cat, s.selection["tables"], s.scout, s.spec, today=TODAY))
    s.register("data_quality", {"overall_score": 95, "findings": []})
    s.run_next(_analysis_df())
    seen = {}
    monkeypatch.setattr(ci, "analyze", lambda df, session=None, **kw: seen.update(kw) or {"verdict": "valid", "findings": []})
    s.run_next()
    assert seen["randomized"] is randomized and ("outcome_pre" in seen["covariates"]) is adjust_pre
    assert seen["pre_outcome_col"] == "outcome_pre"


def test_unknown_assignment_is_a_gap_in_the_spec():
    spec = _spec("causal")
    assert spec["questions"][0]["assignment"] == "unknown"
    assert any("assigned" in m["item"] and not m["blocking"] for m in spec["missing"])
    assert not any("assigned" in m["item"] for m in _spec("causal", assignment="randomized")["missing"])


def test_the_recipe_never_takes_covariates_from_the_exposure_and_lists_only_measures(cat):
    scout = {"candidates": [
        _cand(EXPOSURE, [("q1.exposure", "sent_at"), ("covariate", "channel")], [(EXPOSURE, "email_hash", CUSTOMERS, "email_hash")]),
        _cand(ORDERS, [("q1.outcome", "net_amount_eur"), ("q1.outcome", "order_id"), ("q1.outcome", "order_date")],
              [(CUSTOMERS, "customer_id", ORDERS, "customer_id")]),
        _cand(CUSTOMERS, [("q1.unit", "customer_id")])]}
    text = "\n".join(build_plan(_state(cat, _spec("causal"), scout, [EXPOSURE, ORDERS, CUSTOMERS]))["recipe"])
    assert "Don't use channel from campaign_exposure as covariates" in text
    assert "aggregate net_amount_eur from customer_orders" in text

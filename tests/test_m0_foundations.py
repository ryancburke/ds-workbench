"""M0 foundations: provider-configurable client, declared required checks, failed session-start
checks recorded (never dropped), and safely quoted Databricks identifiers."""
import pytest

from ds_workbench.core.readiness import assess, required_checks


# ── Client routing ────────────────────────────────────────────────────────────

class _Fake:
    def __init__(self, **kw):
        self.kw = kw


@pytest.fixture
def fake_sdk(monkeypatch):
    import anthropic
    from skills.api_retry import PROVIDERS
    for cls in set(PROVIDERS.values()):
        monkeypatch.setattr(anthropic, cls, type(cls, (_Fake,), {}))


def test_the_anthropic_api_gets_the_key(fake_sdk):
    from skills.api_retry import build_client
    c = build_client("anthropic", api_key="k", base_url="")
    assert type(c).__name__ == "Anthropic" and c.kw["api_key"] == "k" and "base_url" not in c.kw


@pytest.mark.parametrize("provider,cls", [("aws", "AnthropicAWS"), ("bedrock", "AnthropicBedrock"),
                                          ("vertex", "AnthropicVertex"), ("google_cloud", "AnthropicGoogleCloud"),
                                          ("foundry", "AnthropicFoundry"), (" Bedrock ", "AnthropicBedrock")])
def test_platform_providers_use_their_own_credentials(fake_sdk, provider, cls):
    from skills.api_retry import build_client
    c = build_client(provider, api_key="should-not-be-passed", base_url="https://gateway.example")
    assert type(c).__name__ == cls
    assert "api_key" not in c.kw and c.kw["base_url"] == "https://gateway.example"


def test_an_unknown_provider_is_a_clear_error():
    from skills.api_retry import build_client
    with pytest.raises(ValueError, match="ANTHROPIC_PROVIDER"):
        build_client("openai")


def test_get_client_follows_config_and_is_still_instrumented(fake_sdk, monkeypatch):
    from config import config
    from skills import api_retry, usage
    seen = []
    monkeypatch.setattr(api_retry, "_client", None)
    monkeypatch.setattr(config, "anthropic_provider", "bedrock")
    monkeypatch.setattr(config, "anthropic_base_url", "")
    monkeypatch.setattr(usage, "instrument", lambda c: seen.append(c) or c)
    c = api_retry.get_client()
    assert type(c).__name__ == "AnthropicBedrock" and seen == [c]
    monkeypatch.setattr(api_retry, "_client", None)


def test_platform_model_ids_are_priced():
    from config import call_cost, price_key
    assert price_key("anthropic.claude-opus-5-5-v1:0") == "claude-opus-5-5"
    assert call_cost("anthropic.claude-opus-5-5-v1:0", 1_000_000, 0) == call_cost("claude-opus-5-5", 1_000_000, 0) > 0
    assert call_cost("some-other-model", 1_000_000, 1_000_000) == 0.0


# ── Declared required checks ──────────────────────────────────────────────────

DQ = {"overall_score": 95, "findings": []}
CAUSAL = {"verdict": "valid", "findings": []}
MS = {"data_readiness": "ready", "findings": []}
LD = {"verdict": "clean", "findings": []}


def test_without_a_declaration_behaviour_is_unchanged():
    from ds_workbench.core.readiness import mandatory_for
    for results in ({}, {"data_quality": DQ}, {"causal_inference": CAUSAL}, {"model_selector": MS}):
        assert required_checks(results) == mandatory_for(results)


def test_a_causal_investigation_can_declare_its_checks_without_leakage():
    declared = ["data_quality", "causal_inference"]
    assert required_checks({}, declared) == ("data_quality", "causal_inference")
    assert assess({"data_quality": DQ}, declared=declared)["missing"] == ["causal_inference"]
    assert assess({"data_quality": DQ, "causal_inference": CAUSAL}, declared=declared)["status"] == "ready_for_review"


def test_a_declaration_never_removes_the_floor_set_by_what_ran():
    # Model selection ran, so leakage is required even though only data quality was declared
    r = assess({"data_quality": DQ, "model_selector": MS}, declared=["data_quality"])
    assert r["status"] == "incomplete" and r["missing"] == ["leakage_detector"]
    r = assess({"data_quality": DQ, "causal_inference": CAUSAL, "model_selector": MS},
               declared=["data_quality", "causal_inference"])
    assert r["missing"] == ["leakage_detector"]
    assert assess({"data_quality": DQ, "model_selector": MS, "leakage_detector": LD},
                  declared=["data_quality"])["status"] == "ready_for_review"


@pytest.mark.parametrize("bad", [[], ["data_quality", "made_up_check"]])
def test_an_empty_or_unknown_declaration_is_rejected(bad):
    with pytest.raises(ValueError):
        required_checks({}, bad)
    from skills.session import WorkflowSession
    with pytest.raises(ValueError):
        WorkflowSession("q", persist=False, required_checks=bad)


def test_sessions_and_the_digest_use_the_declaration():
    from skills import team_sessions
    from skills.session import WorkflowSession, build_digest
    declared = ["data_quality", "causal_inference"]
    s = WorkflowSession("did the campaign work?", persist=False, required_checks=declared)
    s.register("data_quality", DQ)
    s.register("causal_inference", CAUSAL)
    assert s.digest()["overall_verdict"] == "ready_for_review"
    assert build_digest({"data_quality": DQ}, required_checks=declared)["missing"] == ["causal_inference"]
    record = {"results": dict(s._results), "runs": [], "required_checks": declared}
    assert team_sessions.summarize_session(record)["verdict"] == "ready_for_review"
    # an older record without the field keeps today's inference
    assert team_sessions.summarize_session({"results": {"data_quality": DQ}, "runs": []})["missing"] == \
        ["leakage_detector"]


def test_the_declaration_is_saved_with_the_session(tmp_path, monkeypatch):
    from skills import team_sessions
    monkeypatch.setattr(team_sessions, "SESSIONS_FILE", tmp_path / "sessions.json")
    team_sessions.save_session("s1", owner="ana", question="q", datasets=[], results={}, runs=[],
                               source="notebook", required_checks=["data_quality", "causal_inference"])
    assert team_sessions.load_sessions()[0]["required_checks"] == ["data_quality", "causal_inference"]


# ── Session start records failed data-quality checks ─────────────────────────

def test_a_failed_dq_check_at_session_start_is_recorded_not_dropped(monkeypatch):
    import app
    import skills.data_quality as dq
    from pages import home
    from workbench.session_panels import latest_results

    def boom(df, table_name):
        raise RuntimeError("model declined the request")
    monkeypatch.setattr(dq, "assess", boom)
    store, *_ = home.start_session(1, "why did sales drop?", ["sales_clean"], dict(app.EMPTY_STORE))
    [run] = store["data_quality"]
    assert run["status"] == "error" and "model declined" in run["error"]
    r = assess(latest_results(store))
    assert r["status"] == "incomplete" and r["failed"][0]["skill"] == "data_quality"

    titles = str([t.to_plotly_json() for t in home.update_skill_titles(store)])
    assert "failed" in titles and "0/100" not in titles
    assert "DQ failed" in str(home.render_session(store))


def test_an_error_result_at_session_start_is_recorded_as_failed(monkeypatch):
    import app
    import skills.data_quality as dq
    from pages import home
    monkeypatch.setattr(dq, "assess", lambda df, table_name: {"error": "truncated response"})
    store, *_ = home.start_session(1, "q", ["sales_clean"], dict(app.EMPTY_STORE))
    assert store["data_quality"][0]["status"] == "error" and "truncated" in store["data_quality"][0]["error"]


# ── Databricks identifiers ────────────────────────────────────────────────────

@pytest.mark.parametrize("name,quoted", [
    ("main.crm.customers", "`main`.`crm`.`customers`"),
    ("customers", "`customers`"),
    ("main.`my-schema`.t", "`main`.`my-schema`.`t`"),
    ("main.`a``b`.t", "`main`.`a``b`.`t`"),
])
def test_table_names_are_quoted(name, quoted):
    from databricks_client import quote_name
    assert quote_name(name) == quoted


@pytest.mark.parametrize("bad", ["x; DROP TABLE y", "a..b", "a.b.c.d", "", "t`; --", "a. .b", "a.", ".a",
                                 "a b", "`a`x", "t\nDROP"])
def test_unsafe_or_malformed_names_are_rejected(bad):
    from databricks_client import quote_name
    with pytest.raises(ValueError):
        quote_name(bad)


def test_queries_use_quoted_names(monkeypatch):
    import databricks_client as dc
    from config import config
    executed = []

    class Cur:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def execute(self, sql): executed.append(sql)
        def fetchall(self): return []

    class Conn(Cur):
        def cursor(self): return Cur()

    monkeypatch.setattr(config, "local_dev_mode", False)
    monkeypatch.setattr(dc, "_get_connection", lambda: Conn())
    dc.list_tables("main", "crm")
    dc.get_table_schema("main.crm.customers")
    assert executed == ["SHOW TABLES IN `main`.`crm`", "DESCRIBE `main`.`crm`.`customers`"]
    with pytest.raises(ValueError):
        dc.get_table_schema("customers; DROP TABLE x")

"""Unity Catalog adapter: the same catalog contract as SyntheticCatalog, with its real SQL run on
SQLite (loaded from the same fixture and rows), plus its safety guarantees."""
import re
import sqlite3

import pandas as pd
import pytest

from ds_workbench.catalogs.unity_catalog import UnityCatalogAdapter
from ds_workbench.core.catalog import NotAvailable, SyntheticCatalog

EXPOSURE, ORDERS, CUSTOMERS = "demo.marketing.campaign_exposure", "demo.sales.customer_orders", "demo.crm.customers"
FEATURES, WEEKLY, PROMO = "demo.crm.customer_features_daily", "demo.sales.store_weekly_sales", "demo.marketing.promo_calendar"


def _sqlite_executor(synthetic: SyntheticCatalog, log: list):
    """Loads the synthetic catalog into SQLite: data tables, and information_schema views in
    Unity Catalog's shape. Rewrites the adapter's 3-part names, nothing else."""
    db = sqlite3.connect(":memory:")
    tables, columns, tags = [], [], []
    for name in synthetic.table_names():
        cat, schema, table = name.split(".")
        meta = synthetic.describe(name)
        tables.append({"table_catalog": cat, "table_schema": schema, "table_name": table, "comment": meta["comment"],
                       "table_owner": meta["owner"], "last_altered": meta["updated_at"]})
        for i, c in enumerate(meta["columns"]):
            columns.append({"table_catalog": cat, "table_schema": schema, "table_name": table, "column_name": c["name"],
                            "ordinal_position": i, "full_data_type": c["type"], "data_type": c["type"].split("(")[0].upper(),
                            "comment": c["comment"]})
        tags += [{"catalog_name": cat, "schema_name": schema, "table_name": table, "tag_name": k, "tag_value": v}
                 for k, v in meta["tags"].items()]
        df = synthetic._frame(name).copy()
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].dt.strftime("%Y-%m-%d %H:%M:%S").where(df[col].notna(), None)
        df.to_sql(f"{cat}__{schema}__{table}", db, index=False)
    pd.DataFrame(tables).to_sql("demo__information_schema__tables", db, index=False)
    pd.DataFrame(columns).to_sql("demo__information_schema__columns", db, index=False)
    pd.DataFrame(tags).to_sql("demo__information_schema__table_tags", db, index=False)

    def execute(sql: str, params: dict) -> list[dict]:
        log.append(sql)
        s = re.sub(r"`([^`]+)`\.`([^`]+)`\.`([^`]+)`", r"`\1__\2__\3`", sql)          # 3-part → one table
        s = re.sub(r"`([^`]+)`\.information_schema\.(\w+)", r"`\1__information_schema__\2`", s)
        s = re.sub(r"TABLESAMPLE \(\d+ ROWS\)", "", s)                              # SQLite has no TABLESAMPLE
        cur = db.execute(s, params)
        names = [d[0] for d in cur.description]
        return [dict(zip(names, r)) for r in cur.fetchall()]
    return execute


@pytest.fixture(scope="module")
def synthetic():
    return SyntheticCatalog.load("retail_marketing")


@pytest.fixture(scope="module")
def sql_log():
    return []


@pytest.fixture(scope="module")
def uc(synthetic, sql_log):
    return UnityCatalogAdapter(_sqlite_executor(synthetic, sql_log), ["demo.*"])


# ── The same answers as SyntheticCatalog ──────────────────────────────────────

@pytest.mark.parametrize("table", [EXPOSURE, ORDERS, CUSTOMERS, "demo.sales.sales_daily_v1"])
def test_describe_matches(uc, synthetic, table):
    a, b = uc.describe(table), synthetic.describe(table)
    assert {k: a[k] for k in ("name", "comment", "tags", "owner", "updated_at", "columns")} == \
        {k: b[k] for k in ("name", "comment", "tags", "owner", "updated_at", "columns")}
    assert a["row_count"] is None          # not in information_schema; profile() counts rows


@pytest.mark.parametrize("terms", [["campaign"], ["sales", "forecast"], ["customer purchases", "orders"],
                                   ["deprecated"], ["store", "weekly", "revenue"], ["nothing_matches_this"]])
def test_search_ranks_the_same(uc, synthetic, terms):
    assert uc.search(terms) == synthetic.search(terms)


@pytest.mark.parametrize("table,cols", [(EXPOSURE, ["sent_at", "email_hash", "opened"]), (ORDERS, None),
                                        (FEATURES, ["snapshot_date", "spend_prior_90d_eur"]), (WEEKLY, None)])
def test_profile_matches(uc, synthetic, table, cols):
    assert uc.profile(table, cols) == synthetic.profile(table, cols)


@pytest.mark.parametrize("left,lk,right,rk", [
    (EXPOSURE, ["email_hash"], CUSTOMERS, ["email_hash"]),
    (CUSTOMERS, ["customer_id"], ORDERS, ["customer_id"]),          # null keys on the right
    ("demo.finance.product_margins", ["sku"], ORDERS, ["sku"]),     # many-to-many
    (PROMO, ["store_id"], WEEKLY, ["store_id"]),
])
def test_join_profile_matches(uc, synthetic, left, lk, right, rk):
    assert uc.join_profile(left, lk, right, rk) == synthetic.join_profile(left, lk, right, rk)


def test_verification_gives_the_same_result_on_both(uc, synthetic):
    from datetime import date
    from ds_workbench.core.verify import verify_tables
    scout = {"candidates": [
        {"table": EXPOSURE, "recommendation": "investigate", "role_matches": [{"concept": "q1.exposure", "column": "sent_at"}],
         "join_keys": [{"left_table": EXPOSURE, "left_column": "email_hash", "right_table": CUSTOMERS, "right_column": "email_hash"}]},
        {"table": CUSTOMERS, "recommendation": "investigate", "role_matches": [], "join_keys": []}]}
    args = ([EXPOSURE, CUSTOMERS], scout, None)
    assert verify_tables(uc, *args, today=date(2026, 10, 21)) == verify_tables(synthetic, *args, today=date(2026, 10, 21))


def test_samples_are_capped_and_typed(uc):
    df = uc.sample(ORDERS, ["order_id", "net_amount_eur"], n=10**9)
    assert len(df) == 20_000 and list(df.columns) == ["order_id", "net_amount_eur"]


# ── Safety ────────────────────────────────────────────────────────────────────

def test_every_statement_was_a_read_only_select(uc, sql_log):
    uc.describe(CUSTOMERS)
    assert sql_log and all(re.match(r"\s*(SELECT|WITH)\b", s) and ";" not in s for s in sql_log)


@pytest.mark.parametrize("sql", ["DROP TABLE x", "SELECT 1; DROP TABLE x", "  insert into t values (1)",
                                 "UPDATE t SET a=1", "CREATE TABLE t AS SELECT 1", "GRANT SELECT ON t TO u"])
def test_writes_are_refused_before_they_are_sent(sql):
    sent = []
    a = UnityCatalogAdapter(lambda s, p: sent.append(s) or [], ["demo.*"])
    with pytest.raises(PermissionError):
        a._run(sql)
    assert sent == []


def test_the_allow_list_bounds_everything(synthetic, sql_log):
    a = UnityCatalogAdapter(_sqlite_executor(synthetic, []), ["demo.crm"])
    assert a.allowed(CUSTOMERS) and not a.allowed(ORDERS)
    assert {h["name"].split(".")[1] for h in a.search(["customer", "orders", "campaign"])} == {"crm"}
    for call in (lambda: a.describe(ORDERS), lambda: a.profile(ORDERS), lambda: a.sample(ORDERS),
                 lambda: a.join_profile(CUSTOMERS, ["customer_id"], ORDERS, ["customer_id"])):
        with pytest.raises(KeyError, match="allow-list"):
            call()
    with pytest.raises(ValueError, match="allow-list"):
        UnityCatalogAdapter(lambda s, p: [], [])
    with pytest.raises(ValueError):
        UnityCatalogAdapter(lambda s, p: [], ["just_a_catalog"])


@pytest.mark.parametrize("bad", ["demo.crm", "demo.crm.customers; DROP", "demo.`crm`.customers", "a.b.c.d", ""])
def test_malformed_names_are_refused(uc, bad):
    with pytest.raises(KeyError):
        uc.describe(bad)


def test_search_terms_are_bound_parameters_not_sql(synthetic):
    seen = []
    a = UnityCatalogAdapter(lambda s, p: seen.append((s, p)) or [], ["demo.*"])
    a.search(["x' OR 1=1 --"])
    sql, params = seen[0]
    assert "OR 1=1" not in sql and any("1=1" in str(v) or "1" in str(v) for v in params.values())


def test_lineage_is_opt_in_and_failures_are_not_available(synthetic):
    off = UnityCatalogAdapter(lambda s, p: [], ["demo.*"])
    with pytest.raises(NotAvailable):
        off.lineage(CUSTOMERS)

    def no_system_tables(sql, params):
        if "system.access" in sql:
            raise RuntimeError("TABLE_OR_VIEW_NOT_FOUND")
        return []
    with pytest.raises(NotAvailable):
        UnityCatalogAdapter(no_system_tables, ["demo.*"], lineage=True).lineage(CUSTOMERS)
    rows = [{"source_table_full_name": ORDERS, "target_table_full_name": CUSTOMERS},
            {"source_table_full_name": "secret.hr.salaries", "target_table_full_name": CUSTOMERS}]
    up = UnityCatalogAdapter(lambda s, p: rows, ["demo.*"], lineage=True).lineage(CUSTOMERS)["upstream"]
    assert up == [ORDERS]                        # names outside the allow-list aren't revealed


def test_describe_works_without_access_to_tags(synthetic):
    real = _sqlite_executor(synthetic, [])

    def no_tags(sql, params):
        if "table_tags" in sql and "SELECT tag_name" in sql:
            raise RuntimeError("PERMISSION_DENIED")
        return real(sql, params)
    assert UnityCatalogAdapter(no_tags, ["demo.*"]).describe(CUSTOMERS)["tags"] == {}


# ── Wiring ────────────────────────────────────────────────────────────────────

def test_the_scout_and_verification_run_on_the_adapter(uc, monkeypatch):
    """The full metadata → aggregates path on the adapter, with the scout's model faked."""
    from types import SimpleNamespace
    from skills import data_scout, data_verification
    from ds_workbench.core.problem_spec import confirm, normalize
    spec = confirm(normalize({"questions": [{"id": "q1", "type": "causal", "primary": True, "statement": "s",
                                             "unit_of_analysis": "customer", "outcome_concept": "purchases",
                                             "exposure_concept": "campaign"}]}, "q"), user="ana")
    turns = [[SimpleNamespace(type="tool_use", name="search_catalog", input={"terms": ["campaign"]}, id="a")],
             [SimpleNamespace(type="tool_use", name="describe_table", input={"table": EXPOSURE}, id="b")],
             [SimpleNamespace(type="tool_use", name="report_candidates", id="c", input={
                 "summary": "", "coverage_notes": [], "candidates": [{
                     "table": EXPOSURE, "rank": 1, "recommendation": "investigate", "why": "", "grain": "",
                     "role_matches": [{"concept": "q1.exposure", "column": "sent_at", "basis": ""}],
                     "join_keys": [], "concerns": [], "expert_assumptions": []}]})]]
    monkeypatch.setattr(data_scout, "api_call", lambda _c, **kw: SimpleNamespace(content=turns.pop(0), stop_reason="tool_use"))
    monkeypatch.setattr(data_scout, "get_client", lambda: SimpleNamespace(messages=SimpleNamespace(create=None)))
    rep = data_scout.scout(spec=spec, catalog=uc)
    assert rep["candidates"][0]["table"] == EXPOSURE and rep["validation"]["dropped"] == []
    v = data_verification.verify(tables=[EXPOSURE], spec=spec, scout=rep, catalog=uc)
    assert v["tables"][EXPOSURE]["time_range"][0].startswith("2026-01-12")


def test_the_factory_needs_an_allow_list_and_credentials(monkeypatch):
    from config import config
    from skills import data_scout
    monkeypatch.setattr(data_scout, "_CATALOG", {})
    monkeypatch.setattr(config, "databricks_host", "")
    with pytest.raises(ValueError, match="DATABRICKS_HOST"):
        data_scout.get_catalog("unity_catalog")
    monkeypatch.setattr(config, "databricks_host", "https://example.cloud.databricks.com")
    monkeypatch.setattr(config, "databricks_http_path", "/sql/1.0/warehouses/x")
    monkeypatch.setattr(config, "databricks_token", "")
    monkeypatch.setattr(config, "databricks_client_id", "")
    with pytest.raises(ValueError, match="service principal"):
        data_scout.get_catalog("unity_catalog")
    monkeypatch.setattr(config, "databricks_client_id", "sp")
    monkeypatch.setattr(config, "databricks_client_secret", "secret")
    monkeypatch.setattr(config, "catalog_allowlist", "")
    with pytest.raises(ValueError, match="allow-list"):
        data_scout.get_catalog("unity_catalog")
    with pytest.raises(ValueError, match="isn't supported"):
        data_scout.get_catalog("hive")


def test_the_executor_uses_service_principal_oauth_and_tags_queries(monkeypatch):
    """Checks the arguments against the installed connector's API (no network)."""
    pytest.importorskip("databricks.sql")
    import databricks.sql as dbsql
    from ds_workbench.catalogs.unity_catalog import databricks_executor
    calls = {}

    class Cur:
        description = [("a",)]
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def execute(self, sql, parameters=None, query_tags=None): calls["execute"] = (sql, parameters, query_tags)
        def fetchall(self): return [(1,)]

    class Conn:
        def cursor(self): return Cur()
    monkeypatch.setattr(dbsql, "connect", lambda **kw: calls.setdefault("connect", kw) and Conn() or Conn())
    ex = databricks_executor("https://x.cloud.databricks.com", "/sql/1.0/warehouses/w", client_id="sp", client_secret="s")
    assert ex("SELECT 1 AS a", {"p": 1}) == [{"a": 1}]
    kw = calls["connect"]
    assert kw["server_hostname"] == "x.cloud.databricks.com" and callable(kw["credentials_provider"])
    assert kw["session_configuration"] == {"STATEMENT_TIMEOUT": "120"} and "access_token" not in kw
    assert calls["execute"][2] == {"app": "ds-workbench"}

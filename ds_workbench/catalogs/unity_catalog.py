"""
Unity Catalog adapter: the catalog interface (ds_workbench.core.catalog) over Databricks SQL.

    from ds_workbench.catalogs.unity_catalog import UnityCatalogAdapter, databricks_executor

    catalog = UnityCatalogAdapter(databricks_executor(host, http_path, client_id=..., client_secret=...),
                                  allowlist=["sandbox.marketing", "sandbox.sales"])

Guarantees (each covered by a test):
  - Read-only: every statement is a single SELECT/WITH; anything else is refused before it is sent.
  - Allow-list: only tables in allow-listed catalog.schema entries ("catalog.*" for a whole
    catalog) are searched, described, profiled, joined or sampled. An empty allow-list is refused.
  - Metadata first: search and describe read information_schema only (names, types, comments,
    tags, owners, last change). profile and join_profile return aggregates computed in the
    warehouse; sample is capped and used only by opt-in steps.
  - Parameters are bound (`:name`), identifiers parsed and backtick-quoted.
  - The SQL runs as whoever the executor authenticates as: a scoped service principal for the
    pilot, the signed-in user later. Unity Catalog grants apply either way.

The adapter's SQL is plain Databricks SQL; the executor is any callable (sql, params) -> rows,
so tests run the same SQL on SQLite and compare every answer with SyntheticCatalog.
"""
from __future__ import annotations

import re

from ds_workbench.core.catalog import SAMPLE_CAP, NotAvailable, keyword_search, tokens

_NUMERIC = ("tinyint", "smallint", "int", "bigint", "long", "short", "byte", "float", "double", "decimal", "numeric")
_TEMPORAL = ("date", "timestamp")
_READ_ONLY = re.compile(r"^\s*(select|with)\b", re.I)
MAX_SEARCH_CANDIDATES = 200


def _q(part: str) -> str:
    return "`" + part.replace("`", "``") + "`"


def _split(table: str) -> tuple[str, str, str]:
    parts = (table or "").split(".")
    if len(parts) != 3 or not all(p.strip() for p in parts) or any("`" in p for p in parts):
        raise KeyError(f"not a catalog.schema.table name: {table!r}")
    return parts[0], parts[1], parts[2]


def _iso(v):
    if v is None:
        return None
    if hasattr(v, "isoformat"):
        return v.isoformat()[:19]
    s = str(v)
    return s.replace(" ", "T", 1)[:19] if re.match(r"\d{4}-\d\d-\d\d", s) else s


class UnityCatalogAdapter:
    """Catalog interface over Unity Catalog via an executor (sql, params) -> list[dict]."""

    def __init__(self, execute, allowlist: list[str], lineage: bool = False, name: str = "unity_catalog"):
        entries = [a.strip() for a in (allowlist or []) if a and a.strip()]
        if not entries:
            raise ValueError("A Unity Catalog adapter needs an allow-list (CATALOG_ALLOWLIST=catalog.schema,...)")
        self._allow: dict[str, set[str] | None] = {}
        for a in entries:
            cat, _, schema = a.partition(".")
            if not cat or not schema:
                raise ValueError(f"allow-list entries are catalog.schema or catalog.*: {a!r}")
            if schema == "*":
                self._allow[cat] = None
            elif self._allow.get(cat, set()) is not None:
                self._allow.setdefault(cat, set()).add(schema)
        self._execute = execute
        self.name = name
        self.supports_lineage = bool(lineage)
        self._meta: dict[str, dict] = {}

    # ── guards ─────────────────────────────────────────────────────────────────

    def _run(self, sql: str, params: dict | None = None) -> list[dict]:
        if not _READ_ONLY.match(sql) or ";" in sql:
            raise PermissionError("the catalog adapter only runs single read-only SELECT statements")
        return list(self._execute(sql, params or {}))

    def allowed(self, table: str) -> bool:
        try:
            cat, schema, _ = _split(table)
        except KeyError:
            return False
        return cat in self._allow and (self._allow[cat] is None or schema in self._allow[cat])

    def _check(self, table: str) -> tuple[str, str, str]:
        parts = _split(table)
        if not self.allowed(table):
            raise KeyError(f"{table!r} is not in the catalog allow-list")
        return parts

    def _schema_filter(self, cat: str, alias: str) -> tuple[str, dict]:
        schemas = self._allow[cat]
        if schemas is None:
            return f"{alias}.table_schema <> 'information_schema'", {}
        names = sorted(schemas)
        params = {f"s{i}": s for i, s in enumerate(names)}
        return f"{alias}.table_schema IN ({', '.join(':' + k for k in params)})", params

    # ── discovery: metadata only ───────────────────────────────────────────────

    def search(self, terms: list[str], limit: int = 8) -> list[dict]:
        """LIKE pushdown on names, comments, columns and tags to find candidates, then the same
        keyword ranking as every catalog (core.catalog.keyword_search) on their full metadata."""
        words = sorted({t for term in terms for t in tokens(term)})[:12]
        if not words:
            return []
        candidates: list[str] = []
        for cat in sorted(self._allow):
            where, params = self._schema_filter(cat, "t")
            like = []
            for i, w in enumerate(words):
                params[f"w{i}"] = f"%{w}%"
                p = f":w{i}"
                like.append(
                    f"lower(t.table_name) LIKE {p} OR lower(coalesce(t.comment, '')) LIKE {p} "
                    f"OR EXISTS (SELECT 1 FROM {_q(cat)}.information_schema.columns c WHERE c.table_schema = t.table_schema "
                    f"AND c.table_name = t.table_name AND (lower(c.column_name) LIKE {p} OR lower(coalesce(c.comment, '')) LIKE {p})) "
                    f"OR EXISTS (SELECT 1 FROM {_q(cat)}.information_schema.table_tags g WHERE g.schema_name = t.table_schema "
                    f"AND g.table_name = t.table_name AND (lower(g.tag_name) LIKE {p} OR lower(coalesce(g.tag_value, '')) LIKE {p}))")
            rows = self._run(
                f"SELECT t.table_catalog, t.table_schema, t.table_name FROM {_q(cat)}.information_schema.tables t "
                f"WHERE {where} AND ({' OR '.join('(' + x + ')' for x in like)}) "
                f"ORDER BY t.table_schema, t.table_name LIMIT {MAX_SEARCH_CANDIDATES}", params)
            candidates += [f"{r['table_catalog']}.{r['table_schema']}.{r['table_name']}" for r in rows]
        metas = [self.describe(t) for t in candidates if self.allowed(t)]
        return keyword_search(metas, terms, limit)

    def describe(self, table: str) -> dict:
        if table in self._meta:
            return _copy(self._meta[table])
        cat, schema, name = self._check(table)
        p = {"schema": schema, "table": name}
        rows = self._run(f"SELECT table_name, comment, table_owner, last_altered FROM {_q(cat)}.information_schema.tables "
                         "WHERE table_schema = :schema AND table_name = :table", p)
        if not rows:
            raise KeyError(f"no table named {table!r} (or no permission to see it)")
        cols = self._run(f"SELECT column_name, full_data_type, data_type, comment FROM {_q(cat)}.information_schema.columns "
                         "WHERE table_schema = :schema AND table_name = :table ORDER BY ordinal_position", p)
        try:
            tags = {r["tag_name"]: r["tag_value"] for r in self._run(
                f"SELECT tag_name, tag_value FROM {_q(cat)}.information_schema.table_tags "
                "WHERE schema_name = :schema AND table_name = :table", p)}
        except PermissionError:
            raise
        except Exception:            # tags view unavailable to this principal: describe without tags
            tags = {}
        r = rows[0]
        meta = {"name": table, "comment": r.get("comment") or "", "tags": tags, "owner": r.get("table_owner"),
                "updated_at": (_iso(r.get("last_altered")) or "")[:10] or None,
                "row_count": None,           # not in information_schema; profile() counts rows when needed
                "columns": [{"name": c["column_name"], "type": (c.get("full_data_type") or c.get("data_type") or "").lower(),
                             "comment": c.get("comment") or ""} for c in cols]}
        self._meta[table] = meta
        return _copy(meta)

    def lineage(self, table: str) -> dict:
        if not self.supports_lineage:
            raise NotAvailable("lineage isn't enabled for this catalog (CATALOG_LINEAGE)")
        self._check(table)
        try:
            rows = self._run("SELECT DISTINCT source_table_full_name, target_table_full_name FROM system.access.table_lineage "
                             "WHERE source_table_full_name = :t OR target_table_full_name = :t", {"t": table})
        except PermissionError:
            raise
        except Exception as exc:
            raise NotAvailable(f"lineage query failed: {str(exc)[:120]}") from exc
        up = sorted({r["source_table_full_name"] for r in rows if r["target_table_full_name"] == table
                     and r["source_table_full_name"] and self.allowed(r["source_table_full_name"])})
        down = sorted({r["target_table_full_name"] for r in rows if r["source_table_full_name"] == table
                       and r["target_table_full_name"] and self.allowed(r["target_table_full_name"])})
        return {"upstream": up, "downstream": down}

    # ── verification: aggregates computed in the warehouse ─────────────────────

    def _types(self, table: str, columns) -> dict[str, str]:
        types = {c["name"]: c["type"] for c in self.describe(table)["columns"]}
        missing = [c for c in columns or [] if c not in types]
        if missing:
            raise KeyError(f"{table} has no column(s) {missing}")
        return {c: types[c] for c in (columns or types)}

    def profile(self, table: str, columns: list[str] | None = None) -> dict:
        cat, schema, name = self._check(table)
        types = self._types(table, columns)
        sel = ["COUNT(*) AS n__rows"]
        for i, (c, t) in enumerate(types.items()):
            sel += [f"COUNT({_q(c)}) AS c{i}__n", f"COUNT(DISTINCT {_q(c)}) AS c{i}__d"]
            if t.startswith(_NUMERIC + _TEMPORAL):
                sel += [f"MIN({_q(c)}) AS c{i}__min", f"MAX({_q(c)}) AS c{i}__max"]
        r = self._run(f"SELECT {', '.join(sel)} FROM {_q(cat)}.{_q(schema)}.{_q(name)}")[0]
        n = int(r["n__rows"] or 0)
        out = {}
        for i, (c, t) in enumerate(types.items()):
            temporal = t.startswith(_TEMPORAL)
            conv = (lambda v: _iso(v)) if temporal else (lambda v: None if v is None else (float(v) if "." in str(v) else int(v)))
            out[c] = {"null_pct": round((n - int(r[f"c{i}__n"] or 0)) / n * 100, 2) if n else 0.0,
                      "distinct": int(r[f"c{i}__d"] or 0),
                      "min": conv(r.get(f"c{i}__min")), "max": conv(r.get(f"c{i}__max"))}
        return {"row_count": n, "columns": out}

    def join_profile(self, left: str, left_keys: list[str], right: str, right_keys: list[str]) -> dict:
        from ds_workbench.core.joins import stats_from_counts
        if len(left_keys) != len(right_keys) or not left_keys:
            raise ValueError("join_profile needs the same number of keys on each side")
        lc, ls, ln = self._check(left)
        rc, rs, rn = self._check(right)
        self._types(left, left_keys)
        self._types(right, right_keys)
        L, R = f"{_q(lc)}.{_q(ls)}.{_q(ln)}", f"{_q(rc)}.{_q(rs)}.{_q(rn)}"
        lk = ", ".join(f"{_q(k)} AS k{i}" for i, k in enumerate(left_keys))
        rk = ", ".join(f"{_q(k)} AS k{i}" for i, k in enumerate(right_keys))
        kk = ", ".join(f"k{i}" for i in range(len(left_keys)))
        lnn = " AND ".join(f"{_q(k)} IS NOT NULL" for k in left_keys)
        rnn = " AND ".join(f"{_q(k)} IS NOT NULL" for k in right_keys)
        lnull = " OR ".join(f"{_q(k)} IS NULL" for k in left_keys)
        rnull = " OR ".join(f"{_q(k)} IS NULL" for k in right_keys)
        on = " AND ".join(f"l.k{i} = r.k{i}" for i in range(len(left_keys)))
        r = self._run(f"""WITH
  lk AS (SELECT {kk}, COUNT(*) AS n FROM (SELECT {lk} FROM {L} WHERE {lnn}) x GROUP BY {kk}),
  rk AS (SELECT {kk}, COUNT(*) AS n FROM (SELECT {rk} FROM {R} WHERE {rnn}) y GROUP BY {kk}),
  m AS (SELECT l.n AS ln, r.n AS rn FROM lk l JOIN rk r ON {on})
SELECT
  (SELECT COUNT(*) FROM {L}) AS n_left, (SELECT COUNT(*) FROM {R}) AS n_right,
  (SELECT COUNT(*) FROM {L} WHERE {lnull}) AS left_null_rows,
  (SELECT COUNT(*) FROM {R} WHERE {rnull}) AS right_null_rows,
  (SELECT COALESCE(SUM(n - 1), 0) FROM lk) AS left_dups, (SELECT COALESCE(SUM(n - 1), 0) FROM rk) AS right_dups,
  (SELECT COUNT(*) FROM lk) AS left_distinct, (SELECT COUNT(*) FROM rk) AS right_distinct,
  (SELECT COUNT(*) FROM m) AS common_keys, (SELECT COALESCE(SUM(ln * rn), 0) FROM m) AS inner_rows,
  (SELECT COALESCE(SUM(ln), 0) FROM m) AS matched_left, (SELECT COALESCE(SUM(rn), 0) FROM m) AS matched_right,
  (SELECT COUNT(*) FROM m WHERE ln > 1 AND rn > 1) AS m2m, (SELECT COALESCE(MAX(ln * rn), 0) FROM m) AS max_rows""")[0]
        i = {k: int(v or 0) for k, v in r.items()}
        return stats_from_counts(
            left_name=left, right_name=right, keys=[f"_k{j}" for j in range(len(left_keys))], how="inner",
            n_left=i["n_left"], n_right=i["n_right"], inner_rows=i["inner_rows"], matched_left=i["matched_left"],
            matched_right=i["matched_right"], left_null_rows=i["left_null_rows"], right_null_rows=i["right_null_rows"],
            left_key_duplicates=i["left_dups"], right_key_duplicates=i["right_dups"],
            left_distinct_keys=i["left_distinct"], right_distinct_keys=i["right_distinct"],
            common_keys=i["common_keys"], many_to_many_keys=i["m2m"], max_rows_from_one_key=i["max_rows"])

    def sample(self, table: str, columns: list[str] | None = None, n: int = SAMPLE_CAP):
        import pandas as pd
        cat, schema, name = self._check(table)
        types = self._types(table, columns)
        n = max(1, min(int(n), SAMPLE_CAP))
        rows = self._run(f"SELECT {', '.join(_q(c) for c in types)} FROM {_q(cat)}.{_q(schema)}.{_q(name)} "
                         f"TABLESAMPLE ({n} ROWS) LIMIT {n}")
        return pd.DataFrame(rows, columns=list(types))


def _copy(meta: dict) -> dict:
    return {**meta, "tags": dict(meta["tags"]), "columns": [dict(c) for c in meta["columns"]]}


# ── Executor over databricks-sql-connector (optional dependency) ───────────────

def databricks_executor(host: str, http_path: str, *, token: str = "", client_id: str = "", client_secret: str = "",
                        timeout_s: int = 120, query_tags: dict | None = None):
    """An executor for UnityCatalogAdapter. Authenticates as a service principal (OAuth M2M:
    client_id + client_secret) when given, else with a token. Queries carry a statement timeout
    and query tags so the platform team can see what the workbench ran."""
    try:
        from databricks import sql
    except ImportError as exc:
        raise ImportError("Unity Catalog access needs the optional Databricks libraries: "
                          ".venv/bin/pip install -r requirements-databricks.txt") from exc
    if not host or not http_path:
        raise ValueError("DATABRICKS_HOST and DATABRICKS_HTTP_PATH are required for CATALOG_BACKEND=unity_catalog")
    kwargs = {"server_hostname": host.replace("https://", "").rstrip("/"), "http_path": http_path,
              "session_configuration": {"STATEMENT_TIMEOUT": str(int(timeout_s))}}
    if client_id and client_secret:
        def credentials_provider():
            # Built when a connection opens, not before: the SDK resolves auth over the network
            from databricks.sdk.core import Config, oauth_service_principal
            return oauth_service_principal(Config(host=host if host.startswith("https://") else f"https://{host}",
                                                  client_id=client_id, client_secret=client_secret))
        kwargs["credentials_provider"] = credentials_provider
    elif token:
        kwargs["access_token"] = token
    else:
        raise ValueError("Set DATABRICKS_CLIENT_ID and DATABRICKS_CLIENT_SECRET (a service principal) or DATABRICKS_TOKEN")
    tags = {"app": "ds-workbench", **(query_tags or {})}
    state = {"conn": None}

    def execute(statement: str, params: dict) -> list[dict]:
        if state["conn"] is None:
            state["conn"] = sql.connect(**kwargs)
        with state["conn"].cursor() as cur:
            cur.execute(statement, parameters=params or None, query_tags=tags)
            names = [d[0] for d in cur.description or []]
            return [dict(zip(names, row)) for row in cur.fetchall()]
    return execute

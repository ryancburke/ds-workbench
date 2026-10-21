"""
Catalog interface for data discovery: the only way the Data Scout sees data assets.

A catalog answers metadata questions — search, describe, (optionally) lineage — and nothing
else: no rows leave it at this stage. Two implementations share this contract:

  - SyntheticCatalog: a JSON fixture (ds_workbench/catalogs/<name>.json) for repeatable tests,
    benchmarks and open-source demos.
  - A Unity Catalog adapter (a later milestone): the same three methods over information_schema
    / the Databricks SDK, run as the signed-in user so their grants apply.

Table metadata shape (describe):
    {"name", "comment", "tags": {k: v}, "owner", "updated_at", "row_count",
     "columns": [{"name", "type", "comment"}]}
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Protocol

CATALOGS_DIR = Path(__file__).parent.parent / "catalogs"


class NotAvailable(Exception):
    """The catalog can't answer this (e.g. no access to lineage)."""


SAMPLE_CAP = 20_000         # most rows sample() ever returns


class Catalog(Protocol):
    name: str
    supports_lineage: bool

    # Discovery: metadata only
    def search(self, terms: list[str], limit: int = 8) -> list[dict]: ...
    def describe(self, table: str) -> dict: ...
    def lineage(self, table: str) -> dict: ...

    # Verification (after a person selects tables): aggregates computed where the data lives —
    # for Unity Catalog, single SQL aggregate queries — so no table is copied into pandas.
    # {"row_count", "columns": {name: {"null_pct", "distinct", "min", "max"}}}
    def profile(self, table: str, columns: list[str] | None = None) -> dict: ...

    # Counts for `left INNER JOIN right`, in ds_workbench.core.joins.join_stats' shape (no rows)
    def join_profile(self, left: str, left_keys: list[str], right: str, right_keys: list[str]) -> dict: ...

    # At most min(n, SAMPLE_CAP) rows as a DataFrame, for skills that need rows (e.g. DQ)
    def sample(self, table: str, columns: list[str] | None = None, n: int = SAMPLE_CAP): ...


_TOKEN = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> list[str]:
    """Lower-case word tokens with a light plural strip ("orders" → "order")."""
    out = []
    for t in _TOKEN.findall((text or "").lower().replace("_", " ")):
        out.append(t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith("ss") else t)
    return out


def keyword_search(tables: list[dict], terms: list[str], limit: int = 8) -> list[dict]:
    """Plain keyword search over names, tags, comments and columns — what a catalog search box
    does. Also the baseline the Data Scout's ranking is benchmarked against."""
    query = {t for term in terms for t in tokens(term)}
    hits = []
    for tb in tables:
        fields = [(tokens(tb["name"].split(".")[-1]), 3.0),
                  (tokens(" ".join(f"{k} {v}" for k, v in (tb.get("tags") or {}).items())), 2.0),
                  (tokens(tb.get("comment", "")), 1.0),
                  (tokens(" ".join(c["name"] for c in tb.get("columns", []))), 1.0),
                  (tokens(" ".join(c.get("comment", "") for c in tb.get("columns", []))), 0.5)]
        score, matched = 0.0, set()
        for words, weight in fields:
            m = query & set(words)
            score += weight * len(m)
            matched |= m
        if score:
            hits.append({"name": tb["name"], "comment": tb.get("comment", ""), "score": score,
                         "matched": sorted(matched)})
    hits.sort(key=lambda h: (-h["score"], h["name"]))
    return hits[:limit]


class SyntheticCatalog:
    """A catalog read from a JSON fixture: {"name", "description", "tables": [...],
    "lineage": {table: {"upstream": [...], "downstream": [...]}} (optional)}."""

    def __init__(self, data: dict):
        self.name = data["name"]
        self.description = data.get("description", "")
        self._tables = {t["name"]: t for t in data["tables"]}
        self._lineage = data.get("lineage")
        self.supports_lineage = self._lineage is not None
        self._data_module = data.get("data_module")
        if self._data_module and not re.fullmatch(r"[a-z0-9_]+", self._data_module):
            raise ValueError(f"not a data module name: {self._data_module!r}")
        self._frames: dict = {}

    @classmethod
    def load(cls, name: str) -> "SyntheticCatalog":
        if not re.fullmatch(r"[a-z0-9_]+", name or ""):
            raise ValueError(f"not a catalog fixture name: {name!r}")
        return cls(json.loads((CATALOGS_DIR / f"{name}.json").read_text()))

    def table_names(self) -> list[str]:
        return sorted(self._tables)

    def search(self, terms: list[str], limit: int = 8) -> list[dict]:
        return keyword_search(list(self._tables.values()), terms, limit)

    def describe(self, table: str) -> dict:
        if table not in self._tables:
            raise KeyError(f"no table named {table!r} in catalog {self.name!r}")
        return json.loads(json.dumps(self._tables[table]))      # a copy: callers can't edit the fixture

    # ── Verification: computed on the fixture's synthetic rows ────────────────
    # A fixture names a generator module in ds_workbench/catalogs ("data_module"); without one,
    # rows are generated from the column types. Synthetic rows are scaled down: profile()
    # reports the rows that exist, not the metadata's row_count.

    def _frame(self, table: str):
        if table not in self._tables:
            raise KeyError(f"no table named {table!r} in catalog {self.name!r}")
        if table not in self._frames:
            import importlib
            meta = self._tables[table]
            gen = None
            if self._data_module:
                gen = getattr(importlib.import_module(f"ds_workbench.catalogs.{self._data_module}"), "generate")
            df = gen(table) if gen else None
            self._frames[table] = df if df is not None else generic_frame(meta)
        return self._frames[table]

    def _columns(self, table: str, columns):
        df = self._frame(table)
        missing = [c for c in columns or [] if c not in df.columns]
        if missing:
            raise KeyError(f"{table} has no column(s) {missing}")
        return df[list(columns)] if columns else df

    def profile(self, table: str, columns: list[str] | None = None) -> dict:
        return profile_frame(self._columns(table, columns))

    def join_profile(self, left: str, left_keys: list[str], right: str, right_keys: list[str]) -> dict:
        from ds_workbench.core.joins import join_stats
        if len(left_keys) != len(right_keys) or not left_keys:
            raise ValueError("join_profile needs the same number of keys on each side")
        lf = self._columns(left, left_keys).copy()
        rf = self._columns(right, right_keys).copy()
        keys = [f"_k{i}" for i in range(len(left_keys))]
        lf.columns, rf.columns = keys, keys
        return join_stats(lf, rf, keys, how="inner", left_name=left, right_name=right)

    def sample(self, table: str, columns: list[str] | None = None, n: int = SAMPLE_CAP):
        df = self._columns(table, columns)
        n = max(1, min(int(n), SAMPLE_CAP))
        return df.sample(n=n, random_state=0) if len(df) > n else df.copy()

    def lineage(self, table: str) -> dict:
        if not self.supports_lineage:
            raise NotAvailable("lineage isn't available in this catalog")
        if table not in self._tables:
            raise KeyError(f"no table named {table!r} in catalog {self.name!r}")
        entry = self._lineage.get(table, {})
        return {"upstream": list(entry.get("upstream", [])), "downstream": list(entry.get("downstream", []))}


# ── Aggregates (the same numbers a SQL adapter computes with COUNT / MIN / MAX) ──

def _scalar(v):
    if v is None:
        return None
    try:
        import pandas as pd
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(v, "isoformat"):
        return v.isoformat()[:19]
    if hasattr(v, "item"):
        return v.item()
    return v


def profile_frame(df) -> dict:
    """Row count and, per column, null %, distinct count and min/max (numbers and dates)."""
    import pandas as pd
    n = len(df)
    cols = {}
    for c in df.columns:
        s = df[c]
        entry = {"null_pct": round(float(s.isna().mean() * 100), 2) if n else 0.0,
                 "distinct": int(s.nunique(dropna=True)), "min": None, "max": None}
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s) or \
                pd.api.types.is_datetime64_any_dtype(s):
            nn = s.dropna()
            if len(nn):
                entry["min"], entry["max"] = _scalar(nn.min()), _scalar(nn.max())
        cols[c] = entry
    return {"row_count": n, "columns": cols}


def generic_frame(meta: dict, n: int = 500, seed: int = 0):
    """Plausible rows from column types, for fixture tables without a generator."""
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(seed + sum(map(ord, meta["name"])))
    out = {}
    for c in meta.get("columns", []):
        t = (c.get("type") or "").lower()
        if t.startswith(("int", "bigint", "smallint")):
            out[c["name"]] = rng.integers(0, 100, n)
        elif t.startswith(("decimal", "double", "float")):
            out[c["name"]] = rng.gamma(2.0, 20.0, n).round(2)
        elif t.startswith("bool"):
            out[c["name"]] = rng.random(n) < 0.3
        elif t.startswith("timestamp"):
            out[c["name"]] = pd.Timestamp("2026-01-01") + pd.to_timedelta(rng.integers(0, 270 * 24, n), unit="h")
        elif t.startswith("date"):
            out[c["name"]] = pd.Timestamp("2026-01-01") + pd.to_timedelta(rng.integers(0, 270, n), unit="D")
        else:
            out[c["name"]] = [f"{c['name']}_{i}" for i in rng.integers(0, max(5, n // 2), n)]
    return pd.DataFrame(out)

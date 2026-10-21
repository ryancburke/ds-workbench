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


class Catalog(Protocol):
    name: str
    supports_lineage: bool

    def search(self, terms: list[str], limit: int = 8) -> list[dict]: ...
    def describe(self, table: str) -> dict: ...
    def lineage(self, table: str) -> dict: ...


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

    def lineage(self, table: str) -> dict:
        if not self.supports_lineage:
            raise NotAvailable("lineage isn't available in this catalog")
        if table not in self._tables:
            raise KeyError(f"no table named {table!r} in catalog {self.name!r}")
        entry = self._lineage.get(table, {})
        return {"upstream": list(entry.get("upstream", [])), "downstream": list(entry.get("downstream", []))}

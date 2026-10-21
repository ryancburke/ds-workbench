"""
Verification: check the tables a person selected from the scout's shortlist against the data.

Deterministic, aggregates only (Catalog.profile / join_profile): nothing is copied out of the
catalog. What it checks:

  - each table: rows, freshness, deprecated tags, and how full the columns the scout mapped
    to the question's concepts actually are
  - each proposed join between selected tables: match rate in both directions, key nulls,
    cardinality (one-to-one, one-to-many, many-to-many). One side must find its partner: the
    side holding the exposure (its unmatched partners are the units without exposure — the
    comparison group), otherwise the side with repeated keys, or fewer keys. The other side's
    unmatched rows are reported, not flagged. Only this step can mark a join "verified".
  - temporal alignment for causal questions: the outcome must be observed after the exposure,
    and covariates must exist from before it
  - a selected table that no join links to the others

Every problem is a finding (critical / warning / info), and the result's verdict follows from
them: failed (any critical), verified_with_issues (any warning), verified. It is a readiness
check like any other ("data_verification").
"""
from __future__ import annotations

from datetime import date, datetime

STALE_DAYS = 180
MATCH_FAILED, MATCH_ISSUE = 0.50, 0.95      # child-side match rate
KEY_NULL_ISSUE = 5.0                         # % of rows with a null key
EMPTY, SPARSE, SOME_NULLS = 99.0, 50.0, 5.0  # % null in a mapped column


def _finding(severity, check, table, column, description, recommendation=""):
    return {"severity": severity, "check": check, "table": table, "column": column or "—",
            "description": description, "recommendation": recommendation}


def _day(v) -> date | None:
    if not v:
        return None
    try:
        return datetime.fromisoformat(str(v)[:19]).date()
    except ValueError:
        try:
            return date.fromisoformat(str(v)[:10])
        except ValueError:
            return None


def _roles(scout: dict | None, table: str) -> list[dict]:
    for c in (scout or {}).get("candidates", []):
        if c["table"] == table:
            return c.get("role_matches", [])
    return []


def _kind(role: dict) -> str:
    """'q1.exposure' → 'exposure'; extra roles ('covariate', 'time', ...) as they are."""
    return role["concept"].split(".")[-1]


def verify_tables(catalog, tables: list[str], scout: dict | None = None, spec: dict | None = None,
                  today: date | None = None) -> dict:
    today = today or date.today()
    findings: list[dict] = []
    out_tables: dict[str, dict] = {}

    for t in tables:
        meta = catalog.describe(t)
        prof = catalog.profile(t)
        cols = prof["columns"]
        updated = _day(meta.get("updated_at"))
        age = (today - updated).days if updated else None
        if str((meta.get("tags") or {}).get("deprecated", "")).lower() == "true":
            findings.append(_finding("critical", "deprecated", t, None, "the table is tagged deprecated",
                                     "use the table that replaced it"))
        if age is not None and age > STALE_DAYS:
            findings.append(_finding("warning", "freshness", t, None, f"last updated {age} days ago ({updated})",
                                     "confirm it's still maintained"))
        if prof["row_count"] == 0:
            findings.append(_finding("critical", "empty_table", t, None, "the table has no rows"))
        roles = _roles(scout, t)
        for r in roles:
            c = cols.get(r["column"])
            if c is None:
                findings.append(_finding("critical", "role_column", t, r["column"], "mapped column not found in the data"))
                continue
            for limit, sev, text in ((EMPTY, "critical", "is empty"), (SPARSE, "warning", "is mostly empty"),
                                     (SOME_NULLS, "info", "has some nulls")):
                if c["null_pct"] >= limit:
                    findings.append(_finding(sev, "role_column", t, r["column"],
                                             f"{r['column']} ({_kind(r)}) {text}: {c['null_pct']:.1f}% null",
                                             "check how it's recorded" if sev != "info" else ""))
                    break
        time_col = next((r["column"] for r in roles if _kind(r) == "time" and r["column"] in cols), None) or \
            next((name for name, c in cols.items() if c["min"] and _day(c["min"]) and isinstance(c["min"], str)), None)
        out_tables[t] = {
            "row_count": prof["row_count"], "updated_at": meta.get("updated_at"), "age_days": age,
            "time_column": time_col,
            "time_range": [cols[time_col]["min"], cols[time_col]["max"]] if time_col else None,
            "roles": sorted({_kind(r) for r in roles}),
            "mapped_columns": {r["column"]: cols.get(r["column"]) for r in roles},
        }

    # ── Joins between selected tables ────────────────────────────────────────
    pairs = {}
    for c in (scout or {}).get("candidates", []):
        for j in c.get("join_keys", []):
            if j["left_table"] in tables and j["right_table"] in tables:
                key = tuple(sorted([(j["left_table"], j["left_column"]), (j["right_table"], j["right_column"])]))
                pairs[key] = j
    joins = []
    for (a, ak), (b, bk) in pairs:
        ab = catalog.join_profile(a, [ak], b, [bk])
        ba = catalog.join_profile(b, [bk], a, [ak])
        a_rate, b_rate = 1 - ab["left_rows_dropped_pct"] / 100, 1 - ba["left_rows_dropped_pct"] / 100
        a_dup, b_dup = ab["left_key_duplicates"] > 0, ab["right_key_duplicates"] > 0
        cardinality = {(False, False): "one-to-one", (True, False): "many-to-one",
                       (False, True): "one-to-many", (True, True): "many-to-many"}[(a_dup, b_dup)]
        a_exp, b_exp = "exposure" in out_tables[a]["roles"], "exposure" in out_tables[b]["roles"]
        if a_exp != b_exp:      # the exposure must link; units without it are the comparison group
            child = a if a_exp else b
        elif a_dup != b_dup:
            child = a if a_dup else b
        else:      # same cardinality on both sides: the side with fewer distinct keys should be contained
            a_keys = catalog.profile(a, [ak])["columns"][ak]["distinct"]
            b_keys = catalog.profile(b, [bk])["columns"][bk]["distinct"]
            child = a if a_keys <= b_keys else b
        parent = b if child == a else a
        child_rate, parent_rate = (a_rate, b_rate) if child == a else (b_rate, a_rate)
        label = f"{a.split('.')[-1]}.{ak} = {b.split('.')[-1]}.{bk}"
        jf = []
        if child_rate < MATCH_FAILED:
            jf.append(_finding("critical", "join_match", child, ak if child == a else bk,
                               f"{label}: only {child_rate:.0%} of {child.split('.')[-1]} rows find a match",
                               "this isn't the right key, or a bridge table is missing"))
        elif child_rate < MATCH_ISSUE:
            jf.append(_finding("warning", "join_match", child, ak if child == a else bk,
                               f"{label}: {1 - child_rate:.0%} of {child.split('.')[-1]} rows have no match",
                               "find out who they are; they drop out of the analysis"))
        if cardinality == "many-to-many":
            jf.append(_finding("warning", "join_cardinality", a, ak, f"{label}: many-to-many keys",
                               "aggregate one side to the unit of analysis before joining"))
        for name, k, prof_ in ((a, ak, ab), (b, bk, ba)):
            nulls = prof_["left_key_null_rows"] / prof_["left_rows"] * 100 if prof_["left_rows"] else 0
            # the child's unmatched rows already include its null keys: say it once
            if nulls >= KEY_NULL_ISSUE and not (name == child and child_rate < MATCH_ISSUE):
                jf.append(_finding("warning", "join_key_nulls", name, k, f"{nulls:.0f}% of {name.split('.')[-1]} rows have no {k}",
                                   "those rows can't be linked"))
        findings += jf
        joins.append({
            "left_table": a, "left_column": ak, "right_table": b, "right_column": bk,
            "cardinality": cardinality, "child": child, "child_match_rate": round(child_rate, 4),
            "parent_match_rate": round(parent_rate, 4),
            "status": ("failed" if any(f["severity"] == "critical" for f in jf)
                       else "verified_with_issues" if any(f["severity"] == "warning" for f in jf) else "verified"),
            "note": (f"{1 - parent_rate:.0%} of {parent.split('.')[-1]} rows have no match "
                     + ("(units without exposure: the comparison group)" if a_exp != b_exp
                        else "(expected if not every parent has children)"))
            if parent_rate < 1 and child_rate >= MATCH_FAILED else "",
            "join_audit": {"verdict": ab["verdict"], "reasons": ab["verdict_reasons"],
                           "fan_out_factor": ab["fan_out_factor"]},
        })
    linked = {t for j in joins if j["status"] != "failed" for t in (j["left_table"], j["right_table"])}
    if len(tables) > 1:
        for t in tables:
            if t not in linked:
                findings.append(_finding("warning", "unlinked", t, None,
                                         "no verified join links this table to the other selected tables",
                                         "propose a key or a bridge table"))

    # ── Temporal alignment (causal questions) ────────────────────────────────
    temporal = []
    q = next((q for q in (spec or {}).get("questions", []) if q.get("primary")), {})
    if q.get("type") == "causal":
        def ranges(kind):
            return [(t, _day(v["time_range"][0]), _day(v["time_range"][1])) for t, v in out_tables.items()
                    if kind in v["roles"] and v["time_range"]]
        exp = ranges("exposure")
        if exp:
            e_min, e_max = min(r[1] for r in exp), max(r[2] for r in exp)
            for t, lo, hi in ranges("outcome"):
                if hi <= e_max:
                    temporal.append(_finding("critical", "temporal", t, out_tables[t]["time_column"],
                                             f"outcome data ends {hi}, not after the last exposure ({e_max})",
                                             "the effect can't be observed"))
                if lo >= e_min:
                    temporal.append(_finding("warning", "temporal", t, out_tables[t]["time_column"],
                                             f"outcome data starts {lo}, not before exposure ({e_min}): no pre-period",
                                             "pre-exposure outcomes help adjust for differences between groups"))
            for t, lo, hi in ranges("covariate"):
                if lo > e_max:
                    temporal.append(_finding("critical", "temporal", t, out_tables[t]["time_column"],
                                             f"covariates start {lo}, after every exposure ({e_min} to {e_max}): "
                                             "nothing here was measured before treatment",
                                             "use covariates recorded before exposure, or reconstruct them from history"))
                elif lo > e_min:
                    temporal.append(_finding("warning", "temporal", t, out_tables[t]["time_column"],
                                             f"covariates start {lo}, after the first exposure ({e_min})",
                                             "early exposures have no pre-treatment covariates"))
        findings += temporal

    verdict = ("failed" if any(f["severity"] == "critical" for f in findings)
               else "verified_with_issues" if any(f["severity"] == "warning" for f in findings) else "verified")
    return {"verdict": verdict, "tables": out_tables, "joins": joins, "temporal": temporal,
            "findings": sorted(findings, key=lambda f: ["critical", "warning", "info"].index(f["severity"]))}

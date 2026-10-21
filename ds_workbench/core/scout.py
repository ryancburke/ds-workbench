"""
Data Scout rules: what to look for (concepts from a ProblemSpec) and what a scout report may
claim. Deterministic; the model's report is validated here before anyone sees it.

  - Concepts come from the spec, with ids the model must use (e.g. "q1.outcome").
  - Every table and column in the report must have come back from describe() in this run;
    anything else is dropped and listed under validation.dropped — never shown as a finding.
    The one exception: a table that search returned but the scout didn't describe may be listed
    as reject or maybe (its name is real) — with no column, role or join claims.
  - Every role match is evidence level "metadata": the scout has read names, types and
    comments, not data. Verification (bounded profiling, join audits) is a later step.
  - Join keys get a status from code: name_match (same column name and type),
    type_compatible (different names, same type) — never "verified" here. Different types
    are dropped. A join the model proposes between two tables with no compatible columns
    can't survive.
  - Coverage is computed, not claimed: a required concept no "investigate" candidate measures
    is a coverage gap.
"""
from __future__ import annotations

from ds_workbench.core.problem_spec import primary_question

RECOMMENDATIONS = ("investigate", "maybe", "reject")
# Roles a column can play besides measuring a spec concept
EXTRA_ROLES = ("covariate", "join_key", "time", "context")
_CONCEPT_FIELDS = (("outcome_concept", "outcome"), ("exposure_concept", "exposure"),
                   ("unit_of_analysis", "unit"), ("population", "population"), ("comparison", "comparison"),
                   ("time_window", "time_window"), ("prediction_point", "prediction_point"))
# Concepts a primary question can't be investigated without
_REQUIRED = {"outcome", "exposure", "unit"}


def concepts_from_spec(spec: dict) -> list[dict]:
    """[{id, kind, text, question, required}] — what the scout must find data for."""
    out = []
    for q in spec.get("questions", []):
        for field, kind in _CONCEPT_FIELDS:
            if q.get(field):
                out.append({"id": f"{q['id']}.{kind}", "kind": kind, "text": q[field],
                            "question": q["id"], "required": bool(q.get("primary")) and kind in _REQUIRED})
    for i, inp in enumerate((spec.get("decision") or {}).get("inputs_needed", [])):
        out.append({"id": f"decision.input{i + 1}", "kind": "decision_input", "text": inp,
                    "question": "decision", "required": False})
    return out


def search_terms(spec: dict) -> list[str]:
    """The words a person would type into a catalog search box for this spec (the baseline)."""
    q = primary_question(spec)
    return [q.get(f) for f in ("outcome_concept", "exposure_concept", "unit_of_analysis") if q.get(f)]


def _base_type(t: str) -> str:
    t = (t or "").lower()
    for family, names in (("string", ("string", "varchar", "char")), ("int", ("int", "bigint", "smallint", "long")),
                          ("decimal", ("decimal", "double", "float", "numeric")), ("date", ("date",)),
                          ("timestamp", ("timestamp",)), ("boolean", ("bool",))):
        if any(t.startswith(n) for n in names):
            return family
    return t


def validate_report(report: dict, described: dict[str, dict], concepts: list[dict],
                    searched: set[str] | None = None) -> dict:
    """Keep only claims grounded in described metadata; add code-computed statuses and coverage.
    `searched`: table names search returned this run (may be rejected without being described)."""
    searched = set(searched or ())
    concept_ids = {c["id"] for c in concepts}
    dropped: list[dict] = []
    cols = {t: {c["name"]: c for c in meta.get("columns", [])} for t, meta in described.items()}

    def drop(what: str, why: str):
        dropped.append({"claim": what, "reason": why})

    candidates = []
    for c in report.get("candidates") or []:
        table = c.get("table")
        rec = c.get("recommendation") if c.get("recommendation") in RECOMMENDATIONS else "maybe"
        if table not in described:
            if table in searched and rec in ("reject", "maybe"):
                for r in c.get("role_matches") or []:
                    drop(f"{table}.{r.get('column')}", "table not described: no column claims")
                candidates.append({"table": table, "recommendation": rec, "why": c.get("why", ""),
                                   "role_matches": [], "join_keys": [],
                                   "grain": {"value": "", "evidence": "not described"}, "freshness": None,
                                   "concerns": [x for x in c.get("concerns") or [] if x],
                                   "expert_assumptions": [], "described": False,
                                   "rank": c.get("rank") if isinstance(c.get("rank"), int) else 999})
                continue
            drop(f"table {table}", "not described in this run (the scout may not claim a table it hasn't read)")
            continue
        roles = []
        for r in c.get("role_matches") or []:
            col, role = r.get("column"), r.get("concept")
            if col not in cols[table]:
                drop(f"{table}.{col}", "no such column")
                continue
            if role not in concept_ids and role not in EXTRA_ROLES:
                drop(f"{table}.{col} as {role}", "unknown concept")
                continue
            roles.append({"concept": role, "column": col, "basis": r.get("basis", ""), "evidence": "metadata"})
        joins = []
        for j in c.get("join_keys") or []:
            lt, lc, rt, rc = j.get("left_table"), j.get("left_column"), j.get("right_table"), j.get("right_column")
            if lt not in described or rt not in described:
                drop(f"join {lt}.{lc} = {rt}.{rc}", "a table in the join wasn't described")
                continue
            if lc not in cols[lt] or rc not in cols[rt]:
                drop(f"join {lt}.{lc} = {rt}.{rc}", "no such column")
                continue
            lty, rty = _base_type(cols[lt][lc]["type"]), _base_type(cols[rt][rc]["type"])
            if lty != rty:
                drop(f"join {lt}.{lc} = {rt}.{rc}", f"types differ ({cols[lt][lc]['type']} vs {cols[rt][rc]['type']})")
                continue
            joins.append({"left_table": lt, "left_column": lc, "right_table": rt, "right_column": rc,
                          "status": "name_match" if lc == rc else "type_compatible",
                          "note": j.get("note", "")})
        candidates.append({
            "table": table, "recommendation": rec, "why": c.get("why", ""),
            "role_matches": roles, "join_keys": joins,
            "grain": {"value": c.get("grain", ""), "evidence": "metadata (inferred)"},
            "freshness": described[table].get("updated_at"),
            "concerns": [x for x in c.get("concerns") or [] if x],
            "expert_assumptions": [x for x in c.get("expert_assumptions") or [] if x],
            "described": True,
            "rank": c.get("rank") if isinstance(c.get("rank"), int) else 999,
        })
    order = {"investigate": 0, "maybe": 1, "reject": 2}
    candidates.sort(key=lambda c: (order[c["recommendation"]], c["rank"]))
    for i, c in enumerate(candidates, 1):
        c["rank"] = i

    covered = {r["concept"] for c in candidates if c["recommendation"] == "investigate" for r in c["role_matches"]}
    coverage = [{"concept": k["id"], "kind": k["kind"], "text": k["text"], "required": k["required"],
                 "covered": k["id"] in covered,
                 "by": sorted({f"{c['table']}.{r['column']}" for c in candidates if c["recommendation"] == "investigate"
                               for r in c["role_matches"] if r["concept"] == k["id"]})}
                for k in concepts]
    gaps = [g for g in coverage if not g["covered"]]
    feasibility = ("not_covered" if any(g["required"] for g in gaps)
                   else "partially_covered" if gaps else "covered")
    model_gaps = [{"concept": g.get("concept"), "note": g.get("note", "")}
                  for g in report.get("coverage_notes") or [] if isinstance(g, dict)]
    return {
        "candidates": candidates,
        "coverage": coverage,
        "coverage_gaps": [g["concept"] for g in gaps],
        "feasibility": feasibility,
        "coverage_notes": model_gaps,
        "summary": report.get("summary", ""),
        "evidence_level": "metadata",
        "validation": {"dropped": dropped, "described_tables": sorted(described)},
    }

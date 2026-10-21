"""
Data Scout: find and assess the data assets that could answer a framed question.

    from ds_workbench import WorkflowSession, problem_framer, data_scout

    session = WorkflowSession()
    problem_framer.frame("Did our spring campaign increase sales?", session=session)
    session.confirm_spec(note="...")                  # recommended (REQUIRE_CONFIRMED_SPEC to enforce)
    report = data_scout.scout(session=session)        # catalog from CATALOG_BACKEND

The scout works on metadata only — table and column names, types, comments, tags, owners,
freshness and (if available) lineage — through a bounded loop of read-only catalog calls.
It returns a ranked shortlist (investigate / maybe / reject) with the columns that would
measure each concept in the spec, candidate join keys, concerns and assumptions for a domain
expert, and which concepts no table covers. Code validates every claim against what the
catalog returned (ds_workbench.core.scout): the scout can't invent a table, a column or a join.
Nothing is verified against the data yet: that's the next step, after you pick the shortlist.
"""
from __future__ import annotations

import json

from ds_workbench.core.catalog import NotAvailable, SyntheticCatalog
from ds_workbench.core.scout import EXTRA_ROLES, RECOMMENDATIONS, concepts_from_spec, validate_report
from skills.api_retry import api_call, get_client, skill_entry, strict_tool
from skills.models import THINKING_ADAPTIVE, role
from skills.result import SkillResult

MAX_TOOL_CALLS = 24         # catalog calls per run
MAX_TURNS = 14              # model turns per run

_SEARCH = {
    "name": "search_catalog",
    "description": "Keyword search over table names, comments, tags and column names. Returns up to 8 tables.",
    "input_schema": {"type": "object", "properties": {"terms": {"type": "array", "items": {"type": "string"}}},
                     "required": ["terms"]},
}
_DESCRIBE = {
    "name": "describe_table",
    "description": "Full metadata for one table: comment, tags, owner, last update, row count, columns with types and comments.",
    "input_schema": {"type": "object", "properties": {"table": {"type": "string"}}, "required": ["table"]},
}
_LINEAGE = {
    "name": "get_lineage",
    "description": "Upstream and downstream tables of one table.",
    "input_schema": {"type": "object", "properties": {"table": {"type": "string"}}, "required": ["table"]},
}


def _report_tool(concept_ids: list[str]) -> dict:
    return {
        "name": "report_candidates",
        "description": "Report the assessed shortlist. Call once, at the end.",
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "candidates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "table": {"type": "string"},
                            "rank": {"type": "integer"},
                            "recommendation": {"type": "string", "enum": list(RECOMMENDATIONS)},
                            "why": {"type": "string"},
                            "grain": {"type": "string", "description": "what one row is, as far as metadata shows"},
                            "role_matches": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "concept": {"type": "string", "enum": [*concept_ids, *EXTRA_ROLES]},
                                        "column": {"type": "string"},
                                        "basis": {"type": "string", "description": "the metadata that suggests it"},
                                    },
                                    "required": ["concept", "column", "basis"],
                                },
                            },
                            "join_keys": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "left_table": {"type": "string"}, "left_column": {"type": "string"},
                                        "right_table": {"type": "string"}, "right_column": {"type": "string"},
                                        "note": {"type": "string"},
                                    },
                                    "required": ["left_table", "left_column", "right_table", "right_column", "note"],
                                },
                            },
                            "concerns": {"type": "array", "items": {"type": "string"}},
                            "expert_assumptions": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["table", "rank", "recommendation", "why", "grain", "role_matches",
                                     "join_keys", "concerns", "expert_assumptions"],
                    },
                },
                "coverage_notes": {
                    "type": "array",
                    "items": {"type": "object",
                              "properties": {"concept": {"type": "string"}, "note": {"type": "string"}},
                              "required": ["concept", "note"]},
                },
            },
            "required": ["summary", "candidates", "coverage_notes"],
        },
    }


_PROMPT = """You are a senior data scientist scouting a data catalog for an analysis. You can only see
metadata (names, types, comments, tags, owners, last-updated dates{lineage_note}); you can't read rows.

The analysis, as framed:
{spec_block}

Concepts to find data for (use these ids in role_matches):
{concepts_block}

How to work:
- Search with several phrasings (business words, likely column names, synonyms), then describe
  every table you might recommend or reject: you may only report tables you have described, and
  only columns that describe_table returned.
- Judge suitability for THIS question, not keyword overlap. Things that make a table unsuitable
  even when its name matches: it holds predictions or plans rather than what happened; it is
  deprecated, temporary or stale; it measures a different thing or population; its grain can't
  be linked to the unit of analysis.
- For a causal question, look for: the exposure as it actually happened (who/what was exposed,
  when), the outcome after it, covariates measured BEFORE exposure (to adjust for how exposure
  was chosen), and a comparison group.
- Join keys: propose only columns that exist on both sides. If two tables share no key, look for
  a bridge table that carries both keys; if none exists, say so in a concern — don't force a join.
- If no table measures a concept, say so in coverage_notes. A plausible-looking proxy is a concern
  for a domain expert, not a match.
- Put assumptions only a domain expert can confirm (what a column really means, how a flag is set)
  in expert_assumptions.
- Recommend: investigate (worth verifying next), maybe (useful context or depends on an
  assumption), reject (looked relevant but unsuitable — say why). Include rejected look-alikes so
  the person sees why they were excluded.
You have at most {max_calls} catalog calls. When done, call report_candidates once."""


def get_catalog(backend: str | None = None, fixture: str | None = None):
    """The catalog to scout, from CATALOG_BACKEND / CATALOG_FIXTURE."""
    from config import config
    backend = (backend or config.catalog_backend).strip().lower()
    if backend == "synthetic":
        return SyntheticCatalog.load(fixture or config.catalog_fixture)
    raise ValueError(f"CATALOG_BACKEND={backend!r} isn't available yet (supported: synthetic)")


def _spec_block(spec: dict) -> str:
    keep = {k: spec.get(k) for k in ("question", "decision", "questions", "assumptions")}
    return json.dumps(keep, indent=1)


def _run_tool(catalog, name: str, args: dict, described: dict, searched: set) -> str:
    try:
        if name == "search_catalog":
            hits = catalog.search([str(t) for t in args.get("terms", [])][:8])
            searched.update(h["name"] for h in hits)
            return json.dumps(hits)
        if name == "describe_table":
            meta = catalog.describe(str(args.get("table", "")))
            described[meta["name"]] = meta
            return json.dumps(meta)
        if name == "get_lineage":
            return json.dumps(catalog.lineage(str(args.get("table", ""))))
        return json.dumps({"error": f"unknown tool {name}"})
    except (KeyError, NotAvailable, ValueError) as exc:
        return json.dumps({"error": str(exc).strip("'\"")})


@skill_entry("data_scout")
def scout(spec: dict | None = None, session=None, catalog=None) -> dict:
    """Shortlist and assess data assets for a framed question (metadata only).

    Args:
        spec: A ProblemSpec (problem_framer.frame). Taken from the session when not given.
        session: Optional WorkflowSession; the report is stored on it.
        catalog: A catalog (ds_workbench.core.catalog). Default: from CATALOG_BACKEND.

    Returns:
        Dict: candidates (ranked; role_matches, join_keys with code-set status, concerns,
        expert_assumptions), coverage per concept, coverage_gaps, feasibility
        (covered | partially_covered | not_covered), validation (claims dropped as ungrounded),
        catalog_calls, warnings.
    """
    try:
        return _scout(spec, session, catalog)
    except Exception as exc:
        return SkillResult({"error": str(exc)}, skill="data_scout")


def _scout(spec, session, catalog) -> dict:
    from config import config
    spec = spec or (session.spec if session is not None else None)
    if not spec or spec.get("error"):
        raise ValueError("The scout needs a problem spec: run problem_framer.frame(question, session=session) first")
    blocking = [m["item"] for m in spec.get("missing", []) if m.get("blocking")]
    if blocking:
        raise ValueError("Resolve the spec's blocking gaps before scouting for data: " + "; ".join(blocking))
    warnings = []
    if spec.get("status") != "confirmed":
        if config.require_confirmed_spec:
            raise ValueError("Confirm the problem spec first (session.confirm_spec); REQUIRE_CONFIRMED_SPEC is on")
        warnings.append("The problem spec isn't confirmed yet: the shortlist follows the draft's concepts.")
    catalog = catalog or get_catalog()
    concepts = concepts_from_spec(spec)
    if not concepts:
        raise ValueError("The spec names no concepts to look for")

    tools = [strict_tool(_SEARCH), strict_tool(_DESCRIBE)]
    if catalog.supports_lineage:
        tools.append(strict_tool(_LINEAGE))
    report_tool = strict_tool(_report_tool([c["id"] for c in concepts]))
    tools.append(report_tool)
    prompt = _PROMPT.format(
        lineage_note=", lineage" if catalog.supports_lineage else "",
        spec_block=_spec_block(spec),
        concepts_block="\n".join(f"  {c['id']}: {c['text']}{' (required)' if c['required'] else ''}"
                                 for c in concepts),
        max_calls=MAX_TOOL_CALLS)

    messages = [{"role": "user", "content": prompt}]
    described: dict[str, dict] = {}
    searched: set[str] = set()
    calls: list[dict] = []
    report = None
    for _ in range(MAX_TURNS):
        resp = api_call(get_client().messages.create, **role("scout"), max_tokens=16000,
                        thinking=THINKING_ADAPTIVE, tools=tools, tool_choice={"type": "auto"},
                        messages=messages)
        messages.append({"role": "assistant", "content": resp.content})
        uses = [b for b in resp.content if b.type == "tool_use"]
        final = next((b for b in uses if b.name == "report_candidates"), None)
        if final is not None:
            report = final.input
            break
        if not uses:
            messages.append({"role": "user", "content": "Call report_candidates now with your assessment."})
            continue
        results = []
        for b in uses:
            if len(calls) >= MAX_TOOL_CALLS:
                out = json.dumps({"error": "catalog call budget used up: call report_candidates now"})
            else:
                out = _run_tool(catalog, b.name, dict(b.input), described, searched)
                calls.append({"tool": b.name, "input": dict(b.input)})
            results.append({"type": "tool_result", "tool_use_id": b.id, "content": out})
        messages.append({"role": "user", "content": results})
    if report is None:
        raise RuntimeError(f"The scout didn't report within {MAX_TURNS} turns ({len(calls)} catalog calls)")

    out = validate_report(report, described, concepts, searched)
    out.update({"question": spec.get("question", ""), "catalog": catalog.name, "catalog_calls": calls,
                "warnings": warnings + ([f"{len(out['validation']['dropped'])} ungrounded claim(s) removed"]
                                        if out["validation"]["dropped"] else [])})
    result = SkillResult(out, skill="data_scout")
    if session is not None:
        session.set_scout(result)
    return result

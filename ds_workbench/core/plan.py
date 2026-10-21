"""
Investigation planner: from a ProblemSpec and the state of a session to an ordered plan and the
next action. Deterministic templates per question type — no model call. The plan is recomputed
from the session's state every time, so it can't go stale.

  frame → scout → select (a person) → verify → assemble the analysis table (a person, from a
  recipe) → analysis path for the question type:
      causal:                  data quality → causal inference (analyze or did, chosen here)
      predictive:              data quality → leakage → feature engineering → model selection
      descriptive, diagnostic: data quality → EDA
      optimization:            data quality → (no optimisation skill yet: the plan says so)

A causal question never goes down the modelling path. Steps that are readiness checks are
declared on the session (required_checks), so readiness follows the plan. A failed step, or a
failed verification, blocks everything after it, and the next action says what to fix instead
of proceeding.

State (a plain dict, from WorkflowSession or a saved session record):
    {"spec", "scout", "selection", "results": {skill: result}, "analysis_table": {"name", "columns", "rows"}}
"""
from __future__ import annotations

from ds_workbench.core.problem_spec import primary_question

# steps that are readiness checks, and so are declared as required when mandatory
_CHECK_STEPS = {"data_verification", "data_quality", "causal_inference", "leakage_detector"}


def _kind(role: dict) -> str:
    return role["concept"].split(".")[-1]


def _roles_by_table(scout: dict | None, tables: list[str]) -> dict[str, list[dict]]:
    out = {t: [] for t in tables}
    for c in (scout or {}).get("candidates", []):
        if c["table"] in out:
            out[c["table"]] = c.get("role_matches", [])
    return out


def choose_causal_method(state: dict) -> dict:
    """analyze (unit-level exposure events) or did (a policy period applied to groups of units over
    time), from the shape of the verified data. A person can override it."""
    sel = (state.get("selection") or {}).get("tables", [])
    roles = _roles_by_table(state.get("scout"), sel)
    ver = (state.get("results") or {}).get("data_verification") or {}
    exp_tables = [t for t, rs in roles.items() if any(_kind(r) == "exposure" for r in rs)]
    for t in exp_tables:
        dated = [c.lower() for c in ver.get("tables", {}).get(t, {}).get("date_columns", [])]
        if any("start" in c for c in dated) and any("end" in c for c in dated):     # a period, not an event
            return {"method": "did", "why": f"{t.split('.')[-1]} records a period (start/end) applied to "
                                            "groups of units, and the outcome is observed over time: "
                                            "difference-in-differences compares treated and untreated units "
                                            "before and after the start"}
    return {"method": "analyze", "why": "exposure is recorded per unit (who received it, when): compare "
                                        "exposed and unexposed units, adjusting for covariates measured "
                                        "before exposure"}


def _join_path(start: str, goal: str, joins: list[dict]) -> list[dict] | None:
    """Shortest chain of usable (not failed) joins from start to goal."""
    usable = [j for j in joins if j["status"] != "failed"]
    frontier, seen = [(start, [])], {start}
    while frontier:
        node, path = frontier.pop(0)
        if node == goal:
            return path
        for j in usable:
            for a, b in ((j["left_table"], j["right_table"]), (j["right_table"], j["left_table"])):
                if a == node and b not in seen:
                    seen.add(b)
                    frontier.append((b, path + [j]))
    return None


def contract(state: dict) -> dict:
    """The columns the analysis table must have, so each skill can be called without guessing."""
    q = primary_question(state.get("spec") or {})
    if q.get("type") == "causal":
        if choose_causal_method(state)["method"] == "did":
            return {"unit": "unit_id", "time": "period", "group": "treated_group", "outcome": "outcome",
                    "treatment_start": "<first exposure date>"}
        return {"treatment": "treated", "outcome": "outcome", "covariates": "<pre-exposure columns>",
                "pre_outcome": "outcome_pre (optional)"}
    if q.get("type") == "predictive":
        return {"target": "target", "features": "<columns known at the prediction point>"}
    return {}


def assembly_recipe(state: dict) -> list[str]:
    """Steps to build the analysis table from the selected, verified tables (for a person to run)."""
    spec, scout = state.get("spec") or {}, state.get("scout")
    q = primary_question(spec)
    sel = (state.get("selection") or {}).get("tables", [])
    ver = (state.get("results") or {}).get("data_verification") or {}
    vt, joins = ver.get("tables", {}), ver.get("joins", [])
    roles = _roles_by_table(scout, sel)

    def with_role(kind):
        return [t for t in sel if any(_kind(r) == kind for r in roles[t])]

    def cols(t, kind):
        return [r["column"] for r in roles[t] if _kind(r) == kind]

    def measures(t):
        """Outcome columns that are measurements, not the table's dates or identifiers."""
        mapped = vt.get(t, {}).get("mapped_columns") or {}
        return [c for c in cols(t, "outcome") if not c.endswith("_id") and c not in vt.get(t, {}).get("date_columns", [])
                and (not mapped.get(c) or mapped[c].get("min") is None or isinstance(mapped[c]["min"], (int, float)))]

    def span(t):
        rng = vt.get(t, {}).get("time_range")
        return f"{rng[0][:10]} to {rng[1][:10]}" if rng else "an unverified range"

    unit_tables = with_role("unit") or with_role("population") or sel[:1]
    base = unit_tables[0] if unit_tables else None
    steps = []
    if base:
        steps.append(f"Start from {base} — one row per {q.get('unit_of_analysis') or 'unit'}.")
    for t in sel:
        if t == base:
            continue
        path = _join_path(base, t, joins) if base else None
        if path:
            hops = " → ".join(f"{j['left_table'].split('.')[-1]}.{j['left_column']} = "
                              f"{j['right_table'].split('.')[-1]}.{j['right_column']}" for j in path)
            steps.append(f"Join {t.split('.')[-1]} via {hops}.")
        elif base:
            steps.append(f"{t.split('.')[-1]} has no verified join path to {base.split('.')[-1]}: "
                         "leave it out or find a key first.")
    method = choose_causal_method(state)["method"] if q.get("type") == "causal" else None
    for t in with_role("exposure"):
        steps.append(f"Exposure: {t.split('.')[-1]} spans {span(t)}. Keep only the exposure the question is about "
                     f"({q.get('exposure_concept', '?')}" + (f", {q['time_window']}" if q.get("time_window") else "")
                     + ") — other campaigns or periods in the same table are not this treatment.")
        if method == "analyze":
            steps.append("Define `treated` = 1 for units with that exposure, 0 otherwise.")
        elif method == "did":
            steps.append("Define `treated_group` = 1 for units that received it, `period` per time step, "
                         "and note the first exposure date as treatment_start.")
    for t in with_role("outcome"):
        c = ", ".join(measures(t)) or "the outcome column"
        if method == "analyze":
            steps.append(f"Outcome: aggregate {c} from {t.split('.')[-1]} per unit over a fixed window after "
                         "exposure → `outcome`; the same window before exposure → `outcome_pre` (the analysis "
                         + ("uses it to reduce variance)." if q.get("assignment") == "randomized"
                            else "adjusts for it: it's the best proxy for how units were chosen)."))
        elif method == "did":
            steps.append(f"Outcome: {c} from {t.split('.')[-1]} per unit per period → `outcome`, covering "
                         "periods before and after the start.")
        else:
            steps.append(f"Outcome: {c} from {t.split('.')[-1]} → `target`" if q.get("type") == "predictive"
                         else f"Measure: {c} from {t.split('.')[-1]}.")
    temporal_bad = {f["table"] for f in ver.get("findings", []) if f["check"] == "temporal" and f["severity"] == "critical"}
    exposure_tables = set(with_role("exposure"))
    for t in with_role("covariate"):
        if t in exposure_tables and q.get("type") == "causal":
            steps.append(f"Don't use {', '.join(cols(t, 'covariate'))} from {t.split('.')[-1]} as covariates: "
                         "attributes of the exposure itself aren't measured before treatment.")
        elif t in temporal_bad:
            steps.append(f"Don't use {t.split('.')[-1]} for covariates: verification found it starts after exposure.")
        elif q.get("type") == "causal":
            steps.append(f"Covariates from {t.split('.')[-1]} ({', '.join(cols(t, 'covariate'))}), "
                         "measured before each unit's exposure.")
    if q.get("type") == "predictive" and q.get("prediction_point"):
        steps.append(f"Every feature must be known at the prediction point ({q['prediction_point']}); "
                     "build them as of that date.")
    for f in ver.get("findings", []):
        if f["severity"] == "warning" and f["check"] in ("join_match", "join_key_nulls", "join_cardinality"):
            steps.append(f"Watch: {f['description']} — {f['recommendation']}")
    return steps


def _status(step: dict, results: dict, state: dict) -> str:
    sk = step["skill"]
    if sk == "frame":
        spec = state.get("spec")
        if not spec:
            return "ready"
        return "failed" if any(m.get("blocking") for m in spec.get("missing", [])) else "done"
    if sk == "scout":
        return "done" if state.get("scout") else "ready"
    if sk == "select":
        return "done" if state.get("selection") else "ready"
    if sk == "assemble":
        return "done" if state.get("analysis_table") else "ready"
    if sk == "unavailable":
        return "unavailable"
    r = results.get(sk)
    if r is None:
        return "ready"
    if r.get("error") or (sk == "data_verification" and r.get("verdict") not in ("verified", "verified_with_issues")):
        return "failed"
    return "done"


def build_plan(state: dict) -> dict:
    spec = state.get("spec") or {}
    q = primary_question(spec)
    qtype = q.get("type")
    results = state.get("results") or {}
    steps = [
        {"id": "frame", "skill": "frame", "mandatory": True, "why": "define the question before touching data"},
        {"id": "scout", "skill": "scout", "mandatory": True, "why": "find candidate tables in the catalog"},
        {"id": "select", "skill": "select", "mandatory": True, "human": True,
         "why": "you choose which tables to use"},
        {"id": "verify", "skill": "data_verification", "mandatory": True,
         "why": "check the selected tables, joins and timing against the data"},
        {"id": "assemble", "skill": "assemble", "mandatory": True, "human": True,
         "why": "build one analysis table from the recipe"},
        {"id": "dq", "skill": "data_quality", "mandatory": True, "why": "check the analysis table itself"},
    ]
    method = None
    if qtype == "causal":
        method = choose_causal_method(state)
        steps.append({"id": "causal", "skill": "causal_inference", "method": method["method"], "mandatory": True,
                      "randomized": {"randomized": True, "not_randomized": False}.get(q.get("assignment")),
                      "why": method["why"]})
    elif qtype == "predictive":
        steps += [{"id": "leakage", "skill": "leakage_detector", "mandatory": True,
                   "why": "features must exist at the prediction point"},
                  {"id": "features", "skill": "feature_engineering", "mandatory": False, "why": "plan features"},
                  {"id": "model", "skill": "model_selector", "mandatory": False, "why": "choose a model and metric"}]
    elif qtype in ("descriptive", "diagnostic"):
        steps.append({"id": "eda", "skill": "eda", "mandatory": False, "why": "summarise and decompose"})
    elif qtype == "optimization":
        steps.append({"id": "optimize", "skill": "unavailable", "mandatory": True,
                      "why": "no optimisation skill yet (ROADMAP.md, phase 3): the analysis can't be completed here"})
    for i, s in enumerate(steps):
        s["depends_on"] = steps[i - 1]["id"] if i else None

    # After a failed mandatory step everything not yet done is blocked; otherwise only the first
    # open step is ready and the rest wait for it.
    blocked = False
    first_open = None
    for s in steps:
        status = _status(s, results, state)
        if status != "done":
            if blocked:
                status = "blocked"
            elif first_open is None:
                first_open = s
            elif status == "ready":
                status = "waiting"
            if s["mandatory"] and status in ("failed", "unavailable"):
                blocked = True
        s["status"] = status

    return {
        "question_type": qtype, "steps": steps, "method": method,
        "required_checks": [s["skill"] for s in steps if s["mandatory"] and s["skill"] in _CHECK_STEPS],
        "contract": contract(state), "recipe": assembly_recipe(state) if state.get("selection") else [],
        "next": next_action(steps, state),
        "done": sum(s["status"] == "done" for s in steps), "total": len(steps),
    }


def next_action(steps: list[dict], state: dict) -> dict:
    """What to do now: run the next step, fix what failed, or stop."""
    for s in steps:
        if s["status"] == "done":
            continue
        if s["status"] == "failed":
            fix = {
                "frame": "Answer the spec's blocking gaps, then frame the question again.",
                "data_verification": "Resolve the critical verification findings, or change the selection "
                                     "(session.select_tables), then verify again.",
            }.get(s["skill"], f"{s['skill']} failed: fix the cause and run it again.")
            return {"step": s["id"], "kind": "fix", "text": fix}
        if s["status"] == "unavailable":
            return {"step": s["id"], "kind": "stop", "text": s["why"]}
        if s.get("human"):
            text = ("Choose the tables: session.select_tables([...], note=...)" if s["id"] == "select" else
                    "Build the analysis table from the recipe, then session.run_next(df)")
            return {"step": s["id"], "kind": "human", "text": text}
        return {"step": s["id"], "kind": "run", "text": f"session.run_next() runs {s['skill']}"
                + (f" ({s['method']})" if s.get("method") else "")}
    return {"step": None, "kind": "complete", "text": "Every step has run. Review the digest (session.digest())."}

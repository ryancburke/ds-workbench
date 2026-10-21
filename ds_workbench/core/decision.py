"""
Decision readiness: is the evidence enough to support the decision the question was asked for?

Separate from analytical readiness (ds_workbench.core.readiness), which asks whether the required
checks ran and passed. Decision readiness takes that as one input and adds what a decision needs:
an identified effect, its direction and size, the decision inputs (costs, value) and the open
assumptions. Deterministic; the narrative a model writes (skills/decision_synthesis.py) can cite
this but never change it.

Statuses (most to least severe):
  not_answerable         the question can't be answered as posed (from the spec)
  insufficient_evidence  analysis incomplete or blocked, or the effect isn't identified
  supported_with_caveats the evidence supports a conclusion, with named caveats
  supported              everything required ran clean, the effect is identified and the
                         decision inputs are in

A conclusion that the effect is null is still a conclusion: "no detectable effect" can be
supported. Insufficient evidence is a valid outcome, not a failure.
"""
from __future__ import annotations

from ds_workbench.core.problem_spec import primary_question
from ds_workbench.core.readiness import assess

STATUSES = ("not_answerable", "insufficient_evidence", "supported_with_caveats", "supported")
STATUS_LABEL = {"not_answerable": "Not answerable as posed", "insufficient_evidence": "Insufficient evidence",
                "supported_with_caveats": "Supported, with caveats", "supported": "Supported"}
INPUT_KEYS = ("cost", "units", "value_per_outcome_unit", "roi_threshold")


def roi(effect: dict, inputs: dict) -> dict | None:
    """Return on the exposure from the estimated effect, computed in code.

    inputs: cost (total cost of the exposure), units (units exposed; default: the analysis's
    treated count), value_per_outcome_unit (e.g. a margin rate when the outcome is revenue;
    default 1 — the outcome is already value), roi_threshold (optional).
    ROI = (effect × units × value − cost) / cost, also at the ends of the effect's interval.
    """
    if not inputs or inputs.get("cost") in (None, 0) or not effect or effect.get("effect") is None:
        return None
    units = inputs.get("units")
    if not units:
        return None
    value = float(inputs.get("value_per_outcome_unit", 1.0))
    cost = float(inputs["cost"])

    def at(e):
        return None if e is None else round((float(e) * units * value - cost) / cost, 4)
    out = {"roi": at(effect["effect"]), "roi_low": at(effect.get("ci_low")), "roi_high": at(effect.get("ci_high")),
           "incremental_value": round(float(effect["effect"]) * units * value, 2),
           "cost": cost, "units": units, "value_per_outcome_unit": value,
           "assumed_value_per_unit": "value_per_outcome_unit" not in inputs}
    t = inputs.get("roi_threshold")
    if t is not None and out["roi_low"] is not None and out["roi_high"] is not None:
        out["threshold"] = float(t)
        out["clears_threshold"] = ("yes" if out["roi_low"] >= t else "no" if out["roi_high"] < t else "uncertain")
    return out


def _conclusion(est: dict | None) -> str:
    if not est or est.get("effect") is None:
        return "none"
    lo, hi = est.get("ci_low"), est.get("ci_high")
    if lo is None or hi is None:
        return "estimate_without_interval"
    if lo > 0:
        return "positive_effect"
    if hi < 0:
        return "negative_effect"
    return "no_detectable_effect"


def evidence_ledger(state: dict, analytical: dict, roi_out: dict | None) -> list[dict]:
    """Numbered facts the narrative may cite: what was asked, checked and estimated."""
    ev, spec = [], state.get("spec") or {}
    results = state.get("results") or {}

    def add(source, text):
        ev.append({"id": f"E{len(ev) + 1}", "source": source, "text": text})
    q = primary_question(spec)
    add("spec", f"Question ({q.get('type', '?')}): {q.get('statement', spec.get('question', ''))}; "
                f"assignment: {q.get('assignment', 'n/a')}; spec {spec.get('status', 'missing')}")
    for a in spec.get("assumptions", []):
        if a.get("must_confirm"):
            add("spec", f"Assumption to confirm: {a['statement']}")
    add("readiness", f"Analytical readiness: {analytical['status']} — {analytical['summary']}")
    ver = results.get("data_verification") or {}
    for f in ver.get("findings", []):
        if f["severity"] in ("critical", "warning"):
            add("verification", f"{f['severity']}: {f['table'].split('.')[-1]} — {f['description']}")
    dq = results.get("data_quality") or {}
    if dq and not dq.get("error"):
        add("data_quality", f"Analysis table data quality {dq.get('overall_score', '?')}/100, "
                            f"{sum(f.get('severity') == 'critical' for f in dq.get('findings', []))} critical")
    ci = results.get("causal_inference") or {}
    if ci and not ci.get("error"):
        est = ci.get("estimate") or {}
        if est.get("effect") is not None:
            add("causal", f"Estimate ({est.get('method')}): {est['effect']:+.4g} "
                          f"[{est.get('ci_low', float('nan')):+.4g}, {est.get('ci_high', float('nan')):+.4g}], "
                          f"p={est.get('p_value', float('nan')):.3g}; verdict {ci.get('verdict')}")
        for f in ci.get("findings", []):
            if f.get("severity") in ("critical", "warning"):
                add("causal", f"{f['severity']}: {f.get('check', '')} — {(f.get('evidence') or f.get('description') or '')[:220]}")
    if roi_out:
        rng = (f" (interval {roi_out['roi_low']:+.0%} to {roi_out['roi_high']:+.0%})"
               if roi_out["roi_low"] is not None and roi_out["roi_high"] is not None else "")
        add("roi", f"ROI {roi_out['roi']:+.0%}{rng} on cost {roi_out['cost']:,.0f} over {roi_out['units']:,} units"
                   + (" — assumes the outcome is already value" if roi_out["assumed_value_per_unit"] else ""))
    return ev


def decision_readiness(state: dict, inputs: dict | None = None, declared: list | None = None) -> dict:
    """Decision status, conclusion, reasons and caveats, ROI and the evidence ledger."""
    spec = state.get("spec") or {}
    results = state.get("results") or {}
    q = primary_question(spec)
    inputs = {k: v for k, v in (inputs or {}).items() if k in INPUT_KEYS and v is not None}
    analytical = assess(results, confirmations=state.get("confirmations"), declared=declared)
    reasons, caveats = [], []

    ci = results.get("causal_inference") or {}
    est = ci.get("estimate") if ci and not ci.get("error") else None
    if est and "units" not in inputs:
        n = (ci.get("evidence") or {}).get("n_treated")
        if n:
            inputs["units"] = int(n)

    if not spec:
        status = "insufficient_evidence"
        reasons.append("No problem spec: the question was never framed.")
    elif spec.get("answerability", {}).get("status") == "not_answerable_as_posed":
        status = "not_answerable"
        reasons.append(spec["answerability"].get("reason") or "The question can't be answered as posed.")
    else:
        if analytical["status"] in ("blocked", "incomplete"):
            reasons.append(f"Analytical readiness is {analytical['label'].lower()}: {analytical['summary']}")
        if q.get("type") == "causal":
            v = ci.get("verdict") if ci else None
            if not ci or ci.get("error"):
                reasons.append("No causal estimate was produced.")
            elif v == "not_identified":
                reasons.append("The causal effect is not identified with this data.")
            elif v == "caution":
                caveats.append("The causal analysis reports caveats (verdict: caution).")
            elif v != "valid":
                reasons.append(f"Unrecognised causal verdict {v!r}.")
        if spec.get("status") != "confirmed":
            caveats.append("The problem spec was never confirmed by a person.")
        open_assumptions = [a["statement"] for a in spec.get("assumptions", []) if a.get("must_confirm")]
        if open_assumptions:
            caveats.append(f"{len(open_assumptions)} assumption(s) still to confirm.")
        if analytical["status"] == "caution":
            caveats.append(f"Checks raised {len(analytical['cautions'])} issue(s) to review.")
        needed = (spec.get("decision") or {}).get("inputs_needed", [])
        if needed and "cost" not in inputs:
            caveats.append("Decision inputs (costs, value) not provided: the decision itself isn't evaluated, "
                           "only the evidence.")
        status = "insufficient_evidence" if reasons else ("supported_with_caveats" if caveats else "supported")

    # A return computed on an effect that isn't established would be the most misleading number
    # in the summary: only compute it when the evidence supports a conclusion.
    roi_out = roi(est, inputs) if est and status in ("supported", "supported_with_caveats") else None
    if est and "cost" in inputs and roi_out is None and status == "insufficient_evidence":
        caveats.append("ROI not computed: the effect it would rest on isn't established.")
    if roi_out and roi_out.get("clears_threshold") == "uncertain" and status == "supported":
        status = "supported_with_caveats"
    if roi_out and roi_out.get("clears_threshold") == "uncertain":
        caveats.append("Whether the return clears the threshold depends on where in its interval the effect lies.")
    if roi_out and roi_out["assumed_value_per_unit"]:
        caveats.append("ROI assumes the outcome is already in money; pass value_per_outcome_unit (e.g. a margin rate).")

    return {
        "status": status, "label": STATUS_LABEL[status],
        # The conclusion can't outrun the status: an estimate behind insufficient evidence isn't one
        "conclusion": ("none" if status == "not_answerable" else
                       "not_established" if status == "insufficient_evidence" else _conclusion(est)),
        "estimate_direction": _conclusion(est),
        "estimate": est, "roi": roi_out, "inputs": inputs,
        "reasons": reasons, "caveats": caveats,
        "analytical": {k: analytical[k] for k in ("status", "label", "summary", "missing", "failed")},
        "evidence": evidence_ledger(state, analytical, roi_out),
    }


# ── The narrative: every claim must cite the ledger ──────────────────────────

SECTIONS = ("established", "not_established", "alternative_explanations", "assumptions_and_limitations",
            "options_and_tradeoffs", "next_steps")


def validate_narrative(narrative: dict, evidence: list[dict]) -> dict:
    """Keep claims that cite at least one existing evidence id; drop the rest and list them.
    Only known sections survive; nothing in the narrative can set a status."""
    ids = {e["id"] for e in evidence}
    dropped, out = [], {}
    for sec in SECTIONS:
        kept = []
        for c in narrative.get(sec) or []:
            if not isinstance(c, dict) or not str(c.get("text", "")).strip():
                continue
            cites = [x for x in c.get("cites") or [] if x in ids]
            bad = [x for x in c.get("cites") or [] if x not in ids]
            if not cites:
                dropped.append({"section": sec, "text": c.get("text", ""), "reason": "no valid citation"
                                + (f" (cited {bad})" if bad else "")})
                continue
            kept.append({"text": str(c["text"]).strip(), "cites": cites})
        out[sec] = kept
    head = narrative.get("headline") or {}
    hc = [x for x in head.get("cites") or [] if x in ids] if isinstance(head, dict) else []
    out["headline"] = {"text": str(head.get("text", "")).strip(), "cites": hc} if hc else None
    if isinstance(head, dict) and head.get("text") and not hc:
        dropped.append({"section": "headline", "text": head.get("text", ""), "reason": "no valid citation"})
    return {"narrative": out, "dropped": dropped}

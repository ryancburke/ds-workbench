"""
Decision synthesis: what the investigation established, what it didn't, and what to do next.

    summary = session.decide(inputs={"cost": 85_000, "value_per_outcome_unit": 0.35, "roi_threshold": 0.2})

The decision status, the conclusion's direction and any ROI are computed in code
(ds_workbench.core.decision) from the session: the spec, the readiness of the checks, the
verification findings, the estimate and the decision inputs you pass (all optional). The model
then writes the narrative — what was and wasn't established, alternative explanations,
assumptions and limitations, options and trade-offs, next steps — where every claim must cite
the numbered evidence ledger. Claims without a valid citation are removed and listed. The
narrative can't change the status: "insufficient evidence" is a valid conclusion, not a failure.
"""
from __future__ import annotations

import json

from ds_workbench.core.decision import SECTIONS, decision_readiness, validate_narrative
from skills.api_retry import api_call, get_client, skill_entry, tool_input
from skills.models import THINKING_ADAPTIVE, role
from skills.result import SkillResult

_CLAIMS = {"type": "array", "items": {"type": "object", "properties": {
    "text": {"type": "string"}, "cites": {"type": "array", "items": {"type": "string"}}},
    "required": ["text", "cites"]}}

_TOOL = {
    "name": "report_decision_summary",
    "description": "Report the decision summary. Every claim cites evidence ids (E1, E2, ...).",
    "input_schema": {
        "type": "object",
        "properties": {"headline": {"type": "object", "properties": {
                           "text": {"type": "string"}, "cites": {"type": "array", "items": {"type": "string"}}},
                           "required": ["text", "cites"]},
                       **{s: _CLAIMS for s in SECTIONS}},
        "required": ["headline", *SECTIONS],
    },
}

_PROMPT = """You are writing the decision summary of a data investigation for the person who has to decide.

Business question: {question}
Decision: {decision}

Computed by the workbench (facts — don't restate them differently, don't contradict them):
- Decision status: {status} — {label}
- Conclusion about the effect: {conclusion}
- Reasons the evidence falls short: {reasons}
- Caveats: {caveats}

Evidence ledger (cite these ids; nothing else is evidence):
{ledger}

Write, in plain language for a business reader:
- headline: one sentence that matches the status. If the status is insufficient_evidence, say
  what can't be concluded and why — never imply the effect is known. If the conclusion is
  no_detectable_effect, say that, and what size of effect could still be hiding.
- established: what the evidence does show (can be empty).
- not_established: what it doesn't show, and why.
- alternative_explanations: other explanations for what was observed (e.g. who was selected).
- assumptions_and_limitations: what the conclusion rests on, and what the data couldn't check.
- options_and_tradeoffs: the decision options and what each risks, given the evidence and any ROI.
  Don't recommend acting on an effect the status says isn't established.
- next_steps: the cheapest steps that would settle the question (e.g. a holdout, a missing
  variable, a confirmation from a named owner).
Every claim cites at least one evidence id that supports it. Be specific and brief."""


@skill_entry("decision_synthesis")
def synthesize(session=None, inputs: dict | None = None, state: dict | None = None) -> dict:
    """Decision status (code) plus a cited narrative (model) for a session's investigation.

    Args:
        session: WorkflowSession (spec, results, confirmations, required checks).
        inputs: Optional decision inputs: cost, units, value_per_outcome_unit, roi_threshold.
        state: Instead of a session, a state dict ({"spec", "results", "confirmations"}).
    """
    try:
        return _synthesize(session, inputs, state)
    except Exception as exc:
        return SkillResult({"error": str(exc)}, skill="decision_synthesis")


def _synthesize(session, inputs, state):
    if session is not None:
        state = {"spec": session.spec, "results": dict(session._results), "confirmations": session.confirmations}
        declared = session.required_checks
    else:
        declared = None
    if not state:
        raise ValueError("decide() needs a session or a state")
    d = decision_readiness(state, inputs, declared=declared)
    spec = state.get("spec") or {}
    prompt = _PROMPT.format(
        question=spec.get("question", "?"),
        decision=json.dumps(spec.get("decision") or {}),
        status=d["status"], label=d["label"], conclusion=d["conclusion"],
        reasons="; ".join(d["reasons"]) or "none", caveats="; ".join(d["caveats"]) or "none",
        ledger="\n".join(f"{e['id']} [{e['source']}] {e['text']}" for e in d["evidence"]))
    r1 = api_call(get_client().messages.create, **role("analysis"), max_tokens=8000,
                  thinking=THINKING_ADAPTIVE, messages=[{"role": "user", "content": prompt}])
    draft = next((b.text for b in r1.content if b.type == "text"), "")
    if not draft.strip():
        raise RuntimeError("decision_synthesis Pass 1 returned no text")
    r2 = api_call(get_client().messages.create, **role("structured"), max_tokens=4096, tools=[_TOOL],
                  tool_choice={"type": "tool", "name": _TOOL["name"]},
                  messages=[{"role": "user", "content": prompt}, {"role": "assistant", "content": draft},
                            {"role": "user", "content": "Now report it with the report_decision_summary tool."}])
    checked = validate_narrative(tool_input(r2, _TOOL["name"]), d["evidence"])
    out = {**d, **checked, "question": spec.get("question", "")}
    result = SkillResult(out, skill="decision_synthesis")
    if session is not None:
        session.set_decision(result)
    return result

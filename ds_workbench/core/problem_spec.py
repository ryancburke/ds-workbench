"""
ProblemSpec: the structured analytical specification a business question turns into.

The Problem Framer (skills/problem_framer.py) drafts one with a model; everything here is
deterministic and runs on the model's draft, so the rules hold whatever the model wrote:

  - only known fields survive, and a draft is always status "draft": only a person confirms
  - each question has a recognised type; an unrecognised one blocks (fail closed)
  - a causal question needs an exposure and an outcome; a missing one blocks for the primary
    question (a secondary, exploratory causal question only records the gap)
  - a predictive question without a prediction point records the gap (leakage then runs on
    statistics only)
  - decision inputs (costs, margins, thresholds) are optional: listed with the decision, never
    blocking, but while any are needed the spec can't claim to be fully answerable
  - answerability can't claim more than the gaps allow
  - the checks the work must pass (required_checks) follow from the question types, in code

The spec names concepts ("sales", "campaign exposure"), never columns: finding the data that
measures them is the Data Scout's job.
"""
from __future__ import annotations

from datetime import datetime, timezone

QUESTION_TYPES = ("descriptive", "diagnostic", "predictive", "causal", "optimization")
ANSWERABILITY = ("answerable", "answerable_with_gaps", "needs_clarification", "not_answerable_as_posed")
ASSIGNMENT = ("randomized", "not_randomized", "unknown")     # how a causal exposure was assigned

# Checks each kind of question requires (ds_workbench.core.readiness check names)
CHECKS_BY_TYPE = {
    "descriptive":  ("data_quality",),
    "diagnostic":   ("data_quality",),
    "predictive":   ("data_quality", "leakage_detector"),
    "causal":       ("data_quality", "causal_inference"),
    "optimization": ("data_quality",),
}
# Question types the workbench can frame but has no analysis skill for yet
UNSUPPORTED_TYPES = {"optimization": "No optimisation skill yet (see ROADMAP.md, phase 3)."}

_QUESTION_KEYS = ("id", "type", "statement", "primary", "unit_of_analysis", "population",
                  "outcome_concept", "exposure_concept", "comparison", "time_window", "prediction_point",
                  "assignment")


def _text(v) -> str:
    return " ".join(str(v).split()) if v is not None else ""


def _missing(item: str, why: str, blocking: bool) -> dict:
    return {"item": item, "why_it_matters": why, "blocking": blocking}


def normalize(raw: dict, question: str = "") -> dict:
    """Apply the spec rules to a model draft. Returns a new dict with status "draft"."""
    raw = raw if isinstance(raw, dict) else {}
    dec = raw.get("decision") if isinstance(raw.get("decision"), dict) else {}
    decision = {
        "statement": _text(dec.get("statement")),
        "options": [_text(o) for o in dec.get("options") or [] if _text(o)],
        "criteria": _text(dec.get("criteria")),
        "inputs_needed": [_text(i) for i in dec.get("inputs_needed") or [] if _text(i)],
    }
    missing = [_missing(_text(m.get("item")), _text(m.get("why_it_matters")), m.get("blocking") is True)
               for m in raw.get("missing") or [] if isinstance(m, dict) and _text(m.get("item"))]

    questions = []
    for i, q in enumerate(raw.get("questions") or []):
        if not isinstance(q, dict):
            continue
        q = {k: _text(q[k]) if k != "primary" else q[k] is True for k in _QUESTION_KEYS if k in q}
        q["id"] = q.get("id") or f"q{i + 1}"
        if q.get("type") not in QUESTION_TYPES:
            missing.append(_missing(f"What kind of question is '{q.get('statement', q['id'])}'?",
                                    f"unrecognised question type {q.get('type')!r}", True))
            q["type"] = "unrecognised"
        questions.append(q)
    if questions and not any(q.get("primary") for q in questions):
        questions[0]["primary"] = True
    seen_primary = False
    for q in questions:                       # exactly one primary question
        q["primary"] = bool(q.get("primary")) and not seen_primary
        seen_primary = seen_primary or q["primary"]
    for q in questions:
        if q["type"] == "causal":
            if q.get("assignment") not in ASSIGNMENT:
                q["assignment"] = "unknown"
            if q["primary"] and q["assignment"] == "unknown":
                missing.append(_missing("How was the exposure assigned: randomised, or chosen (by whom, how)?",
                                        "it decides between an experiment readout and adjusting for confounding",
                                        False))
            for field, label in (("exposure_concept", "exposure (what was done)"), ("outcome_concept", "outcome")):
                if not q.get(field):
                    missing.append(_missing(f"Define the {label} for '{q.get('statement', q['id'])}'",
                                            "a causal effect is only defined for a stated exposure and outcome",
                                            q["primary"]))
        if q["type"] == "predictive" and not q.get("prediction_point"):
            missing.append(_missing("When will predictions be made (the prediction point)?",
                                    "without it, leakage can only be judged statistically", False))
    if not questions:
        missing.append(_missing("What is the analytical question?", "nothing can be planned without one", True))

    for t in sorted({q["type"] for q in questions} & UNSUPPORTED_TYPES.keys()):
        missing.append(_missing(UNSUPPORTED_TYPES[t], f"{t} questions can be framed but not yet analysed", False))

    missing.sort(key=lambda m: not m["blocking"])      # blocking gaps first
    ans = raw.get("answerability") if isinstance(raw.get("answerability"), dict) else {}
    status = ans.get("status") if ans.get("status") in ANSWERABILITY else "needs_clarification"
    reason = _text(ans.get("reason")) or ("unrecognised answerability" if ans.get("status") not in ANSWERABILITY else "")
    # The status can't claim more than the gaps allow
    if any(m["blocking"] for m in missing) and status in ("answerable", "answerable_with_gaps"):
        status, reason = "needs_clarification", (reason + " — blocking information is missing").strip(" —")
    elif (missing or decision["inputs_needed"]) and status == "answerable":
        status = "answerable_with_gaps"

    spec = {
        "question": _text(question),
        "decision": decision,
        "questions": questions,
        "assumptions": [{"statement": _text(a.get("statement")),
                         "source": a.get("source") if a.get("source") in ("user", "inferred") else "inferred",
                         "must_confirm": a.get("must_confirm") is not False}
                        for a in raw.get("assumptions") or [] if isinstance(a, dict) and _text(a.get("statement"))],
        "missing": missing,
        "clarifying_questions": [_text(c) for c in raw.get("clarifying_questions") or [] if _text(c)],
        "answerability": {"status": status, "reason": reason},
        "status": "draft",
    }
    spec["required_checks"] = list(required_checks(spec))
    return spec


def primary_question(spec: dict) -> dict:
    return next((q for q in spec.get("questions", []) if q.get("primary")), {})


def required_checks(spec: dict) -> tuple[str, ...]:
    """Checks the framed work must pass: the union over its question types (data quality
    always). Unrecognised types add nothing but have already blocked the spec."""
    out = ["data_quality"]
    for q in spec.get("questions", []):
        out += CHECKS_BY_TYPE.get(q.get("type"), ())
    return tuple(dict.fromkeys(out))


def confirm(spec: dict, *, user: str, note: str = "", ts: str | None = None) -> dict:
    """A person's confirmation of the spec (the model can never produce a confirmed spec)."""
    if not _text(user):
        raise ValueError("confirming a problem spec needs a user")
    return {**spec, "status": "confirmed", "confirmed_by": _text(user), "note": _text(note),
            "confirmed_at": ts or datetime.now(timezone.utc).isoformat(timespec="seconds")}

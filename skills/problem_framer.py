"""
Problem Framer: turn a business question into a structured analytical specification.

    from ds_workbench import WorkflowSession, problem_framer

    session = WorkflowSession()
    spec = problem_framer.frame("Did our spring campaign increase sales, and should we expand it?",
                                context="Campaign ran in 12 regions; targeting was chosen by marketing",
                                session=session)
    session.confirm_spec(note="Exposure = received the campaign email")   # recommended

The spec separates the business decision from the analytical questions, types each question
(descriptive, diagnostic, predictive, causal, optimization), names the concepts to measure
(never columns), lists assumptions and missing information, and says whether the question is
answerable as posed. The required checks follow from the question types, in code
(ds_workbench.core.problem_spec), and are declared on the session.
"""
from __future__ import annotations

from ds_workbench.core.problem_spec import ANSWERABILITY, QUESTION_TYPES, normalize
from skills.api_retry import api_call, get_client, skill_entry, tool_input
from skills.models import THINKING_ADAPTIVE, role
from skills.result import SkillResult

_TOOL = {
    "name": "report_problem_spec",
    "description": "Report the structured specification of a business question",
    "input_schema": {
        "type": "object",
        "properties": {
            "decision": {
                "type": "object",
                "description": "The business decision the analysis supports, kept apart from the analytical questions",
                "properties": {
                    "statement": {"type": "string"},
                    "options": {"type": "array", "items": {"type": "string"}},
                    "criteria": {"type": "string", "description": "How the decision would be made, if stated or implied"},
                    "inputs_needed": {"type": "array", "items": {"type": "string"},
                                      "description": "Non-analytical inputs the decision needs (costs, margins, thresholds)"},
                },
                "required": ["statement", "options", "criteria", "inputs_needed"],
            },
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "type": {"type": "string", "enum": list(QUESTION_TYPES)},
                        "statement": {"type": "string"},
                        "primary": {"type": "boolean"},
                        "unit_of_analysis": {"type": "string"},
                        "population": {"type": "string"},
                        "outcome_concept": {"type": "string"},
                        "exposure_concept": {"type": "string", "description": "causal questions: what was done"},
                        "comparison": {"type": "string", "description": "causal questions: compared with what"},
                        "time_window": {"type": "string"},
                        "prediction_point": {"type": "string", "description": "predictive questions: when the model scores"},
                    },
                    "required": ["id", "type", "statement", "primary", "unit_of_analysis", "population",
                                 "outcome_concept"],
                },
            },
            "assumptions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "statement": {"type": "string"},
                        "source": {"type": "string", "enum": ["user", "inferred"]},
                        "must_confirm": {"type": "boolean"},
                    },
                    "required": ["statement", "source", "must_confirm"],
                },
            },
            "missing": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "item": {"type": "string"},
                        "why_it_matters": {"type": "string"},
                        "blocking": {"type": "boolean"},
                    },
                    "required": ["item", "why_it_matters", "blocking"],
                },
            },
            "clarifying_questions": {"type": "array", "items": {"type": "string"}},
            "answerability": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": list(ANSWERABILITY)},
                    "reason": {"type": "string"},
                },
                "required": ["status", "reason"],
            },
        },
        "required": ["decision", "questions", "assumptions", "missing", "clarifying_questions", "answerability"],
    },
}

_PROMPT = """You are a principal data scientist framing a business question before any data is touched.
Your job is to say what kind of analysis the question needs, what would have to be measured,
and whether it can be answered — not to answer it.

Business question: {question}
{context_block}
Work through it:

1. The decision. What will someone do differently depending on the answer? Keep the decision
   apart from the analytical questions: "did X work?" is a question about evidence; "should we
   do more of X?" is a decision that also depends on costs, risks and alternatives. List the
   non-analytical inputs the decision needs (costs, margins, capacity, thresholds) — these are
   optional gaps, not blockers.

2. The analytical questions, each with one type:
   - descriptive: what happened / how much (counts, trends, segments)
   - diagnostic: why something changed, by decomposing it (which segments, which drivers moved);
     it explains a past change without claiming a causal effect of an intervention
   - predictive: what will happen for a unit, scored at some point in time; the use is acting on
     individual predictions or forecasts
   - causal: what an action, policy or treatment did (or would do) to an outcome, compared with
     not doing it. Words like "drive", "impact", "because of", "should we roll out" usually mean
     causal, even when the person asks for "factors" or "a model". A correlation-based model
     answers a predictive question, not a causal one.
   - optimization: choosing the best action under constraints (allocation, pricing, scheduling)
   Mark one question primary. Name concepts (e.g. "weekly revenue per store"), never column names.
   For causal questions state the exposure, the outcome, the comparison and the time window.
   For predictive questions state the prediction point if it is known; if not, list it as missing.

3. Assumptions. Separate what the person said (source=user) from what you are inferring
   (source=inferred). Mark must_confirm when a wrong assumption would change the analysis — e.g.
   whether an exposure was randomised, how the population was selected, what the outcome means.

4. Missing information: at most 6 items, most important first. Mark blocking only when the
   question itself can't be defined without it (too vague to type, or the main causal question
   has no definable exposure or outcome). Whether data exists, how it's recorded and what it
   covers is never blocking here: finding and checking the data comes next. List such points as
   non-blocking gaps or assumptions to confirm.

5. Answerability, as posed:
   - answerable: the question is well defined and the kind of evidence needed plausibly exists
   - answerable_with_gaps: can proceed, but known gaps limit the conclusion
   - needs_clarification: too vague or ambiguous to plan an analysis
   - not_answerable_as_posed: the evidence the question needs can't exist as described (e.g. the
     effect of something applied to every unit at the same moment with no comparison group, or
     of something that hasn't happened); say what would make it answerable (an experiment, a
     holdout, a staggered rollout)
   Then list the clarifying questions you'd ask the person, most important first (at most 5).

CALIBRATION EXAMPLES (an airline; judge the question above on its own terms):
- "Which passengers are likely to miss their connection tomorrow so ground staff can help?" →
  predictive; prediction point = the evening before departure.
- "Did the new boarding process reduce departure delays at the hubs where we piloted it?" →
  causal; exposure = new boarding process; comparison = non-pilot hubs or the pre-pilot period;
  assumption to confirm: how the pilot hubs were chosen.
- "Why did on-time performance fall in August?" → diagnostic.
- "Raise all fares by 5% everywhere on Monday — what did it do to bookings?" with no comparison
  group → causal, not_answerable_as_posed without a comparison (suggest a staggered or regional
  rollout)."""


def _context_block(context: str) -> str:
    return f"\nWhat the person told us about it: {context}\n" if context.strip() else ""


@skill_entry("problem_framer")
def frame(question: str, context: str = "", session=None) -> dict:
    """Frame a business question as an analytical specification (a draft for you to confirm).

    Args:
        question: The business question, in the asker's words.
        context: Anything known about it: how an intervention was assigned, data that exists,
            the decision deadline, costs.
        session: Optional WorkflowSession. The spec is stored on it, its required checks are
            declared on it, and its question becomes the session context if none was set.

    Returns:
        Dict (a ProblemSpec): decision, questions (typed), assumptions, missing,
        clarifying_questions, answerability, required_checks, status="draft".
    """
    try:
        return _frame(question, context, session)
    except Exception as exc:
        return SkillResult({"error": str(exc)}, skill="problem_framer")


def _frame(question: str, context: str, session) -> dict:
    if not (question or "").strip():
        raise ValueError("frame() needs a question")
    prompt = _PROMPT.format(question=question.strip(), context_block=_context_block(context or ""))

    r1 = api_call(
        get_client().messages.create,
        **role("analysis"),
        max_tokens=8000,
        thinking=THINKING_ADAPTIVE,
        messages=[{"role": "user", "content": prompt}],
    )
    analysis = next((b.text for b in r1.content if b.type == "text"), "")
    if not analysis.strip():
        raise RuntimeError("problem_framer Pass 1 returned no text — aborting to avoid empty structured output")

    response = api_call(
        get_client().messages.create,
        **role("structured"),
        max_tokens=4096,
        tools=[_TOOL],
        tool_choice={"type": "tool", "name": _TOOL["name"]},
        messages=[
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": analysis},
            {"role": "user", "content": "Now report the specification using the report_problem_spec tool."},
        ],
    )
    spec = normalize(tool_input(response, _TOOL["name"]), question=question)
    if context.strip():
        spec["context"] = context.strip()
    result = SkillResult(spec, skill="problem_framer")
    if session is not None:
        session.set_spec(result)
    return result

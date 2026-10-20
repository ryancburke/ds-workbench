"""
Leakage Detector skill.

Detects target leakage in a DataFrame before model training.
Combines statistical pre-screening (correlation, conditional null rates) with
LLM analysis to flag features that would inflate training accuracy but fail in production.
"""
import json
import pandas as pd
from skills.result import SkillResult, findings_of
from ds_workbench.core.leakage import leakage_profile
from skills.api_retry import api_call, get_client, skill_entry, tool_input
from skills.models import THINKING_ADAPTIVE, role

build_leakage_profile = leakage_profile  # former name, kept for callers


_TOOL = {
    "name": "report_leakage",
    "description": "Report target leakage findings for features in a dataset",
    "input_schema": {
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["leakage_detected", "suspicious", "clean"],
            },
            "risk_level": {
                "type": "string",
                "enum": ["high", "medium", "low", "none"],
            },
            "summary": {"type": "string"},
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "column": {"type": "string"},
                        "leakage_type": {
                            "type": "string",
                            "enum": ["direct", "proxy", "temporal", "high_correlation", "identifier"],
                        },
                        "severity": {"type": "string", "enum": ["critical", "warning", "info"]},
                        "evidence_type": {
                            "type": "string",
                            "enum": ["statistical", "timing"],
                            "description": "timing only when the prediction point and what is known "
                                           "about the column establish it exists after that point; "
                                           "otherwise statistical (the signal alone)",
                        },
                        "evidence": {"type": "string"},
                        "recommendation": {"type": "string"},
                    },
                    "required": ["column", "leakage_type", "severity", "evidence_type", "evidence",
                                 "recommendation"],
                },
            },
        },
        "required": ["verdict", "risk_level", "summary", "findings"],
    },
}




def _maybe_sample(
    df: pd.DataFrame, target_col: str, n: int = 50_000
) -> tuple[pd.DataFrame, str | None]:
    if len(df) <= n:
        return df, None
    try:
        sampled = (
            df.groupby(target_col, group_keys=False)
            .apply(lambda x: x.sample(min(len(x), max(1, int(n * len(x) / len(df))))))
            .reset_index(drop=True)
        )
    except Exception:
        sampled = df.sample(n, random_state=42).reset_index(drop=True)
    return sampled, f"Sampled {len(sampled):,} of {len(df):,} rows (stratified) for analysis"


@skill_entry("leakage_detector")
def check(
    df: pd.DataFrame,
    target_col: str,
    context: str = "",
    dq_findings: list | None = None,
    session=None,
    *,
    prediction_point: str = "",
    column_notes: dict[str, str] | None = None,
    dataset: str = "",
) -> dict:
    """Check a DataFrame for target leakage before model training.

    Args:
        df: Full training DataFrame including the target column.
        target_col: Name of the column being predicted.
        context: Optional description of the prediction task.
        dq_findings: Optional findings from a prior data_quality.assess() run.
            Populated automatically when a WorkflowSession is provided.
        session: Optional WorkflowSession. Enables prerequisite warnings and
            automatic cross-skill context.
        prediction_point: When the model will score, in plain words (e.g. "1st of each month,
            predicting churn in the next 30 days"). Without it, findings rest on statistics
            only and the output says so. Taken from the session when not given.
        column_notes: What is known about when or how columns are recorded, e.g.
            {"tickets_mtd": "support tickets opened from the 1st to the 14th"}.
        dataset: Table name, so a person's availability confirmation applies to this data only.
            Taken from the session when it has exactly one dataset.

    Returns:
        Dict with verdict, risk_level, findings (each with evidence_type), summary and
        timing_basis ("prediction_point" or "statistics_only").
    """
    try:
        return _check(df, target_col, context, dq_findings, session,
                      prediction_point=prediction_point, column_notes=column_notes, dataset=dataset)
    except Exception as exc:
        return SkillResult({"error": str(exc)}, skill="leakage_detector")


STATISTICS_ONLY_CAVEAT = (
    "Leakage findings are based on statistics only: no prediction point was given, so whether each "
    "column exists at prediction time wasn't checked. Pass prediction_point=... (and column_notes) "
    "to check timing."
)
STATISTICAL_ACTION = "Confirm availability at prediction time, or drop."

_FINDING_KEYS = ("column", "leakage_type", "severity", "evidence_type", "evidence", "recommendation")
_RESULT_KEYS = ("verdict", "risk_level", "summary", "findings")


def finalize(tool_args: dict, *, prediction_point: str = "", dataset: str = "",
             column_notes: dict | None = None) -> dict:
    """Code-side rules applied to the model's report, independent of what the model wrote:
      - only known fields survive, so the model can never record a confirmation or anything else
      - a missing or unknown evidence_type is statistical (fail closed); without a prediction
        point every finding is statistical, since timing can't have been established
      - a statistical critical/warning finding's action is "confirm availability, or drop"
      - the timing basis and, without a prediction point, the statistics-only caveat
    """
    prediction_point = (prediction_point or "").strip()
    out = {k: tool_args.get(k) for k in _RESULT_KEYS if k in tool_args}
    findings = []
    for f in tool_args.get("findings") or []:
        if not isinstance(f, dict):
            continue
        f = {k: f[k] for k in _FINDING_KEYS if k in f}
        if f.get("evidence_type") not in ("statistical", "timing") or not prediction_point:
            f["evidence_type"] = "statistical"
        if f["evidence_type"] == "statistical" and f.get("severity") in ("critical", "warning"):
            rec = (f.get("recommendation") or "").strip()
            f["recommendation"] = STATISTICAL_ACTION + (f" {rec}" if rec else "")
        findings.append(f)
    out["findings"] = findings
    out["dataset"] = (dataset or "").strip()
    out["prediction_point"] = prediction_point
    if column_notes:
        out["column_notes"] = dict(column_notes)
    out["timing_basis"] = "prediction_point" if prediction_point else "statistics_only"
    if not prediction_point:
        out["caveat"] = STATISTICS_ONLY_CAVEAT
    return out


def _check(
    df: pd.DataFrame,
    target_col: str,
    context: str = "",
    dq_findings: list | None = None,
    session=None,
    *,
    prediction_point: str = "",
    column_notes: dict[str, str] | None = None,
    dataset: str = "",
) -> dict:
    if session is not None:
        if dq_findings is None:
            dq_findings = session.get_findings("data_quality")
        if not context:
            context = getattr(session, "context", "")
        prediction_point = prediction_point or getattr(session, "prediction_point", "")
        column_notes = column_notes or getattr(session, "column_notes", None)
        if not dataset and len(getattr(session, "datasets", [])) == 1:
            dataset = session.datasets[0]

    dq_findings = findings_of(dq_findings)  # accepts a prior result or its findings list
    session_warnings = session.prerequisite_warnings("leakage_detector") if session else []

    if session is not None and session.has_run("feature_engineering"):
        session_warnings.append(
            '<div style="background:#fde8e8;border-left:4px solid #c62828;border-radius:4px;'
            'padding:8px 12px;margin-bottom:6px;font-size:12px">'
            '<strong style="color:#c62828">⚠ Feature pipeline ran before leakage check</strong><br>'
            '<span style="color:#666;font-size:11px">'
            'Your feature_engineering plan may include leaking columns. '
            'Review the findings below and re-run feature_engineering after resolving any leakage.'
            '</span></div>'
        )

    df, sample_note = _maybe_sample(df, target_col)
    profile = leakage_profile(df, target_col)
    context_line = f"\nTask context: {context}\n" if context else ""
    if prediction_point:
        context_line += (f"\nPrediction point: {prediction_point}\n"
                         "A column is legitimate only if its value exists at this point for the row "
                         "being scored. Judge each strong signal against it.\n")
    else:
        context_line += ("\nNo prediction point was given, so timing cannot be established: report every "
                         "finding with evidence_type=statistical.\n")
    if column_notes:
        context_line += ("\nWhat the data owner says about when or how columns are recorded:\n"
                         + "\n".join(f"  - {c}: {n}" for c, n in column_notes.items()) + "\n")

    dq_block = ""
    if dq_findings:
        crit = [f for f in dq_findings if f.get("severity") == "critical"]
        warn = [f for f in dq_findings if f.get("severity") == "warning"]
        dq_lines = [
            f"  - {f.get('column','?')} [{f.get('issue_type','?')}]: {f.get('description','')}"
            for f in (crit + warn)[:5]
        ]
        dq_block = (
            f"\nData quality context ({len(crit)} critical, {len(warn)} warnings from earlier DQ scan):\n"
            + "\n".join(dq_lines)
            + "\nFactor these into severity judgments — a null pattern already flagged by DQ "
            "may not be leakage, just a data quality issue.\n"
        )

    _prompt = f"""You are a senior ML engineer performing a pre-training leakage audit on '{target_col}'.
{context_line}{dq_block}
Leakage profile (computed from the dataset):
{json.dumps(profile, indent=2)}

Identify any features that constitute target leakage — information about the target that would
not be available at prediction time in production.

Leakage types:
- direct: feature IS the target or a trivial transformation of it
- proxy: feature only exists as a downstream consequence of the target
  (e.g. a refund_id only assigned after a return; a claim_id only exists after filing a claim)
- high_correlation: the feature separates the target far better than any pre-event signal
  plausibly could — correlation >0.8, or single_feature_auc ≥ 0.97 even when correlation is
  low (a non-linear leak) — with no plausible causal explanation
  (a legitimate feature can have 0.5-0.7 correlation — that's predictive, not leakage)
- temporal: feature uses data from after the prediction event
- identifier: ID whose values encode the target (e.g. fraud_case_id → only fraudulent rows)

Profile signals:
- single_feature_auc: how well this one column alone separates the classes (0.5 = not at all,
  1.0 = perfectly). Legitimate strong predictors usually sit at 0.6–0.9; ≥ 0.97 from a single
  column is rare without leakage. Pearson r can be low for a near-perfect non-linear leak.
- single_class_values: category values (≥1% of rows) that occur in only one target class, e.g. a
  status written by the event itself.
- zero_pct_by_class: a column that is zero for every row of one class but not the other, e.g. a
  credit or fee issued after the event (the zero-valued twin of conditional_null_pct).
A strong signal says where to look; whether it is leakage depends on when the value exists.
An attribute of the thing being scored (a transaction's amount, an applicant's income) is known
at prediction time, so near-perfect separation there is the signal the model is for: at most a
warning to confirm how it was recorded. Call it critical when something says the value is made
by or after the outcome: a derived score or index, a status or ID written by the outcome flow, a
value present or non-zero for one class only.

Severity rules:
- critical: almost certainly leakage (correlation >0.85, conditional null gap >90%, direct proxy).
  Must be removed before training.
- warning: strong signal but worth manual review (correlation 0.7-0.85, gap 60-90%)
- info: weak signal, possibly legitimate — document for awareness only

Verdict:
- leakage_detected: one or more critical findings
- suspicious: only warning-level findings
- clean: no findings or only info-level notes

Do NOT flag:
- Legitimate predictors with moderate correlation (0.3-0.65) — that is predictive signal, not leakage
- Null values uniformly distributed across target classes (equal conditional null rate)
- Sequential or random ID columns (e.g. account_number, ticket_uuid, row_key) that exist
  for EVERY row regardless of the target value. These are not target leakage — they're identifiers
  that should be excluded from feature sets as good practice. Only flag an ID as leakage if the
  conditional_null_pct shows the ID is nearly 100% null for one target class (post-event proxy).
  If you mention an ID column, use info severity at most — never critical or warning.
- Event timestamps that represent WHEN the event was recorded (not post-event data)

CALIBRATION EXAMPLES — illustrative cases from a hospital-readmission dataset (target
'readmitted'); judge this dataset on its own profile:

CLEAN — legitimate predictor, do NOT flag:
  length_of_stay_days, correlation_with_target: 0.38
  → Moderate correlation is predictive signal. Not leakage.

CRITICAL LEAKAGE — post-event proxy, must flag:
  readmission_case_id, conditional_null_pct: {{"True": 3.0, "False": 98.9}}
  → Almost always null when readmitted=False, present when readmitted=True.
    Created by the readmission itself. leakage_type=proxy, severity=critical.

IDENTIFIER — not leakage, info only:
  patient_mrn, unique_count=5000, n_rows=5000,
  conditional_null_pct: {{"True": 0.2, "False": 0.2}}
  → Equal null rates across classes. Row identifier, not leakage.
    leakage_type=identifier, severity=info at most.

SUSPICIOUS HIGH CORRELATION — flag critical:
  care_team_risk_score, correlation_with_target: 0.91
  → r=0.91 with 'readmitted'. Assigned by the care team after the patient came back, so it
    encodes the outcome. leakage_type=high_correlation, severity=critical.

STRONG LEGITIMATE FEATURE — do NOT flag:
  prior_admissions_12m, correlation_with_target: 0.55
  → r=0.55. Prior admissions plausibly drive readmission. Predictive, not leaky.

Evidence type, for every finding:
- timing: the prediction point together with what is known about the column (its notes, or
  what it unambiguously is) shows the value is recorded after the prediction point. Say how in
  the evidence.
- statistical: the signal alone (separation, null or zero patterns). Keep a strong statistical
  signal critical, but don't claim timing you haven't established.
A column whose value exists before the prediction point is a legitimate predictor however strong
it is: report it as info at most.

Reference the specific statistics from the profile in your evidence fields."""

    # Pass 1: reason through leakage patterns with extended thinking
    _r1 = api_call(
        get_client().messages.create,
        **role("leakage"),
        max_tokens=12000,
        thinking=THINKING_ADAPTIVE,
        messages=[{"role": "user", "content": _prompt}],
    )
    _analysis = next((b.text for b in _r1.content if b.type == "text"), "")
    if not _analysis.strip():
        raise RuntimeError("leakage_detector Pass 1 returned no text — aborting to avoid empty structured output")

    # Pass 2: format findings into structured output
    response = api_call(
        get_client().messages.create,
        **role("structured"),
        max_tokens=4096,
        tools=[_TOOL],
        tool_choice={"type": "tool", "name": "report_leakage"},
        messages=[
            {"role": "user", "content": _prompt},
            {"role": "assistant", "content": _analysis},
            {"role": "user", "content": "Now report your findings using the report_leakage tool."},
        ],
    )

    tool_args = finalize(tool_input(response, _TOOL["name"]), prediction_point=prediction_point,
                         dataset=dataset, column_notes=column_notes)
    result = SkillResult(tool_args, skill="leakage_detector",
                         session_warnings=session_warnings)
    # Ready for the next cell: df.drop(columns=result["drop_columns"])
    result["drop_columns"] = list(dict.fromkeys(
        f["column"] for f in result.get("findings", [])
        if f.get("severity") == "critical" and f.get("column")))
    if sample_note:
        result["sample_note"] = sample_note

    # Cross-reference: columns flagged by both DQ and leakage
    if dq_findings:
        dq_cols = {f.get("column") for f in dq_findings
                   if f.get("column") and f.get("severity") in ("critical", "warning")}
        ld_cols = {f.get("column") for f in result.get("findings", [])
                   if f.get("column") and f.get("severity") in ("critical", "warning")}
        overlap = sorted(dq_cols & ld_cols)
        if overlap:
            result["cross_referenced_cols"] = overlap

    if session is not None:
        session.register("leakage_detector", result)
    return result

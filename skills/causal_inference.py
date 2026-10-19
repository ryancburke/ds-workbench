"""
Causal Inference skill.

Estimates the effect of a binary treatment on an outcome — for randomised experiments
(A/B tests) and observational data — and says whether the estimate can be trusted.

Every number is computed in code (skills/causal_stats.py): sample-ratio mismatch, covariate
balance, propensity overlap, power, and effect estimates with confidence intervals. The model
reads that evidence, chooses the design and estimate, and names the threats to validity —
especially bad controls (covariates measured after treatment) and confounding. The estimate
it picks is recomputed in code, so no number in the result is model-written.

    from ds_workbench import causal_inference
    r = causal_inference.analyze(df, treatment_col="variant", outcome_col="revenue",
                                 randomized=True, pre_outcome_col="revenue_prev_30d")
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from skills import causal_stats as cs
from skills.api_retry import api_call, get_client, skill_entry, strict_tool, tool_input
from skills.dtypes import is_text
from skills.models import role
from skills.result import SkillResult

_CHECKS = ["sample_ratio_mismatch", "covariate_imbalance", "confounding", "bad_control",
           "low_overlap", "underpowered", "time_trend_or_novelty", "multiple_comparisons",
           "interference", "other"]
_ESTIMATES = ["difference_in_means", "regression_adjustment", "cuped", "inverse_propensity_weighting"]

_TOOL = {
    "name": "report_causal_analysis",
    "description": "Report whether the treatment effect is identified, which estimate to trust, and the threats to validity",
    "input_schema": {
        "type": "object",
        "required": ["verdict", "design", "identification_strategy", "recommended_estimate",
                     "findings", "assumptions", "summary", "next_steps"],
        "properties": {
            "verdict": {
                "type": "string", "enum": ["valid", "caution", "not_identified"],
                "description": ("valid: the recommended estimate can be acted on; caution: usable with stated "
                                "caveats; not_identified: no estimate here can be trusted as causal"),
            },
            "design": {"type": "string", "enum": ["randomized_experiment", "observational"]},
            "identification_strategy": {
                "type": "string",
                "enum": ["randomization", "selection_on_observables", "difference_in_differences",
                         "instrumental_variables", "regression_discontinuity", "none_credible"],
            },
            "recommended_estimate": {
                "type": "object",
                "required": ["method", "exclude_covariates", "rationale"],
                "properties": {
                    "method": {"type": "string", "enum": _ESTIMATES,
                               "description": "Which of the computed estimates to report"},
                    "exclude_covariates": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Covariates to leave out of adjustment, e.g. ones measured after treatment",
                    },
                    "rationale": {"type": "string"},
                },
            },
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["check", "severity", "column", "evidence", "recommendation"],
                    "properties": {
                        "check": {"type": "string", "enum": _CHECKS},
                        "severity": {"type": "string", "enum": ["critical", "warning", "info"]},
                        "column": {"type": "string", "description": "Column involved, or '' if none"},
                        "evidence": {"type": "string", "description": "The computed numbers behind the finding"},
                        "recommendation": {"type": "string"},
                    },
                },
            },
            "assumptions": {"type": "array", "items": {"type": "string"},
                            "description": "Assumptions the estimate relies on, in plain language"},
            "summary": {"type": "string", "description": "2-3 sentences a stakeholder can act on"},
            "next_steps": {"type": "array", "items": {"type": "string"}},
        },
    },
}


# ── Evidence (all computed) ───────────────────────────────────────────────────

def _binary_treatment(s: pd.Series, treated_value=None) -> np.ndarray:
    values = sorted(s.dropna().unique(), key=str)
    if len(values) != 2:
        raise ValueError(f"Treatment column must have exactly 2 values, found {len(values)}: {values[:5]}")
    if treated_value is None:
        treated_value = True if True in values else (1 if 1 in values else values[1])
    if treated_value not in values:
        raise ValueError(f"treated_value {treated_value!r} not in {values}")
    return (s == treated_value).astype(int).to_numpy()


def _covariate_matrix(df: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, list[str]]:
    """Numeric covariates as-is; low-cardinality text one-hot (drop first); IDs skipped."""
    parts, names = [], []
    for c in cols:
        s = df[c]
        if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
            parts.append(s.astype(float).fillna(s.astype(float).median()).to_numpy()[:, None]); names.append(c)
        elif is_text(s) and 1 < s.nunique() <= 20:
            dummies = pd.get_dummies(s.astype(str), prefix=c, drop_first=True).astype(float)
            parts.append(dummies.to_numpy()); names.extend(dummies.columns)
    return (np.hstack(parts) if parts else np.empty((len(df), 0))), names


def _default_covariates(df, exclude: set[str]) -> list[str]:
    out = []
    for c in df.columns:
        if c in exclude:
            continue
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            continue
        if is_text(s) and s.nunique() > 20:   # identifiers / free text
            continue
        out.append(c)
    return out


def _fmt(est: dict) -> dict:
    keep = ("method", "effect", "se", "ci_low", "ci_high", "p_value", "mean_treated", "mean_control")
    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in est.items() if k in keep}


def compute_evidence(df: pd.DataFrame, treatment_col: str, outcome_col: str,
                     covariates: list[str] | None = None, randomized: bool | None = None,
                     pre_outcome_col: str | None = None, expected_treated_share: float = 0.5,
                     treated_value=None) -> dict:
    """Every statistic the analysis rests on, plus the estimate functions to recompute choices."""
    for col in [treatment_col, outcome_col] + ([pre_outcome_col] if pre_outcome_col else []):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found")
    data = df.dropna(subset=[treatment_col, outcome_col]).reset_index(drop=True)
    t = _binary_treatment(data[treatment_col], treated_value)
    y = data[outcome_col].astype(float).to_numpy()
    if covariates is None:
        covariates = _default_covariates(data, {treatment_col, outcome_col, pre_outcome_col})
    X, x_names = _covariate_matrix(data, covariates)

    ev: dict = {
        "n": int(len(data)), "dropped_missing_rows": int(len(df) - len(data)),
        "treatment_col": treatment_col, "outcome_col": outcome_col,
        "design_declared": ("randomized" if randomized else "observational") if randomized is not None else "not stated",
        "outcome_binary": bool(set(np.unique(y)) <= {0.0, 1.0}),
        "covariates": covariates,
    }
    # A designed split only exists in an experiment; observational groups are whatever they are
    if randomized is False:
        ev["sample_ratio"] = {"n_treated": int(t.sum()), "n_control": int((1 - t).sum()),
                              "note": "not applicable: observational data has no designed split"}
    else:
        ev["sample_ratio"] = cs.srm(int(t.sum()), int((1 - t).sum()), expected_treated_share)

    estimates = {"difference_in_means": cs.diff_in_means(y, t)}
    if X.shape[1]:
        estimates["regression_adjustment"] = cs.ols_effect(y, t, X)
    if pre_outcome_col:
        pre = data[pre_outcome_col].astype(float).fillna(data[pre_outcome_col].median()).to_numpy()
        estimates["cuped"] = cs.cuped(y, t, pre)
    ps = None
    if X.shape[1]:
        ps = cs.propensity_scores(t, X)
        estimates["inverse_propensity_weighting"] = cs.ipw(y, t, ps)
        ev["propensity_overlap"] = cs.overlap(ps, t)
    ev["estimates"] = {k: _fmt(v) for k, v in estimates.items()}
    ev["mde_80pct_power"] = round(cs.mde(estimates["difference_in_means"]["se"]), 6)

    # Balance, outcome correlation, and how much the adjusted estimate moves without each
    # covariate — a covariate that shifts it a lot and is imbalanced in a randomised test is
    # the signature of a bad control (measured after treatment).
    balance = []
    for c in covariates:
        Xc, _ = _covariate_matrix(data, [c])
        if not Xc.shape[1]:
            continue
        col_smd = max((abs(cs.smd(Xc[:, j], t)) for j in range(Xc.shape[1])), default=0.0)
        corr = max((abs(float(np.corrcoef(Xc[:, j], y)[0, 1])) for j in range(Xc.shape[1])
                    if Xc[:, j].std() > 0), default=0.0)
        row = {"covariate": c, "abs_smd": round(col_smd, 3), "abs_corr_with_outcome": round(corr, 3)}
        if X.shape[1] > Xc.shape[1]:
            keep = [i for i, name in enumerate(x_names) if not (name == c or name.startswith(f"{c}_"))]
            without = cs.ols_effect(y, t, X[:, keep])["effect"]
            row["adjusted_effect_without_it"] = round(without, 6)
        balance.append(row)
    ev["balance"] = sorted(balance, key=lambda r: -r["abs_smd"])

    ev["_recompute"] = {"y": y, "t": t, "X": X, "x_names": x_names, "ps": ps,
                        "pre": pre if pre_outcome_col else None, "data": data}
    return ev


def _final_estimate(ev: dict, method: str, exclude: list[str]) -> dict:
    """Recompute the model's chosen estimate (minus excluded covariates) in code."""
    r = ev["_recompute"]
    keep = [i for i, name in enumerate(r["x_names"])
            if not any(name == c or name.startswith(f"{c}_") for c in exclude)]
    X = r["X"][:, keep] if r["X"].shape[1] else r["X"]
    if method == "cuped" and r["pre"] is not None:
        est = cs.cuped(r["y"], r["t"], r["pre"])
    elif method == "regression_adjustment" and X.shape[1]:
        est = cs.ols_effect(r["y"], r["t"], X)
    elif method == "inverse_propensity_weighting" and X.shape[1]:
        est = cs.ipw(r["y"], r["t"], cs.propensity_scores(r["t"], X))
    else:
        est = cs.diff_in_means(r["y"], r["t"])
    out = _fmt(est)
    out["excluded_covariates"] = [c for c in exclude if c in ev["covariates"]]
    return out


# ── Skill entry point ─────────────────────────────────────────────────────────

@skill_entry("causal_inference")
def analyze(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    covariates: list[str] | None = None,
    randomized: bool | None = None,
    pre_outcome_col: str | None = None,
    expected_treated_share: float = 0.5,
    treated_value=None,
    context: str = "",
    session=None,
) -> dict:
    """Estimate a treatment effect and judge whether it is causally credible.

    Args:
        df: One row per unit (user, account, store...).
        treatment_col: Binary treatment/variant column.
        outcome_col: Numeric or 0/1 outcome.
        covariates: Columns to check balance on and adjust for. Default: all other numeric and
            low-cardinality columns. Include only things measured before treatment.
        randomized: True for an A/B test, False for observational data, None if unsure.
        pre_outcome_col: The outcome measured before treatment, enabling CUPED.
        expected_treated_share: The designed treatment share, for the sample-ratio check.
        treated_value: Which treatment value means "treated" (default: True/1/the second value).
        context: What decision the estimate informs.
    """
    try:
        return _analyze(df, treatment_col, outcome_col, covariates, randomized, pre_outcome_col,
                        expected_treated_share, treated_value, context, session)
    except Exception as exc:
        return SkillResult({"error": str(exc)}, skill="causal_inference")


def _analyze(df, treatment_col, outcome_col, covariates, randomized, pre_outcome_col,
             expected_treated_share, treated_value, context, session):
    if session is not None and not context:
        context = getattr(session, "context", "")
    ev = compute_evidence(df, treatment_col, outcome_col, covariates, randomized,
                          pre_outcome_col, expected_treated_share, treated_value)
    shown = {k: v for k, v in ev.items() if not k.startswith("_")}
    context_line = f"\nDecision this informs: {context}\n" if context else ""

    response = api_call(
        get_client().messages.create,
        **role("analysis"),
        max_tokens=16000,
        tools=[strict_tool(_TOOL)],
        tool_choice={"type": "auto"},
        messages=[{
            "role": "user",
            "content": f"""You are a senior data scientist reviewing a causal analysis before anyone acts on it.
Treatment: '{treatment_col}'. Outcome: '{outcome_col}'.{context_line}
Evidence (computed from the data; every number below is exact):
{json.dumps(shown, indent=1, default=str)}

Decide whether the effect of the treatment on the outcome is identified, and which computed
estimate to report. What to weigh:
- Design. In a randomised experiment, the split should match the design (sample_ratio p_value
  far below 0.01 means assignment or logging is broken, and no estimate is trustworthy until
  it's explained), and pre-treatment covariates should be balanced (abs_smd under ~0.1).
- Bad controls. A covariate affected by the treatment (measured after it, or a mediator) must
  not be adjusted for: it biases the estimate. In a randomised test, a covariate that is badly
  imbalanced, or that moves the adjusted estimate a lot when dropped
  (adjusted_effect_without_it), is a likely bad control; name it in exclude_covariates. Judge
  timing from names and context too.
- Observational data. Treated and control groups differ, so the difference in means is
  confounded unless adjusted; adjustment only removes measured confounding, so state what
  unmeasured confounders could remain. Poor propensity overlap means some units have no
  comparable counterparts.
- Precision. If the confidence interval is wide or the observed effect is below
  mde_80pct_power, say what effect the test could and could not have detected.
- Prefer CUPED or regression adjustment on pre-treatment covariates in a clean randomised test
  (same target, smaller interval); prefer regression adjustment or IPW for observational data.

Your verdict and findings must agree with the evidence. Report your results by calling the
report_causal_analysis tool.""",
        }],
    )

    out = tool_input(response, _TOOL["name"])
    choice = out["recommended_estimate"]
    out["estimate"] = _final_estimate(ev, choice["method"], choice.get("exclude_covariates") or [])
    out["evidence"] = shown
    result = SkillResult(out, skill="causal_inference")
    if session is not None:
        session.register("causal_inference", result)
    return result

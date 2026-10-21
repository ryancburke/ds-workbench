"""
Benchmark scoring functions.

Each scorer takes (skill_output: dict, answer_key: dict) and returns a ScoreResult.
ScoreResult.overall is 0-100 and is what gets tracked over time in benchmark_history.json.
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from pathlib import Path


# ── Severity ordering ─────────────────────────────────────────────────────────

_SEV_RANK = {"critical": 0, "warning": 1, "info": 2}


def _at_least_as_severe(output_sev: str, required_sev: str) -> bool:
    """True if output_sev is equal to or more severe than required_sev."""
    return _SEV_RANK.get(output_sev, 2) <= _SEV_RANK.get(required_sev, 2)


def _more_severe_than(output_sev: str, max_sev: str) -> bool:
    """True if output_sev is strictly more severe than max_sev."""
    return _SEV_RANK.get(output_sev, 2) < _SEV_RANK.get(max_sev, 2)


# ── Model family keyword matching ─────────────────────────────────────────────

_FAMILY_KEYWORDS: dict[str, list[str]] = {
    "gradient_boosting": ["gradient_boost", "gbm", "gbt", "xgboost", "xgb", "lightgbm", "lgbm", "catboost"],
    "xgboost":           ["xgboost", "xgb"],
    "lightgbm":          ["lightgbm", "lgbm"],
    "catboost":          ["catboost"],
    "random_forest":     ["random_forest", "random forest", "rf"],
    "logistic_regression": ["logistic_regression", "logistic regression", "logistic"],
    "linear_regression": ["linear_regression", "linear regression", "ols", "ridge", "lasso"],
    "ridge":             ["ridge"],
    "lasso":             ["lasso"],
    "decision_tree":     ["decision_tree", "decision tree"],
    "svm":               ["svm", "svc", "support vector"],
    "neural_network":    ["neural_network", "neural network", "deep learning", "mlp", "ann"],
    "naive_bayes":       ["naive_bayes", "naive bayes", "bayes"],
    "isolation_forest":  ["isolation_forest", "isolation forest"],
    "one_class_svm":     ["one_class_svm", "one-class svm", "one class svm"],
    "time_series":       ["time_series", "time series", "arima", "sarima", "prophet",
                          "exponential_smoothing", "holt"],
    "prophet":           ["prophet"],
    "arima":             ["arima", "sarima"],
    "clustering":        ["cluster", "k-means", "kmeans", "dbscan", "hdbscan"],
    "regression":        ["linear_regression", "linear regression", "ols"],
    "anomaly_detection": ["isolation_forest", "one_class", "anomaly"],
}


def _normalise_family(name: str) -> str:
    """Normalise a model family string to a canonical key (or return as-is lowercase).
    Uses one-directional matching: keyword must appear IN the name, not the reverse.
    This prevents 'regression' from matching 'logistic_regression'."""
    # Spark MLlib class names (GBTClassifier, RandomForestClassifier) → gbt_classifier, ...
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", name)
    name_lower = name.lower().replace("-", "_").replace(" ", "_")
    for canonical, keywords in _FAMILY_KEYWORDS.items():
        if any(kw.replace(" ", "_") in name_lower for kw in keywords):
            return canonical
    return name_lower


def _family_matches_any(model_family: str, acceptable_list: list[str]) -> bool:
    normalised = _normalise_family(model_family)
    for acceptable in acceptable_list:
        if normalised == _normalise_family(acceptable):
            return True
        # Substring check: only apply if the target canonical is >= 10 chars to avoid
        # "regression" matching "logistic_regression" etc.
        canonical_acc = _normalise_family(acceptable)
        if len(canonical_acc) >= 10:
            if canonical_acc in normalised or normalised in canonical_acc:
                return True
    return False


def _family_is_forbidden(model_family: str, forbidden_list: list[str]) -> bool:
    """Use exact canonical matching for forbidden checks to avoid false positives
    like 'logistic_regression' being caught by a forbidden 'regression' pattern."""
    normalised = _normalise_family(model_family)
    for forbidden in forbidden_list:
        if normalised == _normalise_family(forbidden):
            return True
    return False


# ── Result types ─────────────────────────────────────────────────────────────

@dataclass
class ScoreResult:
    benchmark_id: str          # e.g. "orders_dirty/data_quality"
    overall: float             # 0-100, the number tracked over time
    passed: bool               # overall >= 70
    components: dict           # per-dimension scores
    matched_findings: list     # which required findings were caught
    missed_findings: list      # which required findings were missed
    false_positives: list      # forbidden findings that appeared
    severity_violations: list  # findings whose severity exceeds max_severity
    notes: list[str]           # human-readable diagnostic notes

    def summary_line(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        # score_in_range for DQ, verdict_correct for leakage, '?' for model_selector
        if "score_in_range" in self.components:
            extra = f"score_in_range={self.components['score_in_range']}"
        elif "verdict_correct" in self.components:
            extra = f"verdict={'OK' if self.components['verdict_correct'] else 'WRONG'}"
        else:
            extra = "score_in_range=?"
        return (f"[{status}] {self.benchmark_id}: {self.overall:.1f}/100  "
                f"(recall={self.components.get('recall',0):.0%}  "
                f"fp={len(self.false_positives)}  {extra})")


# ── Shared: leakage-safe code order ──────────────────────────────────────────

_SPLIT_RE = re.compile(r"randomSplit\(|train_test_split\(|sampleBy\(|\btrain(?:_df|_sdf)?\s*=")
_LEARN_RE = re.compile(r"\.fit\(|groupBy\([^)]*\)\s*\.\s*(?:count|agg)\(")


def split_before_learning(code: str) -> tuple[bool, str]:
    """Heuristic: is the train/test split made before anything is learned from the data
    (model or transformer .fit, frequency counts / class weights via groupBy)?
    Fitting on all rows, or with no holdout at all, leaks evaluation data into training."""
    learn = _LEARN_RE.search(code or "")
    if not learn:
        return True, "nothing is fitted"
    split = _SPLIT_RE.search(code)
    if not split:
        return False, f"'{learn.group(0)}' runs on all rows; the code never holds data out"
    if learn.start() < split.start():
        return False, f"'{learn.group(0)}' runs before the train/test split"
    return True, "split comes before any fitting"


# ── DQ scoring ───────────────────────────────────────────────────────────────

RANGE_FADE = 10  # points outside the expected overall_score range at which range credit hits 0

def _match_required_finding(output_finding: dict, spec: dict) -> bool:
    """True if output_finding satisfies a required/optional finding spec."""
    col = output_finding.get("column", "").lower()
    issue = output_finding.get("issue_type", "")
    sev = output_finding.get("severity", "info")

    if "column_hint" in spec:
        if spec["column_hint"].lower() not in col:
            return False

    if "issue_type" in spec:
        if issue != spec["issue_type"]:
            return False

    # Some problems are fairly reported under more than one type (an 'N/A' placeholder is
    # a null_rate or an inconsistent_format finding)
    if "issue_types" in spec:
        if issue not in spec["issue_types"]:
            return False

    if "min_severity" in spec:
        if not _at_least_as_severe(sev, spec["min_severity"]):
            return False

    return True


def _check_forbidden_finding(output_finding: dict, spec: dict) -> bool:
    """True if output_finding matches a forbidden finding pattern."""
    col = output_finding.get("column", "").lower()
    issue = output_finding.get("issue_type", "")

    if "issue_type" in spec:
        if issue != spec["issue_type"]:
            return False

    if "column_hint" in spec:
        if spec["column_hint"].lower() not in col:
            return False

    # Optional: findings up to max_severity are allowed (e.g. an info note on a clean column)
    if "max_severity" in spec:
        if not _more_severe_than(output_finding.get("severity", "info"), spec["max_severity"]):
            return False

    return True


def _check_severity_override(output_finding: dict, spec: dict) -> bool:
    """True if output_finding violates a max_severity constraint."""
    col = output_finding.get("column", "").lower()
    issue = output_finding.get("issue_type", "")
    sev = output_finding.get("severity", "info")

    if "column_hint" in spec:
        if spec["column_hint"].lower() not in col:
            return False

    if "issue_type" in spec:
        if issue != spec["issue_type"]:
            return False

    if "max_severity" in spec:
        if _more_severe_than(sev, spec["max_severity"]):
            return True

    return False


def score_dq(output: dict, answer_key: dict) -> ScoreResult:
    """Score a data_quality skill output against its answer key."""
    benchmark_id = f"{answer_key['_meta']['dataset']}/data_quality"
    findings = output.get("findings", [])
    overall_score = output.get("overall_score", 0)

    notes = []
    matched = []
    missed = []
    false_positives = []
    severity_violations = []

    # ── 1. Recall: did we catch all required findings? ────────────────────────
    required = answer_key.get("required_findings", [])
    for spec in required:
        matched_any = any(_match_required_finding(f, spec) for f in findings)
        if matched_any:
            matched.append(spec["id"])
        else:
            missed.append(spec["id"])
            notes.append(f"MISSED required finding '{spec['id']}': {spec.get('note','')}")

    recall = len(matched) / len(required) if required else 1.0

    # ── 2. False positives: did we report forbidden findings? ─────────────────
    forbidden = answer_key.get("forbidden_findings", [])
    for f in findings:
        for spec in forbidden:
            if _check_forbidden_finding(f, spec):
                false_positives.append({
                    "finding": f,
                    "rule_id": spec["id"],
                    "reason": spec.get("note", ""),
                })
                notes.append(f"FALSE POSITIVE '{spec['id']}': reported {f.get('issue_type')} on '{f.get('column')}' which should not exist")

    fp_penalty = min(len(false_positives) * 15, 40)

    # ── 3. Severity violations: findings reported at wrong severity ───────────
    sev_overrides = answer_key.get("forbidden_severity_overrides", [])
    for f in findings:
        for spec in sev_overrides:
            if _check_severity_override(f, spec):
                severity_violations.append({
                    "finding": f,
                    "rule_id": spec["id"],
                    "reason": spec.get("note", ""),
                })
                notes.append(f"SEVERITY VIOLATION '{spec['id']}': '{f.get('column')}' reported as {f.get('severity')} but max is {spec['max_severity']}")

    sev_penalty = min(len(severity_violations) * 10, 20)

    # ── 4. Score in range? ────────────────────────────────────────────────────
    # A score computed in code from the findings adds nothing the recall, false-positive
    # and severity checks don't already measure, so only a model-stated score is checked.
    score_range = answer_key.get("score", {}) if output.get("score_source") != "computed" else {}
    score_in_range = True
    range_distance = 0.0
    if score_range:
        lo, hi = score_range.get("min", 0), score_range.get("max", 100)
        score_in_range = lo <= overall_score <= hi
        range_distance = max(lo - overall_score, overall_score - hi, 0)
        if not score_in_range:
            notes.append(
                f"SCORE OUT OF RANGE: got {overall_score}, expected {score_range.get('min')}-{score_range.get('max')}"
            )

    # ── 5. Max findings check ─────────────────────────────────────────────────
    max_total = answer_key.get("max_total_findings")
    max_critical = answer_key.get("max_critical_findings")
    max_warning = answer_key.get("max_warning_findings")

    if max_total is not None and len(findings) > max_total:
        noise_count = len(findings) - max_total
        notes.append(f"NOISY: {len(findings)} findings reported, expected at most {max_total} ({noise_count} likely noise)")

    if max_critical is not None:
        n_critical = sum(1 for f in findings if f.get("severity") == "critical")
        if n_critical > max_critical:
            notes.append(f"OVER-CRITICAL: {n_critical} critical findings, expected at most {max_critical}")

    if max_warning is not None:
        n_warning = sum(1 for f in findings if f.get("severity") == "warning")
        if n_warning > max_warning:
            notes.append(f"OVER-WARNING: {n_warning} warning findings, expected at most {max_warning}")

    # ── 6. Composite score ────────────────────────────────────────────────────
    #   recall:         50 points (catching real issues is the most important thing)
    #   score_in_range: 25 points (calibration)
    #   no false positives: 25 points (precision on forbidden findings)
    recall_pts      = recall * 50
    # Full credit inside the range, fading to 0 at RANGE_FADE points outside it: the
    # skill's own score is model arithmetic, so 1 point over shouldn't cost all 25.
    range_pts       = 25 * max(0.0, 1 - range_distance / RANGE_FADE)
    fp_pts          = max(0, 25 - fp_penalty)
    sev_pts_penalty = sev_penalty
    overall = max(0.0, recall_pts + range_pts + fp_pts - sev_pts_penalty)

    components = {
        "recall":              round(recall, 3),
        "recall_pts":          round(recall_pts, 1),
        "score_in_range":      score_in_range,
        "range_pts":           range_pts,
        "false_positive_count": len(false_positives),
        "fp_pts":              round(fp_pts, 1),
        "severity_violations": len(severity_violations),
        "sev_penalty":         sev_penalty,
    }

    return ScoreResult(
        benchmark_id=benchmark_id,
        overall=round(overall, 1),
        passed=overall >= 70,
        components=components,
        matched_findings=matched,
        missed_findings=missed,
        false_positives=false_positives,
        severity_violations=severity_violations,
        notes=notes,
    )


# ── Model selector scoring ────────────────────────────────────────────────────

def _coerce_list(val):
    """Claude occasionally returns a JSON array as a string in tool calls when the response is long.
    If val is a string that looks like a JSON array, parse it."""
    if isinstance(val, str):
        stripped = val.strip()
        if stripped.startswith("["):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                pass
    return val if isinstance(val, list) else []


def score_model_selector(output: dict, answer_key: dict, benchmark_id: str | None = None) -> ScoreResult:
    """Score a model_selector skill output against its answer key."""
    dataset = answer_key["_meta"]["dataset"]
    if benchmark_id is None:
        target = answer_key["_meta"]["target_col"]
        benchmark_id = f"{dataset}/model_selector_{target}"

    notes = []
    false_positives = []
    severity_violations = []
    matched = []
    missed = []

    components: dict = {}
    total_pts = 0.0
    max_pts = 0.0

    # ── task_type (35 pts) ────────────────────────────────────────────────────
    required_fields = answer_key.get("required_fields", {})
    task_spec = required_fields.get("task_type", {})
    if task_spec:
        acceptable = task_spec.get("acceptable_values", [])
        forbidden_vals = task_spec.get("forbidden_values", [])
        got = output.get("task_type", "")
        correct = got in acceptable
        if forbidden_vals and got in forbidden_vals:
            correct = False
            notes.append(f"WRONG task_type: '{got}' is explicitly forbidden (acceptable: {acceptable})")
        elif not correct:
            notes.append(f"WRONG task_type: got '{got}', acceptable: {acceptable}")
        pts = 35 if correct else 0
        total_pts += pts
        max_pts += 35
        components["task_type_correct"] = correct
        components["task_type_pts"] = pts
        (matched if correct else missed).append("task_type")

    # ── class_balance_issue (25 pts) ─────────────────────────────────────────
    bal_spec = required_fields.get("class_balance_issue", {})
    if bal_spec and "expected_value" in bal_spec:
        expected = bal_spec["expected_value"]
        got_bal = output.get("target_analysis", {}).get("class_balance_issue")
        correct_bal = got_bal == expected
        if not correct_bal:
            notes.append(
                f"WRONG class_balance_issue: got {got_bal!r}, expected {expected!r}  — {bal_spec.get('note','')}"
            )
        pts = 25 if correct_bal else 0
        total_pts += pts
        max_pts += 25
        components["class_balance_correct"] = correct_bal
        components["class_balance_pts"] = pts
        (matched if correct_bal else missed).append("class_balance_issue")

    # ── imbalance_ratio range (10 pts) ────────────────────────────────────────
    target_analysis_spec = answer_key.get("target_analysis", {})
    ratio_spec = target_analysis_spec.get("imbalance_ratio", {})
    if ratio_spec:
        got_ratio = output.get("target_analysis", {}).get("imbalance_ratio")
        in_range = (got_ratio is not None and
                    ratio_spec["min"] <= got_ratio <= ratio_spec["max"])
        if got_ratio is None:
            notes.append(f"MISSING imbalance_ratio field (expected {ratio_spec['min']}-{ratio_spec['max']})")
        elif not in_range:
            notes.append(f"IMBALANCE RATIO out of range: got {got_ratio}, expected {ratio_spec['min']}-{ratio_spec['max']}")
        pts = 10 if in_range else 0
        total_pts += pts
        max_pts += 10
        components["imbalance_ratio_correct"] = in_range
        components["imbalance_ratio_pts"] = pts

    # ── data_readiness (10 pts) ───────────────────────────────────────────────
    dr_spec = required_fields.get("data_readiness", {})
    if dr_spec:
        acceptable_dr = dr_spec.get("acceptable_values", [])
        got_dr = output.get("data_readiness", "")
        correct_dr = got_dr in acceptable_dr
        if not correct_dr:
            notes.append(f"WRONG data_readiness: got '{got_dr}', acceptable: {acceptable_dr}")
        pts = 10 if correct_dr else 0
        total_pts += pts
        max_pts += 10
        components["data_readiness_correct"] = correct_dr
        components["data_readiness_pts"] = pts
        (matched if correct_dr else missed).append("data_readiness")

    # ── top recommendation family (20 pts) ────────────────────────────────────
    rec_spec = answer_key.get("recommendations", {})
    if rec_spec:
        recs = _coerce_list(output.get("recommendations", []))
        top1_acceptable = rec_spec.get("top_1_acceptable_families", [])
        all_acceptable = rec_spec.get("acceptable_families", [])
        forbidden_fams = rec_spec.get("forbidden_families", [])

        top1_ok = False
        if recs and isinstance(recs[0], dict):
            top1_family = recs[0].get("model_family", "")
            top1_ok = _family_matches_any(top1_family, top1_acceptable)
            if not top1_ok:
                notes.append(
                    f"TOP-1 model '{top1_family}' not in acceptable set {top1_acceptable}"
                )
        else:
            notes.append("NO RECOMMENDATIONS returned")

        pts = 20 if top1_ok else 0
        total_pts += pts
        max_pts += 20
        components["top1_family_correct"] = top1_ok
        components["top1_pts"] = pts
        (matched if top1_ok else missed).append("top1_recommendation")

        # Forbidden family check (exact canonical match only)
        for rec in (r for r in recs if isinstance(r, dict)):
            fam = rec.get("model_family", "")
            if _family_is_forbidden(fam, forbidden_fams):
                false_positives.append({"model_family": fam, "reason": "forbidden family for this task type"})
                notes.append(f"FORBIDDEN FAMILY: '{fam}' recommended but is forbidden for this task type")

        # Key feature insight (for sales_forecasting)
        key_insight = rec_spec.get("key_insight_required")
        if key_insight:
            # Check if any recommendation's rationale or caveats mention the key feature
            all_text = " ".join(
                ((r.get("rationale") or "") + " " + " ".join(r.get("caveats") or []))
                for r in recs if isinstance(r, dict)
            ).lower()
            # Also check assumption_checks
            assumption_checks = _coerce_list(output.get("assumption_checks", []))
            all_text += " ".join(
                a.get("evidence", "") for a in assumption_checks if isinstance(a, dict)
            ).lower()
            if key_insight.lower() in all_text:
                pts = 0  # no extra pts — just note
                notes.append(f"GOOD: key feature '{key_insight}' mentioned in output")
                matched.append(f"key_insight_{key_insight}")
            else:
                notes.append(
                    f"MISSING KEY INSIGHT: '{key_insight}' not mentioned — "
                    f"{rec_spec.get('key_insight_note','')}"
                )
                missed.append(f"key_insight_{key_insight}")

    # ── imbalance handling mention (bonus 0 pts, just diagnostic) ─────────────
    if rec_spec.get("imbalance_handling_required"):
        recs = _coerce_list(output.get("recommendations", []))
        all_text = " ".join(
            (r.get("rationale") or "") + " " + (r.get("starter_code") or "") +
            " ".join(r.get("caveats") or [])
            for r in recs if isinstance(r, dict)
        ).lower()
        all_text += " ".join(output.get("readiness_notes", [])).lower()
        # Also scan target_analysis summary which often contains imbalance strategy discussion
        all_text += (output.get("target_analysis") or {}).get("summary", "").lower()
        imbalance_signals = [
            "scale_pos_weight", "class_weight", "balanced",
            "smote", "oversample", "undersample",
            "precision_recall", "pr_auc", "f1", "f-beta", "fbeta",
            "resampl", "imbalance",
        ]
        found_signal = any(s in all_text for s in imbalance_signals)
        if found_signal:
            notes.append("GOOD: imbalance handling strategy mentioned in output")
            matched.append("imbalance_handling")
        else:
            notes.append(
                f"WARNING: imbalance_handling_required=true but no imbalance strategy found — "
                f"{rec_spec.get('imbalance_handling_note','')}"
            )
            missed.append("imbalance_handling")

    # The decision's primary metric must suit the target (optional per key), e.g. not
    # plain accuracy at 32:1, where predicting the majority class scores 97%
    decision_specs = list(answer_key.get("decision_requirements", []))
    if answer_key.get("decision_primary_metric"):
        decision_specs.append({"id": "decision_primary_metric", "field": "primary_metric",
                               **answer_key["decision_primary_metric"]})
    for spec in decision_specs:
        value = str((output.get("decision") or {}).get(spec["field"], "")).lower()
        max_pts += 10
        if any(term in value for term in spec["any_of"]):
            total_pts += 10; matched.append(spec["id"])
        else:
            missed.append(spec["id"])
            notes.append(f"DECISION {spec['field'].upper()}: {value[:80]!r} — expected one of {spec['any_of']}")

    # Starter code must not learn from the evaluation rows (optional per key)
    if answer_key.get("split_before_learning"):
        code = "\n".join(r.get("starter_code") or "" for r in output.get("recommendations", []))
        ok, why = split_before_learning(code)
        max_pts += 10
        if ok:
            total_pts += 10; matched.append("split_before_learning")
        else:
            missed.append("split_before_learning"); notes.append(f"LEAKY STARTER CODE: {why}")

    # ── Composite score ───────────────────────────────────────────────────────
    extra_ids = {"imbalance_handling", "split_before_learning"} | {d["id"] for d in decision_specs}
    recall = len([m for m in matched if m not in extra_ids]) \
        / max(len(required_fields), 1)
    overall = (total_pts / max_pts * 100) if max_pts > 0 else 0.0

    # Penalise forbidden families
    if false_positives:
        overall = max(0.0, overall - len(false_positives) * 10)

    components["recall"] = round(recall, 3)
    components["false_positive_count"] = len(false_positives)

    return ScoreResult(
        benchmark_id=benchmark_id,
        overall=round(overall, 1),
        passed=overall >= 70,
        components=components,
        matched_findings=matched,
        missed_findings=missed,
        false_positives=false_positives,
        severity_violations=severity_violations,
        notes=notes,
    )


# ── Leakage detector scoring ─────────────────────────────────────────────────

def _match_required_leakage_finding(output_finding: dict, spec: dict) -> bool:
    col = output_finding.get("column", "").lower()
    leakage_type = output_finding.get("leakage_type", "")
    sev = output_finding.get("severity", "info")

    if "column_hint" in spec:
        if spec["column_hint"].lower() not in col:
            return False

    acceptable_types = spec.get("acceptable_leakage_types")
    if acceptable_types and leakage_type not in acceptable_types:
        return False

    # Timing cases: the leak must be established by timing, not only by its signal
    evidence_types = spec.get("acceptable_evidence_types")
    if evidence_types and output_finding.get("evidence_type", "statistical") not in evidence_types:
        return False

    if "min_severity" in spec:
        if not _at_least_as_severe(sev, spec["min_severity"]):
            return False

    return True


def score_leakage_detector(output: dict, answer_key: dict, benchmark_id: str | None = None) -> ScoreResult:
    """Score a leakage_detector skill output against its answer key."""
    dataset = answer_key["_meta"]["dataset"]
    if benchmark_id is None:
        target = answer_key["_meta"]["target_col"]
        benchmark_id = f"{dataset}/leakage_detector_{target}"

    findings = output.get("findings", [])
    verdict = output.get("verdict", "")

    notes: list[str] = []
    matched: list[str] = []
    missed: list[str] = []
    false_positives: list = []
    severity_violations: list = []

    # ── 1. Recall: required leaky features caught? (50 pts) ───────────────────
    required = answer_key.get("required_findings", [])
    for spec in required:
        matched_any = any(_match_required_leakage_finding(f, spec) for f in findings)
        if matched_any:
            matched.append(spec["id"])
        else:
            missed.append(spec["id"])
            notes.append(
                f"MISSED LEAKAGE '{spec['id']}': did not flag '{spec['column_hint']}' — {spec.get('note','')}"
            )

    recall = len(matched) / len(required) if required else 1.0

    # ── 2. Verdict correct? (25 pts) ─────────────────────────────────────────
    expected = answer_key.get("expected_verdict")
    expected_list = answer_key.get("expected_verdicts", [expected] if expected else [])

    verdict_correct = verdict in expected_list
    if not verdict_correct:
        # Acceptable downgrade: "suspicious" when "leakage_detected" was expected
        if "leakage_detected" in expected_list and verdict == "suspicious":
            verdict_correct = True
            notes.append(f"VERDICT DOWNGRADED: expected leakage_detected, got 'suspicious' (acceptable)")
        else:
            notes.append(f"VERDICT WRONG: expected one of {expected_list}, got '{verdict}'")

    # ── 3. No critical false positives (25 pts) ───────────────────────────────
    forbidden = answer_key.get("forbidden_findings", [])
    for f in findings:
        col = f.get("column", "").lower()
        sev = f.get("severity", "info")
        for spec in forbidden:
            hint = spec.get("column_hint", "").lower()
            max_sev = spec.get("max_severity", "info")
            if hint and hint not in col:
                continue
            if _more_severe_than(sev, max_sev):
                false_positives.append({
                    "finding": f,
                    "rule_id": spec["id"],
                    "reason": spec.get("note", ""),
                })
                notes.append(
                    f"FALSE POSITIVE '{spec['id']}': '{f.get('column')}' flagged as {sev} but max is {max_sev}"
                )

    fp_penalty = min(len(false_positives) * 15, 40)

    # ── 4. Noisy check ────────────────────────────────────────────────────────
    max_total = answer_key.get("max_total_findings")
    if max_total is not None and len(findings) > max_total:
        noise = len(findings) - max_total
        notes.append(f"NOISY: {len(findings)} findings, expected at most {max_total} ({noise} likely noise)")

    # ── 5. Composite score ────────────────────────────────────────────────────
    recall_pts  = recall * 50
    verdict_pts = 25 if verdict_correct else 0
    fp_pts      = max(0, 25 - fp_penalty)
    overall     = max(0.0, recall_pts + verdict_pts + fp_pts)

    components = {
        "recall":               round(recall, 3),
        "recall_pts":           round(recall_pts, 1),
        "verdict_correct":      verdict_correct,
        "verdict_pts":          verdict_pts,
        "false_positive_count": len(false_positives),
        "fp_pts":               round(fp_pts, 1),
    }

    return ScoreResult(
        benchmark_id=benchmark_id,
        overall=round(overall, 1),
        passed=overall >= 70,
        components=components,
        matched_findings=matched,
        missed_findings=missed,
        false_positives=false_positives,
        severity_violations=severity_violations,
        notes=notes,
    )


# ── Feature Engineering scoring ──────────────────────────────────────────────

def score_feature_engineering(output: dict, answer_key: dict, benchmark_id: str | None = None) -> ScoreResult:
    """Score a feature_engineering skill output against its answer key."""
    dataset = answer_key["_meta"]["dataset"]
    if benchmark_id is None:
        target = answer_key["_meta"]["target_col"]
        benchmark_id = f"{dataset}/feature_engineering_{target}"

    features = output.get("features", [])
    dropped_cols = [c.lower() for c in output.get("dropped_cols", [])]
    pyspark_code = output.get("pyspark_code", "")
    # Trees don't need scaling, so a tree-based plan may keep numerics as-is
    tree_based = output.get("model_family") == "tree_based"

    notes: list[str] = []
    matched: list[str] = []
    missed: list[str] = []
    false_positives: list = []
    severity_violations: list = []

    # Build lookup: col_name_lower → feature entry
    feature_by_col = {f["column"].lower(): f for f in features}

    # ── 1. Required transformations recall (40 pts) ───────────────────────────
    required_transforms = answer_key.get("required_transformations", [])
    for spec in required_transforms:
        hint = spec["column_hint"].lower()
        acceptable = spec.get("acceptable_actions", [])
        requires_imp = spec.get("requires_imputation", False)

        matching = {col: feat for col, feat in feature_by_col.items() if hint in col}

        if not matching:
            if any(hint in dc for dc in dropped_cols):
                missed.append(spec["id"])
                notes.append(
                    f"WRONG ACTION '{spec['id']}': '{spec['column_hint']}' was dropped but should be transformed"
                )
            else:
                missed.append(spec["id"])
                notes.append(
                    f"MISSING TRANSFORM '{spec['id']}': '{spec['column_hint']}' not found in plan"
                )
            continue

        feat_entry = list(matching.values())[0]
        action = feat_entry.get("action", "")
        imputation = feat_entry.get("imputation", "none")

        action_ok = (
            action in acceptable
            or (action == "log_then_scale" and "scale" in acceptable)
            or (tree_based and action == "keep_as_is" and "scale" in acceptable)
        )
        imputation_ok = not requires_imp or imputation != "none"

        if action_ok and imputation_ok:
            matched.append(spec["id"])
        else:
            missed.append(spec["id"])
            if not action_ok:
                notes.append(
                    f"WRONG ACTION '{spec['id']}': '{spec['column_hint']}' action='{action}' not in {acceptable}"
                )
            if not imputation_ok:
                notes.append(
                    f"MISSING IMPUTATION '{spec['id']}': '{spec['column_hint']}' has nulls but imputation='{imputation}'"
                )

    recall = len(matched) / len(required_transforms) if required_transforms else 1.0
    recall_pts = recall * 40

    # ── 2. Required identifier drops (20 pts) ─────────────────────────────────
    required_drops = answer_key.get("required_dropped_cols", [])
    drop_matched: list[str] = []
    drop_missed: list[str] = []
    for spec in required_drops:
        hint = spec["column_hint"].lower()
        in_dropped = any(hint in dc for dc in dropped_cols)
        # Also OK if it appears with action="drop" in features (belt-and-suspenders)
        in_features_as_drop = any(
            hint in col and feature_by_col[col].get("action") == "drop"
            for col in feature_by_col
        )
        if in_dropped or in_features_as_drop:
            drop_matched.append(spec["id"])
        else:
            drop_missed.append(spec["id"])
            notes.append(
                f"NOT DROPPED '{spec['id']}': '{spec['column_hint']}' should be dropped — {spec.get('note','')}"
            )

    drop_recall = len(drop_matched) / len(required_drops) if required_drops else 1.0
    drop_pts = drop_recall * 20

    # ── 3. Code structural check (20 pts) ─────────────────────────────────────
    required_code = answer_key.get("required_code_elements", [])
    code_matched: list[str] = []
    code_missed: list[str] = []
    if tree_based:
        required_code = [e for e in required_code if e != "StandardScaler"]
    for elem in required_code:
        if elem.lower() in pyspark_code.lower():
            code_matched.append(elem)
        else:
            code_missed.append(elem)
            notes.append(f"MISSING CODE ELEMENT: '{elem}' not found in pyspark_code")

    code_recall = len(code_matched) / len(required_code) if required_code else 1.0
    code_pts = code_recall * 20

    # ── 4. Target / identifier not used as features (20 pts) ──────────────────
    forbidden_in_features = answer_key.get("forbidden_in_features", [])
    fp_count = 0
    for spec in forbidden_in_features:
        hint = spec["column_hint"].lower()
        # False positive: column appears in features with a non-drop action
        in_active_features = any(
            hint in col and feature_by_col[col].get("action") not in ("drop",)
            for col in feature_by_col
        )
        if in_active_features:
            false_positives.append({"rule_id": spec["id"], "reason": spec.get("note", "")})
            notes.append(
                f"FALSE INCLUSION '{spec['id']}': '{spec['column_hint']}' should not be in the feature vector"
            )
            fp_count += 1

    fp_pts = max(0, 20 - fp_count * 20)

    # ── 5. Harder checks (optional per answer key; 10 pts each) ───────────────
    extra_possible = extra_pts = 0.0

    # Encodings that are wrong for the stated model family, e.g. a StringIndexer
    # index on a high-cardinality ID fed to a linear model as a number
    for spec in answer_key.get("forbidden_actions", []):
        extra_possible += 10
        hint = spec["column_hint"].lower()
        exempt = output.get("model_family") in spec.get("allowed_for_model_families", [])
        bad = [col for col, f in feature_by_col.items()
               if hint in col and f.get("action") in spec["forbidden"] and not exempt]
        if bad:
            missed.append(spec["id"])
            notes.append(f"BAD ENCODING '{spec['id']}': {bad} used {spec['forbidden']} — {spec.get('note', '')}")
        else:
            matched.append(spec["id"])
            extra_pts += 10

    # Concepts the plan must address (imbalance handling, case normalisation, ...)
    for spec in answer_key.get("required_concepts", []):
        extra_possible += 10
        haystack = " ".join(
            str(output.get(field, "")) for field in spec.get("fields", ["pyspark_code"])
        ).lower()
        if any(term.lower() in haystack for term in spec["any_of"]):
            matched.append(spec["id"])
            extra_pts += 10
        else:
            missed.append(spec["id"])
            notes.append(f"MISSING CONCEPT '{spec['id']}': none of {spec['any_of']} — {spec.get('note', '')}")

    # Better-than-acceptable transforms earn credit (e.g. log1p on a skewed amount)
    for spec in answer_key.get("preferred_actions", []):
        extra_possible += 10
        hint = spec["column_hint"].lower()
        if any(hint in col and f.get("action") in spec["preferred"] for col, f in feature_by_col.items()):
            matched.append(spec["id"])
            extra_pts += 10
        else:
            missed.append(spec["id"])
            notes.append(f"NOT PREFERRED '{spec['id']}': '{spec['column_hint']}' not {spec['preferred']} — {spec.get('note', '')}")

    if answer_key.get("split_before_learning"):
        extra_possible += 10
        ok, why = split_before_learning(pyspark_code)
        if ok:
            matched.append("split_before_learning"); extra_pts += 10
        else:
            missed.append("split_before_learning"); notes.append(f"LEAKY CODE ORDER: {why}")

    # dropped_cols must agree with the per-column plan
    if answer_key.get("dropped_cols_consistent"):
        extra_possible += 10
        plan_drops = {col for col, f in feature_by_col.items() if f.get("action") == "drop"}
        missing = sorted(plan_drops - set(dropped_cols))
        if missing:
            missed.append("dropped_cols_consistent")
            notes.append(f"INCONSISTENT DROPS: plan drops {missing} but dropped_cols omits them")
        else:
            matched.append("dropped_cols_consistent")
            extra_pts += 10

    # ── 6. Composite score (base 100 + harder checks, normalised to 100) ──────
    base = recall_pts + drop_pts + code_pts + fp_pts
    overall = max(0.0, (base + extra_pts) / (100 + extra_possible) * 100)

    components = {
        "recall":               round(recall, 3),
        "recall_pts":           round(recall_pts, 1),
        "drop_recall":          round(drop_recall, 3),
        "drop_pts":             round(drop_pts, 1),
        "code_recall":          round(code_recall, 3),
        "code_pts":             round(code_pts, 1),
        "false_positive_count": fp_count,
        "fp_pts":               round(fp_pts, 1),
        "extra_pts":            round(extra_pts, 1),
        "extra_possible":       round(extra_possible, 1),
    }

    return ScoreResult(
        benchmark_id=benchmark_id,
        overall=round(overall, 1),
        passed=overall >= 70,
        components=components,
        matched_findings=matched + drop_matched + code_matched,
        missed_findings=missed + drop_missed + code_missed,
        false_positives=false_positives,
        severity_violations=severity_violations,
        notes=notes,
    )


# ── Experiment tracker scoring ───────────────────────────────────────────────

def score_experiment_tracker(output: dict, answer_key: dict, benchmark_id: str | None = None) -> ScoreResult:
    """Verdict (40), required red flags raised as concern/critical (40), and no critical
    finding in a dimension the case rules out (20) — the precision check that stops a
    skill from crying wolf on healthy results."""
    benchmark_id = benchmark_id or f"experiment_metrics/{answer_key['_meta'].get('case', '?')}"
    findings = output.get("findings", [])
    raised = {f.get("dimension") for f in findings if f.get("severity") in ("concern", "critical")}
    critical = {f.get("dimension") for f in findings if f.get("severity") == "critical"}
    notes, matched, missed, false_pos = [], [], [], []

    verdict_ok = output.get("verdict") in answer_key["acceptable_verdicts"]
    if not verdict_ok:
        notes.append(f"WRONG VERDICT: {output.get('verdict')!r}, expected one of {answer_key['acceptable_verdicts']}")

    required = answer_key.get("required_dimensions", [])
    for dim in required:
        (matched if dim in raised else missed).append(dim)
        if dim not in raised:
            notes.append(f"MISSED: no concern/critical finding for {dim}")
    recall = len(matched) / len(required) if required else 1.0

    for dim in answer_key.get("forbidden_critical_dimensions", []):
        if dim in critical:
            false_pos.append({"rule_id": dim, "reason": "critical finding in a dimension this case rules out"})
            notes.append(f"FALSE ALARM: critical {dim} finding")

    overall = 40 * verdict_ok + 40 * recall + max(0, 20 - 10 * len(false_pos))
    return ScoreResult(
        benchmark_id=benchmark_id, overall=round(overall, 1), passed=overall >= 70,
        components={"verdict_correct": verdict_ok, "recall": round(recall, 3),
                    "false_positive_count": len(false_pos)},
        matched_findings=matched, missed_findings=missed, false_positives=false_pos,
        severity_violations=[], notes=notes,
    )


# ── Causal inference scoring ─────────────────────────────────────────────────

_CAUSAL_SEV_LEVEL = {"info": 0, "warning": 1, "critical": 2}


def score_causal_inference(output: dict, answer_key: dict, benchmark_id: str | None = None) -> ScoreResult:
    """Verdict (25), required findings (35), reported estimate within 3 SE of the known true
    effect (25), and no critical finding in a check the case rules out (15)."""
    benchmark_id = benchmark_id or f"{answer_key['_meta']['dataset']}/causal_inference"
    findings = output.get("findings", [])
    notes, matched, missed, false_pos = [], [], [], []

    verdict_ok = output.get("verdict") in answer_key["acceptable_verdicts"]
    if not verdict_ok:
        notes.append(f"WRONG VERDICT: {output.get('verdict')!r}, expected one of {answer_key['acceptable_verdicts']}")

    required = answer_key.get("required_findings", [])
    for spec in required:
        hit = any(f.get("check") in spec["checks"]
                  and spec.get("column_hint", "").lower() in (f.get("column") or "").lower()
                  and _CAUSAL_SEV_LEVEL.get(f.get("severity"), 0) >= _CAUSAL_SEV_LEVEL[spec.get("min_severity", "info")]
                  for f in findings)
        (matched if hit else missed).append(spec["id"])
        if not hit:
            notes.append(f"MISSED: {spec['id']} ({'/'.join(spec['checks'])} ≥ {spec.get('min_severity', 'info')})")
    recall = len(matched) / len(required) if required else 1.0

    estimate_ok = True
    truth = answer_key.get("true_effect")
    if truth is not None:
        est = output.get("estimate") or {}
        effect, se = est.get("effect"), est.get("se")
        estimate_ok = effect is not None and se is not None and abs(effect - truth) <= 3 * se
        (matched if estimate_ok else missed).append("estimate_near_truth")
        if not estimate_ok:
            notes.append(f"ESTIMATE OFF: {est.get('method')} gives {effect} (se {se}), true effect {truth}")

    for f in findings:
        if f.get("severity") == "critical" and f.get("check") in answer_key.get("forbidden_critical_checks", []):
            false_pos.append({"rule_id": f["check"], "reason": "critical finding the case rules out"})
            notes.append(f"FALSE ALARM: critical {f['check']}")

    overall = 25 * verdict_ok + 35 * recall + 25 * estimate_ok + max(0.0, 15 - 7.5 * len(false_pos))
    return ScoreResult(
        benchmark_id=benchmark_id, overall=round(overall, 1), passed=overall >= 70,
        components={"verdict_correct": verdict_ok, "recall": round(recall, 3), "estimate_ok": estimate_ok,
                    "false_positive_count": len(false_pos)},
        matched_findings=matched, missed_findings=missed, false_positives=false_pos,
        severity_violations=[], notes=notes,
    )


# ── Answer key loader ─────────────────────────────────────────────────────────

# ── Problem Framer scoring ───────────────────────────────────────────────────

def score_problem_framer(output: dict, answer_key: dict, benchmark_id: str | None = None) -> ScoreResult:
    """Type (40) + answerability (25) + required mentions (20) + concept naming (15).
    A wrong primary type or answerability fails the case whatever the other points: they
    decide which analysis runs and whether it should run at all."""
    from ds_workbench.core.problem_spec import primary_question
    bid = benchmark_id or "framing/problem_framer"
    notes: list[str] = []
    pq = primary_question(output)
    types = {q.get("type") for q in output.get("questions", [])}

    type_ok = pq.get("type") in answer_key["primary_type"]
    if not type_ok:
        notes.append(f"PRIMARY TYPE WRONG: got {pq.get('type')!r}, expected one of {answer_key['primary_type']}")
    missing_types = [t for t in answer_key.get("required_types", []) if t not in types]
    if missing_types:
        notes.append(f"MISSING QUESTION TYPES: {missing_types}")
    ans = (output.get("answerability") or {}).get("status")
    ans_ok = ans in answer_key["answerability"]
    if not ans_ok:
        notes.append(f"ANSWERABILITY WRONG: got {ans!r}, expected one of {answer_key['answerability']}")

    text = " ".join([*(m.get("item", "") + " " + m.get("why_it_matters", "") for m in output.get("missing", [])),
                     *(a.get("statement", "") for a in output.get("assumptions", [])),
                     *output.get("clarifying_questions", []),
                     " ".join((output.get("decision") or {}).get("inputs_needed", [])),
                     (output.get("answerability") or {}).get("reason", "")]).lower()
    matched, missed = [], []
    for m in answer_key.get("must_mention", []):
        (matched if any(k.lower() in text for k in m["any_of"]) else missed).append(m["id"])
    for mid in missed:
        notes.append(f"NOT RAISED '{mid}'")
    mention_frac = len(matched) / len(answer_key["must_mention"]) if answer_key.get("must_mention") else 1.0

    concepts = answer_key.get("concepts", {})
    concept_hits = [f for f, kws in concepts.items() if any(k.lower() in (pq.get(f) or "").lower() for k in kws)]
    for f in set(concepts) - set(concept_hits):
        notes.append(f"CONCEPT '{f}' not named as expected: got {pq.get(f)!r}")
    concept_frac = len(concept_hits) / len(concepts) if concepts else 1.0

    overall = (40 * type_ok * (0.75 if missing_types else 1.0) + 25 * ans_ok
               + 20 * mention_frac + 15 * concept_frac)
    passed = type_ok and ans_ok and overall >= 70
    return ScoreResult(
        benchmark_id=bid, overall=round(overall, 1), passed=passed,
        components={"type_correct": type_ok, "answerability_correct": ans_ok,
                    "recall": round(mention_frac, 3), "concepts": round(concept_frac, 3),
                    "verdict_correct": type_ok and ans_ok},
        matched_findings=matched, missed_findings=missed, false_positives=[], severity_violations=[],
        notes=notes)


# ── Data Scout scoring ───────────────────────────────────────────────────────

def scout_baseline(answer_key: dict, k: int = 5) -> dict:
    """What plain catalog keyword search finds for the same spec: top-k recall of the required
    tables and how many look-alikes it ranks in its top k."""
    from ds_workbench.core.catalog import SyntheticCatalog
    from ds_workbench.core.scout import search_terms
    cat = SyntheticCatalog.load(answer_key["input"]["catalog"])
    top = [h["name"] for h in cat.search(search_terms(answer_key["input"]["spec"]), limit=k)]
    req = answer_key["required_tables"]
    hit = [any(t in top for t in (r["any_of"] if isinstance(r, dict) else [r])) for r in req]
    return {"top": top, "recall": (sum(hit) / len(req)) if req else 1.0,
            "lookalikes_in_top": sum(t in top for t in answer_key["forbidden_tables"])}


def score_data_scout(output: dict, answer_key: dict, benchmark_id: str | None = None) -> ScoreResult:
    """Recall of required tables as 'investigate' (40) + no look-alike recommended (25) +
    required roles (20) + required joins (15). Fails on a missed required table, a recommended
    look-alike, a forbidden join, or a missing required coverage gap."""
    bid = benchmark_id or "scout/data_scout"
    notes: list[str] = []
    cands = output.get("candidates", [])
    investigate = {c["table"] for c in cands if c["recommendation"] == "investigate"}

    # A requirement is a table, or {"any_of": [...]} when several tables serve equally well
    req = answer_key["required_tables"]
    def _met(r):
        return any(t in investigate for t in r["any_of"]) if isinstance(r, dict) else r in investigate
    missed = [r if isinstance(r, str) else " | ".join(r["any_of"]) for r in req if not _met(r)]
    for t in missed:
        notes.append(f"MISSED TABLE {t}")
    recall = (len(req) - len(missed)) / len(req) if req else 1.0

    fps = [t for t in answer_key["forbidden_tables"] if t in investigate]
    for t in fps:
        notes.append(f"LOOK-ALIKE RECOMMENDED {t}")

    role_hits = []
    for r in answer_key.get("required_roles", []):
        ok = any(c["table"] == r["table"] and c["recommendation"] == "investigate"
                 and any(m["concept"].endswith("." + r["concept_kind"]) for m in c["role_matches"]) for c in cands)
        role_hits.append(ok)
        if not ok:
            notes.append(f"ROLE MISSED: {r['concept_kind']} from {r['table']}")
    role_frac = sum(role_hits) / len(role_hits) if role_hits else 1.0

    pairs = {tuple(sorted((j["left_table"], j["right_table"]))) for c in cands for j in c["join_keys"]}
    # A join requirement is a table pair, or {"any_of": [pairs]}
    req_joins = [r["any_of"] if isinstance(r, dict) else [r] for r in answer_key.get("required_joins", [])]
    join_hits = [any(tuple(sorted(p)) in pairs for p in opts) for opts in req_joins]
    for opts, ok in zip(req_joins, join_hits):
        if not ok:
            notes.append("JOIN MISSED: " + " or ".join(f"{p[0]} ↔ {p[1]}" for p in opts))
    join_frac = sum(join_hits) / len(join_hits) if join_hits else 1.0
    bad_joins = [p for p in answer_key.get("forbidden_joins", []) if tuple(p) in pairs]
    for p in bad_joins:
        notes.append(f"FORBIDDEN JOIN PROPOSED: {p[0]} ↔ {p[1]}")

    gaps = output.get("coverage_gaps", [])
    missing_gaps = [g for g in answer_key.get("required_gaps", []) if not any(x.endswith("." + g) for x in gaps)]
    for g in missing_gaps:
        notes.append(f"COVERAGE GAP NOT REPORTED: {g}")
    feas_ok = answer_key.get("acceptable_feasibility") is None or output.get("feasibility") in answer_key["acceptable_feasibility"]
    if not feas_ok:
        notes.append(f"FEASIBILITY {output.get('feasibility')!r}, expected one of {answer_key['acceptable_feasibility']}")

    dropped = len((output.get("validation") or {}).get("dropped", []))
    if dropped:
        notes.append(f"{dropped} ungrounded claim(s) were removed by validation")
    baseline = scout_baseline(answer_key)

    overall = max(0.0, 40 * recall + max(0, 25 - 12.5 * len(fps)) + 20 * role_frac + 15 * join_frac - 5 * dropped)
    passed = (not missed and not fps and not bad_joins and not missing_gaps and feas_ok and overall >= 70)
    return ScoreResult(
        benchmark_id=bid, overall=round(overall, 1), passed=passed,
        components={"recall": round(recall, 3), "lookalikes_recommended": len(fps), "roles": round(role_frac, 3),
                    "joins": round(join_frac, 3), "ungrounded_claims": dropped,
                    "catalog_calls": len(output.get("catalog_calls", [])),
                    "baseline_recall": round(baseline["recall"], 3),
                    "baseline_lookalikes_in_top5": baseline["lookalikes_in_top"],
                    "verdict_correct": not missing_gaps and feas_ok},
        matched_findings=[r if isinstance(r, str) else " | ".join(r["any_of"]) for r in req if _met(r)],
        missed_findings=missed,
        false_positives=fps + [f"{a}↔{b}" for a, b in bad_joins], severity_violations=[], notes=notes)


def load_answer_key(dataset: str, benchmark: str) -> dict:
    """Load an answer key JSON file. benchmark is the filename stem, e.g. 'data_quality'."""
    path = Path(__file__).parent / "answer_keys" / dataset / f"{benchmark}.json"
    if not path.exists():
        raise FileNotFoundError(f"No answer key at {path}")
    return json.loads(path.read_text())


def score(skill_output: dict, dataset: str, benchmark: str) -> ScoreResult:
    """Top-level scoring entry point. benchmark is e.g. 'data_quality', 'model_selector_attrition'."""
    key = load_answer_key(dataset, benchmark)
    skill = key["_meta"]["skill"]
    canonical_id = f"{dataset}/{benchmark}"
    if skill == "data_quality":
        return score_dq(skill_output, key)
    if skill == "model_selector":
        return score_model_selector(skill_output, key, benchmark_id=canonical_id)
    if skill == "leakage_detector":
        return score_leakage_detector(skill_output, key, benchmark_id=canonical_id)
    if skill == "feature_engineering":
        return score_feature_engineering(skill_output, key, benchmark_id=canonical_id)
    if skill == "experiment_tracker":
        return score_experiment_tracker(skill_output, key, benchmark_id=canonical_id)
    if skill == "causal_inference":
        return score_causal_inference(skill_output, key, benchmark_id=canonical_id)
    if skill == "problem_framer":
        return score_problem_framer(skill_output, key, benchmark_id=canonical_id)
    if skill == "data_scout":
        return score_data_scout(skill_output, key, benchmark_id=canonical_id)
    raise ValueError(f"No scorer implemented for skill '{skill}'")

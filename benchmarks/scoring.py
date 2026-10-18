"""
Benchmark scoring functions.

Each scorer takes (skill_output: dict, answer_key: dict) and returns a ScoreResult.
ScoreResult.overall is 0-100 and is what gets tracked over time in benchmark_history.json.
"""
from __future__ import annotations
import json
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
    "gradient_boosting": ["gradient_boost", "gbm", "xgboost", "xgb", "lightgbm", "lgbm", "catboost"],
    "xgboost":           ["xgboost", "xgb"],
    "lightgbm":          ["lightgbm", "lgbm"],
    "catboost":          ["catboost"],
    "random_forest":     ["random_forest", "random forest", "rf"],
    "logistic_regression": ["logistic_regression", "logistic regression", "logistic"],
    "linear_regression": ["linear_regression", "linear regression", "ols", "ridge", "lasso"],
    "ridge":             ["ridge"],
    "lasso":             ["lasso"],
    "decision_tree":     ["decision_tree", "decision tree"],
    "svm":               ["svm", "support vector"],
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


# ── DQ scoring ───────────────────────────────────────────────────────────────

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
    score_range = answer_key.get("score", {})
    score_in_range = True
    if score_range:
        score_in_range = score_range.get("min", 0) <= overall_score <= score_range.get("max", 100)
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
    range_pts       = 25 if score_in_range else 0
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

    # ── Composite score ───────────────────────────────────────────────────────
    recall = len([m for m in matched if m != "imbalance_handling"]) / max(len(required_fields), 1)
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

        action_ok = action in acceptable
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

    # ── 5. Composite score ────────────────────────────────────────────────────
    overall = max(0.0, recall_pts + drop_pts + code_pts + fp_pts)

    components = {
        "recall":               round(recall, 3),
        "recall_pts":           round(recall_pts, 1),
        "drop_recall":          round(drop_recall, 3),
        "drop_pts":             round(drop_pts, 1),
        "code_recall":          round(code_recall, 3),
        "code_pts":             round(code_pts, 1),
        "false_positive_count": fp_count,
        "fp_pts":               round(fp_pts, 1),
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


# ── Answer key loader ─────────────────────────────────────────────────────────

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
    raise ValueError(f"No scorer implemented for skill '{skill}'")

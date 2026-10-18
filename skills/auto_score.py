"""
Quantitative skill scoring against ground-truth answer keys.

Each scorer returns a dict with at minimum:
  score (float 0-5), recall (float 0-1), and diagnostic fields.
Returns None if no answer key exists for the given dataset/scenario.

Scoring philosophy:
- Primary metric is recall: what fraction of known issues did the skill surface?
- Required-tier findings (clear/unambiguous) are weighted 2× over expected-tier.
- Precision is tracked but not penalised heavily — better to over-report than miss.
- For clean datasets, scored by absence of false critical/warning alarms.
"""

from skills.answer_keys import DQ_ANSWER_KEYS, JOIN_ANSWER_KEYS, MLOPS_ANSWER_KEYS, EDA_ANSWER_KEYS


# ── Data Quality ──────────────────────────────────────────────────────────────

def score_data_quality(result: dict, dataset: str) -> dict | None:
    key = DQ_ANSWER_KEYS.get(dataset)
    if key is None:
        return None

    findings = result.get("findings", [])
    expected = key["expected_findings"]

    # Clean dataset: no expected issues — score by absence of false alarms
    if not expected:
        false_alarms = [f for f in findings if f.get("severity") in ("critical", "warning")]
        n_false = len(false_alarms)
        precision = max(0.0, 1.0 - n_false * 0.15)
        # Also check overall_score meets minimum expectation
        overall = result.get("overall_score", 0)
        score_min = key.get("expected_score_min", 80)
        score_penalty = max(0.0, (score_min - overall) / score_min) if overall < score_min else 0.0
        raw = precision - score_penalty * 0.3
        return {
            "recall": 1.0,
            "precision": round(precision, 3),
            "score": round(max(0.0, min(5.0, raw * 5)), 1),
            "issues_found": 0,
            "issues_total": 0,
            "false_alarms": n_false,
            "note": "Clean dataset — scored by absence of false alarms + overall_score check",
        }

    # Build lookup: (col.lower(), issue_type) → True
    found_pairs = {
        (f.get("column", "").lower(), f.get("issue_type", ""))
        for f in findings
    }
    found_types_only = {f.get("issue_type", "") for f in findings}
    found_cols_only  = {f.get("column", "").lower() for f in findings}

    total_weight = 0.0
    earned_weight = 0.0
    matched = []
    missed = []

    for exp in expected:
        w = 2.0 if exp.get("tier") == "required" else 1.0
        col   = exp["column"].lower()
        itype = exp["issue_type"]
        total_weight += w

        if col == "_table":
            hit = itype in found_types_only
        else:
            hit = (col, itype) in found_pairs
            if not hit and col in found_cols_only:
                # Partial: right column, different issue type → half credit
                earned_weight += w * 0.5
                missed.append({**exp, "partial": True})
                continue

        if hit:
            earned_weight += w
            matched.append(exp)
        else:
            missed.append(exp)

    recall = earned_weight / total_weight if total_weight else 1.0
    score  = round(min(recall, 1.0) * 5, 1)

    return {
        "recall": round(recall, 3),
        "score": score,
        "issues_found": len(matched),
        "issues_total": len(expected),
        "matched": [f"{m['column']}:{m['issue_type']}" for m in matched],
        "missed":  [f"{m['column']}:{m['issue_type']}" for m in missed],
    }


# ── Join Auditor ──────────────────────────────────────────────────────────────

def score_join_auditor(result: dict, join_scenario: str) -> dict | None:
    key = JOIN_ANSWER_KEYS.get(join_scenario)
    if key is None:
        return None

    findings = result.get("findings", [])
    found_types = {f.get("issue_type", "") for f in findings}
    actual_verdict = result.get("overall_verdict", "")

    # Verdict correctness
    if actual_verdict == key["expected_verdict"]:
        verdict_score = 1.0
    elif actual_verdict in key.get("acceptable_verdicts", []):
        verdict_score = 0.7
    else:
        verdict_score = 0.0

    # Issue type recall
    expected_types = key["expected_issue_types"]
    matched = [t for t in expected_types if t in found_types]
    recall = len(matched) / len(expected_types) if expected_types else 1.0

    # 50% verdict, 50% recall
    combined = 0.5 * verdict_score + 0.5 * recall
    score = round(combined * 5, 1)

    return {
        "recall": round(recall, 3),
        "verdict_correct": actual_verdict == key["expected_verdict"],
        "verdict_actual": actual_verdict,
        "verdict_expected": key["expected_verdict"],
        "score": score,
        "issue_types_found": len(matched),
        "issue_types_total": len(expected_types),
    }


# ── MLOps Review ──────────────────────────────────────────────────────────────

def score_mlops(result: dict, scenario_id: str) -> dict | None:
    key = MLOPS_ANSWER_KEYS.get(scenario_id)
    if key is None:
        return None

    findings = result.get("findings", [])
    found_cats = {f.get("category", "") for f in findings}
    expected_cats = key["expected_categories"]

    matched = [c for c in expected_cats if c in found_cats]
    recall  = len(matched) / len(expected_cats) if expected_cats else 1.0
    score   = round(recall * 5, 1)

    return {
        "recall": round(recall, 3),
        "score": score,
        "categories_found": len(matched),
        "categories_total": len(expected_cats),
        "matched":  matched,
        "missed":   [c for c in expected_cats if c not in found_cats],
    }


# ── EDA ───────────────────────────────────────────────────────────────────────

def score_eda(result: dict, dataset: str) -> dict | None:
    key = EDA_ANSWER_KEYS.get(dataset)
    if key is None:
        return None

    targets   = result.get("potential_targets", [])
    insights  = " ".join(result.get("key_insights", [])).lower()
    narrative = result.get("narrative", "").lower()
    all_text  = insights + " " + narrative

    # Target identification
    found_target_pairs = {
        (t.get("column", "").lower(), t.get("task_type", ""))
        for t in targets
    }
    found_target_cols = {t.get("column", "").lower() for t in targets}

    exp_targets = key.get("expected_targets", [])
    target_score = 0.0
    for exp in exp_targets:
        col = exp["column"].lower()
        if (col, exp["task_type"]) in found_target_pairs:
            target_score += 1.0
        elif col in found_target_cols:
            target_score += 0.5  # right column, wrong task type

    target_recall = target_score / len(exp_targets) if exp_targets else 1.0

    # Correlation detection
    exp_corr = key.get("expected_correlation_pairs", [])
    if exp_corr:
        corr_found = sum(
            1 for c1, c2 in exp_corr
            if c1.lower() in all_text and c2.lower() in all_text
        )
        corr_recall = corr_found / len(exp_corr)
    else:
        corr_recall = 1.0

    # Key topic coverage
    key_topics = key.get("key_topics", [])
    if key_topics:
        topics_covered = sum(1 for t in key_topics if t.lower() in all_text)
        topic_recall = topics_covered / len(key_topics)
    else:
        topic_recall = 1.0

    # Combine: 50% targets, 25% correlations, 25% topics
    weights = [0.50, 0.25, 0.25] if exp_targets else [0.0, 0.5, 0.5]
    combined = weights[0] * target_recall + weights[1] * corr_recall + weights[2] * topic_recall
    score = round(combined * 5, 1)

    return {
        "target_recall":      round(target_recall, 3),
        "correlation_recall": round(corr_recall, 3),
        "topic_recall":       round(topic_recall, 3),
        "score": score,
    }


# ── NL → PySpark ──────────────────────────────────────────────────────────────

def score_nl_to_spark(result: dict) -> dict:
    """Structural check: valid Python syntax + both outputs non-empty."""
    import ast as _ast

    pyspark_code = result.get("pyspark_code", "") or ""
    sql_query    = result.get("sql_query",    "") or ""

    pyspark_nonempty = bool(pyspark_code.strip())
    sql_nonempty     = bool(sql_query.strip())

    pyspark_valid = False
    if pyspark_nonempty:
        try:
            _ast.parse(pyspark_code)
            pyspark_valid = True
        except SyntaxError:
            pass

    # Has assumptions or explanation (quality signal)
    has_explanation  = bool((result.get("explanation") or "").strip())
    has_assumptions  = bool(result.get("assumptions"))

    structural_score = (
        (1.0 if pyspark_valid    else 0.0) * 0.40 +
        (1.0 if sql_nonempty     else 0.0) * 0.25 +
        (1.0 if pyspark_nonempty else 0.0) * 0.15 +
        (1.0 if has_explanation  else 0.0) * 0.10 +
        (1.0 if has_assumptions  else 0.0) * 0.10
    )
    score = round(structural_score * 5, 1)

    return {
        "pyspark_valid_syntax": pyspark_valid,
        "sql_generated":        sql_nonempty,
        "pyspark_generated":    pyspark_nonempty,
        "has_explanation":      has_explanation,
        "score": score,
        "note": "Structural check (syntax + completeness). Quality scored qualitatively.",
    }


# ── Dispatcher ────────────────────────────────────────────────────────────────

def auto_score(skill: str, result: dict, dataset: str, scenario_id: str,
               join_scenario: str | None = None) -> dict | None:
    """
    Route to the right scorer and return a quantitative score dict, or None
    if no answer key is available.
    """
    if skill == "data_quality":
        return score_data_quality(result, dataset)
    if skill == "join_auditor":
        return score_join_auditor(result, join_scenario or "")
    if skill == "mlops_review":
        return score_mlops(result, scenario_id)
    if skill == "eda":
        return score_eda(result, dataset)
    if skill == "nl_to_spark":
        return score_nl_to_spark(result)
    return None

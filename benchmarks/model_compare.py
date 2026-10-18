"""
Model comparison experiment: Opus 4.8 vs Fable 5 on DQ and Leakage benchmarks.

Runs each benchmark under three model configs and reports side-by-side scores,
timing, and key quality signals.

Usage:
    .venv/bin/python benchmarks/model_compare.py
    .venv/bin/python benchmarks/model_compare.py --skill data_quality
    .venv/bin/python benchmarks/model_compare.py --skill leakage_detector
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import config  # triggers load_dotenv

import anthropic
import pandas as pd

BENCHMARKS_DIR = Path(__file__).parent
RESULTS_FILE   = BENCHMARKS_DIR / "model_comparison_results.json"

# ── Model configs under test ──────────────────────────────────────────────────

MODELS = [
    {
        "name":     "sonnet-5.5",
        "model":    "claude-sonnet-5-5",
        "thinking": {"type": "adaptive"},
        "extra":    {},
    },
    {
        "name":     "opus-5.5",
        "model":    "claude-opus-5-5",
        "thinking": {"type": "adaptive"},
        "extra":    {},
    },
    {
        "name":     "fable-5.1-medium",
        "model":    "claude-fable-5-1",
        "thinking": {"type": "adaptive"},
        "extra":    {"output_config": {"effort": "medium"}},
    },
]

# ── Datasets per skill ────────────────────────────────────────────────────────

DQ_DATASETS = [
    ("orders_dirty",       "_gen_orders_dirty",       {"table_name": "orders_dirty"}),
    ("sales_clean",        "_gen_sales_clean",        {"table_name": "sales_clean"}),
    ("hr_attrition",       "_gen_hr_attrition",       {"table_name": "hr_attrition"}),
    ("ecommerce",          "_gen_ecommerce",           {"table_name": "ecommerce"}),
    ("transactions_fraud", "_gen_transactions_fraud", {"table_name": "transactions_fraud"}),
]

LD_DATASETS = [
    ("ecommerce_leakage", "_gen_ecommerce_leakage", "converted"),
    ("hr_attrition",      "_gen_hr_attrition",      "attrition"),
]

MS_DATASETS = [
    ("transactions_fraud", "_gen_transactions_fraud", "is_fraud",  "model_selector_fraud"),
    ("ecommerce",          "_gen_ecommerce",           "converted", "model_selector_converted"),
    ("hr_attrition",       "_gen_hr_attrition",        "attrition", "model_selector_attrition"),
    ("sales_forecasting",  "_gen_sales_forecasting",   "revenue",   "model_selector_revenue"),
]


# ── Two-pass runners (model-parameterised) ────────────────────────────────────

def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=config.anthropic_api_key)


def _p1(model_cfg: dict, prompt: str, max_tokens: int) -> str:
    """Run Pass 1 (reasoning) with the given model config. Returns the text block."""
    kwargs = {
        "model":    model_cfg["model"],
        "max_tokens": max_tokens,
        "thinking": model_cfg["thinking"],
        "messages": [{"role": "user", "content": prompt}],
        **model_cfg["extra"],
    }
    r = _client().messages.create(**kwargs)
    return next((b.text for b in r.content if b.type == "text"), "")


def run_dq(df: pd.DataFrame, table_name: str, model_cfg: dict) -> dict:
    from skills.data_quality import build_profile, _TOOL, _get_client as _dq_client
    import skills.data_quality as dq_mod

    profile   = build_profile(df)
    n_rows    = profile["shape"]["rows"]
    col_names = list(profile["columns"].keys())

    # Rebuild the prompt (copy from data_quality._assess)
    import json as _json
    prompt = dq_mod._assess.__code__.co_consts  # we'll reconstruct below

    # Build prompt inline (same as the skill)
    _prompt = f"""You are a senior data engineer reviewing dataset '{table_name}' before ML training.

Profile:
{_json.dumps(profile, indent=2)}

Work through each step below. For every candidate you find, reason about whether it is a genuine
quality problem — don't just list it. Think through the domain implications.

STEP 1 — NULL RATES
Check null_pct for every column. For each column where null_pct > 5.0:
  • State the exact rate.
  • Reason: is this on a critical column (identifier, join key, widely-used feature) or peripheral?
  • Severity: >20% on a key column → critical; 5–20% → usually warning;
    very peripheral features at 5–8% may warrant only info.

STEP 2 — DUPLICATE ROWS
Read the top-level "duplicate_rows" field. State the count and its percentage of {n_rows} rows.
If > 0, assess the training impact: exact duplicates bias model weights toward over-represented patterns.

STEP 3 — FORMAT CONSISTENCY
For every string/object column, inspect top_values.
Flag columns where:
  (a) The same concept appears with different casing in that column's own values, OR
  (b) Different values each use a different casing convention.
Do NOT flag: all-lowercase different words, all-Title-Case sets, datetime/numeric/boolean columns.

STEP 4 — NUMERIC OUTLIERS
For every numeric column where outlier_count > 0 (pre-computed via 3×IQR fence),
work through all three questions:
  Q1 Rate: outlier_count / {n_rows} rows — is this above 1%?
  Q2 Volume: is outlier_count ≥ 10?
  Q3 Plausibility: are the extreme min/max values realistic for what this column measures?
Severity: warning if any of Q1, Q2, Q3 triggers; info if all three say "probably fine".

STEP 5 — IDENTIFIER INTEGRITY
Find columns whose name suggests an identifier (contains 'id', 'key', 'code', 'ref', 'uuid').
Flag any where unique_pct < 95%.

STEP 6 — TYPE MISMATCHES
Find object/string columns where ALL top_values consist only of digits.

FINAL ASSESSMENT
After all six steps:
1. Write a 1–2 sentence overall verdict.
2. List confirmed findings in this format: column | issue_type | severity | brief justification
3. State an overall_score out of 100."""

    analysis = _p1(model_cfg, _prompt, max_tokens=14000)
    if not analysis.strip():
        raise RuntimeError("Pass 1 returned no text")

    from skills.data_quality import _TOOL as DQ_TOOL
    response = _client().messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=4096,
        tools=[DQ_TOOL],
        tool_choice={"type": "tool", "name": "report_data_quality"},
        messages=[{"role": "user", "content": (
            f"Format this data quality analysis for '{table_name}' into structured output.\n\n"
            f"Analysis:\n{analysis}\n\n"
            f"Formatting rules:\n"
            f"- Use ONLY column names from this list: {col_names}\n"
            f"- For duplicate_rows findings, use '{table_name}' as the column field\n"
            f"- Match severity exactly to what the analysis concluded\n"
            f"- Include affected_rows_pct wherever the analysis provides a percentage\n"
            f"- The overall_score must match what the analysis calculated\n"
            f"- Format only what the analysis identified — do not add or remove findings"
        )}],
    )
    tb = next(b for b in response.content if b.type == "tool_use")
    return dict(tb.input)


def run_ld(df: pd.DataFrame, target_col: str, model_cfg: dict) -> dict:
    from skills.leakage_detector import build_leakage_profile, _TOOL as LD_TOOL
    import json as _json

    profile = build_leakage_profile(df, target_col)

    _prompt = f"""You are a senior ML engineer performing a pre-training leakage audit on '{target_col}'.

Leakage profile (computed from the dataset):
{_json.dumps(profile, indent=2)}

Assess every feature for:
1. DIRECT LEAKAGE: feature derived from or identical to the target
2. POST-EVENT PROXY: feature that only exists after the target event
3. HIGH CORRELATION: r > 0.85 with target with no causal mechanism
4. IDENTIFIER: row ID, user ID, transaction ID — not leakage but flag as info

For each finding state the leakage_type, severity (critical/warning/info), and evidence.
Overall verdict: leakage_detected (critical found) / suspicious (warnings only) / clean (no issues)."""

    analysis = _p1(model_cfg, _prompt, max_tokens=12000)
    if not analysis.strip():
        raise RuntimeError("Pass 1 returned no text")

    response = _client().messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=4096,
        tools=[LD_TOOL],
        tool_choice={"type": "tool", "name": "report_leakage"},
        messages=[
            {"role": "user",      "content": _prompt},
            {"role": "assistant", "content": analysis},
            {"role": "user",      "content": "Now report your findings using the report_leakage tool."},
        ],
    )
    tb = next(b for b in response.content if b.type == "tool_use")
    return dict(tb.input)


def run_ms(df: pd.DataFrame, target_col: str, model_cfg: dict) -> dict:
    from skills.model_selector import _build_profile, _TOOL as MS_TOOL
    from skills.models import STRUCTURED_MODEL
    import json as _json

    profile = _build_profile(df, target_col)

    _prompt = f"""You are a senior ML engineer advising a data scientist on model selection.

Data profile:
{profile}

## Your task

1. **Identify the task type** from the target column's distribution (binary_classification,
   multiclass_classification, regression, time_series_forecasting, clustering, anomaly_detection).

2. **Check statistical assumptions** relevant to that task type. For each, cite actual numbers
   from the profile. Cover:
   - Target distribution / class balance (imbalance ratio, skewness, outliers)
   - Feature–target relationship strength (linear vs. non-linear signal)
   - Multicollinearity (any high feature–feature correlations)
   - Data sufficiency (row count vs. model complexity)
   - Missing data patterns (any features with >10% nulls that constrain model choice)

3. **Recommend 3–4 model families**, ranked. For rank 1, include a PySpark MLlib starter
   code snippet (with Databricks in mind). Be specific — cite actual numbers in rationale
   (e.g. "the 5.3:1 imbalance makes gradient boosting preferable to logistic regression
   without resampling").

4. **Assess data readiness**: ready (train immediately), needs_prep (specific transforms
   required first), or insufficient (fundamental gaps).

Ground all reasoning in the actual numbers above, not generic ML advice."""

    analysis = _p1(model_cfg, _prompt, max_tokens=12000)
    if not analysis.strip():
        raise RuntimeError("Pass 1 returned no text")

    response = _client().messages.create(
        model=STRUCTURED_MODEL,
        max_tokens=8192,
        tools=[MS_TOOL],
        tool_choice={"type": "tool", "name": "report_model_selection"},
        messages=[
            {"role": "user",      "content": _prompt},
            {"role": "assistant", "content": analysis},
            {"role": "user",      "content": "Now report your recommendations using the report_model_selection tool."},
        ],
    )
    tb = next(b for b in response.content if b.type == "tool_use")
    return dict(tb.input)


# ── Scoring ───────────────────────────────────────────────────────────────────

def _score(output: dict, dataset: str, benchmark: str) -> dict:
    from benchmarks.scoring import score as score_fn
    try:
        result = score_fn(output, dataset, benchmark)
        return {
            "overall":          result.overall,
            "passed":           result.passed,
            "recall":           result.components.get("recall"),
            "precision":        result.components.get("precision"),
            "correctness":      result.components.get("correctness"),
            "missed_findings":  result.missed_findings,
            "false_positives":  len(result.false_positives),
            "notes":            result.notes,
        }
    except Exception as exc:
        return {"overall": None, "error": str(exc)}


# ── Comparison runner ─────────────────────────────────────────────────────────

def run_comparison(skill_filter: str | None = None):
    import databricks_client as dc

    ts      = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    rows    = []
    all_out = {}

    print(f"\n{'='*72}")
    print(f"  Model Comparison Experiment — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Models: {', '.join(m['name'] for m in MODELS)}")
    print(f"{'='*72}\n")

    # ── DQ benchmarks ────────────────────────────────────────────────────────
    if not skill_filter or skill_filter == "data_quality":
        print("── DATA QUALITY ──────────────────────────────────────────────────────\n")
        for ds_name, gen_fn_name, run_kwargs in DQ_DATASETS:
            df = getattr(dc, gen_fn_name)()
            benchmark_id = f"{ds_name}/data_quality"
            print(f"  Dataset: {ds_name}")
            for mc in MODELS:
                try:
                    t0     = time.perf_counter()
                    output = run_dq(df, run_kwargs["table_name"], mc)
                    dur    = time.perf_counter() - t0
                    scored = _score(output, ds_name, "data_quality")
                except Exception as exc:
                    dur    = time.perf_counter() - t0 if 't0' in dir() else 0
                    output = {}
                    scored = {"overall": None, "error": str(exc)}

                row = {
                    "ts": ts, "skill": "data_quality", "dataset": ds_name,
                    "model": mc["name"], "duration_s": round(dur, 1), **scored,
                }
                rows.append(row)
                all_out[f"{benchmark_id}/{mc['name']}"] = output

                recall_str = f"recall={scored.get('recall', '?')}"
                prec_str   = f"prec={scored.get('precision', '?')}"
                score_str  = f"overall={scored.get('overall', 'ERR')}"
                pass_str   = "✓" if scored.get("passed") else "✗"
                err_str    = f"  ⚠ {scored['error'][:60]}" if scored.get("error") else ""
                print(f"    {mc['name']:18}  {pass_str}  {score_str:14}  {recall_str:14}  "
                      f"{prec_str:12}  {dur:.0f}s{err_str}")
            print()

    # ── Leakage benchmarks ───────────────────────────────────────────────────
    if not skill_filter or skill_filter == "leakage_detector":
        print("── LEAKAGE DETECTOR ──────────────────────────────────────────────────\n")
        for ds_name, gen_fn_name, target_col in LD_DATASETS:
            df           = getattr(dc, gen_fn_name)()
            benchmark    = f"leakage_detector_{target_col}"
            benchmark_id = f"{ds_name}/{benchmark}"
            print(f"  Dataset: {ds_name}  (target={target_col})")
            for mc in MODELS:
                try:
                    t0     = time.perf_counter()
                    output = run_ld(df, target_col, mc)
                    dur    = time.perf_counter() - t0
                    scored = _score(output, ds_name, benchmark)
                except Exception as exc:
                    dur    = time.perf_counter() - t0 if 't0' in dir() else 0
                    output = {}
                    scored = {"overall": None, "error": str(exc)}

                row = {
                    "ts": ts, "skill": "leakage_detector", "dataset": ds_name,
                    "model": mc["name"], "duration_s": round(dur, 1), **scored,
                }
                rows.append(row)
                all_out[f"{benchmark_id}/{mc['name']}"] = output

                verdict    = output.get("verdict", "?")
                recall_str = f"recall={scored.get('recall', '?')}"
                prec_str   = f"prec={scored.get('precision', '?')}"
                score_str  = f"overall={scored.get('overall', 'ERR')}"
                pass_str   = "✓" if scored.get("passed") else "✗"
                err_str    = f"  ⚠ {scored['error'][:60]}" if scored.get("error") else ""
                print(f"    {mc['name']:18}  {pass_str}  {score_str:14}  {recall_str:14}  "
                      f"{prec_str:12}  {dur:.0f}s  [{verdict}]{err_str}")
            print()

    # ── Model selector benchmarks ────────────────────────────────────────────
    if not skill_filter or skill_filter == "model_selector":
        print("── MODEL SELECTOR ────────────────────────────────────────────────────\n")
        for ds_name, gen_fn_name, target_col, benchmark_key in MS_DATASETS:
            df           = getattr(dc, gen_fn_name)()
            benchmark_id = f"{ds_name}/{benchmark_key}"
            print(f"  Dataset: {ds_name}  (target={target_col})")
            for mc in MODELS:
                try:
                    t0     = time.perf_counter()
                    output = run_ms(df, target_col, mc)
                    dur    = time.perf_counter() - t0
                    scored = _score(output, ds_name, benchmark_key)
                except Exception as exc:
                    dur    = time.perf_counter() - t0 if 't0' in dir() else 0
                    output = {}
                    scored = {"overall": None, "error": str(exc)}

                row = {
                    "ts": ts, "skill": "model_selector", "dataset": ds_name,
                    "model": mc["name"], "duration_s": round(dur, 1), **scored,
                }
                rows.append(row)
                all_out[f"{benchmark_id}/{mc['name']}"] = output

                task_type  = output.get("task_type", "?")
                recall_str = f"recall={scored.get('recall', '?')}"
                score_str  = f"overall={scored.get('overall', 'ERR')}"
                pass_str   = "✓" if scored.get("passed") else "✗"
                err_str    = f"  ⚠ {scored['error'][:60]}" if scored.get("error") else ""
                print(f"    {mc['name']:18}  {pass_str}  {score_str:14}  {recall_str:14}  "
                      f"{dur:.0f}s  [{task_type}]{err_str}")
            print()

    # ── Summary table ────────────────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("  SUMMARY (mean across all benchmarks per model)\n")
    for mc in MODELS:
        model_rows  = [r for r in rows if r["model"] == mc["name"] and r.get("overall") is not None]
        if not model_rows:
            continue
        mean_score  = sum(r["overall"] for r in model_rows) / len(model_rows)
        mean_dur    = sum(r["duration_s"] for r in model_rows) / len(model_rows)
        n_pass      = sum(1 for r in model_rows if r.get("passed"))
        recalls     = [r["recall"] for r in model_rows if r.get("recall") is not None]
        mean_recall = sum(recalls) / len(recalls) if recalls else None
        recall_str  = f"{mean_recall:.0%}" if mean_recall is not None else "n/a"
        print(f"  {mc['name']:18}  score={mean_score:.1f}/100  "
              f"recall={recall_str}  pass={n_pass}/{len(model_rows)}  "
              f"avg_time={mean_dur:.0f}s")
    print(f"{'─'*72}\n")

    # ── Persist results ───────────────────────────────────────────────────────
    existing = json.loads(RESULTS_FILE.read_text()) if RESULTS_FILE.exists() else {"runs": []}
    existing["runs"].append({"ts": ts, "rows": rows})
    RESULTS_FILE.write_text(json.dumps(existing, indent=2, default=str))
    print(f"  Results saved → {RESULTS_FILE.name}")

    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model comparison for DQ and Leakage skills")
    parser.add_argument("--skill", choices=["data_quality", "leakage_detector", "model_selector"],
                        help="Run only this skill (default: all)")
    args = parser.parse_args()
    run_comparison(skill_filter=args.skill)

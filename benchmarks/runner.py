"""
Benchmark runner.

Usage:
    # Full run (re-calls Claude API for all benchmarks):
    .venv/bin/python benchmarks/runner.py

    # Score only (uses cached skill outputs, no API calls):
    .venv/bin/python benchmarks/runner.py --score-only

    # Single dataset/benchmark:
    .venv/bin/python benchmarks/runner.py --filter orders_dirty/data_quality

    # List available benchmarks:
    .venv/bin/python benchmarks/runner.py --list
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config  # triggers load_dotenv

BENCHMARKS_DIR = Path(__file__).parent
CACHE_DIR = BENCHMARKS_DIR / "cache"
HISTORY_FILE = BENCHMARKS_DIR / "benchmark_history.json"
PASS_THRESHOLD = 70.0

# ── Benchmark registry ────────────────────────────────────────────────────────
# Each entry: (dataset_name, benchmark_id, skill_fn, skill_kwargs)
# skill_fn is called as skill_fn(df, **skill_kwargs) → dict

def _get_benchmarks() -> list[dict]:
    """Build the full benchmark list from available answer keys."""
    from databricks_client import (
        _gen_orders_dirty, _gen_sales_clean, _gen_hr_attrition,
        _gen_ecommerce, _gen_transactions_fraud, _gen_sales_forecasting,
        _gen_ecommerce_leakage,
    )
    return [
        # ── Data Quality ──────────────────────────────────────────────────────
        {
            "id":        "orders_dirty/data_quality",
            "dataset":   "orders_dirty",
            "benchmark": "data_quality",
            "skill":     "data_quality",
            "df_fn":     _gen_orders_dirty,
            "run_fn":    _run_dq,
            "run_kwargs": {"table_name": "orders_dirty"},
        },
        {
            "id":        "sales_clean/data_quality",
            "dataset":   "sales_clean",
            "benchmark": "data_quality",
            "skill":     "data_quality",
            "df_fn":     _gen_sales_clean,
            "run_fn":    _run_dq,
            "run_kwargs": {"table_name": "sales_clean"},
        },
        {
            "id":        "hr_attrition/data_quality",
            "dataset":   "hr_attrition",
            "benchmark": "data_quality",
            "skill":     "data_quality",
            "df_fn":     _gen_hr_attrition,
            "run_fn":    _run_dq,
            "run_kwargs": {"table_name": "hr_attrition"},
        },
        {
            "id":        "ecommerce/data_quality",
            "dataset":   "ecommerce",
            "benchmark": "data_quality",
            "skill":     "data_quality",
            "df_fn":     _gen_ecommerce,
            "run_fn":    _run_dq,
            "run_kwargs": {"table_name": "ecommerce"},
        },
        {
            "id":        "transactions_fraud/data_quality",
            "dataset":   "transactions_fraud",
            "benchmark": "data_quality",
            "skill":     "data_quality",
            "df_fn":     _gen_transactions_fraud,
            "run_fn":    _run_dq,
            "run_kwargs": {"table_name": "transactions_fraud"},
        },
        {
            "id":        "sales_forecasting/data_quality",
            "dataset":   "sales_forecasting",
            "benchmark": "data_quality",
            "skill":     "data_quality",
            "df_fn":     _gen_sales_forecasting,
            "run_fn":    _run_dq,
            "run_kwargs": {"table_name": "sales_forecasting"},
        },
        # ── Feature Engineering ───────────────────────────────────────────────
        {
            "id":        "ecommerce/feature_engineering_converted",
            "dataset":   "ecommerce",
            "benchmark": "feature_engineering_converted",
            "skill":     "feature_engineering",
            "df_fn":     _gen_ecommerce,
            "run_fn":    _run_fe,
            "run_kwargs": {"target_col": "converted"},
        },
        {
            "id":        "hr_attrition/feature_engineering_attrition",
            "dataset":   "hr_attrition",
            "benchmark": "feature_engineering_attrition",
            "skill":     "feature_engineering",
            "df_fn":     _gen_hr_attrition,
            "run_fn":    _run_fe,
            "run_kwargs": {"target_col": "attrition"},
        },
        # ── Leakage Detector ──────────────────────────────────────────────────
        {
            "id":        "ecommerce_leakage/leakage_detector_converted",
            "dataset":   "ecommerce_leakage",
            "benchmark": "leakage_detector_converted",
            "skill":     "leakage_detector",
            "df_fn":     _gen_ecommerce_leakage,
            "run_fn":    _run_ld,
            "run_kwargs": {"target_col": "converted"},
        },
        {
            "id":        "hr_attrition/leakage_detector_attrition",
            "dataset":   "hr_attrition",
            "benchmark": "leakage_detector_attrition",
            "skill":     "leakage_detector",
            "df_fn":     _gen_hr_attrition,
            "run_fn":    _run_ld,
            "run_kwargs": {"target_col": "attrition"},
        },
        # ── Model Selector ────────────────────────────────────────────────────
        {
            "id":        "hr_attrition/model_selector_attrition",
            "dataset":   "hr_attrition",
            "benchmark": "model_selector_attrition",
            "skill":     "model_selector",
            "df_fn":     _gen_hr_attrition,
            "run_fn":    _run_ms,
            "run_kwargs": {"target_col": "attrition"},
        },
        {
            "id":        "ecommerce/model_selector_converted",
            "dataset":   "ecommerce",
            "benchmark": "model_selector_converted",
            "skill":     "model_selector",
            "df_fn":     _gen_ecommerce,
            "run_fn":    _run_ms,
            "run_kwargs": {"target_col": "converted"},
        },
        {
            "id":        "transactions_fraud/model_selector_fraud",
            "dataset":   "transactions_fraud",
            "benchmark": "model_selector_fraud",
            "skill":     "model_selector",
            "df_fn":     _gen_transactions_fraud,
            "run_fn":    _run_ms,
            "run_kwargs": {"target_col": "is_fraud"},
        },
        {
            "id":        "sales_forecasting/model_selector_revenue",
            "dataset":   "sales_forecasting",
            "benchmark": "model_selector_revenue",
            "skill":     "model_selector",
            "df_fn":     _gen_sales_forecasting,
            "run_fn":    _run_ms,
            "run_kwargs": {"target_col": "revenue"},
        },
    ]


# ── Skill runners ─────────────────────────────────────────────────────────────

def _run_dq(df, table_name: str = "dataset") -> dict:
    from skills.data_quality import assess
    return assess(df, table_name=table_name)


def _run_fe(df, target_col: str, context: str = "") -> dict:
    from skills.feature_engineering import suggest
    return suggest(df, target_col=target_col, context=context)


def _run_ld(df, target_col: str, context: str = "") -> dict:
    from skills.leakage_detector import check
    return check(df, target_col=target_col, context=context)


def _run_ms(df, target_col: str, problem_context: str = "") -> dict:
    from skills.model_selector import select
    return select(df, target_col=target_col, problem_context=problem_context)


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _cache_path(benchmark_id: str, ts: str) -> Path:
    safe_id = benchmark_id.replace("/", "_")
    return CACHE_DIR / f"{ts}_{safe_id}.json"


def _latest_cache(benchmark_id: str) -> Path | None:
    safe_id = benchmark_id.replace("/", "_")
    matches = sorted(CACHE_DIR.glob(f"*_{safe_id}.json"), reverse=True)
    return matches[0] if matches else None


def _write_cache(benchmark_id: str, ts: str, output: dict, duration_s: float) -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    path = _cache_path(benchmark_id, ts)
    path.write_text(json.dumps({
        "benchmark_id": benchmark_id,
        "timestamp": ts,
        "duration_s": round(duration_s, 2),
        "output": output,
    }, indent=2, default=str))
    return path


def _read_cache(path: Path) -> tuple[dict, float]:
    data = json.loads(path.read_text())
    return data["output"], data.get("duration_s", 0.0)


# ── History helpers ───────────────────────────────────────────────────────────

def _load_history() -> dict:
    if HISTORY_FILE.exists():
        return json.loads(HISTORY_FILE.read_text())
    return {"runs": []}


def _append_history(run_record: dict) -> None:
    history = _load_history()
    history["runs"].append(run_record)
    HISTORY_FILE.write_text(json.dumps(history, indent=2, default=str))


# ── Main runner ───────────────────────────────────────────────────────────────

def run_benchmarks(
    score_only: bool = False,
    filter_id: str | None = None,
) -> dict:
    from benchmarks.scoring import score as score_fn, ScoreResult

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    benchmarks = _get_benchmarks()
    if filter_id:
        benchmarks = [b for b in benchmarks if filter_id in b["id"]]
        if not benchmarks:
            print(f"No benchmarks match filter '{filter_id}'")
            return {}

    results: dict[str, dict] = {}
    all_scores: list[float] = []

    print(f"\n{'='*62}")
    print(f"  DS Workbench Benchmark Suite  —  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  mode: {'score-only (using cached outputs)' if score_only else 'full run (calling Claude API)'}")
    print(f"{'='*62}\n")

    for bm in benchmarks:
        bid = bm["id"]
        print(f"  ▶ {bid}")

        # ── Obtain skill output ───────────────────────────────────────────────
        output: dict | None = None
        duration_s = 0.0

        if score_only:
            cached = _latest_cache(bid)
            if cached:
                output, duration_s = _read_cache(cached)
                print(f"    using cache: {cached.name}")
            else:
                print(f"    ⚠ no cache found — skipping (run without --score-only first)")
                continue
        else:
            try:
                df = bm["df_fn"]()
                t0 = time.perf_counter()
                output = bm["run_fn"](df, **bm["run_kwargs"])
                duration_s = time.perf_counter() - t0
                _write_cache(bid, ts, output, duration_s)
                print(f"    Claude call completed in {duration_s:.1f}s")
            except Exception as exc:
                print(f"    ✗ SKILL ERROR: {exc}")
                results[bid] = {"error": str(exc), "overall": 0.0, "passed": False}
                continue

        # ── Score the output ──────────────────────────────────────────────────
        try:
            result: ScoreResult = score_fn(output, bm["dataset"], bm["benchmark"])
        except Exception as exc:
            print(f"    ✗ SCORING ERROR: {exc}")
            results[bid] = {"error": str(exc), "overall": 0.0, "passed": False}
            continue

        all_scores.append(result.overall)
        status = "✓ PASS" if result.passed else "✗ FAIL"
        print(f"    {result.summary_line()}")
        if result.notes:
            for note in result.notes:
                print(f"      → {note}")

        results[bid] = {
            "overall":            result.overall,
            "passed":             result.passed,
            "components":         result.components,
            "matched_findings":   result.matched_findings,
            "missed_findings":    result.missed_findings,
            "false_positives":    len(result.false_positives),
            "severity_violations": len(result.severity_violations),
            "notes":              result.notes,
            "duration_s":         round(duration_s, 2),
        }

    # ── Summary ───────────────────────────────────────────────────────────────
    n_total  = len(results)
    n_pass   = sum(1 for r in results.values() if r.get("passed"))
    n_fail   = n_total - n_pass
    mean_score = sum(all_scores) / len(all_scores) if all_scores else 0.0

    print(f"\n{'─'*62}")
    print(f"  Passed: {n_pass}/{n_total}   Mean score: {mean_score:.1f}/100")
    if n_fail:
        failing = [bid for bid, r in results.items() if not r.get("passed")]
        print(f"  Failing: {', '.join(failing)}")
    print(f"{'─'*62}\n")

    # ── Append to history ─────────────────────────────────────────────────────
    run_record = {
        "timestamp":   ts,
        "score_only":  score_only,
        "results":     results,
        "summary": {
            "n_total":    n_total,
            "n_pass":     n_pass,
            "n_fail":     n_fail,
            "mean_score": round(mean_score, 1),
        },
    }
    _append_history(run_record)
    print(f"  Results appended to {HISTORY_FILE.relative_to(Path.cwd()) if HISTORY_FILE.is_relative_to(Path.cwd()) else HISTORY_FILE}")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cmd_list():
    for bm in _get_benchmarks():
        ak_path = BENCHMARKS_DIR / "answer_keys" / bm["dataset"] / f"{bm['benchmark']}.json"
        exists = "✓" if ak_path.exists() else "✗ (no answer key)"
        cached = _latest_cache(bm["id"])
        cache_str = f"cached {cached.name[:15]}…" if cached else "no cache"
        print(f"  {bm['id']:55s} {exists}  [{cache_str}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DS Workbench benchmark runner")
    parser.add_argument("--score-only", action="store_true",
                        help="Score cached outputs only — no Claude API calls")
    parser.add_argument("--filter", metavar="ID",
                        help="Only run benchmarks whose ID contains this substring")
    parser.add_argument("--list", action="store_true",
                        help="List available benchmarks and exit")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable benchmarks:")
        _cmd_list()
        sys.exit(0)

    run_benchmarks(score_only=args.score_only, filter_id=args.filter)

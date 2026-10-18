"""
Display benchmark history.

Usage:
    .venv/bin/python benchmarks/report.py            # latest run
    .venv/bin/python benchmarks/report.py --all      # all runs (trend view)
    .venv/bin/python benchmarks/report.py --trend orders_dirty/data_quality
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

HISTORY_FILE = Path(__file__).parent / "benchmark_history.json"
PASS_THRESHOLD = 70.0


def _load() -> list[dict]:
    if not HISTORY_FILE.exists():
        print("No benchmark history found. Run: .venv/bin/python benchmarks/runner.py")
        sys.exit(0)
    return json.loads(HISTORY_FILE.read_text()).get("runs", [])


def _score_bar(score: float, width: int = 20) -> str:
    filled = round(score / 100 * width)
    bar = "█" * filled + "░" * (width - filled)
    color = "PASS" if score >= PASS_THRESHOLD else "FAIL"
    return f"[{bar}] {score:5.1f}  {color}"


def show_latest():
    runs = _load()
    run = runs[-1]
    ts = run["timestamp"]
    s = run["summary"]

    print(f"\n{'='*65}")
    print(f"  Benchmark run: {ts[:4]}-{ts[4:6]}-{ts[6:8]} {ts[9:11]}:{ts[11:13]} UTC")
    print(f"  {s['n_pass']}/{s['n_total']} passed   mean score: {s['mean_score']:.1f}/100")
    print(f"{'='*65}\n")

    # Group by skill
    dq_rows = []
    fe_rows = []
    ld_rows = []
    ms_rows = []
    for bid, r in sorted(run["results"].items()):
        if "data_quality" in bid:
            dq_rows.append((bid, r))
        elif "feature_engineering" in bid:
            fe_rows.append((bid, r))
        elif "leakage_detector" in bid:
            ld_rows.append((bid, r))
        else:
            ms_rows.append((bid, r))

    for group_name, rows in [
        ("Data Quality", dq_rows),
        ("Feature Engineering", fe_rows),
        ("Leakage Detector", ld_rows),
        ("Model Selector", ms_rows),
    ]:
        if not rows:
            continue
        print(f"  ── {group_name} ────────────────────────────────────────────")
        for bid, r in rows:
            if "error" in r:
                print(f"  {bid:45s}  ERROR: {r['error'][:40]}")
                continue
            score = r.get("overall", 0.0)
            bar = _score_bar(score)
            recall = r.get("components", {}).get("recall", 0)
            fp = r.get("false_positives", 0)
            missed = r.get("missed_findings", [])
            label = bid.split("/")[0]
            extra = f"  recall={recall:.0%}  fp={fp}"
            if missed:
                extra += f"  missed={missed}"
            print(f"  {label:30s} {bar}{extra}")
        print()


def show_all():
    runs = _load()
    if not runs:
        print("No runs recorded.")
        return

    # Collect all benchmark IDs
    all_ids = sorted({bid for run in runs for bid in run["results"]})
    print(f"\nBenchmark trend  ({len(runs)} runs)\n")
    print(f"  {'Benchmark':<45}", end="")
    for run in runs[-5:]:
        ts = run["timestamp"]
        print(f"  {ts[6:8]}/{ts[4:6]} {ts[9:11]}:{ts[11:13]}", end="")
    print()
    print("  " + "─" * 45 + ("─" * 15) * min(len(runs), 5))

    for bid in all_ids:
        print(f"  {bid:<45}", end="")
        for run in runs[-5:]:
            r = run["results"].get(bid, {})
            score = r.get("overall", None)
            if score is None:
                print(f"  {'—':>13}", end="")
            else:
                mark = "✓" if score >= PASS_THRESHOLD else "✗"
                print(f"  {mark} {score:5.1f}      ", end="")
        print()


def show_trend(benchmark_id: str):
    runs = _load()
    rows = [(r["timestamp"], r["results"].get(benchmark_id)) for r in runs
            if benchmark_id in r["results"]]
    if not rows:
        print(f"No history for '{benchmark_id}'")
        return

    print(f"\nTrend: {benchmark_id}\n")
    for ts, r in rows:
        if r is None:
            continue
        score = r.get("overall", 0.0)
        bar = _score_bar(score, width=30)
        date = f"{ts[6:8]}/{ts[4:6]}/{ts[:4]} {ts[9:11]}:{ts[11:13]}"
        print(f"  {date}  {bar}")
        if r.get("missed_findings"):
            print(f"              missed: {r['missed_findings']}")
        if r.get("false_positives", 0) > 0:
            print(f"              false positives: {r['false_positives']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DS Workbench benchmark reporter")
    parser.add_argument("--all", action="store_true", help="Show all runs in a trend table")
    parser.add_argument("--trend", metavar="ID", help="Show history for a specific benchmark ID")
    args = parser.parse_args()

    if args.all:
        show_all()
    elif args.trend:
        show_trend(args.trend)
    else:
        show_latest()

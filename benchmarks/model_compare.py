"""
Model / effort sweep over the benchmark suite.

Runs the real skills (same prompts as production) with one role swapped to each arm via
skills.models.override_roles, scores every output against its answer key, and reports
mean score, spread across trials, cost and time per arm. Use it to decide model and
effort settings from evidence instead of list prices.

Usage:
    # Leakage on its configured model vs Opus 5.5 vs Sonnet 5.5, 3 trials each
    .venv/bin/python benchmarks/model_compare.py --filter leakage \\
        --arms claude-fable-5-1:medium claude-opus-5-5:medium claude-sonnet-5-5:high --trials 3

    # Effort sweep for DQ on the configured reasoning model
    .venv/bin/python benchmarks/model_compare.py --filter data_quality --efforts low medium high

Arms default to the configured model for each benchmark's role. Results append to
benchmarks/model_comparison_results.json.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import call_cost  # noqa: E402  (importing config triggers load_dotenv)

from anthropic.resources.messages import Messages  # noqa: E402

from skills.models import SKILL_ROLE, override_roles, role  # noqa: E402

RESULTS_FILE = Path(__file__).parent / "model_comparison_results.json"

_spend = defaultdict(float)
_current = {"arm": None}
_orig_create = Messages.create


def _metered_create(self, **kw):
    resp = _orig_create(self, **kw)
    _spend[_current["arm"]] += call_cost(kw["model"], resp.usage.input_tokens, resp.usage.output_tokens)
    return resp


def _parse_arm(text: str) -> tuple[str, str | None]:
    model, _, effort = text.partition(":")
    return model, effort or None


def run_sweep(filter_id: str | None, arms: list[tuple[str, str | None]] | None,
              efforts: list[str] | None, trials: int) -> list[dict]:
    from benchmarks.runner import _get_benchmarks
    from benchmarks.scoring import score

    Messages.create = _metered_create
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    benches = [b for b in _get_benchmarks() if not filter_id or filter_id in b["id"]]
    if not benches:
        sys.exit(f"No benchmark matches {filter_id!r}")
    rows = []

    for bm in benches:
        role_name = SKILL_ROLE[bm["skill"]]
        default = role(role_name)
        default_arm = (default["model"], default.get("output_config", {}).get("effort"))
        bm_arms = arms or ([(default_arm[0], e) for e in efforts] if efforts else [default_arm])
        df = bm["df_fn"]()
        print(f"\n{bm['id']}  (role: {role_name})")
        for model, effort in bm_arms:
            name = f"{model.removeprefix('claude-')}:{effort or 'default'}"
            _current["arm"] = (bm["id"], name)
            scores, secs = [], []
            for _ in range(trials):
                t0 = time.perf_counter()
                with override_roles(**{role_name: (model, effort)}):
                    out = bm["run_fn"](df, **bm["run_kwargs"])
                secs.append(time.perf_counter() - t0)
                if out.get("error"):
                    print(f"  {name:24s} error: {out['error'][:100]}")
                    continue
                scores.append(score(dict(out), bm["dataset"], bm["benchmark"]).overall)
            cost = _spend[(bm["id"], name)]
            row = {
                "ts": ts, "benchmark": bm["id"], "role": role_name, "arm": name,
                "trials": trials, "scores": scores,
                "mean": round(statistics.mean(scores), 1) if scores else None,
                "stdev": round(statistics.stdev(scores), 1) if len(scores) > 1 else 0.0,
                "cost_per_run": round(cost / trials, 4),
                "secs_per_run": round(sum(secs) / len(secs), 1),
            }
            rows.append(row)
            print(f"  {name:24s} mean={row['mean']}  sd={row['stdev']}  "
                  f"${row['cost_per_run']:.3f}/run  {row['secs_per_run']:.0f}s/run  {scores}")

    Messages.create = _orig_create
    existing = json.loads(RESULTS_FILE.read_text()) if RESULTS_FILE.exists() else {"runs": []}
    existing["runs"].append({"ts": ts, "kind": "sweep", "rows": rows})
    RESULTS_FILE.write_text(json.dumps(existing, indent=2, default=str))
    print(f"\nResults appended to {RESULTS_FILE.name}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--filter", help="Substring of benchmark ids to run, e.g. 'leakage' or 'transactions_fraud'")
    ap.add_argument("--arms", nargs="+", metavar="MODEL[:EFFORT]",
                    help="Model/effort arms for the benchmark's role (default: configured model)")
    ap.add_argument("--efforts", nargs="+", help="Effort levels to sweep on the configured model")
    ap.add_argument("--trials", type=int, default=1, help="Runs per arm (use ≥3 to see spread)")
    args = ap.parse_args()
    run_sweep(args.filter, [_parse_arm(a) for a in args.arms] if args.arms else None,
              args.efforts, args.trials)

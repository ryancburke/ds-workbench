"""
CLI runner: run one or more simulations and print a summary.

Usage:
  .venv/bin/python run_sim.py                    # coverage-weighted scenario
  .venv/bin/python run_sim.py churn_risk          # specific scenario
  .venv/bin/python run_sim.py q3_revenue churn_risk  # multiple scenarios
"""
import sys
import time
from agents.scheduler import _run_simulation, get_history, get_recommendations
from agents.simulation import SCENARIOS


def _pick_scenario() -> str:
    """Pick the scenario with the fewest recent benchmark runs (coverage-weighted)."""
    from agents.scheduler import get_benchmarks
    counts = {}
    for s in SCENARIOS:
        focus = s.get("focus_skills", [])
        total = sum(len(get_benchmarks(skill=sk)) for sk in focus)
        counts[s["id"]] = total
    return min(counts, key=counts.get)


def run(scenario_id: str):
    scenario = next((s for s in SCENARIOS if s["id"] == scenario_id), None)
    if not scenario:
        print(f"Unknown scenario: {scenario_id}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Scenario: {scenario['title']}")
    print(f"  ID:       {scenario_id}")
    print(f"{'='*60}")
    t0 = time.time()

    _run_simulation(scenario_id, triggered_by="cli")

    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.0f}s")

    history = get_history()
    last = next((r for r in history if r["scenario_id"] == scenario_id), None)
    if last and last.get("error"):
        print(f"  ERROR: {last['error']}")
        return

    recs_added = last.get("recommendations_added", 0) if last else 0
    print(f"  Recommendations added/upvoted: {recs_added}")


if __name__ == "__main__":
    ids = sys.argv[1:] if sys.argv[1:] else [_pick_scenario()]
    for sid in ids:
        run(sid)

    pending = get_recommendations(status="pending")
    print(f"\nTotal pending recommendations: {len(pending)}")
    if pending:
        print("\nTop 5 by priority×votes:")
        for r in pending[:5]:
            print(f"  [{r.get('priority','?'):8}] {r.get('votes',1):2}v  {r['title']}")

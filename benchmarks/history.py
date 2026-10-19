"""Read benchmark_history.json into the shape the dashboard needs (no Dash dependency)."""
from __future__ import annotations

import json
from pathlib import Path

HISTORY_FILE = Path(__file__).parent / "benchmark_history.json"


def load_runs(path: Path = HISTORY_FILE) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text()).get("runs", [])


def suite_summary(runs: list[dict], trend_points: int = 20) -> dict | None:
    """Latest score per benchmark, its change since the previous run that scored it,
    and the mean score of recent runs. None when there is no history."""
    runs = [r for r in runs if r.get("results")]
    if not runs:
        return None
    latest = runs[-1]

    def scored(res: dict) -> float | None:
        # the runner records a crashed case as overall=0.0 plus an error; that is not a score
        return None if res.get("error") else res.get("overall")

    rows = []
    for bid, res in sorted(latest["results"].items()):
        score = scored(res)
        prev = next((scored(r["results"][bid]) for r in reversed(runs[:-1])
                     if scored(r["results"].get(bid, {})) is not None), None)
        rows.append({
            "id": bid,
            "skill": res.get("skill") or _skill_of(bid),
            "score": score,
            "previous": prev,
            "delta": round(score - prev, 1) if score is not None and prev is not None else None,
            "passed": res.get("passed"),
            "missed": res.get("missed_findings") or [],
            "notes": res.get("notes") or [],
            "error": res.get("error"),
        })

    trend = [{"timestamp": r["timestamp"], "mean": r.get("summary", {}).get("mean_score"),
              "n_pass": r.get("summary", {}).get("n_pass"), "n_total": r.get("summary", {}).get("n_total"),
              "score_only": r.get("score_only", False)}
             for r in runs[-trend_points:]]
    return {"timestamp": latest["timestamp"], "summary": latest.get("summary", {}),
            "models": latest.get("models", {}), "score_only": latest.get("score_only", False),
            "rows": rows, "trend": trend}


def _skill_of(benchmark_id: str) -> str:
    name = benchmark_id.split("/", 1)[-1]
    for skill in ("data_quality", "feature_engineering", "leakage_detector",
                  "model_selector", "experiment_tracker"):
        if name.startswith(skill):
            return skill
    return name

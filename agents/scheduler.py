"""
Background simulation scheduler and persistent recommendation store.

The scheduler runs full simulations on a configurable interval,
collects usability reflections from each persona, and surfaces
improvement suggestions in the feedback dashboard.
"""

import json
import os
import py_compile
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Recommendation store ───────────────────────────────────────────────────────

FEEDBACK_DIR     = Path(__file__).parent.parent / "feedback"
RECS_FILE        = FEEDBACK_DIR / "recommendations.json"
HISTORY_FILE     = FEEDBACK_DIR / "sim_history.json"
BENCHMARKS_FILE  = FEEDBACK_DIR / "benchmarks.json"

PRIORITY_WEIGHT = {"critical": 4, "high": 3, "medium": 2, "low": 1}


def _init_files():
    FEEDBACK_DIR.mkdir(exist_ok=True)
    if not RECS_FILE.exists():
        RECS_FILE.write_text(json.dumps({"recommendations": []}, indent=2))
    if not HISTORY_FILE.exists():
        HISTORY_FILE.write_text(json.dumps({"runs": []}, indent=2))
    if not BENCHMARKS_FILE.exists():
        BENCHMARKS_FILE.write_text(json.dumps({"benchmarks": []}, indent=2))


def _read_recs() -> dict:
    _init_files()
    try:
        return json.loads(RECS_FILE.read_text())
    except Exception:
        return {"recommendations": []}


def _write_recs(data: dict):
    _init_files()
    tmp = RECS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(RECS_FILE)  # atomic rename — prevents partial reads


def _read_history() -> dict:
    _init_files()
    try:
        return json.loads(HISTORY_FILE.read_text())
    except Exception:
        return {"runs": []}


def _write_history(data: dict):
    _init_files()
    tmp = HISTORY_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(HISTORY_FILE)


def _similar_title(a: str, b: str) -> bool:
    """True if titles share > 35% of meaningful words (Jaccard on the smaller set)."""
    stop = {
        "a", "an", "the", "to", "add", "in", "for", "and", "with", "of", "on",
        "be", "is", "are", "it", "this", "that", "when", "more", "not", "no",
        "show", "surface", "display", "output", "skill", "finding", "findings",
        "should", "needs", "need", "before", "after", "just", "only", "also",
    }
    # Normalise: strip punctuation, lowercase, split
    import re
    clean = lambda s: set(re.sub(r"[^a-z0-9_ ]", " ", s.lower()).split()) - stop
    w1, w2 = clean(a), clean(b)
    if not w1 or not w2:
        return False
    return len(w1 & w2) / min(len(w1), len(w2)) > 0.35


def add_recommendation(rec: dict) -> str:
    """
    Add a recommendation or upvote an existing similar one.
    Returns the recommendation ID.
    """
    data = _read_recs()
    recs = data["recommendations"]

    # Match on title similarity alone (skill_affected removed from gate — same concept
    # can arrive with different or empty skill labels across personas/scenarios).
    existing = next(
        (r for r in recs
         if r.get("status", "pending") == "pending"
         and _similar_title(r.get("title", ""), rec.get("title", ""))),
        None,
    )

    if existing:
        existing["votes"] = existing.get("votes", 1) + 1
        existing["last_seen"] = _now()
        # Append context from new sighting
        seen_by = existing.get("seen_by", [])
        seen_by.append(f"{rec.get('persona_name','?')} · {rec.get('scenario_title','?')}")
        existing["seen_by"] = seen_by[-5:]  # keep last 5
        _write_recs(data)
        return existing["id"]

    rec_id = f"rec_{int(time.time() * 1000)}"
    full_rec = {
        "id":           rec_id,
        "votes":        1,
        "status":       "pending",
        "created_at":   _now(),
        "last_seen":    _now(),
        "seen_by":      [f"{rec.get('persona_name','?')} · {rec.get('scenario_title','?')}"],
        "drafted_code": None,
        "drafted_file": None,
        **rec,
    }
    recs.append(full_rec)
    _write_recs(data)
    return rec_id


def get_recommendations(status: str | None = None) -> list[dict]:
    data = _read_recs()
    recs = data.get("recommendations", [])
    if status:
        recs = [r for r in recs if r.get("status") == status]
    return sorted(
        recs,
        key=lambda r: PRIORITY_WEIGHT.get(r.get("priority", "low"), 1) * r.get("votes", 1),
        reverse=True,
    )


def update_rec_status(rec_id: str, status: str):
    data = _read_recs()
    for r in data["recommendations"]:
        if r["id"] == rec_id:
            r["status"] = status
            r["status_updated_at"] = _now()
            break
    _write_recs(data)


def save_draft(rec_id: str, file_path: str, code: str):
    data = _read_recs()
    for r in data["recommendations"]:
        if r["id"] == rec_id:
            r["drafted_file"] = file_path
            r["drafted_code"] = code
            r["status"] = "drafted"
            r["drafted_at"] = _now()
            break
    _write_recs(data)


def get_rec(rec_id: str) -> dict | None:
    data = _read_recs()
    return next((r for r in data["recommendations"] if r["id"] == rec_id), None)


def pending_count() -> int:
    return len(get_recommendations(status="pending"))


def get_history() -> list[dict]:
    return _read_history().get("runs", [])


# ── Benchmark store ────────────────────────────────────────────────────────────

def _read_benchmarks() -> dict:
    _init_files()
    try:
        return json.loads(BENCHMARKS_FILE.read_text())
    except Exception:
        return {"benchmarks": []}


def _write_benchmarks(data: dict):
    _init_files()
    BENCHMARKS_FILE.write_text(json.dumps(data, indent=2))


def record_benchmark(bench: dict):
    """Append a benchmark record. Keeps last 2000 entries."""
    data = _read_benchmarks()
    data["benchmarks"] = [bench] + data.get("benchmarks", [])[:1999]
    _write_benchmarks(data)


def get_benchmarks(skill: str | None = None) -> list[dict]:
    data = _read_benchmarks()
    benches = data.get("benchmarks", [])
    if skill:
        benches = [b for b in benches if b.get("skill") == skill]
    return benches


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Run state (module-level, visible to all Dash callbacks) ───────────────────

_state_lock = threading.Lock()
_running_scenarios: set[str] = set()   # scenario_ids currently in flight
_last_run_meta: dict | None = None


def is_running(scenario_id: str | None = None) -> bool:
    """True if any simulation is running, or if the given scenario is running."""
    with _state_lock:
        if scenario_id:
            return scenario_id in _running_scenarios
        return bool(_running_scenarios)


def running_scenarios() -> set[str]:
    """Return a copy of the set of currently-running scenario IDs."""
    with _state_lock:
        return set(_running_scenarios)


def last_run_meta() -> dict | None:
    return _last_run_meta


# ── Simulation runner ──────────────────────────────────────────────────────────

def _run_simulation(scenario_id: str, triggered_by: str = "scheduler"):
    global _last_run_meta

    with _state_lock:
        if scenario_id in _running_scenarios:
            return  # already running this exact scenario — skip
        _running_scenarios.add(scenario_id)

    started_at = _now()
    run_id = f"sim_{int(time.time())}"

    try:
        from agents.simulation import run_all_personas, SCENARIOS
        _, recommendations, benchmarks = run_all_personas(scenario_id)
        for bench in benchmarks:
            record_benchmark(bench)

        scenario = next(s for s in SCENARIOS if s["id"] == scenario_id)
        rec_ids = [add_recommendation(r) for r in recommendations]

        history = _read_history()
        run_record = {
            "id":                     run_id,
            "scenario_id":            scenario_id,
            "scenario_title":         scenario["title"],
            "started_at":             started_at,
            "completed_at":           _now(),
            "triggered_by":           triggered_by,
            "recommendations_added":  len(rec_ids),
        }
        history["runs"] = [run_record] + history.get("runs", [])[:49]
        _write_history(history)

        _last_run_meta = run_record

    except Exception as e:
        history = _read_history()
        try:
            from agents.simulation import SCENARIOS
            _scenario = next((s for s in SCENARIOS if s["id"] == scenario_id), None)
            _title = _scenario["title"] if _scenario else scenario_id
        except Exception:
            _title = scenario_id
        history["runs"] = [{
            "id": run_id, "scenario_id": scenario_id, "scenario_title": _title,
            "started_at": started_at, "completed_at": _now(),
            "triggered_by": triggered_by, "error": str(e),
            "recommendations_added": 0,
        }] + history.get("runs", [])[:49]
        _write_history(history)

    finally:
        with _state_lock:
            _running_scenarios.discard(scenario_id)


def trigger_manual_run(scenario_id: str):
    """Start one simulation in a background thread. Non-blocking."""
    t = threading.Thread(
        target=_run_simulation,
        args=(scenario_id, "manual"),
        daemon=True,
    )
    t.start()


def trigger_bulk_run(scenario_ids: list[str], max_concurrent: int = 3):
    """
    Start multiple simulations. Up to max_concurrent run truly in parallel;
    the rest queue behind a semaphore. Non-blocking for the caller.
    Skips any scenario_id that is already running.
    """
    sem = threading.Semaphore(max_concurrent)

    def _run_with_sem(sid: str):
        with sem:
            _run_simulation(sid, triggered_by="manual_bulk")

    for sid in scenario_ids:
        with _state_lock:
            already = sid in _running_scenarios
        if not already:
            threading.Thread(target=_run_with_sem, args=(sid,), daemon=True).start()


# ── APScheduler setup ─────────────────────────────────────────────────────────

_scenario_rotation = ["q3_revenue", "churn_risk", "model_readiness", "pipeline_handoff", "fraud_detection"]
_rotation_idx = 0


def _coverage_weighted_scenario() -> str:
    """
    Pick the scenario whose focus_skills have the fewest benchmark runs.
    Ties broken by fewest simulation runs (so identical-focus scenarios like
    model_readiness and fraud_detection alternate rather than one dominating).
    Falls back to round-robin if benchmark data is unavailable.
    """
    global _rotation_idx
    try:
        from agents.simulation import SCENARIOS
        bench_data = _read_benchmarks()
        benches = bench_data.get("benchmarks", [])
        skill_counts: dict[str, int] = {}
        for b in benches:
            sk = b.get("skill", "")
            if sk:
                skill_counts[sk] = skill_counts.get(sk, 0) + 1

        hist = _read_history()
        sim_counts: dict[str, int] = {}
        for r in hist.get("runs", []):
            sid = r.get("scenario_id", "")
            if sid:
                sim_counts[sid] = sim_counts.get(sid, 0) + 1

        best_id = None
        best_min = float("inf")
        best_sim = float("inf")
        for s in SCENARIOS:
            focus = s.get("focus_skills", [])
            if not focus:
                continue
            min_count = min(skill_counts.get(sk, 0) for sk in focus)
            sim_count = sim_counts.get(s["id"], 0)
            if min_count < best_min or (min_count == best_min and sim_count < best_sim):
                best_min = min_count
                best_sim = sim_count
                best_id = s["id"]

        return best_id or _scenario_rotation[_rotation_idx % len(_scenario_rotation)]
    except Exception:
        return _scenario_rotation[_rotation_idx % len(_scenario_rotation)]


def _scheduled_job():
    global _rotation_idx
    scenario_id = _coverage_weighted_scenario()
    _rotation_idx += 1
    print(f"[scheduler] Firing — scenario: {scenario_id}", flush=True)
    _run_simulation(scenario_id, triggered_by="scheduler")


def start_scheduler():
    """
    Start the background APScheduler. Safe to call from app startup.

    Werkzeug debug reloader runs app.py in two processes: the monitor (parent,
    WERKZEUG_RUN_MAIN absent) and the serving child (WERKZEUG_RUN_MAIN=true).
    A production WSGI server also has WERKZEUG_RUN_MAIN absent — so there is no
    env-var-only way to tell the monitor apart from production.

    Strategy: skip the parent by requiring WERKZEUG_RUN_MAIN=true when LOCAL_DEV_MODE
    is active (guaranteed dev/debug context). In production LOCAL_DEV_MODE is false so
    we always start regardless of WERKZEUG_RUN_MAIN. This avoids duplicate schedulers
    in dev without blocking production.
    """
    from config import config

    if config.local_dev_mode and os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        # Dev mode: skip the Werkzeug monitor (parent) process.
        # The child, which actually serves requests, has WERKZEUG_RUN_MAIN=true.
        return None

    interval_hours = config.sim_interval_hours
    if interval_hours <= 0:
        print("[scheduler] Manual mode (SIM_INTERVAL_HOURS=0) — run simulations from the "
              "Feedback page or run_sim.py", flush=True)
        return None

    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from datetime import datetime as _dt

        scheduler = BackgroundScheduler(daemon=True)
        scheduler.add_job(
            _scheduled_job,
            trigger="interval",
            hours=interval_hours,
            id="sim_job",
            replace_existing=True,
            misfire_grace_time=3600,
            next_run_time=_dt.now(),  # fire once immediately on startup
        )
        scheduler.start()
        print(f"[scheduler] Started — interval {interval_hours}h, first run: now", flush=True)
        return scheduler
    except Exception as exc:
        print(f"[scheduler] Failed to start: {exc}", flush=True)
        return None


# ── Code drafting ──────────────────────────────────────────────────────────────

_SKILL_FILE = {
    "data_quality":        "pages/data_quality.py",
    "eda":                 "pages/eda.py",
    "join_auditor":        "pages/join_auditor.py",
    "nl_to_spark":         "pages/nl_to_spark.py",
    "mlops_review":        "pages/mlops_review.py",
    "model_selector":      "skills/model_selector.py",
    "leakage_detector":    "skills/leakage_detector.py",
    "feature_engineering": "skills/feature_engineering.py",
    "agent_sim":           "pages/agent_sim.py",
    "general":             "pages/home.py",
}


def draft_code_change(rec_id: str) -> tuple[str, str]:
    """
    Ask Claude to implement the recommendation.
    Returns (file_path, new_code). Saves draft to store.
    Raises on failure.
    """
    from skills.api_retry import check_response, get_client, response_text
    from skills.models import CODE_EXTRA, CODE_MODEL

    rec = get_rec(rec_id)
    if not rec:
        raise ValueError(f"Recommendation {rec_id} not found")

    file_path = _SKILL_FILE.get(rec.get("skill_affected", "general"), "pages/home.py")
    base_dir   = Path(__file__).parent.parent
    full_path  = base_dir / file_path

    current_code = full_path.read_text() if full_path.exists() else "# File not found"

    # A whole-file rewrite can be tens of thousands of tokens (pages/home.py is ~3.3k
    # lines), so stream with a large max_tokens rather than truncating at 8k.
    with get_client().messages.stream(
        model=CODE_MODEL,
        max_tokens=64000,
        **CODE_EXTRA,
        system=(
            "You are an expert Plotly Dash developer implementing a UX improvement for a "
            "data science workbench. The app uses Dash 4.4, dash-bootstrap-components with "
            "the DARKLY theme, and Plotly Express/Graph Objects. All callbacks use the "
            "`@callback` decorator. Pattern-matching callbacks use MATCH/ALL. "
            "A global `runs-store` (session storage) holds run history per skill. "
            "Return ONLY the complete updated Python file — no markdown, no explanation."
        ),
        messages=[{
            "role": "user",
            "content": (
                f"Implement this improvement:\n\n"
                f"TITLE: {rec['title']}\n"
                f"DESCRIPTION: {rec['description']}\n"
                f"USER STORY: {rec['user_story']}\n"
                f"IMPLEMENTATION HINT: {rec['implementation_hint']}\n\n"
                f"Current file `{file_path}`:\n```python\n{current_code}\n```\n\n"
                "Return the complete updated file. Preserve all existing functionality."
            ),
        }],
    ) as stream:
        resp = check_response(stream.get_final_message())

    if resp.stop_reason == "max_tokens":
        raise RuntimeError(f"Draft for {file_path} was cut off at max_tokens — not saving a partial file")

    new_code = response_text(resp)
    # Strip accidental markdown fences
    if new_code.startswith("```"):
        new_code = "\n".join(new_code.split("\n")[1:])
    if new_code.endswith("```"):
        new_code = "\n".join(new_code.split("\n")[:-1])

    save_draft(rec_id, file_path, new_code)
    return file_path, new_code


def apply_code_change(rec_id: str) -> str:
    """
    Write drafted code to disk. Creates a .bak backup first.
    Returns the file path that was written.
    Raises on syntax error or missing draft.
    """
    rec = get_rec(rec_id)
    if not rec or not rec.get("drafted_code"):
        raise ValueError("No draft found for this recommendation")

    base_dir  = Path(__file__).parent.parent
    file_path = base_dir / rec["drafted_file"]
    new_code  = rec["drafted_code"]

    # Syntax check before touching the file
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as tmp:
        tmp.write(new_code)
        tmp_path = tmp.name
    try:
        py_compile.compile(tmp_path, doraise=True)
    except py_compile.PyCompileError as e:
        raise ValueError(f"Syntax error in drafted code: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    # Backup original
    if file_path.exists():
        file_path.with_suffix(".py.bak").write_text(file_path.read_text())

    file_path.write_text(new_code)
    update_rec_status(rec_id, "implemented")
    return str(rec["drafted_file"])

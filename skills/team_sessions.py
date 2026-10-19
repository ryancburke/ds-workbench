"""
Team sessions: a persistent record of every analysis session — Workbench or notebook — and a
30-second summary a lead can scan: is it safe to proceed, which required checks ran, what is
blocking, what changed since the last session on the same data, and what it cost.

No Dash dependency. Sessions are stored in feedback/sessions.json (one record per session,
latest result per skill); the summary is recomputed from those results whenever it's read.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from skills.session import build_digest

SESSIONS_FILE = Path(__file__).parent.parent / "feedback" / "sessions.json"
_lock = threading.RLock()

# Checks a modelling session is expected to pass, in workflow order
GATES = ["data_quality", "leakage_detector", "feature_engineering", "model_selector"]
GATE_LABEL = {"data_quality": "Data quality", "leakage_detector": "Leakage",
              "feature_engineering": "Features", "model_selector": "Model", "causal_inference": "Causal"}
_MODELLING = {"feature_engineering", "model_selector"}
_PREREQ_GATES = ("data_quality", "leakage_detector")
_ATTENTION_ORDER = {"red": 0, "amber": 1, "green": 2, "grey": 3}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── Store ─────────────────────────────────────────────────────────────────────

def _read() -> dict:
    try:
        return json.loads(SESSIONS_FILE.read_text())
    except (OSError, ValueError):
        return {"sessions": []}


def save_session(session_id: str, *, owner: str, question: str, datasets: list[str],
                 results: dict, runs: list[dict], source: str, started_at: str | None = None) -> dict:
    """Create or update one session. `results` maps skill → its latest result dict;
    `runs` lists every run in order ({skill, label, ts, status, run_id})."""
    with _lock:
        data = _read()
        sessions = data["sessions"]
        existing = next((s for s in sessions if s["id"] == session_id), None)
        record = {
            "id": session_id, "owner": owner, "question": question, "datasets": list(datasets),
            "source": source, "started_at": (existing or {}).get("started_at") or started_at or _now(),
            "updated_at": _now(),
            "results": {k: _jsonable(v) for k, v in results.items()},
            "runs": runs,
        }
        if existing:
            sessions[sessions.index(existing)] = record
        else:
            sessions.append(record)
        try:
            SESSIONS_FILE.parent.mkdir(exist_ok=True)
            tmp = SESSIONS_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, default=str))
            tmp.replace(SESSIONS_FILE)
        except OSError:
            pass  # read-only install: the session still works locally
        return record


def load_sessions() -> list[dict]:
    return _read()["sessions"]


def _jsonable(result) -> dict:
    """Drop bulky internals (profiles) that the summary doesn't need."""
    out = dict(result) if result else {}
    out.pop("profile", None)
    return json.loads(json.dumps(out, default=str))


# ── Summary ───────────────────────────────────────────────────────────────────

def _gate_status(skill: str, result: dict | None) -> str:
    if not result:
        return "skipped"
    if result.get("error"):
        return "error"
    findings = result.get("findings", [])
    sev = {f.get("severity") for f in findings}
    if (skill == "leakage_detector" and result.get("verdict") == "leakage_detected") or "critical" in sev:
        return "blocked"
    if skill == "model_selector":
        return {"ready": "clear", "insufficient": "blocked"}.get(result.get("data_readiness"), "warn")
    if "warning" in sev:
        return "warn"
    return "clear"


def headline(skill: str, result: dict) -> str:
    """One line per skill: the conclusion, not the mechanics."""
    if result.get("error"):
        return f"failed: {result['error'][:80]}"
    f = result.get("findings", [])
    crit = sum(1 for x in f if x.get("severity") == "critical")
    warn = sum(1 for x in f if x.get("severity") == "warning")
    if skill == "data_quality":
        return f"score {result.get('overall_score', '?')}/100 · {crit} critical, {warn} warnings"
    if skill == "leakage_detector":
        drop = result.get("drop_columns") or []
        return (f"{result.get('verdict', '?').replace('_', ' ')}"
                + (f" · drop {', '.join(drop[:3])}" if drop else ""))
    if skill == "feature_engineering":
        return (f"{len(result.get('features', []))} features · designed for "
                f"{result.get('model_family', '?').replace('_', '-')} models")
    if skill == "model_selector":
        d = result.get("decision") or {}
        model = d.get("model", "?")
        model = model if len(model) <= 40 else model[:39] + "…"
        metric = d.get("primary_metric", "").split(";")[0][:40]
        return f"{model} · {metric} · {result.get('data_readiness', '').replace('_', ' ')}"
    if skill == "causal_inference":
        est = result.get("estimate") or {}
        eff = f" · effect {est['effect']:+.3g}" if est.get("effect") is not None else ""
        return f"{result.get('verdict', '?').replace('_', ' ')}{eff}"
    if skill == "join_auditor":
        return f"{result.get('overall_verdict', '?').replace('_', ' ')} join"
    if skill == "eda":
        return f"{len(result.get('key_insights', []))} insights"
    return (result.get("summary") or "")[:90]


def summarize_session(session: dict) -> dict:
    results = session.get("results", {})
    digest = build_digest(results) if results else {"blockers": [], "cautions": []}
    blockers = list(digest.get("blockers", []))
    cautions = list(digest.get("cautions", []))

    gates = {g: _gate_status(g, results.get(g)) for g in GATES}
    if "causal_inference" in results:
        r = results["causal_inference"]
        gates["causal_inference"] = ("error" if r.get("error") else
                                     {"valid": "clear", "caution": "warn"}.get(r.get("verdict"), "blocked"))
        if r.get("verdict") == "not_identified":
            blockers.append({"source": "causal_inference", "column": "—",
                             "issue": "Effect not identified: no estimate can be trusted yet"})

    # Modelling output without the checks it depends on is the risk a lead most needs to see
    unresolved = [g for g in _PREREQ_GATES if gates[g] == "skipped"] if _MODELLING & results.keys() else []
    for g in unresolved:
        blockers.insert(0, {"source": "workflow", "column": "—",
                            "issue": f"{GATE_LABEL[g]} check skipped before modelling"})

    if not results:
        verdict, attention = "in_progress", "grey"
    elif blockers:
        verdict, attention = "blocked", "red"
    elif cautions or any(s in ("warn", "error") for s in gates.values()):
        verdict, attention = "caution", "amber"
    else:
        verdict, attention = "ready", "green"

    order = [r["skill"] for r in session.get("runs", [])]
    seen = list(dict.fromkeys(order))
    return {
        "verdict": verdict, "attention": attention, "gates": gates,
        "unresolved_gates": unresolved,
        "blockers": blockers, "n_blockers": len(blockers), "n_cautions": len(cautions),
        "headlines": [{"skill": s, "line": headline(s, results[s])} for s in seen if s in results],
    }


def _cost_by_run() -> dict[str, float]:
    try:
        from skills.usage import load
        df = load()
        if df.empty:
            return {}
        return df.dropna(subset=["run_id"]).groupby("run_id")["cost_usd"].sum().to_dict()
    except Exception:
        return {}


def team_overview(sessions: list[dict] | None = None) -> list[dict]:
    """Every session with its summary, the change in blockers since the previous session on the
    same dataset, and its cost; most urgent first."""
    sessions = load_sessions() if sessions is None else sessions
    costs = _cost_by_run()
    rows = []
    for s in sessions:
        summary = summarize_session(s)
        run_ids = [r.get("run_id") for r in s.get("runs", []) if r.get("run_id")]
        rows.append({**{k: s.get(k) for k in ("id", "owner", "question", "datasets", "source",
                                              "started_at", "updated_at")},
                     **summary,
                     "n_runs": len(s.get("runs", [])),
                     "cost_usd": round(sum(costs.get(r, 0.0) for r in run_ids), 4) if costs else None})
    by_time = sorted(rows, key=lambda r: r["updated_at"] or "")
    for i, row in enumerate(by_time):
        prev = next((p for p in reversed(by_time[:i])
                     if set(p["datasets"] or []) & set(row["datasets"] or [])), None)
        row["blocker_delta"] = row["n_blockers"] - prev["n_blockers"] if prev else None
        row["previous_session"] = prev["id"] if prev else None
    return sorted(rows, key=lambda r: (_ATTENTION_ORDER[r["attention"]], -(_ts(r["updated_at"]))))


def _ts(iso: str | None) -> float:
    try:
        return datetime.fromisoformat(iso).timestamp()
    except (TypeError, ValueError):
        return 0.0

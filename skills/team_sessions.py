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

from ds_workbench.core.readiness import STATUSES, assess
from ds_workbench.core.readiness import CHECK_LABEL as GATE_LABEL  # noqa: F401  (used by pages/sessions)

SESSIONS_FILE = Path(__file__).parent.parent / "feedback" / "sessions.json"
_lock = threading.RLock()

# Checks a modelling session is expected to pass, in workflow order
GATES = ["data_quality", "leakage_detector", "feature_engineering", "model_selector"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── Store ─────────────────────────────────────────────────────────────────────

def _read() -> dict:
    try:
        return json.loads(SESSIONS_FILE.read_text())
    except (OSError, ValueError):
        return {"sessions": []}


def save_session(session_id: str, *, owner: str, question: str, datasets: list[str],
                 results: dict, runs: list[dict], source: str, started_at: str | None = None,
                 prediction_point: str = "", confirmations: list[dict] | None = None,
                 required_checks: list[str] | None = None, spec: dict | None = None,
                 scout: dict | None = None, selection: dict | None = None,
                 analysis_table: dict | None = None) -> dict:
    """Create or update one session. `results` maps skill → its latest result dict;
    `runs` lists every run in order ({skill, label, ts, status, run_id}); `confirmations` are
    people's availability confirmations, kept apart from the skill results."""
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
            "prediction_point": prediction_point,
            "confirmations": list(confirmations or []),
            "required_checks": list(required_checks) if required_checks is not None else None,
            "spec": _jsonable(spec) if spec else None,
            "scout": _scout_summary(scout),
            "scout_report": _jsonable(scout) if scout else None,     # metadata only; the plan needs its roles
            "selection": selection,
            "analysis_table": analysis_table,
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
    if skill == "data_verification":
        crit = sum(1 for x in f if x.get("severity") == "critical")
        joins = result.get("joins", [])
        return (f"{result.get('verdict', '?').replace('_', ' ')} · {len(result.get('tables', {}))} tables · "
                f"{sum(j['status'] == 'verified' for j in joins)}/{len(joins)} joins clean"
                + (f" · {crit} critical" if crit else ""))
    if skill == "eda":
        return f"{len(result.get('key_insights', []))} insights"
    return (result.get("summary") or "")[:90]


def _plan_summary(session: dict) -> dict | None:
    """Plan progress for the Sessions page (recomputed from the saved state)."""
    if not session.get("spec"):
        return None
    from ds_workbench.core.plan import build_plan
    p = build_plan({k: session.get(k) for k in ("spec", "selection", "analysis_table")}
                   | {"scout": session.get("scout_report"), "results": session.get("results", {})})
    return {"done": p["done"], "total": p["total"], "next": p["next"]["text"], "kind": p["next"]["kind"],
            "method": (p["method"] or {}).get("method")}


def _scout_summary(report: dict | None) -> dict | None:
    """What the Sessions page needs from a scout report (the full report stays in the notebook)."""
    if not report:
        return None
    return {"investigate": [c["table"] for c in report.get("candidates", []) if c["recommendation"] == "investigate"],
            "rejected": [c["table"] for c in report.get("candidates", []) if c["recommendation"] == "reject"],
            "feasibility": report.get("feasibility"), "coverage_gaps": report.get("coverage_gaps", []),
            "catalog": report.get("catalog"), "evidence_level": report.get("evidence_level")}


def _framing(spec: dict | None) -> dict | None:
    """One-line view of the problem spec for the Sessions page."""
    if not spec:
        return None
    from ds_workbench.core.problem_spec import primary_question
    q = primary_question(spec)
    return {"type": q.get("type", "?"), "question": q.get("statement", ""),
            "answerability": spec.get("answerability", {}).get("status", "?"),
            "status": spec.get("status", "draft"), "confirmed_by": spec.get("confirmed_by"),
            "blocking_gaps": [m["item"] for m in spec.get("missing", []) if m.get("blocking")]}


# check state (readiness policy) → gate chip
_GATE_FROM_STATE = {"missing": "skipped", "failed": "error", "blocked": "blocked",
                    "caution": "warn", "clean": "clear"}
_ATTENTION_FROM_STATUS = {"blocked": "red", "incomplete": "amber", "caution": "amber",
                          "ready_for_review": "green"}


def summarize_session(session: dict) -> dict:
    """The session's readiness comes from the shared policy (ds_workbench.core.readiness);
    this only adds presentation: gate chips, attention colour and per-step headlines."""
    results = session.get("results", {})
    readiness = assess(results, confirmations=session.get("confirmations"),
                       declared=session.get("required_checks"))
    checks = readiness["checks"]
    gate_skills = [*GATES, *(["causal_inference"] if "causal_inference" in results else [])]
    gates = {g: _GATE_FROM_STATE[checks[g]["state"]] if g in checks else "skipped" for g in gate_skills}

    attention = "grey" if not results else _ATTENTION_FROM_STATUS[readiness["status"]]
    order = [r["skill"] for r in session.get("runs", [])]
    seen = list(dict.fromkeys(order))
    return {
        "verdict": readiness["status"], "verdict_label": readiness["label"], "attention": attention,
        "gates": gates, "missing": readiness["missing"], "failed": readiness["failed"],
        "blockers": readiness["blockers"], "n_blockers": len(readiness["blockers"]),
        "n_cautions": len(readiness["cautions"]), "summary": readiness["summary"],
        "confirmed": readiness["confirmed"], "prediction_point": session.get("prediction_point", ""),
        "leakage_caveat": (results.get("leakage_detector") or {}).get("caveat", ""),
        "framing": _framing(session.get("spec")),
        "scout": session.get("scout"),
        "plan": _plan_summary(session),
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
    # most urgent first, in the policy's order; sessions with nothing run yet last
    rank = {s: i for i, s in enumerate(STATUSES)}
    return sorted(rows, key=lambda r: (r["attention"] == "grey", rank[r["verdict"]], -_ts(r["updated_at"])))


def _ts(iso: str | None) -> float:
    try:
        return datetime.fromisoformat(iso).timestamp()
    except (TypeError, ValueError):
        return 0.0

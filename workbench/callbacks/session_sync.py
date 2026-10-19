"""Saves the active Workbench session to the team session store whenever a run is added."""
import re

from dash import callback, Input, Output

from workbench.constants import SKILL_META

_SKILL_KEYS = list(SKILL_META) + ["experiment_tracker"]


def _order(run: dict) -> int:
    """Runs are ordered by the millisecond timestamp in their id (e.g. 'fe_1729…')."""
    m = re.search(r"(\d{10,})", run.get("id", ""))
    return int(m.group(1)) if m else 0


def store_to_session(store: dict) -> dict | None:
    """Runs-store → the fields team_sessions.save_session needs (None without an active session)."""
    session = (store or {}).get("session") or {}
    if not session.get("active") or not session.get("id"):
        return None
    runs, results = [], {}
    for skill in _SKILL_KEYS:
        for run in store.get(skill, []) or []:
            if not isinstance(run, dict):
                continue
            result = run.get("result") or ({"error": run.get("error")} if run.get("status") == "error" else {})
            runs.append({"skill": skill, "label": run.get("label", ""), "ts": run.get("timestamp", ""),
                         "status": run.get("status"), "run_id": result.get("run_id"), "_order": _order(run)})
            if result:
                results[skill] = result          # later runs overwrite earlier ones
    runs.sort(key=lambda r: r.pop("_order"))
    return {"session_id": session["id"], "question": session.get("problem_statement", ""),
            "datasets": session.get("datasets", []), "runs": runs, "results": results}


@callback(Output("wb-session-sync", "data"), Input("runs-store", "data"), prevent_initial_call=True)
def sync_session(store):
    from skills import team_sessions, usage
    payload = store_to_session(store)
    if payload is None:
        return None
    team_sessions.save_session(payload.pop("session_id"), owner=usage.current_user({}),
                               source="workbench", **payload)
    return {"saved": len(payload["runs"])}

"""
Usage ledger: one record per Claude API call, for cost and reliability tracking.

Records metadata only — who, which skill, which model, tokens, latency, outcome, estimated
cost — never prompts, outputs or data. Every call made through the shared client
(skills.api_retry.get_client) is recorded, including streamed ones.

Local runs append to feedback/usage.jsonl. On Databricks, schedule export_to_delta() to
load the ledger into a Delta table the platform team can query and dashboard.

Costs are list-price estimates from token counts (config.MODEL_PRICES). Actual billing can
differ (negotiated rates, batch discounts); reconcile against the Anthropic Usage & Cost
Admin API for invoice-level numbers.
"""
from __future__ import annotations

import getpass
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

LEDGER_FILE = Path(__file__).parent.parent / "feedback" / "usage.jsonl"
_write_lock = threading.Lock()

# The app can register a function returning the signed-in user (e.g. from request headers);
# skills stay free of any web-framework dependency.
_user_resolver: Callable[[], str | None] | None = None


def set_user_resolver(fn: Callable[[], str | None] | None) -> None:
    global _user_resolver
    _user_resolver = fn


def current_user(ctx: dict) -> str:
    """Explicit context → app resolver → WORKBENCH_USER / DATABRICKS_USER → OS user."""
    if ctx.get("user"):
        return ctx["user"]
    if _user_resolver is not None:
        try:
            user = _user_resolver()
            if user:
                return user
        except Exception:  # outside a request, or the resolver failed
            pass
    return os.getenv("WORKBENCH_USER") or os.getenv("DATABRICKS_USER") or getpass.getuser()


def _status(resp) -> str:
    reason = getattr(resp, "stop_reason", None)
    return {"refusal": "refusal", "max_tokens": "truncated"}.get(reason, "ok")


def record(request: dict, resp=None, error: Exception | None = None, latency_s: float = 0.0) -> dict:
    """Append one ledger row. Never raises: losing a usage row must not break a skill."""
    from config import call_cost
    from skills.api_retry import _call_ctx

    ctx = dict(_call_ctx.get())
    usage = getattr(resp, "usage", None)
    tokens = {
        "input_tokens": getattr(usage, "input_tokens", 0) or 0,
        "output_tokens": getattr(usage, "output_tokens", 0) or 0,
        "cache_read_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
        "cache_write_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
    }
    model = request.get("model", "")
    row = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "user": current_user(ctx),
        "skill": ctx.get("skill") or ("simulation" if ctx.get("scenario") else "other"),
        "run_id": ctx.get("run_id"),
        "dataset": ctx.get("dataset"),
        "scenario": ctx.get("scenario"),
        "benchmark": ctx.get("benchmark"),
        "model": model,
        "effort": (request.get("output_config") or {}).get("effort"),
        **tokens,
        "latency_s": round(latency_s, 2),
        "status": "api_error" if error is not None else _status(resp),
        "error_type": type(error).__name__ if error is not None else None,
        "cost_usd": round(call_cost(model, tokens["input_tokens"], tokens["output_tokens"],
                                    tokens["cache_read_tokens"], tokens["cache_write_tokens"]), 6),
    }
    try:
        with _write_lock:
            LEDGER_FILE.parent.mkdir(exist_ok=True)
            with LEDGER_FILE.open("a") as f:
                f.write(json.dumps(row) + "\n")
    except OSError as exc:  # read-only install: keep working, just log
        log.debug("usage ledger write failed: %s", exc)
    return row


def instrument(client):
    """Wrap client.messages.create and .stream so every call is recorded."""
    messages = client.messages
    cls = type(messages)   # look methods up at call time, so class-level patches (tests) apply

    def create_recorded(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            resp = cls.create(messages, *args, **kwargs)
        except Exception as exc:
            record(kwargs, error=exc, latency_s=time.perf_counter() - t0)
            raise
        record(kwargs, resp, latency_s=time.perf_counter() - t0)
        return resp

    def stream_recorded(*args, **kwargs):
        manager = cls.stream(messages, *args, **kwargs)
        t0 = time.perf_counter()

        class _Recorded:
            _recorded = False

            def __enter__(self):
                self._inner = manager.__enter__()
                return self

            def __exit__(self, *exc):
                # A stream consumed by iteration is recorded when it closes
                if not self._recorded and exc[0] is None:
                    try:
                        self.get_final_message()
                    except Exception:
                        pass
                return manager.__exit__(*exc)

            def __iter__(self):            # dunder lookups bypass __getattr__
                return iter(self._inner)

            def get_final_message(self):
                try:
                    resp = self._inner.get_final_message()
                except Exception as exc:
                    if not self._recorded:
                        record(kwargs, error=exc, latency_s=time.perf_counter() - t0)
                        self._recorded = True
                    raise
                if not self._recorded:
                    record(kwargs, resp, latency_s=time.perf_counter() - t0)
                    self._recorded = True
                return resp

            def __getattr__(self, name):
                return getattr(self._inner, name)

        return _Recorded()

    messages.create = create_recorded
    messages.stream = stream_recorded
    return client


def load(path: Path = LEDGER_FILE):
    """The ledger as a pandas DataFrame (empty if nothing recorded yet)."""
    import pandas as pd
    if not path.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    df = pd.DataFrame(rows)
    if len(df):
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def export_to_delta(spark, table: str, path: Path = LEDGER_FILE) -> int:
    """Append the local ledger to a Delta table (e.g. 'main.ds_workbench.usage') and start a
    fresh file. Schedule it as a Databricks job; returns the number of rows exported."""
    df = load(path)
    if df.empty:
        return 0
    spark.createDataFrame(df).write.format("delta").mode("append").saveAsTable(table)
    path.rename(path.with_suffix(f".exported-{int(time.time())}.jsonl"))
    return len(df)


def summarize(df, days: int | None = None) -> dict | None:
    """Spend and reliability by skill, user, model and day. A 'run' is one skill call (its
    run_id groups e.g. both passes); a run succeeded if every call in it ended ok."""
    import pandas as pd
    if df is None or df.empty:
        return None
    if days:
        df = df[df["ts"] >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)]
        if df.empty:
            return None
    df = df.copy()
    df["day"] = df["ts"].dt.strftime("%Y-%m-%d")
    df["ok"] = df["status"] == "ok"
    runs = (df.dropna(subset=["run_id"]).groupby("run_id")
              .agg(skill=("skill", "first"), user=("user", "first"), cost=("cost_usd", "sum"),
                   ok=("ok", "all"), latency=("latency_s", "sum")))

    def by(col: str) -> list[dict]:
        g = df.groupby(col).agg(calls=("cost_usd", "size"), spend=("cost_usd", "sum"),
                                failed_calls=("ok", lambda s: int((~s).sum())))
        r = runs.groupby(col).agg(runs=("cost", "size"), run_cost=("cost", "mean"),
                                  success_rate=("ok", "mean"),
                                  p50_run_s=("latency", "median")) if col in runs.columns and len(runs) else None
        if r is not None:
            g = g.join(r, how="left")
        out = g.sort_values("spend", ascending=False).reset_index()
        return json.loads(out.to_json(orient="records"))

    ok_runs = runs[runs["ok"]] if len(runs) else runs
    return {
        "total_spend": float(df["cost_usd"].sum()),
        "calls": int(len(df)),
        "runs": int(len(runs)),
        "cost_per_successful_run": float(runs["cost"].sum() / len(ok_runs)) if len(ok_runs) else None,
        "failure_spend": float(df.loc[~df["ok"], "cost_usd"].sum()),
        "users": int(df["user"].nunique()),
        "by_skill": by("skill"),
        "by_user": by("user"),
        "by_model": by("model"),
        "daily": json.loads(df.groupby(["day", "skill"])["cost_usd"].sum().reset_index()
                            .to_json(orient="records")),
    }

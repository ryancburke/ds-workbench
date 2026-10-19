"""Shared retry wrapper for Anthropic API calls."""
import contextvars
import functools
import inspect
import json
import logging
import random
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

REFUSAL_LOG = Path(__file__).parent.parent / "feedback" / "refusals.jsonl"

# Who is calling the API (skill, dataset, ...) — attached to refusal log entries.
_call_ctx: contextvars.ContextVar[dict] = contextvars.ContextVar("call_ctx", default={})


@contextmanager
def call_context(**fields):
    """Tag API calls made inside this block, e.g. call_context(dataset="transactions_fraud")."""
    token = _call_ctx.set({**_call_ctx.get(), **{k: v for k, v in fields.items() if v is not None}})
    try:
        yield
    finally:
        _call_ctx.reset(token)


def skill_entry(skill: str):
    """Decorator for a skill's public entry point: tags its API calls with the skill
    name, and with table_name as the dataset when the caller hasn't set one."""
    def wrap(fn):
        sig = inspect.signature(fn)

        @functools.wraps(fn)
        def inner(*args, **kwargs):
            try:
                table = sig.bind_partial(*args, **kwargs).arguments.get("table_name")
            except TypeError:
                table = None
            dataset = _call_ctx.get().get("dataset") or table
            with call_context(skill=skill, dataset=dataset):
                return fn(*args, **kwargs)
        return inner
    return wrap


def _prompt_summary(request: dict, limit: int = 300) -> dict:
    msgs = request.get("messages") or []
    last_user = next((m for m in reversed(msgs) if m.get("role") == "user"), {})
    content = last_user.get("content", "")
    if not isinstance(content, str):
        content = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
    system = request.get("system") or ""
    return {
        "system_head": system[:150] if isinstance(system, str) else "",
        "user_head": content[:limit],
        "prompt_chars": len(json.dumps(msgs, default=str)) + len(str(system)),
        "tools": [t.get("name") for t in request.get("tools") or []],
    }


def _log_refusal(model: str, category, explanation, request: dict | None):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **_call_ctx.get(),
        "model": model,
        "category": category,
        "explanation": explanation,
        **({"prompt": _prompt_summary(request)} if request else {}),
    }
    log.warning("Model refusal: %s", {k: entry[k] for k in entry if k != "prompt"})
    try:
        REFUSAL_LOG.parent.mkdir(exist_ok=True)
        with REFUSAL_LOG.open("a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except OSError:
        pass  # read-only install (e.g. a notebook) — the log.warning above still fires


class ModelRefusal(RuntimeError):
    """The model declined the request (stop_reason == "refusal")."""

    def __init__(self, model: str, category: str | None, explanation: str | None):
        self.model = model
        self.category = category
        detail = f" ({category})" if category else ""
        super().__init__(f"{model} declined the request{detail}: {explanation or 'no explanation'}")


def check_response(resp, request: dict | None = None):
    """Log and raise on a refusal, so callers don't fail later on a missing
    tool_use/text block. Pass the request kwargs to include a prompt summary."""
    if getattr(resp, "stop_reason", None) == "refusal":
        details = getattr(resp, "stop_details", None)
        model = getattr(resp, "model", None) or (request or {}).get("model", "model")
        category = getattr(details, "category", None)
        explanation = getattr(details, "explanation", None)
        _log_refusal(model, category, explanation, request)
        raise ModelRefusal(model, category, explanation)
    return resp


def response_text(resp) -> str:
    """Join the text blocks of a response. 5.x models can lead with thinking blocks,
    so never read content[0]."""
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def api_call(fn, *args, max_retries: int = 3, **kwargs):
    """
    Call fn(*args, **kwargs) with exponential backoff on transient API errors.
    Covers timeouts, connection errors, server errors, and rate limits.
    Raises ModelRefusal if the model declines.
    """
    import anthropic
    _retryable = (
        anthropic.APITimeoutError,
        anthropic.APIConnectionError,
        anthropic.InternalServerError,
        anthropic.RateLimitError,
    )
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            return check_response(fn(*args, **kwargs), kwargs)
        except _retryable as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                delay = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(delay)
    raise last_exc  # type: ignore[misc]

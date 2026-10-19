"""Shared Anthropic client, request helpers, and refusal logging for skills and agents."""
import contextvars
import copy
import functools
import inspect
import json
import logging
import uuid
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
            # One run_id groups the calls of a skill run (e.g. both passes) in the usage ledger;
            # a skill called from inside another keeps the outer run
            run_id = _call_ctx.get().get("run_id") or uuid.uuid4().hex[:12]
            with call_context(skill=skill, dataset=dataset, run_id=run_id):
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


class TruncatedResponse(RuntimeError):
    """The reply hit max_tokens, so its text or tool input is incomplete."""


def check_response(resp, request: dict | None = None):
    """Raise on a refusal or a reply cut off at max_tokens, so callers never parse a
    partial tool input or analysis. Pass the request kwargs to include a prompt summary
    in the refusal log."""
    if getattr(resp, "stop_reason", None) == "max_tokens":
        model = getattr(resp, "model", None) or (request or {}).get("model", "model")
        limit = (request or {}).get("max_tokens")
        raise TruncatedResponse(
            f"{model} reply was cut off at max_tokens{f'={limit}' if limit else ''}; "
            "raise max_tokens or lower effort")
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


_client = None

# The SDK retries connection errors, 408/409/429 and 5xx with backoff and honours
# retry-after headers, so retries live there rather than in a second loop on top.
SDK_MAX_RETRIES = 4


def get_client():
    """Shared Anthropic client (lazy, so importing a skill never needs a key)."""
    global _client
    if _client is None:
        import anthropic
        from config import config
        from skills import usage
        _client = usage.instrument(
            anthropic.Anthropic(api_key=config.anthropic_api_key, max_retries=SDK_MAX_RETRIES))
    return _client


def api_call(fn, *args, **kwargs):
    """Call fn(*args, **kwargs) and raise ModelRefusal if the model declines."""
    return check_response(fn(*args, **kwargs), kwargs)


# Strict tool schemas reject numeric/length bounds; the range goes into the
# description instead and callers clamp the value in code.
_UNSUPPORTED_IN_STRICT = ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
                          "minLength", "maxLength", "minItems", "maxItems")


def strict_tool(tool: dict) -> dict:
    """Copy of a tool definition with strict: true, so the input always matches the
    schema. Used on models where tool_choice can't force the call (Opus/Sonnet/Fable 5.x)."""
    tool = copy.deepcopy(tool)

    def fix(node):
        if isinstance(node, dict):
            bounds = [f"{k}={node.pop(k)}" for k in _UNSUPPORTED_IN_STRICT if k in node]
            if bounds:
                node["description"] = (node.get("description", "") + f" ({', '.join(bounds)})").strip()
            if node.get("type") == "object":
                node["additionalProperties"] = False
            for v in node.values():
                fix(v)
        elif isinstance(node, list):
            for v in node:
                fix(v)

    fix(tool["input_schema"])
    tool["strict"] = True
    return tool


def tool_input(resp, name: str) -> dict:
    """Input of the named tool call, or a clear error if the model answered without it."""
    for block in resp.content:
        if block.type == "tool_use" and block.name == name:
            return block.input
    text = response_text(resp)[:200]
    raise RuntimeError(
        f"{getattr(resp, 'model', 'model')} did not call {name} "
        f"(stop_reason={resp.stop_reason}){': ' + text if text else ''}"
    )

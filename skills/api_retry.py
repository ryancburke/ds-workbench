"""Shared retry wrapper for Anthropic API calls."""
import random
import time


class ModelRefusal(RuntimeError):
    """The model declined the request (stop_reason == "refusal")."""

    def __init__(self, model: str, category: str | None, explanation: str | None):
        self.model = model
        self.category = category
        detail = f" ({category})" if category else ""
        super().__init__(f"{model} declined the request{detail}: {explanation or 'no explanation'}")


def check_response(resp):
    """Raise on a refusal, so callers don't fail later on a missing tool_use/text block."""
    if getattr(resp, "stop_reason", None) == "refusal":
        details = getattr(resp, "stop_details", None)
        raise ModelRefusal(
            getattr(resp, "model", "model"),
            getattr(details, "category", None),
            getattr(details, "explanation", None),
        )
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
            return check_response(fn(*args, **kwargs))
        except _retryable as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                delay = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(delay)
    raise last_exc  # type: ignore[misc]

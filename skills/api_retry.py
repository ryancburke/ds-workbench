"""Shared retry wrapper for Anthropic API calls."""
import random
import time


def api_call(fn, *args, max_retries: int = 3, **kwargs):
    """
    Call fn(*args, **kwargs) with exponential backoff on transient API errors.
    Covers timeouts, connection errors, server errors, and rate limits.
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
            return fn(*args, **kwargs)
        except _retryable as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                delay = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(delay)
    raise last_exc  # type: ignore[misc]

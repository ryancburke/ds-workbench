"""Offline test fixtures: every Anthropic call is intercepted, nothing is sent."""
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from anthropic.resources.messages import Messages  # noqa: E402

import skills.api_retry as api_retry  # noqa: E402


def text_reply(model: str, text: str = "analysis text"):
    """A 5.x-style reply: thinking block first, then text."""
    return NS(model=model, stop_reason="end_turn", content=[
        NS(type="thinking", thinking=""), NS(type="text", text=text)])


def tool_reply(model: str, name: str, args: dict):
    return NS(model=model, stop_reason="tool_use", content=[
        NS(type="thinking", thinking=""), NS(type="tool_use", name=name, input=args)])


def refusal_reply(model: str):
    return NS(model=model, stop_reason="refusal", content=[],
              stop_details=NS(category="cyber", explanation="test refusal"))


class FakeAPI:
    """Records request kwargs; `responder(kwargs)` builds each reply.

    The default responder answers text calls with text and tool calls with the
    first offered tool and empty input, which is enough for request-shape tests."""

    def __init__(self):
        self.calls: list[dict] = []
        self.responder = self.default

    @staticmethod
    def default(kw):
        tools = kw.get("tools") or []
        if tools:
            return tool_reply(kw["model"], tools[0]["name"], {})
        return text_reply(kw["model"])

    def create(self, **kw):
        self.calls.append(kw)
        return self.responder(kw)

    def stream(self, **kw):
        self.calls.append(kw)
        reply = self.responder(kw)

        class _Stream:
            def __enter__(s): return s
            def __exit__(s, *a): return False
            def get_final_message(s): return reply
        return _Stream()


@pytest.fixture
def fake_api(monkeypatch, tmp_path):
    api = FakeAPI()
    monkeypatch.setattr(Messages, "create", lambda _self, **kw: api.create(**kw))
    monkeypatch.setattr(Messages, "stream", lambda _self, **kw: api.stream(**kw))
    # never touch the real refusal log or usage ledger
    monkeypatch.setattr(api_retry, "REFUSAL_LOG", tmp_path / "refusals.jsonl")
    from skills import usage
    monkeypatch.setattr(usage, "LEDGER_FILE", tmp_path / "usage.jsonl")
    return api

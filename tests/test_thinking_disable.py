"""Q1 fix: verify chat_template_kwargs.enable_thinking lands in the request.

P-phase doctrine had `reasoning_effort="none"` as the off switch but
Ollama-served Gemma silently ignores that field — the actual mechanism
is `chat_template_kwargs={"enable_thinking": False}` in the request body.
This test guards the doctrine against regression.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from indra_belief.model_client import ModelClient


@pytest.fixture
def captured_client(monkeypatch):
    """Construct a ModelClient with an OpenAI-compat client whose
    chat.completions.create captures kwargs."""
    monkeypatch.setattr(ModelClient, "_setup_openai_client", lambda self: None)
    client = ModelClient("gemma-remote")

    captured: dict = {}

    fake_response = MagicMock()
    fake_response.choices[0].message.content = "{}"
    fake_response.choices[0].message.reasoning_content = ""
    fake_response.choices[0].finish_reason = "stop"
    fake_response.usage.completion_tokens = 1
    fake_response.usage.prompt_tokens = 100

    def capture(**kwargs):
        captured.update(kwargs)
        return fake_response

    fake_inner_client = MagicMock()
    fake_inner_client.chat.completions.create = capture
    client._client = fake_inner_client

    return client, captured


class TestThinkingDisable:
    def test_reasoning_effort_none_sends_chat_template_kwargs(self, captured_client):
        """`reasoning_effort="none"` must land as
        `chat_template_kwargs={"enable_thinking": False}` in extra_body."""
        client, captured = captured_client
        client.call(
            system="s", messages=[{"role": "user", "content": "u"}],
            reasoning_effort="none",
        )
        eb = captured.get("extra_body", {})
        assert eb.get("chat_template_kwargs") == {"enable_thinking": False}, (
            f"expected chat_template_kwargs.enable_thinking=False; "
            f"extra_body={eb}"
        )
        # Belt-and-suspenders: reasoning_effort should ALSO be sent
        # (different backends honor different fields).
        assert eb.get("reasoning_effort") == "none"

    def test_reasoning_effort_low_does_not_disable_thinking(self, captured_client):
        """Only `none` translates to chat_template_kwargs. `low` and
        `medium` should leave thinking on."""
        client, captured = captured_client
        client.call(
            system="s", messages=[{"role": "user", "content": "u"}],
            reasoning_effort="low",
        )
        eb = captured.get("extra_body", {})
        assert "chat_template_kwargs" not in eb
        assert eb.get("reasoning_effort") == "low"

    def test_no_reasoning_effort_no_chat_template_kwargs(self, captured_client):
        """Default config-driven path: gemma-remote config has
        reasoning_effort='medium' baked in, so chat_template_kwargs should
        NOT be set unless caller explicitly overrides to 'none'."""
        client, captured = captured_client
        client.call(
            system="s", messages=[{"role": "user", "content": "u"}],
        )
        eb = captured.get("extra_body", {})
        assert "chat_template_kwargs" not in eb

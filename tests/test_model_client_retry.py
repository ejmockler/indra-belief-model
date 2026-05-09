"""ModelClient retry-doctrine tests (P-phase).

The only in-client retry is for 429 rate limits. Timeouts and connection
errors raise on the first occurrence. These tests guard the doctrine.
"""
from __future__ import annotations

import pytest

from indra_belief.model_client import ModelClient, ModelResponse


def _make_client(monkeypatch) -> ModelClient:
    """Construct a ModelClient without touching the network."""
    monkeypatch.setattr(ModelClient, "_setup_openai_client", lambda self: None)
    return ModelClient("gemma-remote")


class TestTimeoutNoRetry:
    """Timeouts and connection errors must raise on the first call.

    Pre-P doctrine retried 3× with exponential backoff, which amplified
    endpoint degradation (one slow record cost ~30 min before failing).
    """

    def test_timeout_raises_immediately(self, monkeypatch):
        client = _make_client(monkeypatch)
        calls = []

        def boom(*args, **kwargs):
            calls.append(1)
            raise TimeoutError("read timed out")

        monkeypatch.setattr(client, "_call_openai_compat", boom)
        with pytest.raises(TimeoutError):
            client.call(system="s", messages=[{"role": "user", "content": "u"}])
        assert len(calls) == 1, f"expected 1 call, got {len(calls)}"

    def test_connection_error_raises_immediately(self, monkeypatch):
        client = _make_client(monkeypatch)
        calls = []

        def boom(*args, **kwargs):
            calls.append(1)
            raise ConnectionError("connection refused")

        monkeypatch.setattr(client, "_call_openai_compat", boom)
        with pytest.raises(ConnectionError):
            client.call(system="s", messages=[{"role": "user", "content": "u"}])
        assert len(calls) == 1


class TestRateLimitRetry:
    """429 retries are the ONLY in-client retry path."""

    def test_429_retries_then_succeeds(self, monkeypatch):
        client = _make_client(monkeypatch)
        calls = []
        ok = ModelResponse(
            content="{}", reasoning="", tokens=1, raw_text="{}", finish_reason="stop"
        )

        def maybe_429(*args, **kwargs):
            calls.append(1)
            if len(calls) < 3:
                raise RuntimeError("429 Too Many Requests; retry in 0s")
            return ok

        monkeypatch.setattr(client, "_call_openai_compat", maybe_429)
        # Patch sleep so the test runs fast.
        import indra_belief.model_client as mc
        monkeypatch.setattr(mc, "_parse_retry_delay", lambda *_a, **_k: 0)

        result = client.call(system="s", messages=[{"role": "user", "content": "u"}])
        assert result is ok
        assert len(calls) == 3

    def test_429_exhaustion_raises(self, monkeypatch):
        """After rate_limit_retries (5) attempts, the original 429 error
        propagates — no silent swallowing."""
        client = _make_client(monkeypatch)
        calls = []

        def always_429(*args, **kwargs):
            calls.append(1)
            raise RuntimeError("429 Too Many Requests")

        monkeypatch.setattr(client, "_call_openai_compat", always_429)
        import indra_belief.model_client as mc
        monkeypatch.setattr(mc, "_parse_retry_delay", lambda *_a, **_k: 0)

        with pytest.raises(RuntimeError, match="429"):
            client.call(system="s", messages=[{"role": "user", "content": "u"}])
        # rate_limit_retries=5 → 1 initial + 5 retries = 6 attempts before
        # the 6th 429 falls through to raise.
        assert len(calls) == 6


class TestNoTimeoutRetryHelper:
    """Negative-assertion guardrail: the deleted retry branch stays gone."""

    def test_call_source_has_no_timeout_retry_branch(self):
        import inspect
        from indra_belief.model_client import ModelClient
        src = inspect.getsource(ModelClient.call)
        # The exact deleted condition — its return would mean the retry
        # branch came back. Doctrine: only 429 retries are allowed.
        assert '"timeout" in msg' not in src
        assert '"connection" in msg' not in src

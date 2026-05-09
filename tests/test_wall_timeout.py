"""Q3 wall-time circuit breaker tests.

The SDK's `timeout` field doesn't bound wall time on streaming
generations. The Q3 wrapper enforces a hard cap via ThreadPoolExecutor.
"""
from __future__ import annotations

import time

import pytest

from indra_belief.model_client import ModelClient, ModelResponse


def _make_client(monkeypatch) -> ModelClient:
    monkeypatch.setattr(ModelClient, "_setup_openai_client", lambda self: None)
    return ModelClient("gemma-remote")


class TestWallTimeoutFires:
    """Hanging _call_openai_compat must raise TimeoutError within ~timeout
    seconds — not block forever."""

    def test_slow_call_raises_timeout(self, monkeypatch):
        client = _make_client(monkeypatch)

        def slow(*a, **kw):
            time.sleep(10)  # well over 1s timeout
            return ModelResponse(content="{}", reasoning="", tokens=1,
                                 raw_text="{}", finish_reason="stop")

        monkeypatch.setattr(client, "_call_openai_compat", slow)
        # Override config timeout to 1s for this test.
        client.config = dict(client.config)
        client.config["timeout"] = 1

        t0 = time.time()
        with pytest.raises(TimeoutError, match="exceeded 1s"):
            client.call(system="s", messages=[{"role": "user", "content": "u"}])
        elapsed = time.time() - t0
        # Some slack for executor wakeup; should be well under 5s.
        assert elapsed < 5, f"timeout took {elapsed}s; should be ~1s"

    def test_fast_call_no_overhead(self, monkeypatch):
        """A successful fast call must not be penalized by the wrapper."""
        client = _make_client(monkeypatch)
        ok = ModelResponse(content="{}", reasoning="", tokens=1,
                           raw_text="{}", finish_reason="stop")
        monkeypatch.setattr(client, "_call_openai_compat", lambda *a, **kw: ok)

        t0 = time.time()
        result = client.call(system="s", messages=[{"role": "user", "content": "u"}])
        elapsed = time.time() - t0
        assert result is ok
        # Threading overhead should be <100ms on any modern host.
        assert elapsed < 0.5, f"fast call took {elapsed}s; should be <0.1s"


class TestTimeoutLogsTelemetry:
    """A timed-out call should still appear in the call_log with error."""

    def test_timeout_records_telemetry(self, monkeypatch):
        client = _make_client(monkeypatch)
        client.config = dict(client.config)
        client.config["timeout"] = 1

        def hang(*a, **kw):
            time.sleep(5)
            return ModelResponse(content="{}", reasoning="", tokens=1,
                                 raw_text="{}", finish_reason="stop")

        monkeypatch.setattr(client, "_call_openai_compat", hang)
        with pytest.raises(TimeoutError):
            client.call(system="s", messages=[{"role": "user", "content": "u"}],
                        kind="parse_evidence")
        log = client.pop_call_log()
        assert len(log) == 1
        assert log[0]["kind"] == "parse_evidence"
        assert log[0]["error"] == "TimeoutError"

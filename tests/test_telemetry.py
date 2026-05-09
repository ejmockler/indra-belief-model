"""P8 per-call telemetry tests.

Asserts that ModelClient.call() records per-invocation telemetry to a
thread-local log that score_evidence_decomposed surfaces in result["call_log"].
"""
from __future__ import annotations

import pytest

from indra_belief.model_client import ModelClient, ModelResponse


def _make_client(monkeypatch) -> ModelClient:
    monkeypatch.setattr(ModelClient, "_setup_openai_client", lambda self: None)
    return ModelClient("gemma-remote")


def _stub_response(content: str = "{}", out_tokens: int = 10,
                   prompt_tokens: int = 100,
                   finish_reason: str = "stop") -> ModelResponse:
    return ModelResponse(
        content=content, reasoning="", tokens=out_tokens,
        raw_text=content, finish_reason=finish_reason,
        prompt_tokens=prompt_tokens,
    )


class TestCallLogPopulated:
    """Each successful call appends one telemetry entry."""

    def test_single_call_records_entry(self, monkeypatch):
        client = _make_client(monkeypatch)
        monkeypatch.setattr(
            client, "_call_openai_compat",
            lambda *a, **kw: _stub_response()
        )
        client.call(system="s", messages=[{"role": "user", "content": "u"}],
                    kind="parse_evidence")
        log = client.pop_call_log()
        assert len(log) == 1
        entry = log[0]
        assert entry["kind"] == "parse_evidence"
        assert entry["finish_reason"] == "stop"
        assert entry["out_tokens"] == 10
        assert entry["prompt_tokens"] == 100
        assert entry["prompt_chars"] > 0  # accumulates from system + messages
        assert "duration_s" in entry

    def test_multiple_calls_accumulate(self, monkeypatch):
        client = _make_client(monkeypatch)
        monkeypatch.setattr(
            client, "_call_openai_compat",
            lambda *a, **kw: _stub_response()
        )
        client.call(system="s", messages=[{"role": "user", "content": "x"}],
                    kind="parse_evidence")
        client.call(system="s", messages=[{"role": "user", "content": "y"}],
                    kind="verify_grounding")
        client.call(system="s", messages=[{"role": "user", "content": "z"}],
                    kind="verify_grounding")
        log = client.pop_call_log()
        assert [e["kind"] for e in log] == [
            "parse_evidence", "verify_grounding", "verify_grounding",
        ]

    def test_pop_clears_log(self, monkeypatch):
        """After pop_call_log() the log is empty for the next batch."""
        client = _make_client(monkeypatch)
        monkeypatch.setattr(
            client, "_call_openai_compat",
            lambda *a, **kw: _stub_response()
        )
        client.call(system="s", messages=[{"role": "user", "content": "x"}],
                    kind="parse_evidence")
        first = client.pop_call_log()
        second = client.pop_call_log()
        assert len(first) == 1
        assert second == []


class TestErrorIsRecorded:
    def test_failed_call_records_error_kind(self, monkeypatch):
        client = _make_client(monkeypatch)

        def boom(*a, **kw):
            raise TimeoutError("read timed out")

        monkeypatch.setattr(client, "_call_openai_compat", boom)
        with pytest.raises(TimeoutError):
            client.call(system="s", messages=[{"role": "user", "content": "u"}],
                        kind="parse_evidence")
        log = client.pop_call_log()
        assert len(log) == 1
        assert log[0]["error"] == "TimeoutError"
        assert log[0]["kind"] == "parse_evidence"
        assert log[0]["out_tokens"] == 0


class TestThreadLocalIsolation:
    """ThreadPoolExecutor workers must not see each other's calls."""

    def test_two_threads_have_isolated_logs(self, monkeypatch):
        from concurrent.futures import ThreadPoolExecutor
        client = _make_client(monkeypatch)
        monkeypatch.setattr(
            client, "_call_openai_compat",
            lambda *a, **kw: _stub_response()
        )

        def worker(label: str) -> list[dict]:
            client.pop_call_log()  # ensure clean
            for _ in range(3):
                client.call(system="s",
                            messages=[{"role": "user", "content": label}],
                            kind=label)
            return client.pop_call_log()

        with ThreadPoolExecutor(max_workers=2) as pool:
            f1 = pool.submit(worker, "alpha")
            f2 = pool.submit(worker, "beta")
            log1 = f1.result()
            log2 = f2.result()

        # Each worker's log contains exactly its own kind labels.
        assert all(e["kind"] == "alpha" for e in log1)
        assert all(e["kind"] == "beta" for e in log2)
        assert len(log1) == 3
        assert len(log2) == 3

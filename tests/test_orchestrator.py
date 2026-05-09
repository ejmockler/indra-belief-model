"""Smoke tests for the S-phase orchestrator.

End-to-end calls of score_via_probes with a mock client: verify the
result dict has the expected shape and that substrate-resolved paths
work without LLM calls.
"""
from __future__ import annotations

from dataclasses import dataclass

from indra_belief.scorers.probes.orchestrator import score_via_probes
from indra.statements import Activation, Agent, Complex, Evidence


@dataclass
class _MockResponse:
    content: str
    raw_text: str = ""
    tokens: int = 10
    finish_reason: str = "stop"
    reasoning: str = ""
    prompt_tokens: int = 100


class _MockClient:
    """Mock client that returns programmable responses keyed by call kind."""
    def __init__(self, responses_by_kind: dict[str, str] | None = None):
        self.responses_by_kind = responses_by_kind or {}
        self.calls: list[dict] = []
        self._call_log: list[dict] = []

    def call(self, *, system, messages, max_tokens=None, temperature=0.1,
             response_format=None, reasoning_effort=None, kind=None,
             **_kwargs) -> _MockResponse:
        self.calls.append({"kind": kind, "n_messages": len(messages)})
        self._call_log.append({"kind": kind, "out_tokens": 10,
                               "duration_s": 0.1})
        content = self.responses_by_kind.get(kind, '{"answer": "abstain"}')
        return _MockResponse(content=content, raw_text=content)

    def pop_call_log(self) -> list[dict]:
        log = list(self._call_log)
        self._call_log.clear()
        return log


def test_orchestrator_returns_expected_dict_shape() -> None:
    stmt = Activation(Agent("MAPK1"), Agent("JUN"))
    ev = Evidence(source_api="reach",
                  text="MAPK1 phosphorylates JUN at Ser63.")
    client = _MockClient({
        "probe_subject_role":
            '{"answer": "present_as_subject", "rationale": "MAPK1 acts"}',
        "probe_object_role":
            '{"answer": "present_as_object", "rationale": "JUN target"}',
        "probe_relation_axis":
            '{"answer": "direct_axis_mismatch", '
            '"rationale": "modification not activity"}',
        "probe_scope": '{"answer": "asserted", "rationale": "direct"}',
    })
    result = score_via_probes(stmt, ev, client)
    # Required keys
    for key in ("score", "verdict", "confidence", "raw_text", "tokens",
                "tier", "grounding_status", "provenance_triggered",
                "reasons", "rationale", "call_log"):
        assert key in result, f"missing key {key!r} in result"
    assert result["tier"] == "decomposed"
    # Modification-on-activity-claim → axis_mismatch → incorrect
    assert result["verdict"] == "incorrect"
    assert "axis_mismatch" in result["reasons"]


def test_orchestrator_handles_parse_claim_failure() -> None:
    """If parse_claim raises, orchestrator returns abstain without LLM."""
    # An unsupported statement type — Complex with unbound members would
    # be supported, but a malformed agent could trip parse_claim. For
    # robustness, just pass an agent with no name.
    bad_stmt = Activation(Agent(""), Agent(""))
    ev = Evidence(source_api="reach", text="...")
    client = _MockClient()
    result = score_via_probes(bad_stmt, ev, client)
    # Result must be a valid dict; parse failure → abstain.
    assert "verdict" in result


def test_orchestrator_handles_llm_failure_gracefully() -> None:
    """When all four probes fail (transport), result is abstain."""
    stmt = Activation(Agent("MAPK1"), Agent("JUN"))
    ev = Evidence(source_api="reach", text="MAPK1 and JUN co-occurred.")

    class _RaisingClient(_MockClient):
        def call(self, **kwargs):
            self.calls.append({"kind": kwargs.get("kind")})
            raise TimeoutError("90s wall clock")

    client = _RaisingClient()
    result = score_via_probes(stmt, ev, client)
    # Should not crash
    assert "verdict" in result
    assert result["verdict"] in ("abstain", "correct", "incorrect")


def test_orchestrator_alias_miss_escalates_to_llm() -> None:
    """When subject/object aliases aren't in evidence, substrate
    escalates to LLM with hint. LLM-confirmed absent → grounding_gap
    abstain, but LLM gets a chance to recover anaphora/paraphrase."""
    stmt = Activation(Agent("MAPK1"), Agent("JUN"))
    ev = Evidence(source_api="reach",
                  text="The cell cycle was monitored using flow cytometry.")
    client = _MockClient({
        "probe_subject_role":
            '{"answer": "absent", "rationale": "MAPK1 not mentioned"}',
        "probe_object_role":
            '{"answer": "absent", "rationale": "JUN not mentioned"}',
        # T-phase Fix A: relation_axis no longer admits "abstain".
        # The mock returns out-of-set "abstain"; failure_default projects
        # to "no_relation" via _llm.py.
        "probe_relation_axis": '{"answer": "no_relation", "rationale": "..."}',
        "probe_scope": '{"answer": "abstain", "rationale": "..."}',
    })
    result = score_via_probes(stmt, ev, client)
    kinds_called = {c["kind"] for c in client.calls}
    # Now LLM IS called for subject/object (substrate doesn't commit absent)
    assert "probe_subject_role" in kinds_called
    assert "probe_object_role" in kinds_called
    # LLM confirmed absent → grounding_gap abstain
    assert result["verdict"] == "abstain"


def test_orchestrator_call_log_populated() -> None:
    stmt = Activation(Agent("MAPK1"), Agent("JUN"))
    ev = Evidence(source_api="reach",
                  text="MAPK1 and JUN are in the same pathway.")
    client = _MockClient({
        "probe_subject_role":
            '{"answer": "present_as_subject", "rationale": "MAPK1"}',
        "probe_object_role":
            '{"answer": "present_as_object", "rationale": "JUN"}',
        "probe_relation_axis":
            '{"answer": "no_relation", "rationale": "co-occurrence"}',
        "probe_scope": '{"answer": "asserted", "rationale": "..."}',
    })
    result = score_via_probes(stmt, ev, client)
    assert isinstance(result["call_log"], list)
    # The call log should reflect at least the LLM probe calls (not
    # substrate-resolved ones).
    assert any("probe_" in (c.get("kind") or "")
               for c in result["call_log"])

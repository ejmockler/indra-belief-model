"""Probe module tests with a mocked ModelClient.

Each probe (subject_role, object_role, relation_axis, scope) is tested
on a small set of representative inputs:
  - LLM returns a valid answer in the set → ProbeResponse with source=llm
  - LLM returns an out-of-set answer → fallback abstain with source=abstain
  - LLM raises a transport error → fallback abstain with source=abstain
  - LLM returns malformed JSON → fallback abstain with source=abstain

The mock captures the messages list it received so we can verify
prompt construction (system + few-shots + user message + substrate
hint when present).
"""
from __future__ import annotations

from dataclasses import dataclass

from indra_belief.scorers.probes import (
    object_role, relation_axis, scope, subject_role,
)
from indra_belief.scorers.probes.types import ProbeRequest


@dataclass
class _MockResponse:
    content: str
    raw_text: str = ""
    tokens: int = 10
    finish_reason: str = "stop"
    reasoning: str = ""
    prompt_tokens: int = 100


class _MockClient:
    """Captures messages and returns a programmable response."""
    def __init__(self, response_content: str = "",
                 raise_exc: Exception | None = None):
        self.response_content = response_content
        self.raise_exc = raise_exc
        self.last_call: dict | None = None

    def call(self, *, system, messages, max_tokens=None, temperature=0.1,
             response_format=None, reasoning_effort=None, kind=None,
             **_kwargs) -> _MockResponse:
        self.last_call = dict(
            system=system, messages=messages, max_tokens=max_tokens,
            temperature=temperature, response_format=response_format,
            reasoning_effort=reasoning_effort, kind=kind,
        )
        if self.raise_exc is not None:
            raise self.raise_exc
        return _MockResponse(content=self.response_content,
                             raw_text=self.response_content)


# --- subject_role probe ----------------------------------------------------

def test_subject_role_llm_success() -> None:
    client = _MockClient(
        response_content='{"answer": "present_as_subject", '
                         '"rationale": "MAPK1 phosphorylates JUN"}'
    )
    req = ProbeRequest(
        kind="subject_role",
        claim_component="MAPK1 (axis=modification, sign=positive, objects=['JUN'])",
        evidence_text="MAPK1 phosphorylates JUN at Ser63.",
    )
    resp = subject_role.answer(req, client)
    assert resp.kind == "subject_role"
    assert resp.answer == "present_as_subject"
    assert resp.source == "llm"
    assert resp.rationale == "MAPK1 phosphorylates JUN"
    # Verify few-shots are threaded
    msgs = client.last_call["messages"]
    assert len(msgs) >= 2  # at least one few-shot pair + the user msg
    assert any(m["role"] == "assistant" for m in msgs)


def test_subject_role_llm_out_of_set_falls_back_to_absent() -> None:
    client = _MockClient(
        response_content='{"answer": "active_kinase", "rationale": "..."}'
    )
    req = ProbeRequest(kind="subject_role", claim_component="MAPK1",
                       evidence_text="MAPK1 phosphorylates JUN.")
    resp = subject_role.answer(req, client)
    assert resp.answer == "absent"
    assert resp.source == "abstain"


def test_subject_role_llm_transport_error_abstains() -> None:
    client = _MockClient(raise_exc=TimeoutError("90s wall clock"))
    req = ProbeRequest(kind="subject_role", claim_component="MAPK1",
                       evidence_text="MAPK1 phosphorylates JUN.")
    resp = subject_role.answer(req, client)
    assert resp.answer == "absent"
    assert resp.source == "abstain"
    assert "transport_error" in resp.rationale


def test_subject_role_llm_malformed_json_abstains() -> None:
    client = _MockClient(response_content='not really json {{{')
    req = ProbeRequest(kind="subject_role", claim_component="MAPK1",
                       evidence_text="MAPK1 phosphorylates JUN.")
    resp = subject_role.answer(req, client)
    assert resp.answer == "absent"
    assert resp.source == "abstain"
    assert "json_parse" in resp.rationale


def test_subject_role_includes_substrate_hint_in_user_message() -> None:
    client = _MockClient(
        response_content='{"answer": "present_as_object", '
                         '"rationale": "role-swap: TNF as target"}'
    )
    req = ProbeRequest(
        kind="subject_role",
        claim_component="TNF",
        evidence_text="TNF receptor binding of TNFalpha activates apoptosis.",
        substrate_hint="substrate observed TNF as TARGET (not agent) of "
                       "non-binding relations",
    )
    resp = subject_role.answer(req, client)
    assert resp.answer == "present_as_object"
    user_msg = client.last_call["messages"][-1]["content"]
    assert "SUBSTRATE HINT:" in user_msg
    assert "TARGET" in user_msg


def test_subject_role_kind_mismatch_raises() -> None:
    import pytest
    client = _MockClient()
    req = ProbeRequest(kind="object_role", claim_component="x",
                       evidence_text="...")
    with pytest.raises(ValueError, match="received kind"):
        subject_role.answer(req, client)


# --- object_role probe -----------------------------------------------------

def test_object_role_llm_success() -> None:
    client = _MockClient(
        response_content='{"answer": "present_as_object", '
                         '"rationale": "JUN is the target"}'
    )
    req = ProbeRequest(kind="object_role", claim_component="JUN",
                       evidence_text="MAPK1 phosphorylates JUN.")
    resp = object_role.answer(req, client)
    assert resp.answer == "present_as_object"
    assert resp.source == "llm"


def test_object_role_role_swap_detection() -> None:
    client = _MockClient(
        response_content='{"answer": "present_as_subject", '
                         '"rationale": "MAPK1 as agent, role swap"}'
    )
    req = ProbeRequest(
        kind="object_role", claim_component="MAPK1",
        evidence_text="MAPK1 in turn phosphorylates RSK and downstream kinases.",
    )
    resp = object_role.answer(req, client)
    assert resp.answer == "present_as_subject"


def test_object_role_kind_mismatch_raises() -> None:
    import pytest
    client = _MockClient()
    req = ProbeRequest(kind="scope", claim_component="x", evidence_text="...")
    with pytest.raises(ValueError, match="received kind"):
        object_role.answer(req, client)


# --- relation_axis probe ---------------------------------------------------

def test_relation_axis_direct_sign_match() -> None:
    client = _MockClient(
        response_content='{"answer": "direct_sign_match", '
                         '"rationale": "MAPK1 activates JUN"}'
    )
    req = ProbeRequest(
        kind="relation_axis",
        claim_component="(MAPK1, JUN) — claim axis=activity, sign=positive",
        evidence_text="MAPK1 activates JUN.",
    )
    resp = relation_axis.answer(req, client)
    assert resp.answer == "direct_sign_match"
    assert resp.source == "llm"


def test_relation_axis_via_mediator() -> None:
    client = _MockClient(
        response_content='{"answer": "via_mediator", '
                         '"rationale": "Myc -> Phlpp2 -> Akt"}'
    )
    req = ProbeRequest(
        kind="relation_axis",
        claim_component="(Myc, Akt) — claim axis=activity, sign=negative",
        evidence_text="Phlpp2 is co-opted by Myc to suppress Akt activity.",
        substrate_hint="L1 chain signal in evidence",
    )
    resp = relation_axis.answer(req, client)
    assert resp.answer == "via_mediator"


def test_relation_axis_partner_mismatch() -> None:
    client = _MockClient(
        response_content='{"answer": "direct_partner_mismatch", '
                         '"rationale": "DNA binding, not protein-protein"}'
    )
    req = ProbeRequest(
        kind="relation_axis",
        claim_component="(p53, DNA-element) — claim axis=binding, sign=neutral",
        evidence_text="p53 binds the consensus DNA element in the promoter.",
    )
    resp = relation_axis.answer(req, client)
    assert resp.answer == "direct_partner_mismatch"


def test_relation_axis_underdetermined_projects_to_no_relation() -> None:
    """T-phase Fix A: 'abstain' is no longer in relation_axis answer set.
    If the LLM emits 'abstain' (out-of-set), _llm.py's failure_default
    mechanism projects to 'no_relation' and treats it as a successful
    classification (source='llm'). Out-of-set string answers go through
    success-with-projection; only transport/JSON/empty failures source
    'abstain'. Doctrine §3.1."""
    client = _MockClient(
        response_content='{"answer": "abstain", '
                         '"rationale": "JUN not mentioned"}'
    )
    req = ProbeRequest(
        kind="relation_axis",
        claim_component="(MAPK1, JUN) — claim axis=activity, sign=positive",
        evidence_text="We characterized MAPK1 substrates in cycling cells.",
    )
    resp = relation_axis.answer(req, client)
    assert resp.answer == "no_relation"  # projected via failure_default
    assert resp.source == "llm"          # LLM responded; just out-of-set


def test_relation_axis_transport_failure_sources_abstain() -> None:
    """T-phase Fix A discrimination: actual LLM failures (transport,
    empty response) still result in source='abstain'. Only out-of-set
    string answers go through success-with-projection."""
    class _RaisingClient:
        def call(self, **kwargs):
            raise TimeoutError("simulated")

    client = _RaisingClient()
    req = ProbeRequest(
        kind="relation_axis",
        claim_component="(MAPK1, JUN) — claim axis=activity, sign=positive",
        evidence_text="MAPK1 activates JUN.",
    )
    resp = relation_axis.answer(req, client)
    assert resp.answer == "no_relation"  # projected via failure_default
    assert resp.source == "abstain"      # transport failure


def test_relation_axis_kind_mismatch_raises() -> None:
    import pytest
    client = _MockClient()
    req = ProbeRequest(kind="subject_role", claim_component="x",
                       evidence_text="...")
    with pytest.raises(ValueError, match="received kind"):
        relation_axis.answer(req, client)


# --- scope probe -----------------------------------------------------------

def test_scope_asserted() -> None:
    client = _MockClient(
        response_content='{"answer": "asserted", '
                         '"rationale": "direct affirmation"}'
    )
    req = ProbeRequest(kind="scope",
                       claim_component="relation between MAPK1 and JUN",
                       evidence_text="MAPK1 activates JUN in stimulated cells.")
    resp = scope.answer(req, client)
    assert resp.answer == "asserted"
    assert resp.source == "llm"


def test_scope_hedged() -> None:
    client = _MockClient(
        response_content='{"answer": "hedged", '
                         '"rationale": "may activate ... remains to be confirmed"}'
    )
    req = ProbeRequest(kind="scope",
                       claim_component="relation between CCR7 and AKT",
                       evidence_text="CCR7 may activate Akt in T-cells, but "
                       "this remains to be confirmed.")
    resp = scope.answer(req, client)
    assert resp.answer == "hedged"


def test_scope_negated() -> None:
    client = _MockClient(
        response_content='{"answer": "negated", '
                         '"rationale": "did not activate"}'
    )
    req = ProbeRequest(kind="scope",
                       claim_component="relation between MAPK1 and JUN",
                       evidence_text="MAPK1 did not activate JUN under any tested condition.")
    resp = scope.answer(req, client)
    assert resp.answer == "negated"


def test_scope_negation_governs_different_clause_returns_asserted() -> None:
    """Critical: 'X activates Y, but Z was not affected' should return
    asserted for the X-Y relation. Few-shot exemplifies this; this test
    verifies the LLM contract is properly framed."""
    client = _MockClient(
        response_content='{"answer": "asserted", '
                         '"rationale": "negation governs different relation"}'
    )
    req = ProbeRequest(
        kind="scope",
        claim_component="relation between MAPK1 and JUN",
        evidence_text="MAPK1 activates JUN robustly, but ELK1 was not "
        "affected by the treatment.",
        substrate_hint="no negation cue within local window",
    )
    resp = scope.answer(req, client)
    assert resp.answer == "asserted"


def test_scope_kind_mismatch_raises() -> None:
    import pytest
    client = _MockClient()
    req = ProbeRequest(kind="relation_axis", claim_component="x",
                       evidence_text="...")
    with pytest.raises(ValueError, match="received kind"):
        scope.answer(req, client)


# --- Cross-probe: telemetry tag in client call -----------------------------

def test_probe_tags_call_log_with_kind() -> None:
    """Each probe should tag the LLM call with its kind so the
    call_log telemetry can attribute latency and tokens correctly."""
    client = _MockClient(
        response_content='{"answer": "present_as_subject", '
                         '"rationale": "..."}'
    )
    req = ProbeRequest(kind="subject_role", claim_component="MAPK1",
                       evidence_text="MAPK1 phosphorylates JUN.")
    subject_role.answer(req, client)
    assert client.last_call["kind"] == "probe_subject_role"

    client = _MockClient(
        response_content='{"answer": "present_as_object", "rationale": "..."}'
    )
    req = ProbeRequest(kind="object_role", claim_component="JUN",
                       evidence_text="MAPK1 phosphorylates JUN.")
    object_role.answer(req, client)
    assert client.last_call["kind"] == "probe_object_role"

    client = _MockClient(
        response_content='{"answer": "direct_sign_match", "rationale": "..."}'
    )
    req = ProbeRequest(kind="relation_axis",
                       claim_component="(MAPK1, JUN) — axis=activity",
                       evidence_text="MAPK1 activates JUN.")
    relation_axis.answer(req, client)
    assert client.last_call["kind"] == "probe_relation_axis"

    client = _MockClient(
        response_content='{"answer": "asserted", "rationale": "..."}'
    )
    req = ProbeRequest(kind="scope",
                       claim_component="MAPK1 and JUN",
                       evidence_text="MAPK1 activates JUN.")
    scope.answer(req, client)
    assert client.last_call["kind"] == "probe_scope"

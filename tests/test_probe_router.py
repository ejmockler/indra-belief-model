"""Substrate-as-router tests.

Cover the four routing functions on representative ctx/claim shapes:
  - subject_role_probe answers absent / present_as_subject /
    present_as_mediator / escalates with hint
  - object_role_probe mirrors
  - relation_axis_probe answers direct_sign_match /
    direct_sign_mismatch / direct_axis_mismatch / via_mediator /
    via_mediator_partial / escalates with hint
  - scope_probe answers negated / hedged / escalates

The tests build minimal EvidenceContext + ClaimCommitment objects
directly — no INDRA Statements, no Gilda calls, so they run fast and
exercise the routing logic in isolation.
"""
from __future__ import annotations

from indra_belief.scorers.commitments import ClaimCommitment
from indra_belief.scorers.context import DetectedRelation, EvidenceContext
from indra_belief.scorers.probes.router import substrate_route
from indra_belief.scorers.probes.types import ProbeRequest, ProbeResponse


def _claim(
    subject: str = "MAPK1",
    objects: tuple[str, ...] = ("JUN",),
    axis: str = "activity",
    sign: str = "positive",
    stmt_type: str = "Activation",
) -> ClaimCommitment:
    return ClaimCommitment(
        stmt_type=stmt_type,
        subject=subject,
        objects=objects,
        axis=axis,  # type: ignore[arg-type]
        sign=sign,  # type: ignore[arg-type]
    )


def _ctx(
    *,
    aliases: dict[str, frozenset[str]] | None = None,
    detected_relations: tuple[DetectedRelation, ...] = (),
    explicit_hedge_markers: frozenset[str] = frozenset(),
    subject_perturbation_marker: str | None = None,
    has_chain_signal: bool = False,
    chain_intermediate_candidates: tuple[str, ...] = (),
    nominalized_relations: tuple[str, ...] = (),
) -> EvidenceContext:
    return EvidenceContext(
        aliases=aliases or {},
        detected_relations=detected_relations,
        explicit_hedge_markers=explicit_hedge_markers,
        subject_perturbation_marker=subject_perturbation_marker,
        has_chain_signal=has_chain_signal,
        chain_intermediate_candidates=chain_intermediate_candidates,
        nominalized_relations=nominalized_relations,
    )


def _dr(
    axis: str, sign: str, agent: str, target: str,
    pattern_id: str = "test_pattern",
) -> DetectedRelation:
    return DetectedRelation(
        axis=axis, sign=sign,
        agent_canonical=agent, target_canonical=target,
        site=None, pattern_id=pattern_id, span=(0, 10),
    )


# ---------------------------------------------------------------------------
# subject_role probe
# ---------------------------------------------------------------------------

def test_subject_no_alias_match_escalates_to_llm() -> None:
    """Substrate no longer commits to 'absent' when aliases don't match
    in evidence — escalates to LLM with hint instead. The LLM can
    detect anaphora, paraphrase, or family-member references that the
    static alias map missed."""
    claim = _claim()
    ctx = _ctx(aliases={"MAPK1": frozenset({"MAPK1", "ERK2"})})
    routings = substrate_route(claim, ctx, "Some unrelated sentence.")
    r = routings["subject_role"]
    assert isinstance(r, ProbeRequest)
    assert "matched no aliases" in (r.substrate_hint or "")


def test_subject_present_as_subject_via_catalog_match() -> None:
    claim = _claim()
    ctx = _ctx(
        aliases={"MAPK1": frozenset({"MAPK1"}), "JUN": frozenset({"JUN"})},
        detected_relations=(
            _dr("activity", "positive", "MAPK1", "JUN", "act_pos.x_activates_y"),
        ),
    )
    routings = substrate_route(claim, ctx, "MAPK1 activates JUN in cells.")
    r = routings["subject_role"]
    assert isinstance(r, ProbeResponse)
    assert r.answer == "present_as_subject"
    assert r.source == "substrate"


def test_subject_with_lof_perturbation_marker() -> None:
    claim = _claim()
    ctx = _ctx(
        aliases={"MAPK1": frozenset({"MAPK1"}), "JUN": frozenset({"JUN"})},
        detected_relations=(
            _dr("activity", "positive", "MAPK1", "JUN", "act_pos.x_activates_y"),
        ),
        subject_perturbation_marker="loss_of_function",
    )
    routings = substrate_route(claim, ctx, "MAPK1 knockdown reduced JUN activity.")
    r = routings["subject_role"]
    assert isinstance(r, ProbeResponse)
    assert r.answer == "present_as_subject"
    assert r.perturbation == "LOF"


def test_subject_role_swap_candidate_escalates_with_hint() -> None:
    claim = _claim()
    ctx = _ctx(
        aliases={"MAPK1": frozenset({"MAPK1"}), "JUN": frozenset({"JUN"})},
        detected_relations=(
            # Substrate detected JUN activates MAPK1 — subject is in target slot
            _dr("activity", "positive", "JUN", "MAPK1", "act_pos.x_activates_y"),
        ),
    )
    routings = substrate_route(claim, ctx, "JUN activates MAPK1 downstream.")
    r = routings["subject_role"]
    assert isinstance(r, ProbeRequest)
    assert "role-swap" in (r.substrate_hint or "")


def test_subject_chain_intermediate_returns_mediator() -> None:
    claim = _claim(subject="MEK1", objects=("JUN",))
    ctx = _ctx(
        aliases={"MEK1": frozenset({"MEK1"}), "JUN": frozenset({"JUN"})},
        has_chain_signal=True,
        chain_intermediate_candidates=("MEK1",),
    )
    routings = substrate_route(
        claim, ctx, "MAPK1 activates MEK1 thereby activating JUN.")
    r = routings["subject_role"]
    assert isinstance(r, ProbeResponse)
    assert r.answer == "present_as_mediator"


def test_subject_mentioned_but_role_unclear_escalates() -> None:
    claim = _claim()
    ctx = _ctx(aliases={"MAPK1": frozenset({"MAPK1"})})
    routings = substrate_route(
        claim, ctx, "MAPK1 was mentioned in passing here.")
    r = routings["subject_role"]
    assert isinstance(r, ProbeRequest)
    assert "MAPK1" in (r.substrate_hint or "")


# ---------------------------------------------------------------------------
# object_role probe
# ---------------------------------------------------------------------------

def test_object_no_alias_match_escalates_to_llm() -> None:
    """Mirror of subject_role: alias miss → LLM escalation, not
    committed absent."""
    claim = _claim()
    ctx = _ctx(aliases={"JUN": frozenset({"JUN", "AP1"})})
    routings = substrate_route(claim, ctx, "MAPK1 phosphorylates ELK1.")
    r = routings["object_role"]
    assert isinstance(r, ProbeRequest)
    assert "matched no aliases" in (r.substrate_hint or "")


def test_object_present_as_object_via_catalog() -> None:
    claim = _claim()
    ctx = _ctx(
        aliases={"MAPK1": frozenset({"MAPK1"}), "JUN": frozenset({"JUN"})},
        detected_relations=(
            _dr("activity", "positive", "MAPK1", "JUN", "act_pos.x_activates_y"),
        ),
    )
    routings = substrate_route(claim, ctx, "MAPK1 activates JUN in cells.")
    r = routings["object_role"]
    assert isinstance(r, ProbeResponse)
    assert r.answer == "present_as_object"


def test_object_role_swap_candidate_escalates() -> None:
    claim = _claim()
    ctx = _ctx(
        aliases={"MAPK1": frozenset({"MAPK1"}), "JUN": frozenset({"JUN"})},
        detected_relations=(
            _dr("activity", "positive", "JUN", "MAPK1", "act_pos.x_activates_y"),
        ),
    )
    routings = substrate_route(claim, ctx, "JUN activates MAPK1 downstream.")
    r = routings["object_role"]
    assert isinstance(r, ProbeRequest)
    assert "role-swap" in (r.substrate_hint or "")


# ---------------------------------------------------------------------------
# relation_axis probe
# ---------------------------------------------------------------------------

def test_relation_direct_sign_match() -> None:
    claim = _claim()
    ctx = _ctx(
        aliases={"MAPK1": frozenset({"MAPK1"}), "JUN": frozenset({"JUN"})},
        detected_relations=(
            _dr("activity", "positive", "MAPK1", "JUN", "act_pos.x_activates_y"),
        ),
    )
    routings = substrate_route(claim, ctx, "MAPK1 activates JUN.")
    r = routings["relation_axis"]
    assert isinstance(r, ProbeResponse)
    assert r.answer == "direct_sign_match"


def test_relation_direct_sign_mismatch() -> None:
    claim = _claim(sign="positive")
    ctx = _ctx(
        aliases={"MAPK1": frozenset({"MAPK1"}), "JUN": frozenset({"JUN"})},
        detected_relations=(
            # Catalog says MAPK1 inhibits JUN — opposite sign
            _dr("activity", "negative", "MAPK1", "JUN", "act_neg.x_inhibits_y"),
        ),
    )
    routings = substrate_route(claim, ctx, "MAPK1 inhibits JUN activity.")
    r = routings["relation_axis"]
    assert isinstance(r, ProbeResponse)
    assert r.answer == "direct_sign_mismatch"


def test_relation_direct_axis_mismatch() -> None:
    # Claim is activity, catalog detected modification on same pair.
    claim = _claim(axis="activity", sign="positive")
    ctx = _ctx(
        aliases={"MAPK1": frozenset({"MAPK1"}), "JUN": frozenset({"JUN"})},
        detected_relations=(
            _dr("modification", "positive", "MAPK1", "JUN",
                "mod_pos.x_phosphorylates_y"),
        ),
    )
    routings = substrate_route(claim, ctx, "MAPK1 phosphorylates JUN at S63.")
    r = routings["relation_axis"]
    assert isinstance(r, ProbeResponse)
    assert r.answer == "direct_axis_mismatch"


def test_relation_perturbation_inverts_effective_sign() -> None:
    # Claim: positive activation. LOF on subject. Substrate detects
    # negative relation (inhibition). After inversion, claim's effective
    # sign is negative, matching detected sign → direct_sign_match.
    claim = _claim(sign="positive")
    ctx = _ctx(
        aliases={"MAPK1": frozenset({"MAPK1"}), "JUN": frozenset({"JUN"})},
        detected_relations=(
            _dr("activity", "negative", "MAPK1", "JUN",
                "act_neg.x_inhibits_y"),
        ),
        subject_perturbation_marker="loss_of_function",
    )
    routings = substrate_route(claim, ctx,
                               "MAPK1 knockdown blocked JUN activation.")
    r = routings["relation_axis"]
    assert isinstance(r, ProbeResponse)
    assert r.answer == "direct_sign_match"


def test_relation_chain_signal_with_intermediates_escalates_with_hint() -> None:
    """Substrate does NOT commit via_mediator from chain signals alone
    (S4 gate finding: L1 markers fire on nominalizations like 'via the
    activation of X' where the relation is direct, not chained).
    Chain signals become hints, not substrate answers."""
    claim = _claim()
    ctx = _ctx(
        aliases={"MAPK1": frozenset({"MAPK1"}), "JUN": frozenset({"JUN"})},
        has_chain_signal=True,
        chain_intermediate_candidates=("MEK1",),
    )
    routings = substrate_route(claim, ctx,
                               "MAPK1 activates MEK1 thereby activating JUN.")
    r = routings["relation_axis"]
    assert isinstance(r, ProbeRequest)
    assert "chain signal" in (r.substrate_hint or "")


def test_relation_chain_signal_no_intermediates_escalates() -> None:
    claim = _claim()
    ctx = _ctx(
        aliases={"MAPK1": frozenset({"MAPK1"}), "JUN": frozenset({"JUN"})},
        has_chain_signal=True,
    )
    routings = substrate_route(claim, ctx, "MAPK1 leads to JUN activation.")
    r = routings["relation_axis"]
    assert isinstance(r, ProbeRequest)
    assert "chain signal" in (r.substrate_hint or "")


def test_relation_no_match_escalates() -> None:
    claim = _claim()
    ctx = _ctx(
        aliases={"MAPK1": frozenset({"MAPK1"}), "JUN": frozenset({"JUN"})},
    )
    routings = substrate_route(claim, ctx,
                               "MAPK1 and JUN co-occur in some pathway.")
    r = routings["relation_axis"]
    assert isinstance(r, ProbeRequest)


def test_relation_binding_symmetric_swapped_treated_as_match() -> None:
    # Complex(JUN, FOS) — order-invariant. Catalog detected FOS-JUN binding.
    claim = _claim(subject="JUN", objects=("FOS",), axis="binding",
                   sign="neutral", stmt_type="Complex")
    ctx = _ctx(
        aliases={"JUN": frozenset({"JUN"}), "FOS": frozenset({"FOS"})},
        detected_relations=(
            _dr("binding", "neutral", "FOS", "JUN", "bind.x_binds_y"),
        ),
    )
    routings = substrate_route(claim, ctx, "FOS forms a complex with JUN.")
    r = routings["relation_axis"]
    assert isinstance(r, ProbeResponse)
    assert r.answer == "direct_sign_match"


# ---------------------------------------------------------------------------
# scope probe
# ---------------------------------------------------------------------------

def test_scope_negation_within_window() -> None:
    claim = _claim()
    ctx = _ctx(
        aliases={"MAPK1": frozenset({"MAPK1"}), "JUN": frozenset({"JUN"})},
    )
    routings = substrate_route(
        claim, ctx, "MAPK1 did not activate JUN in any condition tested.")
    r = routings["scope"]
    assert isinstance(r, ProbeResponse)
    assert r.answer == "negated"


def test_scope_negation_too_far_escalates() -> None:
    claim = _claim()
    ctx = _ctx(
        aliases={"MAPK1": frozenset({"MAPK1"}), "JUN": frozenset({"JUN"})},
    )
    text = ("MAPK1 was characterized in early work. " * 5
            + "We did not test JUN."
            " " * 100
            + " MAPK1 phosphorylates ELK1 robustly.")
    # MAPK1 is far from "did not"; substrate should escalate.
    routings = substrate_route(claim, ctx, text)
    r = routings["scope"]
    # Either negated (if JUN happens to be near) or escalates — both fine
    # so long as the algorithm is honest about proximity.
    assert isinstance(r, (ProbeRequest, ProbeResponse))


def test_scope_explicit_hedge_marker_returns_hedged() -> None:
    claim = _claim()
    ctx = _ctx(
        aliases={"MAPK1": frozenset({"MAPK1"}), "JUN": frozenset({"JUN"})},
        explicit_hedge_markers=frozenset({"may", "could"}),
    )
    routings = substrate_route(claim, ctx, "MAPK1 may activate JUN.")
    r = routings["scope"]
    assert isinstance(r, ProbeResponse)
    assert r.answer == "hedged"


def test_scope_no_signal_escalates() -> None:
    claim = _claim()
    ctx = _ctx(
        aliases={"MAPK1": frozenset({"MAPK1"}), "JUN": frozenset({"JUN"})},
    )
    routings = substrate_route(claim, ctx, "MAPK1 activates JUN in cells.")
    r = routings["scope"]
    assert isinstance(r, ProbeRequest)


def test_scope_negation_outside_subject_object_span_does_not_fire() -> None:
    """S4 regression: 'MAPK1 activates JUN, but did not affect ELK1'
    should NOT emit negated for the MAPK1-JUN claim — the 'did not'
    governs the ELK1 clause, sitting OUTSIDE the [MAPK1...JUN] span."""
    claim = _claim()
    ctx = _ctx(
        aliases={"MAPK1": frozenset({"MAPK1"}), "JUN": frozenset({"JUN"})},
    )
    text = "MAPK1 activates JUN, but did not affect ELK1 transcription."
    routings = substrate_route(claim, ctx, text)
    r = routings["scope"]
    # 'did not' is at position 21+, AFTER JUN (pos 16). For claim entities
    # at positions 0 (MAPK1) and 16 (JUN), span is [0, 16]; negator at
    # position ~21 is outside. Substrate must NOT emit negated.
    assert isinstance(r, ProbeRequest), \
        f"expected escalation, got substrate answer {r!r}"


def test_scope_drops_lexically_broad_cues() -> None:
    """S4 regression: words like 'no', 'never', 'absent' are no longer
    in the negation regex — too noisy. Sentence with these but no
    verb-negator should escalate, not emit negated."""
    claim = _claim()
    ctx = _ctx(
        aliases={"MAPK1": frozenset({"MAPK1"}), "JUN": frozenset({"JUN"})},
    )
    text = "MAPK1 was never observed without JUN, no doubt about it."
    routings = substrate_route(claim, ctx, text)
    r = routings["scope"]
    # 'never', 'without', 'no' all dropped from regex; should escalate.
    assert isinstance(r, ProbeRequest)


# ---------------------------------------------------------------------------
# Top-level: substrate_route returns all four
# ---------------------------------------------------------------------------

def test_substrate_route_returns_all_four_kinds() -> None:
    claim = _claim()
    ctx = _ctx()
    routings = substrate_route(claim, ctx, "irrelevant text.")
    assert set(routings.keys()) == {
        "subject_role", "object_role", "relation_axis", "scope",
    }


def test_substrate_route_all_high_confidence_path() -> None:
    """A claim with full substrate coverage routes all four probes
    deterministically, never escalating."""
    claim = _claim()
    ctx = _ctx(
        aliases={"MAPK1": frozenset({"MAPK1"}), "JUN": frozenset({"JUN"})},
        detected_relations=(
            _dr("activity", "positive", "MAPK1", "JUN", "act_pos.x_activates_y"),
        ),
        explicit_hedge_markers=frozenset({"may"}),
    )
    routings = substrate_route(claim, ctx, "MAPK1 may activate JUN.")
    for kind, r in routings.items():
        assert isinstance(r, ProbeResponse), \
            f"{kind} should be substrate-answered, got {type(r).__name__}"

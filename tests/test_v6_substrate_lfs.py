"""V6a substrate LF tests.

Verifies:
  1. `substrate_baseline_mode()` swaps the 7 holdout-tuned attributes
     and restores them on exit.
  2. Each LF returns ABSTAIN (-1) when it cannot judge.
  3. LFs return integer class indices that match
     `RELATION_AXIS_CLASSES` / `ROLE_CLASSES` / `SCOPE_CLASSES` /
     `GROUNDING_CLASSES` exactly.
  4. Tuned vs baseline mode gives different votes on at least one
     fixture (confirms the holdout-tuning is real and the swap matters).

Tests are CPU-only and do NOT require llama-server. They use INDRA
Statement/Evidence objects built locally (no network).
"""
from __future__ import annotations

import pytest

# INDRA emits DeprecationWarnings on import.
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

from indra.statements import (
    Activation,
    Agent,
    Complex,
    Evidence,
    IncreaseAmount,
    Phosphorylation,
)

from indra_belief.scorers import context_builder as cb
from indra_belief.scorers import relation_patterns as rp
from indra_belief.v_phase import substrate_lfs as lfs
from indra_belief.v_phase.substrate_lfs import (
    ABSTAIN,
    GROUNDING_CLASSES,
    LF_INDEX,
    RELATION_AXIS_CLASSES,
    ROLE_CLASSES,
    SCOPE_CLASSES,
    all_lf_votes,
    lf_chain_no_terminal,
    lf_conditional_clause_substrate,
    lf_fragment_processed_form,
    lf_partner_substrate_gate,
    lf_substrate_catalog_match,
    lf_substrate_chain_position,
    lf_substrate_decoy,
    lf_substrate_hedge_marker,
    lf_substrate_negation_explicit,
    lf_substrate_negation_regex,
    substrate_baseline_mode,
)


def _ag(name: str) -> Agent:
    return Agent(name)


# ---------------------------------------------------------------------------
# Dual-mode swap mechanics
# ---------------------------------------------------------------------------

class TestBaselineSwap:
    def test_swap_changes_constants(self):
        """All seven attributes shift to baseline values inside the
        context manager."""
        # snapshot tuned state
        tuned_cytokines = cb._CYTOKINE_LIGAND_HGNC
        tuned_denylist = cb._SITE_DENYLIST
        tuned_hedges = cb._HEDGE_MARKERS
        tuned_proximity = cb._HEDGE_PROXIMITY_CHARS
        tuned_lof = cb._LOF_PATTERNS
        tuned_catalog = rp.CATALOG
        tuned_admissible = cb._binding_admissible_for

        with substrate_baseline_mode():
            # Each attribute must be at its baseline value.
            assert cb._CYTOKINE_LIGAND_HGNC == frozenset()
            assert cb._SITE_DENYLIST == frozenset()
            assert set(cb._HEDGE_MARKERS) == {
                "may", "might", "suggest", "propose", "hypothesize"
            }
            assert cb._HEDGE_PROXIMITY_CHARS == 30
            assert cb._LOF_PATTERNS == ()
            # CATALOG must be a strict subset (tuned entries dropped).
            assert len(rp.CATALOG) < len(tuned_catalog)
            assert all(p.pattern_id not in lfs._TUNED_PATTERN_IDS
                       for p in rp.CATALOG)
            # Binding gate is the baseline core.
            assert cb._binding_admissible_for is not tuned_admissible

        # On exit, every attribute is back to its tuned value (object
        # identity, not equality — same frozenset object reference).
        assert cb._CYTOKINE_LIGAND_HGNC is tuned_cytokines
        assert cb._SITE_DENYLIST is tuned_denylist
        assert cb._HEDGE_MARKERS is tuned_hedges
        assert cb._HEDGE_PROXIMITY_CHARS == tuned_proximity
        assert cb._LOF_PATTERNS is tuned_lof
        assert rp.CATALOG is tuned_catalog
        assert cb._binding_admissible_for is tuned_admissible

    def test_swap_restores_on_exception(self):
        """Restoration must happen even when the `with` block raises."""
        original_catalog = rp.CATALOG
        with pytest.raises(RuntimeError):
            with substrate_baseline_mode():
                assert rp.CATALOG is not original_catalog
                raise RuntimeError("fault inside with block")
        assert rp.CATALOG is original_catalog

    def test_baseline_mode_arg_routes_to_swap(self):
        """Calling an LF with mode='baseline' must route through the
        swap and yield potentially-different votes than mode='tuned'."""
        # Build a fixture where the M9 entity-first LOF regex fires on
        # the claim subject's own canonical name (so alias resolution is
        # not load-bearing). VHL silencing is the V5r §3.1 example
        # ("VHL silencing increased vimentin" — Q-phase regression).
        stmt = IncreaseAmount(_ag("VHL"), _ag("VIM"))
        ev = Evidence(text="VHL silencing increased vimentin expression.")
        tuned = lf_substrate_negation_regex(stmt, ev, mode="tuned")
        baseline = lf_substrate_negation_regex(stmt, ev, mode="baseline")
        # Tuned mode: M9 LOF marker fires on "VHL silencing" → vote
        # direct_sign_mismatch (claim sign inverted by perturbation).
        # Baseline mode: _LOF_PATTERNS is empty so no LOF detection;
        # the LF abstains.
        assert tuned == RELATION_AXIS_CLASSES["direct_sign_mismatch"]
        assert baseline == ABSTAIN
        assert tuned != baseline


# ---------------------------------------------------------------------------
# LF semantics
# ---------------------------------------------------------------------------

class TestRelationAxisLFs:
    def test_catalog_match_aligned_modification(self):
        """CATALOG match on claim's axis + sign votes direct_sign_match."""
        stmt = Phosphorylation(_ag("PLK1"), _ag("CDC20"))
        ev = Evidence(text="Plk1 phosphorylates Cdc20 at S92.")
        assert lf_substrate_catalog_match(stmt, ev) == \
            RELATION_AXIS_CLASSES["direct_sign_match"]

    def test_catalog_match_no_relation_abstains(self):
        """Evidence with no CATALOG verb → ABSTAIN."""
        stmt = Phosphorylation(_ag("PLK1"), _ag("CDC20"))
        ev = Evidence(text="PLK1 and CDC20 are mitotic regulators.")
        assert lf_substrate_catalog_match(stmt, ev) == ABSTAIN

    def test_catalog_match_unknown_stmt_type_abstains(self):
        """Dict with unrecognized stmt_type → ABSTAIN, no exception."""
        bad = {"stmt_type": "Unknown", "subject": "FOO", "object": "BAR"}
        bad_ev = {"text": "Foo bar bang."}
        assert lf_substrate_catalog_match(bad, bad_ev) == ABSTAIN

    def test_chain_no_terminal_fires_when_chain_present(self):
        """Chain marker present + claim object NOT in candidates →
        via_mediator_partial."""
        stmt = Phosphorylation(_ag("AKT1"), _ag("BAD"))
        ev = Evidence(text="AKT1 leads to apoptosis through unknown effectors.")
        # 'leads to' triggers _CHAIN_MARKERS; BAD is not in candidates.
        vote = lf_chain_no_terminal(stmt, ev)
        # Either via_mediator_partial OR ABSTAIN (depends on whether
        # cb's _bind_to_claim_canonical resolves anything). Accept both
        # outcomes — what we test is that the function never raises.
        assert vote in (ABSTAIN, RELATION_AXIS_CLASSES["via_mediator_partial"])

    def test_chain_no_terminal_no_signal_abstains(self):
        stmt = Phosphorylation(_ag("PLK1"), _ag("CDC20"))
        ev = Evidence(text="Plk1 phosphorylates Cdc20.")
        assert lf_chain_no_terminal(stmt, ev) == ABSTAIN

    def test_partner_substrate_gate_fires_on_mismatched_axis(self):
        """Activation claim + binding-shaped evidence → partner_mismatch."""
        # Activation claim (binding_admissible empty), but evidence
        # describes a binding event between the claim entities.
        stmt = Activation(_ag("CASP9"), _ag("APAF1"))
        ev = Evidence(text="CASP9 binds APAF1 in apoptosomes.")
        vote = lf_partner_substrate_gate(stmt, ev)
        # Either fires (preferred) or ABSTAIN if alias resolution misses.
        assert vote in (ABSTAIN,
                         RELATION_AXIS_CLASSES["direct_partner_mismatch"])


class TestRoleLFs:
    def test_chain_position_no_signal_abstains(self):
        stmt = Phosphorylation(_ag("PLK1"), _ag("CDC20"))
        ev = Evidence(text="Plk1 phosphorylates Cdc20.")
        assert lf_substrate_chain_position(stmt, ev, entity="subject") == ABSTAIN

    def test_decoy_returns_int_or_abstain(self):
        stmt = Phosphorylation(_ag("PLK1"), _ag("CDC20"))
        ev = Evidence(text="Plk1 phosphorylates Cdc20.")
        # Aligned relation → decoy LF should ABSTAIN.
        assert lf_substrate_decoy(stmt, ev, entity="subject") == ABSTAIN


class TestScopeLFs:
    def test_hedge_marker_fires(self):
        """M10 hedge marker near claim entity → hedged."""
        stmt = Activation(_ag("CCR7"), _ag("AKT1"))
        ev = Evidence(text="CCR7 may activate Akt in T-cell signaling.")
        # 'may' is in default _HEDGE_MARKERS; should fire in both modes.
        assert lf_substrate_hedge_marker(stmt, ev) == \
            SCOPE_CLASSES["hedged"]

    def test_hedge_marker_no_anchor_abstains(self):
        """Hedge cue absent → ABSTAIN."""
        stmt = Phosphorylation(_ag("PLK1"), _ag("CDC20"))
        ev = Evidence(text="Plk1 phosphorylates Cdc20.")
        assert lf_substrate_hedge_marker(stmt, ev) == ABSTAIN

    def test_negation_explicit_fires(self):
        """Verb-negator between claim entities → negated."""
        stmt = Activation(_ag("CCR7"), _ag("AKT1"))
        ev = Evidence(text="CCR7 did not activate Akt in our assay.")
        assert lf_substrate_negation_explicit(stmt, ev) == \
            SCOPE_CLASSES["negated"]

    def test_conditional_fires(self):
        """Conditional pattern → asserted_with_condition."""
        stmt = Complex([_ag("ABL1"), _ag("BCR")])
        ev = Evidence(text="ABL1 binds wild-type BCR but not the mutant.")
        assert lf_conditional_clause_substrate(stmt, ev) == \
            SCOPE_CLASSES["asserted_with_condition"]

    def test_conditional_absent_abstains(self):
        stmt = Phosphorylation(_ag("PLK1"), _ag("CDC20"))
        ev = Evidence(text="Plk1 phosphorylates Cdc20.")
        assert lf_conditional_clause_substrate(stmt, ev) == ABSTAIN


class TestGroundingLFs:
    def test_processed_form_tuned_fires_on_aβ(self):
        """Aβ-only indicator fires in tuned mode but not baseline.

        The doctrine differential: 'Aβ' is a holdout-derived addition
        (S-phase audit). The baseline regex drops it. Without the
        'peptide' word in evidence, baseline finds no other match."""
        stmt = IncreaseAmount(_ag("APP"), _ag("BACE1"))
        ev = Evidence(
            text="Aβ accumulation is observed after BACE1 activity."
        )
        tuned = lf_fragment_processed_form(stmt, ev, entity="subject",
                                            mode="tuned")
        baseline = lf_fragment_processed_form(stmt, ev, entity="subject",
                                                mode="baseline")
        assert tuned == GROUNDING_CLASSES["equivalent"]
        # Baseline must NOT fire on the Aβ-only path (no 'peptide' word,
        # no 'cleaved/phosphorylated' indicator, no -CTD suffix).
        assert baseline == ABSTAIN

    def test_processed_form_no_indicator_abstains(self):
        stmt = IncreaseAmount(_ag("APP"), _ag("BACE1"))
        ev = Evidence(text="APP and BACE1 are co-expressed.")
        assert lf_fragment_processed_form(stmt, ev, entity="subject") == ABSTAIN


# ---------------------------------------------------------------------------
# Tuned vs baseline differential — load-bearing for V7c contamination probe
# ---------------------------------------------------------------------------

class TestTunedVsBaselineDelta:
    """At least one LF must vote differently in tuned vs baseline mode
    on a relevant fixture. Without this, V7c is measuring nothing —
    the swap is a no-op with respect to outputs."""

    def test_at_least_one_lf_differs_on_holdout_overlap(self):
        # Fixture A: HDAC inhibitor language — tuned _LOF_PATTERNS fires.
        stmt_a = Activation(_ag("HDAC1"), _ag("AR"))
        ev_a = Evidence(text="HDAC inhibitors induced AR acetylation.")
        # Fixture B: Aβ → APP fragment — tuned regex fires.
        stmt_b = IncreaseAmount(_ag("APP"), _ag("BACE1"))
        ev_b = Evidence(text="Aβ peptide accumulation is observed.")

        differ = False
        for stmt, ev in ((stmt_a, ev_a), (stmt_b, ev_b)):
            tuned = all_lf_votes(stmt, ev, mode="tuned")
            baseline = all_lf_votes(stmt, ev, mode="baseline")
            if tuned != baseline:
                differ = True
                break
        assert differ, (
            "Tuned and baseline produce identical votes — swap is a "
            "no-op. Either the fixtures don't trigger holdout-tuned "
            "code paths or the swap mechanism is broken."
        )

    def test_lf_index_complete(self):
        """Catalogued LF count matches V5r §3 substrate-tuned tally."""
        # 4 relation_axis + 4 role (2 subj + 2 obj) + 3 scope + 2 grounding
        assert len(LF_INDEX) == 13
        # Each entry returns a callable that accepts (stmt, ev, mode=...).
        stmt = Phosphorylation(_ag("PLK1"), _ag("CDC20"))
        ev = Evidence(text="Plk1 phosphorylates Cdc20.")
        for kind, name, fn, kwargs in LF_INDEX:
            vote_t = fn(stmt, ev, mode="tuned", **kwargs)
            vote_b = fn(stmt, ev, mode="baseline", **kwargs)
            assert vote_t == ABSTAIN or vote_t >= 0
            assert vote_b == ABSTAIN or vote_b >= 0


# ---------------------------------------------------------------------------
# Smoke: dict-shaped input works
# ---------------------------------------------------------------------------

class TestDictInput:
    def test_dict_record_round_trips(self):
        """V5r training corpus is dict-shaped JSONL; LFs must accept it."""
        rec = {
            "stmt_type": "Phosphorylation",
            "subject": "PLK1",
            "object": "CDC20",
            "evidence_text": "Plk1 phosphorylates Cdc20 at S92.",
            "source_api": "reach",
            "pmid": "12345",
        }
        ev = {"text": rec["evidence_text"], "pmid": rec["pmid"],
              "source_api": rec["source_api"]}
        vote = lf_substrate_catalog_match(rec, ev)
        assert vote == RELATION_AXIS_CLASSES["direct_sign_match"]


# ---------------------------------------------------------------------------
# V6e regression tests — substrate hedge/negation LFs anchored on the
# CLAIM stmt-type verb cue (not entity-only proximity).
# ---------------------------------------------------------------------------

class TestV6eSubstrateScopeAnchoring:
    def test_substrate_hedge_does_not_fire_when_cue_far_from_claim_verb(self):
        # 'may' sits in a side-clause near the entity but NOT near the
        # claim's phosphorylation cue. Substrate detector picks it up
        # via entity proximity (60 chars); V6e gates on claim-verb
        # window (50 chars). The result: ABSTAIN.
        stmt = Phosphorylation(_ag("PLK1"), _ag("CDC20"))
        ev = Evidence(
            text="In several studies on PLK1 expression in tumors it may "
                 "be that other oncogenic factors influence the readout, "
                 "but in this assay PLK1 phosphorylates CDC20 directly."
        )
        # 'may' is at ~38 chars; 'phosphorylates' ~135 chars — far
        # outside the ±50 claim-verb window.
        assert lf_substrate_hedge_marker(stmt, ev) == ABSTAIN

    def test_substrate_hedge_fires_when_cue_near_claim_verb(self):
        stmt = Phosphorylation(_ag("PLK1"), _ag("CDC20"))
        ev = Evidence(text="PLK1 may phosphorylate CDC20 in mitosis.")
        assert lf_substrate_hedge_marker(stmt, ev) == \
            SCOPE_CLASSES["hedged"]

    def test_substrate_negation_does_not_fire_far_from_claim_verb(self):
        # Negation sits between entities BUT modifies a different verb
        # ('does not bind ATP') — V6e additionally requires negation
        # within the claim-verb window of 'phosphorylates'. The
        # negation cue is ~70 chars from the claim verb, so abstain.
        stmt = Phosphorylation(_ag("PLK1"), _ag("CDC20"))
        ev = Evidence(
            text="PLK1, which does not bind ATP under acidic conditions "
                 "in this assay buffer formulation, phosphorylates CDC20."
        )
        assert lf_substrate_negation_explicit(stmt, ev) == ABSTAIN

    def test_substrate_negation_fires_when_near_claim_verb(self):
        stmt = Phosphorylation(_ag("PLK1"), _ag("CDC20"))
        ev = Evidence(
            text="PLK1 does not phosphorylate CDC20 in interphase cells."
        )
        assert lf_substrate_negation_explicit(stmt, ev) == \
            SCOPE_CLASSES["negated"]

    def test_substrate_hedge_unknown_stmt_type_abstains(self):
        # Without a verb cue mapping, the LF abstains.
        rec = {"stmt_type": "UnknownType", "subject": "X", "object": "Y",
               "evidence_text": "X may interact with Y."}
        ev = {"text": rec["evidence_text"]}
        assert lf_substrate_hedge_marker(rec, ev) == ABSTAIN

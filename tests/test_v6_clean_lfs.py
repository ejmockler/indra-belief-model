"""V6b clean LF tests.

Verifies:
  1. Each LF returns ABSTAIN (-1) when it cannot judge.
  2. LFs return integer class indices that match the probe's class enum.
  3. Positive-vote and ABSTAIN paths fire as expected on fixture inputs.

Tests are CPU-only. The `lf_no_entity_overlap`, `lf_no_grounded_match`,
and `lf_absent_alias_check` LFs invoke Gilda for alias resolution; tests
that exercise these paths use small canonical names so the lookups are
fast and deterministic.
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
    Inhibition,
    Phosphorylation,
)

from indra_belief.v_phase.clean_lfs import (
    LF_INDEX_CLEAN,
    all_clean_grounding_lf_votes,
    all_clean_lf_votes,
    lf_absent_alias_check,
    lf_ambiguous_grounding,
    lf_amount_keyword_negative,
    lf_amount_lexical,
    lf_chain_position_lexical,
    lf_chain_with_named_intermediate,
    lf_clean_assertion,
    lf_conditional_lexical,
    lf_curated_db_axis,
    lf_decoy_lexical,
    lf_epistemics_direct_false,
    lf_epistemics_direct_true,
    lf_evidence_contains_official_symbol,
    lf_evidence_too_short_grounding,
    lf_gilda_alias,
    lf_gilda_exact_symbol,
    lf_gilda_family_member,
    lf_gilda_no_match,
    lf_hedge_lexical,
    lf_low_information_evidence,
    lf_multi_extractor_axis_agreement,
    lf_negation_lexical,
    lf_no_entity_overlap,
    lf_no_grounded_match,
    lf_partner_dna_lexical,
    lf_position_object,
    lf_position_subject,
    lf_reach_found_by_axis_match,
    lf_reach_found_by_axis_mismatch,
    lf_role_swap_lexical,
    lf_text_too_short,
)
from indra_belief.v_phase.substrate_lfs import (
    ABSTAIN,
    GROUNDING_CLASSES,
    RELATION_AXIS_CLASSES,
    ROLE_CLASSES,
    SCOPE_CLASSES,
)


def _ag(name: str) -> Agent:
    return Agent(name)


# ---------------------------------------------------------------------------
# relation_axis LFs
# ---------------------------------------------------------------------------

class TestCuratedDBAxis:
    def test_signor_votes_direct(self):
        rec = {"stmt_type": "Phosphorylation", "subject": "PLK1",
               "object": "CDC20"}
        ev = {"text": "Plk1 phosphorylates Cdc20.", "source_api": "signor"}
        assert lf_curated_db_axis(rec, ev) == \
            RELATION_AXIS_CLASSES["direct_sign_match"]

    def test_reach_abstains(self):
        rec = {"stmt_type": "Phosphorylation", "subject": "PLK1",
               "object": "CDC20"}
        ev = {"text": "Plk1 phosphorylates Cdc20.", "source_api": "reach"}
        assert lf_curated_db_axis(rec, ev) == ABSTAIN


class TestReachFoundBy:
    def test_match_phosphorylation(self):
        rec = {"stmt_type": "Phosphorylation", "subject": "PLK1",
               "object": "CDC20"}
        ev = {"text": "Plk1 phosphorylates Cdc20.", "source_api": "reach",
              "annotations": {"found_by": "Phosphorylation_syntax_1a_noun"}}
        assert lf_reach_found_by_axis_match(rec, ev) == \
            RELATION_AXIS_CLASSES["direct_sign_match"]

    def test_match_positive_activation(self):
        rec = {"stmt_type": "Activation", "subject": "EGF", "object": "EGFR"}
        ev = {"text": "EGF activates EGFR.", "source_api": "reach",
              "annotations": {"found_by": "Positive_activation_syntax_1_verb"}}
        assert lf_reach_found_by_axis_match(rec, ev) == \
            RELATION_AXIS_CLASSES["direct_sign_match"]

    def test_sign_mismatch_negative(self):
        rec = {"stmt_type": "Activation", "subject": "EGF", "object": "EGFR"}
        ev = {"text": "EGF inhibits EGFR.", "source_api": "reach",
              "annotations": {"found_by": "Negative_activation_syntax_1_verb"}}
        assert lf_reach_found_by_axis_match(rec, ev) == \
            RELATION_AXIS_CLASSES["direct_sign_mismatch"]

    def test_axis_mismatch_translocation(self):
        rec = {"stmt_type": "Activation", "subject": "EGF", "object": "EGFR"}
        ev = {"text": "EGF translocates EGFR.", "source_api": "reach",
              "annotations": {"found_by": "Translocation_syntax_2_verb"}}
        # found_by_axis_match returns ABSTAIN (axis differs), the dedicated
        # mismatch LF votes axis_mismatch.
        assert lf_reach_found_by_axis_match(rec, ev) == ABSTAIN
        assert lf_reach_found_by_axis_mismatch(rec, ev) == \
            RELATION_AXIS_CLASSES["direct_axis_mismatch"]

    def test_no_annotations_abstains(self):
        rec = {"stmt_type": "Phosphorylation", "subject": "PLK1",
               "object": "CDC20"}
        ev = {"text": "Plk1 phosphorylates Cdc20.", "source_api": "reach"}
        assert lf_reach_found_by_axis_match(rec, ev) == ABSTAIN
        assert lf_reach_found_by_axis_mismatch(rec, ev) == ABSTAIN


class TestEpistemics:
    def test_direct_true_votes_direct(self):
        rec = {"stmt_type": "Phosphorylation", "subject": "PLK1",
               "object": "CDC20"}
        ev = {"text": "Plk1 phosphorylates Cdc20.",
              "epistemics": {"direct": True}}
        assert lf_epistemics_direct_true(rec, ev) == \
            RELATION_AXIS_CLASSES["direct_sign_match"]
        assert lf_epistemics_direct_false(rec, ev) == ABSTAIN

    def test_direct_false_votes_via_mediator(self):
        rec = {"stmt_type": "Phosphorylation", "subject": "PLK1",
               "object": "CDC20"}
        ev = {"text": "Plk1 indirectly affects Cdc20.",
              "epistemics": {"direct": False}}
        assert lf_epistemics_direct_false(rec, ev) == \
            RELATION_AXIS_CLASSES["via_mediator"]
        assert lf_epistemics_direct_true(rec, ev) == ABSTAIN

    def test_missing_epistemics_abstains(self):
        rec = {"stmt_type": "Phosphorylation", "subject": "PLK1",
               "object": "CDC20"}
        ev = {"text": "Plk1 phosphorylates Cdc20."}
        assert lf_epistemics_direct_true(rec, ev) == ABSTAIN
        assert lf_epistemics_direct_false(rec, ev) == ABSTAIN


class TestMultiExtractor:
    def test_multi_source_votes_direct(self):
        rec = {"stmt_type": "Phosphorylation", "subject": "PLK1",
               "object": "CDC20",
               "source_counts": {"reach": 5, "sparser": 2}}
        ev = {"text": "Plk1 phosphorylates Cdc20.", "source_api": "reach"}
        assert lf_multi_extractor_axis_agreement(rec, ev) == \
            RELATION_AXIS_CLASSES["direct_sign_match"]

    def test_single_source_abstains(self):
        rec = {"stmt_type": "Phosphorylation", "subject": "PLK1",
               "object": "CDC20",
               "source_counts": {"reach": 5}}
        ev = {"text": "Plk1 phosphorylates Cdc20.", "source_api": "reach"}
        assert lf_multi_extractor_axis_agreement(rec, ev) == ABSTAIN


class TestAmountLFs:
    def test_amount_lexical_fires_on_activation_with_expression(self):
        stmt = Activation(_ag("MYC"), _ag("CCND1"))
        ev = Evidence(text="MYC enhances CCND1 expression in T cells.")
        assert lf_amount_lexical(stmt, ev) == \
            RELATION_AXIS_CLASSES["direct_amount_match"]

    def test_amount_lexical_no_amount_word_abstains(self):
        stmt = Activation(_ag("MYC"), _ag("CCND1"))
        ev = Evidence(text="MYC binds CCND1 in vivo.")
        assert lf_amount_lexical(stmt, ev) == ABSTAIN

    def test_amount_keyword_negative_fires(self):
        stmt = IncreaseAmount(_ag("MYC"), _ag("CCND1"))
        ev = Evidence(text="MYC phosphorylates CCND1 directly.")
        assert lf_amount_keyword_negative(stmt, ev) == \
            RELATION_AXIS_CLASSES["direct_axis_mismatch"]

    def test_amount_keyword_negative_abstains_when_amount_present(self):
        stmt = IncreaseAmount(_ag("MYC"), _ag("CCND1"))
        ev = Evidence(
            text="MYC binds the CCND1 promoter and increases expression."
        )
        # Amount lexicon ('expression') is present → LF should not vote
        # axis_mismatch.
        assert lf_amount_keyword_negative(stmt, ev) == ABSTAIN


class TestChainNamedIntermediate:
    def test_fires_with_named_intermediate(self):
        stmt = Phosphorylation(_ag("AKT1"), _ag("BAD"))
        ev = Evidence(
            text="AKT1 phosphorylates GSK3 thereby leading to BAD inhibition."
        )
        # GSK3 is the named intermediate.
        assert lf_chain_with_named_intermediate(stmt, ev) == \
            RELATION_AXIS_CLASSES["via_mediator"]

    def test_abstains_no_chain_marker(self):
        stmt = Phosphorylation(_ag("AKT1"), _ag("BAD"))
        ev = Evidence(text="AKT1 phosphorylates BAD.")
        assert lf_chain_with_named_intermediate(stmt, ev) == ABSTAIN


class TestNoEntityOverlap:
    def test_fires_when_neither_entity_present(self):
        stmt = Phosphorylation(_ag("PLK1"), _ag("CDC20"))
        ev = Evidence(text="MAPK kinase phosphorylates ERK in mitogenic cells.")
        # Neither PLK1 nor CDC20 nor their aliases appear near a verb.
        vote = lf_no_entity_overlap(stmt, ev)
        assert vote == RELATION_AXIS_CLASSES["no_relation"]

    def test_abstains_when_entities_present(self):
        stmt = Phosphorylation(_ag("PLK1"), _ag("CDC20"))
        ev = Evidence(text="Plk1 phosphorylates Cdc20 at S92.")
        # Both names appear within ±20 tokens of the verb. Use a record
        # whose names pass case-insensitive match.
        ev2 = Evidence(text="PLK1 phosphorylates CDC20 at S92.")
        assert lf_no_entity_overlap(stmt, ev2) == ABSTAIN


class TestPartnerDNALexical:
    def test_fires_on_complex_with_promoter(self):
        stmt = Complex([_ag("MYC"), _ag("MAX")])
        ev = Evidence(
            text="MYC binds the CCND1 promoter region in the chromatin."
        )
        assert lf_partner_dna_lexical(stmt, ev) == \
            RELATION_AXIS_CLASSES["direct_partner_mismatch"]

    def test_abstains_no_dna_element(self):
        stmt = Complex([_ag("MYC"), _ag("MAX")])
        ev = Evidence(text="MYC binds MAX directly.")
        assert lf_partner_dna_lexical(stmt, ev) == ABSTAIN

    def test_abstains_non_complex(self):
        stmt = Phosphorylation(_ag("MYC"), _ag("MAX"))
        ev = Evidence(text="MYC binds the MAX promoter.")
        assert lf_partner_dna_lexical(stmt, ev) == ABSTAIN


# ---------------------------------------------------------------------------
# subject_role / object_role LFs
# ---------------------------------------------------------------------------

class TestRolePosition:
    def test_position_subject_for_subject_role(self):
        stmt = Phosphorylation(_ag("PLK1"), _ag("CDC20"))
        ev = Evidence(text="Plk1 phosphorylates Cdc20.")
        assert lf_position_subject(stmt, ev, role="subject") == \
            ROLE_CLASSES["present_as_subject"]

    def test_position_subject_for_object_role_abstains(self):
        stmt = Phosphorylation(_ag("PLK1"), _ag("CDC20"))
        ev = Evidence(text="Plk1 phosphorylates Cdc20.")
        assert lf_position_subject(stmt, ev, role="object") == ABSTAIN

    def test_position_object_for_object_role(self):
        stmt = Phosphorylation(_ag("PLK1"), _ag("CDC20"))
        ev = Evidence(text="Plk1 phosphorylates Cdc20.")
        assert lf_position_object(stmt, ev, role="object") == \
            ROLE_CLASSES["present_as_object"]


class TestRoleSwap:
    def test_role_swap_subject_in_object_position(self):
        # Pattern: <claim_obj> <verb> <claim_subj>: "CDC20 is phosphorylated PLK1".
        # Note: regex requires obj before verb before subj.
        stmt = Phosphorylation(_ag("PLK1"), _ag("CDC20"))
        ev = Evidence(text="CDC20 binds PLK1 in mitosis.")
        # claim subject role: PLK1 in grammatical-object position → vote
        # `present_as_object`.
        assert lf_role_swap_lexical(stmt, ev, role="subject") == \
            ROLE_CLASSES["present_as_object"]

    def test_no_swap_abstains(self):
        stmt = Phosphorylation(_ag("PLK1"), _ag("CDC20"))
        ev = Evidence(text="PLK1 phosphorylates CDC20 in vitro.")
        assert lf_role_swap_lexical(stmt, ev, role="subject") == ABSTAIN


class TestChainPositionLexical:
    def test_via_pattern_fires(self):
        stmt = Phosphorylation(_ag("AKT1"), _ag("BAD"))
        ev = Evidence(text="Inhibition of growth occurs via AKT1 in many cells.")
        assert lf_chain_position_lexical(stmt, ev, role="subject") == \
            ROLE_CLASSES["present_as_mediator"]

    def test_no_chain_pattern_abstains(self):
        stmt = Phosphorylation(_ag("AKT1"), _ag("BAD"))
        ev = Evidence(text="AKT1 phosphorylates BAD.")
        assert lf_chain_position_lexical(stmt, ev, role="subject") == ABSTAIN


class TestDecoyLexical:
    def test_decoy_when_axis_mismatches(self):
        # Claim: Phosphorylation. Evidence: binding language only.
        stmt = Phosphorylation(_ag("PLK1"), _ag("CDC20"))
        ev = Evidence(text="PLK1 binds CDC20 in mitotic complexes.")
        # Only binding-axis observed; claim axis is modification → decoy.
        assert lf_decoy_lexical(stmt, ev, role="subject") == \
            ROLE_CLASSES["present_as_decoy"]

    def test_decoy_abstains_when_axis_matches(self):
        stmt = Phosphorylation(_ag("PLK1"), _ag("CDC20"))
        ev = Evidence(text="PLK1 phosphorylates CDC20 in cells.")
        assert lf_decoy_lexical(stmt, ev, role="subject") == ABSTAIN


class TestAbsentLFs:
    def test_no_grounded_match_fires_when_absent(self):
        stmt = Phosphorylation(_ag("ZZZNONEXISTENT"), _ag("CDC20"))
        ev = Evidence(text="MAPK phosphorylates ERK in cells.")
        # subject not in evidence; LF votes absent.
        assert lf_no_grounded_match(stmt, ev, role="subject") == \
            ROLE_CLASSES["absent"]

    def test_no_grounded_match_abstains_when_present(self):
        stmt = Phosphorylation(_ag("PLK1"), _ag("CDC20"))
        ev = Evidence(text="PLK1 phosphorylates CDC20.")
        assert lf_no_grounded_match(stmt, ev, role="subject") == ABSTAIN

    def test_absent_alias_check_fires_when_no_alias(self):
        stmt = Phosphorylation(_ag("ZZZNONEXISTENT"), _ag("CDC20"))
        ev = Evidence(text="MAPK phosphorylates ERK in cells.")
        assert lf_absent_alias_check(stmt, ev, role="subject") == \
            ROLE_CLASSES["absent"]


# ---------------------------------------------------------------------------
# scope LFs
# ---------------------------------------------------------------------------

class TestScopeLexical:
    def test_hedge_fires(self):
        stmt = Activation(_ag("CCR7"), _ag("AKT1"))
        ev = Evidence(text="CCR7 may activate AKT1 in T cells.")
        assert lf_hedge_lexical(stmt, ev) == SCOPE_CLASSES["hedged"]

    def test_hedge_abstains_no_cue(self):
        stmt = Phosphorylation(_ag("PLK1"), _ag("CDC20"))
        ev = Evidence(text="PLK1 phosphorylates CDC20 in mitosis.")
        assert lf_hedge_lexical(stmt, ev) == ABSTAIN

    def test_negation_fires(self):
        stmt = Activation(_ag("CCR7"), _ag("AKT1"))
        ev = Evidence(text="CCR7 did not activate AKT1 in our assay.")
        assert lf_negation_lexical(stmt, ev) == SCOPE_CLASSES["negated"]

    def test_negation_abstains_no_cue(self):
        stmt = Phosphorylation(_ag("PLK1"), _ag("CDC20"))
        ev = Evidence(text="PLK1 phosphorylates CDC20.")
        assert lf_negation_lexical(stmt, ev) == ABSTAIN

    def test_conditional_fires(self):
        stmt = Phosphorylation(_ag("PLK1"), _ag("CDC20"))
        ev = Evidence(text="In wild-type cells, PLK1 phosphorylates CDC20.")
        assert lf_conditional_lexical(stmt, ev) == \
            SCOPE_CLASSES["asserted_with_condition"]

    def test_conditional_abstains_no_pattern(self):
        stmt = Phosphorylation(_ag("PLK1"), _ag("CDC20"))
        ev = Evidence(text="PLK1 phosphorylates CDC20.")
        assert lf_conditional_lexical(stmt, ev) == ABSTAIN

    def test_clean_assertion_fires(self):
        stmt = Phosphorylation(_ag("PLK1"), _ag("CDC20"))
        ev = Evidence(text="PLK1 phosphorylates CDC20 directly in mitosis.")
        assert lf_clean_assertion(stmt, ev) == SCOPE_CLASSES["asserted"]

    def test_clean_assertion_abstains_with_hedge(self):
        stmt = Phosphorylation(_ag("PLK1"), _ag("CDC20"))
        ev = Evidence(text="PLK1 may phosphorylate CDC20 in mitosis.")
        assert lf_clean_assertion(stmt, ev) == ABSTAIN

    def test_low_information_too_few_tokens(self):
        stmt = Phosphorylation(_ag("PLK1"), _ag("CDC20"))
        ev = Evidence(text="PLK1 acts.")
        assert lf_low_information_evidence(stmt, ev) == SCOPE_CLASSES["abstain"]

    def test_low_information_boilerplate(self):
        stmt = Phosphorylation(_ag("PLK1"), _ag("CDC20"))
        ev = Evidence(text="This is consistent with PLK1 phosphorylating CDC20.")
        assert lf_low_information_evidence(stmt, ev) == SCOPE_CLASSES["abstain"]

    def test_low_information_long_clean_abstains(self):
        stmt = Phosphorylation(_ag("PLK1"), _ag("CDC20"))
        ev = Evidence(
            text="PLK1 phosphorylates CDC20 at serine 92 during mitotic "
                 "entry to drive metaphase progression in human cells."
        )
        assert lf_low_information_evidence(stmt, ev) == ABSTAIN

    def test_text_too_short_fires(self):
        stmt = Phosphorylation(_ag("PLK1"), _ag("CDC20"))
        ev = Evidence(text="PLK1 acts.")
        assert lf_text_too_short(stmt, ev) == SCOPE_CLASSES["abstain"]

    def test_text_too_short_abstains_at_threshold(self):
        stmt = Phosphorylation(_ag("PLK1"), _ag("CDC20"))
        ev = Evidence(text="PLK1 directly phosphorylates CDC20 at S92 in mitotic cells.")
        assert lf_text_too_short(stmt, ev) == ABSTAIN


# ---------------------------------------------------------------------------
# verify_grounding LFs
# ---------------------------------------------------------------------------

class TestGroundingLFs:
    def test_exact_symbol_fires(self):
        ent = {"name": "PLK1", "canonical": "PLK1"}
        ev = {"text": "PLK1 phosphorylates CDC20 in mitosis."}
        assert lf_gilda_exact_symbol(ent, ev) == GROUNDING_CLASSES["mentioned"]

    def test_exact_symbol_abstains_when_absent(self):
        ent = {"name": "PLK1", "canonical": "PLK1"}
        ev = {"text": "MAPK phosphorylates ERK in cells."}
        assert lf_gilda_exact_symbol(ent, ev) == ABSTAIN

    def test_evidence_contains_official_symbol_fires(self):
        ent = {"name": "PLK1"}
        ev = {"text": "PLK1 phosphorylates CDC20."}
        assert lf_evidence_contains_official_symbol(ent, ev) == \
            GROUNDING_CLASSES["mentioned"]

    def test_alias_fires(self):
        # PLK1 is also known as PLK; alias_match expects an alias entry.
        ent = {"name": "PLK1", "canonical": "PLK1",
               "all_names": ["PLK1", "polo-like kinase 1", "PLK"]}
        ev = {"text": "Polo-like kinase 1 phosphorylates CDC20."}
        # Note: "PLK1" appears? No — "Polo-like kinase 1" matches the
        # alias 'polo-like kinase 1'.
        assert lf_gilda_alias(ent, ev) == GROUNDING_CLASSES["equivalent"]

    def test_family_member_fires(self):
        ent = {"name": "MAPK", "is_family": True,
               "family_members": ["MAPK1", "MAPK3", "MAPK14"]}
        ev = {"text": "MAPK1 phosphorylates downstream substrates."}
        assert lf_gilda_family_member(ent, ev) == \
            GROUNDING_CLASSES["equivalent"]

    def test_family_abstains_when_no_member_present(self):
        ent = {"name": "MAPK", "is_family": True,
               "family_members": ["MAPK1", "MAPK3"]}
        ev = {"text": "PLK1 phosphorylates CDC20."}
        assert lf_gilda_family_member(ent, ev) == ABSTAIN

    def test_no_match_fires_when_absent(self):
        ent = {"name": "ZZZNONEXISTENT", "canonical": "ZZZNONEXISTENT"}
        ev = {"text": "MAPK phosphorylates ERK in cells."}
        assert lf_gilda_no_match(ent, ev) == GROUNDING_CLASSES["not_present"]

    def test_no_match_abstains_when_present(self):
        ent = {"name": "PLK1", "canonical": "PLK1"}
        ev = {"text": "PLK1 phosphorylates CDC20."}
        assert lf_gilda_no_match(ent, ev) == ABSTAIN

    def test_evidence_too_short_grounding_fires(self):
        ent = {"name": "PLK1"}
        ev = {"text": "PLK1 acts."}
        assert lf_evidence_too_short_grounding(ent, ev) == \
            GROUNDING_CLASSES["uncertain"]

    def test_evidence_long_enough_abstains(self):
        ent = {"name": "PLK1"}
        ev = {"text": "PLK1 phosphorylates CDC20 at serine 92 in mitotic cells."}
        assert lf_evidence_too_short_grounding(ent, ev) == ABSTAIN

    def test_ambiguous_grounding_returns_int_or_abstain(self):
        # This LF involves Gilda calls; we verify it never raises and
        # returns an int or ABSTAIN.
        ent = {"name": "PLK1"}
        ev = {"text": "MAPK and ERK regulate cell cycle progression."}
        out = lf_ambiguous_grounding(ent, ev)
        assert out in (ABSTAIN, GROUNDING_CLASSES["uncertain"])


# ---------------------------------------------------------------------------
# Convenience runners + LF index integrity
# ---------------------------------------------------------------------------

class TestLFIndex:
    def test_index_runs_without_error(self):
        stmt = Phosphorylation(_ag("PLK1"), _ag("CDC20"))
        ev = Evidence(text="PLK1 phosphorylates CDC20 at serine 92 in mitosis.")
        votes = all_clean_lf_votes(stmt, ev)
        for name, vote in votes.items():
            assert vote == ABSTAIN or vote >= 0, (name, vote)

    def test_grounding_runner(self):
        ent = {"name": "PLK1", "canonical": "PLK1"}
        ev = {"text": "PLK1 phosphorylates CDC20 at S92 in mitotic cells."}
        votes = all_clean_grounding_lf_votes(ent, ev)
        for name, vote in votes.items():
            assert vote == ABSTAIN or vote >= 0, (name, vote)

    def test_lf_index_count(self):
        # 11 (relation_axis) + 7 (subject_role) + 7 (object_role) +
        # 6 (scope) + 7 (verify_grounding) = 38 entries
        assert len(LF_INDEX_CLEAN) == 38

    def test_lf_index_kinds_are_valid(self):
        valid = {"relation_axis", "subject_role", "object_role",
                  "scope", "verify_grounding"}
        for kind, name, fn, kwargs in LF_INDEX_CLEAN:
            assert kind in valid, (kind, name)
            assert callable(fn), name
            assert isinstance(kwargs, dict), name


# ---------------------------------------------------------------------------
# Smoke: dict input round-trips
# ---------------------------------------------------------------------------

class TestDictInput:
    def test_dict_record_relation_axis(self):
        rec = {"stmt_type": "Phosphorylation", "subject": "PLK1",
               "object": "CDC20",
               "evidence_text": "PLK1 phosphorylates CDC20 at S92 in mitotic cells.",
               "source_api": "signor"}
        ev = {"text": rec["evidence_text"], "source_api": "signor"}
        assert lf_curated_db_axis(rec, ev) == \
            RELATION_AXIS_CLASSES["direct_sign_match"]

    def test_dict_record_role_with_kwarg(self):
        rec = {"stmt_type": "Phosphorylation", "subject": "PLK1",
               "object": "CDC20"}
        ev = {"text": "PLK1 phosphorylates CDC20 at S92."}
        assert lf_position_subject(rec, ev, role="subject") == \
            ROLE_CLASSES["present_as_subject"]
        assert lf_position_object(rec, ev, role="object") == \
            ROLE_CLASSES["present_as_object"]


# ---------------------------------------------------------------------------
# V6e regression tests — covers the four fix groups from V7a verdict.
# Per V7a empirical accuracy on U2 holdout: hedge/negation LFs scored
# 0.09-0.18, lf_gilda_alias 0.00 (328 fires), lf_ambiguous_grounding 0.00
# (295 fires), absence LFs 0.19-0.25. V6e fixes the bugs identified.
# ---------------------------------------------------------------------------

class TestV6eHedgeAnchoring:
    """Fix 1: hedge/negation LFs must anchor on the CLAIM stmt-type verb,
    not on any active verb in evidence."""

    def test_hedge_does_not_fire_when_cue_far_from_claim_verb(self):
        # Hedge cue ('may') sits in an unrelated clause, > 50 chars from
        # the claim's phosphorylation verb. Previous LF fired (FP) on
        # any hedge cue near any active verb; V6e must ABSTAIN.
        stmt = Phosphorylation(_ag("PLK1"), _ag("CDC20"))
        ev = Evidence(
            text="The expression of mitotic regulators may be context-"
                 "specific in different cell lines and tissue contexts. "
                 "Specifically PLK1 phosphorylates CDC20 directly at S92."
        )
        # 'may' is in the first sentence; claim verb 'phosphorylates'
        # is in the second sentence, ~80+ chars away. Outside ±50 win.
        assert lf_hedge_lexical(stmt, ev) == ABSTAIN

    def test_hedge_fires_when_cue_near_claim_verb(self):
        stmt = Phosphorylation(_ag("PLK1"), _ag("CDC20"))
        ev = Evidence(text="PLK1 may phosphorylate CDC20 in mitosis.")
        assert lf_hedge_lexical(stmt, ev) == SCOPE_CLASSES["hedged"]

    def test_negation_does_not_fire_when_cue_far_from_claim_verb(self):
        # Negation cue 'not' modifies 'in our hands' clause, not the
        # claim verb. V6e must abstain because the cue is > 50 chars
        # away from the claim verb cue.
        stmt = Phosphorylation(_ag("PLK1"), _ag("CDC20"))
        ev = Evidence(
            text="The downstream substrate is not directly detectable in "
                 "our experimental hands during the early time course "
                 "but PLK1 phosphorylates CDC20 at serine 92 in late "
                 "mitotic entry."
        )
        # 'not' at start; 'phosphorylates' ~80+ chars later.
        assert lf_negation_lexical(stmt, ev) == ABSTAIN

    def test_negation_fires_when_cue_near_claim_verb(self):
        stmt = Phosphorylation(_ag("PLK1"), _ag("CDC20"))
        ev = Evidence(text="PLK1 does not phosphorylate CDC20 in this assay.")
        assert lf_negation_lexical(stmt, ev) == SCOPE_CLASSES["negated"]

    def test_unknown_stmt_type_abstains(self):
        # Without a verb-cue mapping, the LF must abstain even if the
        # text contains hedge cues.
        rec = {"stmt_type": "UnknownType", "subject": "X", "object": "Y",
               "evidence_text": "X may interact with Y."}
        ev = {"text": rec["evidence_text"]}
        assert lf_hedge_lexical(rec, ev) == ABSTAIN


class TestV6eGildaAlias:
    """Fix 2: lf_gilda_alias must verify alias text grounds back to the
    claim entity, not vote `equivalent` on every alias mention."""

    def test_alias_pointing_to_different_entity_abstains(self):
        # Construct an entity claim for one HGNC symbol, but the
        # `aliases` list contains an alias that Gilda would resolve to a
        # different entity. The LF must NOT vote equivalent.
        # Use an alias that grounds elsewhere: 'p53' → TP53, but here
        # claim entity is PLK1 and we synthetically inject 'p53' as if
        # it were in PLK1's alias list.
        ent = {"name": "PLK1", "canonical": "PLK1",
               "db": "HGNC", "db_id": "9077",
               "all_names": ["PLK1", "p53"]}
        # Evidence mentions p53; under the OLD LF this voted
        # `equivalent`. Under V6e it must abstain because gilda
        # grounds 'p53' to TP53, not PLK1.
        ev = {"text": "p53 regulates downstream apoptotic pathways."}
        assert lf_gilda_alias(ent, ev) == ABSTAIN

    def test_alias_pointing_to_claim_entity_fires(self):
        # 'STPK13' is a real PLK1 alias (verified via gilda — in
        # GroundedEntity.resolve('PLK1').all_names). Both forms are
        # valid synonyms.
        ent = {"name": "PLK1", "canonical": "PLK1",
               "db": "HGNC", "db_id": "9077",
               "all_names": ["PLK1", "STPK13"]}
        ev = {"text": "STPK13 phosphorylates downstream substrates."}
        # Gilda should ground 'STPK13' → PLK1 (same HGNC id).
        assert lf_gilda_alias(ent, ev) == GROUNDING_CLASSES["equivalent"]

    def test_official_symbol_in_aliases_skipped(self):
        # If the only matching alias IS the official symbol, the LF
        # should abstain (lf_gilda_exact_symbol covers that case).
        ent = {"name": "PLK1", "canonical": "PLK1",
               "all_names": ["PLK1"]}
        ev = {"text": "PLK1 phosphorylates CDC20."}
        assert lf_gilda_alias(ent, ev) == ABSTAIN


class TestV6eAmbiguousGrounding:
    """Fix 3: lf_ambiguous_grounding must use Gilda's own scoring on the
    CLAIM entity name, not invent ambiguity from co-occurring symbols."""

    def test_unambiguous_grounding_abstains(self):
        # PLK1 grounds unambiguously in Gilda — top-1 is PLK1 with a
        # comfortable score gap. The LF must abstain.
        ent = {"name": "PLK1"}
        ev = {"text": "PLK1 and CDC20 phosphorylate downstream substrates."}
        assert lf_ambiguous_grounding(ent, ev) == ABSTAIN

    def test_no_text_abstains(self):
        ent = {"name": "PLK1"}
        ev = {"text": ""}
        assert lf_ambiguous_grounding(ent, ev) == ABSTAIN

    def test_non_ambiguous_with_unrelated_caps_abstains(self):
        # Previous LF fired uncertain whenever uppercase tokens with
        # similar Gilda scores appeared (e.g., 'CDK', 'PLK', 'AKT').
        # V6e doesn't look at evidence tokens for ambiguity — it looks
        # at the claim entity's own grounding.
        ent = {"name": "PLK1"}
        ev = {"text": "MAPK and ERK regulate cell cycle progression."}
        assert lf_ambiguous_grounding(ent, ev) == ABSTAIN


class TestV6eAbsenceLFs:
    """Fix 4: absence LFs must use broadened alias coverage (HGNC
    all_names + UniProt synonyms) and tolerant case/hyphen matching."""

    def test_no_grounded_match_with_hyphen_variant_abstains(self):
        # Claim is PLK1; evidence references it as 'Plk-1' (hyphen
        # variant). Strict word-boundary regex misses this — but it's
        # a known PLK1 alias. Tolerant match should detect it.
        stmt = Phosphorylation(_ag("PLK1"), _ag("CDC20"))
        ev = Evidence(text="Plk-1 protein phosphorylates targets in mitosis.")
        assert lf_no_grounded_match(stmt, ev, role="subject") == ABSTAIN

    def test_no_grounded_match_with_truly_absent_entity_fires(self):
        stmt = Phosphorylation(_ag("ZZZNONEXISTENT"), _ag("CDC20"))
        ev = Evidence(text="MAPK phosphorylates ERK in cells.")
        assert lf_no_grounded_match(stmt, ev, role="subject") == \
            ROLE_CLASSES["absent"]

    def test_absent_alias_check_alias_form_abstains(self):
        # 'Plk1' (mixed case) is in PLK1's aliases — strict match
        # would catch it (case-insensitive), but ensure the broadened
        # path works with a less-common variant: 'STPK13' (PLK1 alias).
        stmt = Phosphorylation(_ag("PLK1"), _ag("CDC20"))
        ev = Evidence(text="STPK13 phosphorylates the CDC20 substrate.")
        assert lf_absent_alias_check(stmt, ev, role="subject") == ABSTAIN

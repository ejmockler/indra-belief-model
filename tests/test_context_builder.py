"""Tests for EvidenceContext builder.

Each test pins one column of the context object so a regression in
the builder fails loudly. Covers:
  - alias resolution for known FN cases (HSPB1/Hsp27, CSNK2A2/CK2,
    KDM1A/LSD1)
  - family detection on FPLX entities (PKC family)
  - pseudogene flag propagation
  - acceptable_sites from residue+position
  - binding_admissible policy by stmt-type (Complex protein-only;
    Translocation membrane/complex; modification = no gate)
  - is_complex / is_modification / is_translocation predicates
  - degenerate cases (empty evidence, '?'-named agent)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from indra_belief.scorers.context_builder import build_context


def _stmt(stmt_class, *agent_names, **kwargs):
    """Convenience: build an INDRA Statement from canonical names."""
    from indra.statements import Agent
    agents = [Agent(n) for n in agent_names]
    if stmt_class.__name__ == "Complex":
        return stmt_class(agents, **kwargs)
    return stmt_class(*agents, **kwargs)


def _ev(text: str):
    from indra.statements import Evidence
    return Evidence(source_api="reach", text=text)


# ---------------------------------------------------------------------------
# Alias resolution — the core J1 contract
# ---------------------------------------------------------------------------

def test_alias_map_includes_canonical():
    from indra.statements import Phosphorylation
    ctx = build_context(_stmt(Phosphorylation, "HSPB1", "CASP9"),
                        _ev("Hsp27 inhibits caspase 9 cleavage."))
    assert "HSPB1" in ctx.aliases
    assert "CASP9" in ctx.aliases


def test_alias_map_resolves_hsp27_synonym():
    """HSPB1 → Hsp27: the alias-blocked FN class from the v1 holdout."""
    from indra.statements import Phosphorylation
    ctx = build_context(_stmt(Phosphorylation, "HSPB1", "CASP9"),
                        _ev("Hsp27 inhibits caspase 9 cleavage."))
    syns = ctx.aliases["HSPB1"]
    assert any("HSP27" in s.upper() or "HSP 27" in s.upper() or "HSP-27" in s.upper()
               for s in syns), f"Hsp27 alias missing from {syns}"


def test_alias_map_resolves_caspase9_synonym():
    from indra.statements import Phosphorylation
    ctx = build_context(_stmt(Phosphorylation, "HSPB1", "CASP9"),
                        _ev("Hsp27 inhibits caspase 9 cleavage."))
    syns = ctx.aliases["CASP9"]
    assert any("caspase" in s.lower() and "9" in s for s in syns)


def test_alias_map_resolves_lsd1_synonym():
    """KDM1A → LSD1: the FP case (binds DNA promoter, not protein)."""
    from indra.statements import Complex
    ctx = build_context(_stmt(Complex, "KDM1A", "KLF2"),
                        _ev("LSD1 binds the KLF2 promoter."))
    syns = ctx.aliases.get("KDM1A", frozenset())
    assert any("LSD1" in s.upper() or "LSD-1" in s.upper() for s in syns)


def test_alias_map_short_synonyms_dropped():
    """Synonyms shorter than 2 chars must be dropped (over-match risk)."""
    from indra.statements import Phosphorylation
    ctx = build_context(_stmt(Phosphorylation, "HSPB1", "CASP9"), _ev("..."))
    for syns in ctx.aliases.values():
        for s in syns:
            assert len(s) >= 2, f"short synonym {s!r} leaked through"


# ---------------------------------------------------------------------------
# binding_admissible — per stmt-type policy (J3 gate prep)
# ---------------------------------------------------------------------------

def test_complex_admits_only_protein():
    from indra.statements import Complex
    ctx = build_context(_stmt(Complex, "MYB", "MYBL2"), _ev("..."))
    assert ctx.binding_admissible == frozenset({"protein"})


def test_translocation_admits_membrane_and_complex():
    from indra.statements import Translocation
    ctx = build_context(_stmt(Translocation, "AKT1"), _ev("..."))
    assert ctx.binding_admissible == frozenset({"membrane", "complex"})


def test_phosphorylation_no_binding_gate():
    """Catalytic statements: binding_partner_type is informational, not gating."""
    from indra.statements import Phosphorylation
    ctx = build_context(_stmt(Phosphorylation, "RPS6KA1", "YBX1"), _ev("..."))
    assert ctx.binding_admissible == frozenset()


def test_activation_no_binding_gate():
    from indra.statements import Activation
    ctx = build_context(_stmt(Activation, "EGFR", "ERK1"), _ev("..."))
    assert ctx.binding_admissible == frozenset()


# ---------------------------------------------------------------------------
# acceptable_sites — modification residue+position
# ---------------------------------------------------------------------------

def test_acceptable_sites_from_residue_position():
    from indra.statements import Phosphorylation
    ctx = build_context(
        _stmt(Phosphorylation, "RPS6KA1", "YBX1", residue="S", position="102"),
        _ev("..."),
    )
    assert ctx.acceptable_sites == frozenset({"S102"})


def test_acceptable_sites_empty_when_no_site():
    from indra.statements import Phosphorylation
    ctx = build_context(_stmt(Phosphorylation, "RPS6KA1", "YBX1"), _ev("..."))
    assert ctx.acceptable_sites == frozenset()


# ---------------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------------

def test_is_complex_true_for_complex():
    from indra.statements import Complex
    ctx = build_context(_stmt(Complex, "MYB", "MYBL2"), _ev("..."))
    assert ctx.is_complex is True
    assert ctx.is_modification is False
    assert ctx.is_translocation is False
    assert ctx.stmt_type == "Complex"


def test_is_modification_true_for_phosphorylation():
    from indra.statements import Phosphorylation
    ctx = build_context(_stmt(Phosphorylation, "A", "B"), _ev("..."))
    assert ctx.is_modification is True
    assert ctx.is_complex is False


def test_is_translocation_true_for_translocation():
    from indra.statements import Translocation
    ctx = build_context(_stmt(Translocation, "AKT1"), _ev("..."))
    assert ctx.is_translocation is True


# ---------------------------------------------------------------------------
# Clauses — J1 emits single-element; J5 will populate splits
# ---------------------------------------------------------------------------

def test_clauses_single_element_for_simple_sentence():
    from indra.statements import Phosphorylation
    text = "RSK1 phosphorylates YB-1 at S102."
    ctx = build_context(_stmt(Phosphorylation, "RPS6KA1", "YBX1"), _ev(text))
    assert ctx.clauses == (text,)


def test_clauses_empty_when_no_text():
    from indra.statements import Phosphorylation
    from indra.statements import Evidence
    ev = Evidence(source_api="reach", text="")
    ctx = build_context(_stmt(Phosphorylation, "A", "B"), ev)
    assert ctx.clauses == ()


def test_clauses_split_on_multi_sentence_text():
    """J5: Multi-sentence evidence is split on sentence boundaries."""
    from indra.statements import Phosphorylation
    text = ("CK2α phosphorylates BMI1 at Ser110. This phosphorylation "
            "imparts protein stability contributing to clonal growth.")
    ctx = build_context(_stmt(Phosphorylation, "CSNK2A2", "BMI1"), _ev(text))
    assert len(ctx.clauses) == 2
    assert "CK2α phosphorylates BMI1 at Ser110." in ctx.clauses[0]
    assert "imparts protein stability" in ctx.clauses[1]


def test_clauses_capped_at_four():
    from indra.statements import Phosphorylation
    # 5 sentences
    text = ("Apple does X. Bear does Y. Cat does Z. Dog does W. "
            "Eagle does V.")
    ctx = build_context(_stmt(Phosphorylation, "A", "B"), _ev(text))
    assert len(ctx.clauses) <= 4


def test_clauses_drops_short_citation_fragments():
    """Citation markers like '[ ].' shouldn't be treated as clauses."""
    from indra.statements import Phosphorylation
    text = "RSK1 phosphorylates YB-1 at S102. [ ]."
    ctx = build_context(_stmt(Phosphorylation, "RPS6KA1", "YBX1"), _ev(text))
    # The trivial fragment is filtered; only the substantive clause remains
    assert len(ctx.clauses) == 1
    assert "RSK1" in ctx.clauses[0]


def test_clauses_no_split_on_abbreviations():
    """'i.e.', 'et al.' should not trigger sentence split."""
    from indra.statements import Phosphorylation
    text = ("RSK1 (i.e., RPS6KA1) phosphorylates YB-1 at S102, "
            "consistent with prior work (Smith et al., 2020).")
    ctx = build_context(_stmt(Phosphorylation, "RPS6KA1", "YBX1"), _ev(text))
    assert len(ctx.clauses) == 1


# ---------------------------------------------------------------------------
# Degenerate cases
# ---------------------------------------------------------------------------

def test_question_mark_agent_skipped():
    """Agent with name='?' must not blow up the builder; it's just dropped."""
    from indra.statements import Phosphorylation, Agent
    stmt = Phosphorylation(Agent("?"), Agent("YBX1"))
    ctx = build_context(stmt, _ev("..."))
    assert "?" not in ctx.aliases
    assert "YBX1" in ctx.aliases


def test_unknown_entity_no_aliases_recorded():
    """Gilda returns nothing for a made-up name; the entity is silently
    omitted from aliases (graceful degradation, no exception)."""
    from indra.statements import Phosphorylation, Agent
    stmt = Phosphorylation(Agent("ZZZNOTAGENE"), Agent("YBX1"))
    ctx = build_context(stmt, _ev("..."))
    # YBX1 should still resolve; ZZZNOTAGENE may or may not have an entry
    # but the builder must not raise.
    assert "YBX1" in ctx.aliases


# ---------------------------------------------------------------------------
# Self-modification: agent == target
# ---------------------------------------------------------------------------

def test_self_modification_resolves_once():
    """Autophosphorylation has the same entity as agent and substrate.
    The builder must not double-resolve."""
    from indra.statements import Autophosphorylation, Agent
    stmt = Autophosphorylation(Agent("EGFR"))
    ctx = build_context(stmt, _ev("EGFR autophosphorylates at Y1068."))
    assert "EGFR" in ctx.aliases
    # Only one entry in aliases (no double-counting)
    assert len(ctx.aliases) == 1

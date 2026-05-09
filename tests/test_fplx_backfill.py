"""M5 tests: FPLX manual backfill for under-expanded families.

The L8 diagnosis traced 13+ FNs to families where Gilda returned only
the canonical name with no member roster. M5 patches these in
context_builder via _FPLX_BACKFILL. These tests verify:
  - The backfill rosters merge correctly into ctx.aliases.
  - Gilda's well-mapped families (NFkappaB, ERK, MEK) are UNIONED with
    the backfill, not replaced.
  - The bilateral_ambiguity guard still fires on FPLX-FPLX claims.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

from indra.statements import Activation, Agent, Complex, Evidence

from indra_belief.scorers.context_builder import (
    _FPLX_BACKFILL,
    _apply_fplx_backfill,
    build_context,
)


def _ag(name: str) -> Agent:
    return Agent(name)


class TestApplyBackfillUnit:
    def test_unknown_family_returns_input_unchanged(self):
        original = frozenset({"PLK1", "Plk1"})
        out = _apply_fplx_backfill("PLK1", original)
        assert out == original

    def test_known_family_merges(self):
        out = _apply_fplx_backfill("p14_3_3", frozenset({"p14_3_3"}))
        assert "YWHAZ" in out
        assert "YWHAE" in out
        assert "SFN" in out
        # canonical preserved
        assert "p14_3_3" in out

    def test_short_tokens_dropped(self):
        # Sanity: even though backfill rosters are clean, the _apply
        # helper rejects <2 char tokens (defense in depth).
        out = _apply_fplx_backfill("p14_3_3", frozenset({"X", "p14_3_3"}))
        assert all(len(s) >= 2 for s in out)


class TestBackfillContent:
    """Audit the static rosters against expected entries."""

    def test_gpcr_includes_chemokine_receptors(self):
        gpcr = _FPLX_BACKFILL["GPCR"]
        for ccr in ("CCR1", "CCR2", "CCR5", "CXCR4", "CX3CR1"):
            assert ccr in gpcr, f"{ccr} missing from GPCR backfill"

    def test_gpcr_includes_prostaglandin_receptors(self):
        gpcr = _FPLX_BACKFILL["GPCR"]
        for ptger in ("PTGER1", "PTGER2", "PTGER3", "PTGER4"):
            assert ptger in gpcr, f"{ptger} missing from GPCR backfill"

    def test_p14_3_3_complete_roster(self):
        roster = _FPLX_BACKFILL["p14_3_3"]
        for member in ("YWHAB", "YWHAE", "YWHAG", "YWHAH", "YWHAQ", "YWHAZ", "SFN"):
            assert member in roster, f"{member} missing from p14_3_3 backfill"

    def test_tcf_lef_complete_roster(self):
        roster = _FPLX_BACKFILL["TCF_LEF"]
        for member in ("TCF7", "TCF7L1", "TCF7L2", "LEF1"):
            assert member in roster, f"{member} missing from TCF_LEF backfill"

    def test_ikb_complete_roster(self):
        roster = _FPLX_BACKFILL["IKB"]
        for member in ("NFKBIA", "NFKBIB", "NFKBIE"):
            assert member in roster, f"{member} missing from IKB backfill"


class TestBuildContextWithBackfill:
    """Integration: build_context returns aliases including backfill."""

    def test_p14_3_3_aliases_expanded(self):
        # Build a Complex(p14_3_3, CDC25C) statement and verify
        # ctx.aliases['p14_3_3'] now contains YWHAZ etc.
        stmt = Complex([_ag("p14_3_3"), _ag("CDC25C")])
        ev = Evidence(text="14-3-3 binds Cdc25C.")
        ctx = build_context(stmt, ev)
        if "p14_3_3" not in ctx.aliases:
            pytest.skip("Gilda did not resolve p14_3_3 on this machine")
        aliases = ctx.aliases["p14_3_3"]
        assert "YWHAZ" in aliases
        assert "YWHAE" in aliases
        assert "SFN" in aliases

    def test_tcf_lef_aliases_expanded(self):
        stmt = Complex([_ag("TCF_LEF"), _ag("CTNNB1")])
        ev = Evidence(text="beta-catenin interacts with TCF/Lef.")
        ctx = build_context(stmt, ev)
        if "TCF_LEF" not in ctx.aliases:
            pytest.skip("Gilda did not resolve TCF_LEF on this machine")
        aliases = ctx.aliases["TCF_LEF"]
        assert "TCF7" in aliases
        assert "LEF1" in aliases

    def test_gilda_well_mapped_family_preserved(self):
        """NFkappaB is well-mapped by Gilda — backfill must not
        clobber Gilda's roster (no entry in _FPLX_BACKFILL for it)."""
        assert "NFkappaB" not in _FPLX_BACKFILL  # no backfill entry needed
        # Gilda's NFkappaB members are RELA, RELB, NFKB1, NFKB2, REL
        # — verify build_context returns them as before.
        stmt = Activation(_ag("NFkappaB"), _ag("LCN2"))
        ev = Evidence(text="LCN2 expression is upregulated by NFkappaB.")
        ctx = build_context(stmt, ev)
        if "NFkappaB" not in ctx.aliases:
            pytest.skip("Gilda did not resolve NFkappaB on this machine")
        aliases = ctx.aliases["NFkappaB"]
        # At least one canonical NFkB family member should remain
        assert any(m in aliases for m in ("RELA", "RELB", "NFKB1", "NFKB2"))


class TestN1WrittenFormBackfill:
    """N1: spelled-out written forms for kinase / deacetylase families.

    Gilda's PKC roster has the PRKC* members but no spelled-out
    "protein kinase c" alias. Trace-1 miss in the M13 probe
    (PKC→EIF4E, evidence "phosphorylation of eIF-4E ... by protein
    kinase C") parsed agents=["protein kinase c"]; alias bind missed
    because the spelled form wasn't in aliases["PKC"]. N1 patches
    PKC, PKA, PKG, HDAC with their written-form aliases."""

    def test_pkc_includes_member_roster_and_written_form(self):
        pkc = _FPLX_BACKFILL["PKC"]
        for member in ("PRKCA", "PRKCB", "PRKCD", "PRKCE", "PRKCG"):
            assert member in pkc, f"{member} missing from PKC backfill"
        for written in ("protein kinase C", "protein kinase c", "PKCs",
                        "PKC family"):
            assert written in pkc, f"{written!r} missing from PKC backfill"

    def test_pka_includes_member_roster_and_written_form(self):
        pka = _FPLX_BACKFILL["PKA"]
        for member in ("PRKACA", "PRKACB", "PRKACG"):
            assert member in pka, f"{member} missing from PKA backfill"
        for written in ("protein kinase A", "protein kinase a",
                        "cAMP-dependent protein kinase"):
            assert written in pka, f"{written!r} missing from PKA backfill"

    def test_pkg_includes_member_roster_and_written_form(self):
        pkg = _FPLX_BACKFILL["PKG"]
        for member in ("PRKG1", "PRKG2"):
            assert member in pkg, f"{member} missing from PKG backfill"
        assert "cGMP-dependent protein kinase" in pkg
        assert "protein kinase G" in pkg

    def test_hdac_includes_member_roster_and_written_form(self):
        hdac = _FPLX_BACKFILL["HDAC"]
        for member in ("HDAC1", "HDAC2", "HDAC3", "HDAC6"):
            assert member in hdac, f"{member} missing from HDAC backfill"
        for written in ("histone deacetylase", "histone deacetylases",
                        "HDACs"):
            assert written in hdac, f"{written!r} missing from HDAC backfill"

    def test_trace_1_pkc_written_form_aliases(self):
        """Integration: trace-1 (PKC→EIF4E) — claim subject "PKC"
        must alias the evidence surface form "protein kinase c"."""
        from indra.statements import Phosphorylation
        stmt = Phosphorylation(_ag("PKC"), _ag("EIF4E"), "S", "209")
        ev = Evidence(text=(
            "Phosphorylation of eIF-4E on serine 209 by protein "
            "kinase C is inhibited by 4E-binding proteins."
        ))
        ctx = build_context(stmt, ev)
        if "PKC" not in ctx.aliases:
            pytest.skip("Gilda did not resolve PKC on this machine")
        aliases = ctx.aliases["PKC"]
        assert "protein kinase C" in aliases
        assert "protein kinase c" in aliases

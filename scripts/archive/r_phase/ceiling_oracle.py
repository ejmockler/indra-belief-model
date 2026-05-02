"""Ceiling oracle: given hand-crafted *ideal* sub-call commitments, what
verdict does the adjudicator produce?

Different from `ceiling_analysis.py` (parametric per-step reliability).
This script isolates the adjudicator architecture: every input is
hand-crafted to reflect what perfect parse_evidence + verify_grounding
SHOULD have produced for each record. The question:

    Given perfect upstream sub-calls, does the adjudicator emit the
    gold-matching verdict?

If yes → the architecture is sound; investment goes into sub-call
quality (parse_evidence robustness, grounder calibration, hedging
extraction). If no → architecture has a structural ceiling and the
sub-call work is wasted effort.

The fixtures below cover the 8 decomposed losses from the v5 dual-run
(n=24). Each entry includes:
  - record id (matches the dual-run log line number)
  - claim (deterministic from parse_claim — known-good)
  - ideal evidence commitment (what a perfect parse_evidence would emit)
  - ideal groundings (what perfect verify_grounding would emit)
  - gold polarity (what the verdict should be)
  - notes (which sub-call investment is responsible)

Plus 2 spot-check wins to verify that perfect sub-calls don't silently
flip a currently-correct verdict to incorrect.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from indra_belief.scorers.adjudicate import adjudicate
from indra_belief.scorers.commitments import (
    ClaimCommitment,
    EvidenceAssertion,
    EvidenceCommitment,
    GroundingVerdict,
)


@dataclass
class OracleCase:
    rid: int
    label: str           # 'loss' | 'win-control'
    claim: ClaimCommitment
    evidence: EvidenceCommitment | None
    groundings: tuple[GroundingVerdict, ...]
    aliases: dict[str, frozenset[str]]
    gold_polarity: str   # 'correct' | 'incorrect'
    invest: str          # which sub-call investment fixes this
    note: str


def _gv(name: str, status: str) -> GroundingVerdict:
    return GroundingVerdict(claim_entity=name, status=status)


def _ev(*assertions: EvidenceAssertion) -> EvidenceCommitment:
    agents = tuple({n for a in assertions for n in a.agents})
    targets = tuple({n for a in assertions for n in a.targets})
    return EvidenceCommitment(
        agent_candidates=agents,
        target_candidates=targets,
        bystanders=(),
        assertions=assertions,
    )


CASES: list[OracleCase] = [

    # === DEC LOSSES ===

    # [#1] PAPPA→ABCG1 IncreaseAmount, gold=polarity (claim is wrong: PAPPA
    # actually DECREASES ABCG1, evidence sign is negative).
    # Currently dec ABSTAINS (parse_evidence failed on long sentence).
    # Perfect parse: extract assertion(amount, sign=negative, agents={PAPPA},
    # targets={ABCG1}, perturbation="none"). Adjudicator sees axis match but
    # opposite sign → sign_mismatch → incorrect. Matches gold=incorrect.
    OracleCase(
        rid=1, label="loss",
        claim=ClaimCommitment(
            stmt_type="IncreaseAmount", subject="PAPPA", objects=("ABCG1",),
            axis="amount", sign="positive",
        ),
        evidence=_ev(
            EvidenceAssertion(
                agents=("PAPPA", "PAPP-A"), targets=("ABCG1",),
                axis="amount", sign="negative", perturbation="none",
            ),
        ),
        groundings=(_gv("PAPPA", "mentioned"), _gv("ABCG1", "mentioned")),
        aliases={"PAPPA": frozenset({"PAPPA", "PAPP-A"}),
                 "ABCG1": frozenset({"ABCG1"})},
        gold_polarity="incorrect",
        invest="parse_evidence (long-sentence robustness)",
        note="Currently abstains on parse failure; perfect parse yields sign_mismatch.",
    ),

    # [#3] FYN/ABL1 Complex, gold=correct.
    # Evidence: "Dab1 interacts with Fyn and Fe65 interacts with Abl"
    # The evidence describes TWO SEPARATE binary interactions. Neither asserts
    # a direct FYN-ABL1 complex. Gold curator's interpretation seems loose
    # (treating shared pathway as Complex). Perfect parse: two binding
    # assertions, neither has both FYN and ABL1.
    # Adjudicator: no matching binding → absent_relationship → incorrect.
    # Does NOT match gold=correct. Verdict: this is a gold-loose case, not
    # an architecture failure.
    OracleCase(
        rid=3, label="loss",
        claim=ClaimCommitment(
            stmt_type="Complex", subject="FYN", objects=("ABL1",),
            axis="binding", sign="neutral",
        ),
        evidence=_ev(
            EvidenceAssertion(
                agents=("Dab1",), targets=("Fyn",),
                axis="binding", sign="neutral",
            ),
            EvidenceAssertion(
                agents=("Fe65",), targets=("Abl",),
                axis="binding", sign="neutral",
            ),
        ),
        groundings=(_gv("FYN", "mentioned"), _gv("ABL1", "mentioned")),
        aliases={"FYN": frozenset({"FYN", "Fyn"}),
                 "ABL1": frozenset({"ABL1", "Abl"})},
        gold_polarity="correct",
        invest="(none — gold is loose; evidence describes two separate interactions)",
        note="Even with perfect sub-calls dec verdict is incorrect; gold is the issue.",
    ),

    # [#5] TNFRSF14/CD27 Complex, gold=no_relation.
    # Evidence describes CD27 binding TRAFs and HVEM (TNFRSF14) being a
    # separate receptor. NO direct CD27↔TNFRSF14 binding asserted.
    # Perfect parse: assertions about CD27↔TRAF, HVEM separately mentioned.
    # Adjudicator: no Complex(TNFRSF14, CD27) binding → absent_relationship
    # → incorrect. Matches gold (no_relation = incorrect).
    OracleCase(
        rid=5, label="loss",
        claim=ClaimCommitment(
            stmt_type="Complex", subject="TNFRSF14", objects=("CD27",),
            axis="binding", sign="neutral",
        ),
        evidence=_ev(
            EvidenceAssertion(
                agents=("CD27",), targets=("TRAF2", "TRAF3", "TRAF5"),
                axis="binding", sign="neutral",
            ),
            EvidenceAssertion(
                agents=("4-1BB",), targets=("TRAF1", "TRAF2", "TRAF3"),
                axis="binding", sign="neutral",
            ),
        ),
        groundings=(_gv("TNFRSF14", "mentioned"), _gv("CD27", "mentioned")),
        aliases={"TNFRSF14": frozenset({"TNFRSF14", "HVEM"}),
                 "CD27": frozenset({"CD27"})},
        gold_polarity="incorrect",
        invest="parse_evidence (long-sentence robustness)",
        note="Currently abstains on parse failure; perfect parse yields absent_relationship.",
    ),

    # [#6] HSPB1→CASP9 Inhibition, gold=correct.
    # Evidence: "Hsp27 activation... is required for CD133+ cells to inhibit
    # caspase 9 cleavage." HSPB1=Hsp27. This is an indirect-chain pattern:
    # Hsp27 enables CD133+ cells to inhibit CASP9.
    # The strict parse extracts the literal asserted relationship: assertion
    # has CD133+ cells inhibiting CASP9, with Hsp27 as a required upstream.
    # The "perfect" parse for this case would need parse_evidence to ALSO
    # emit an indirect-chain assertion (Hsp27 → CASP9, indirect). That's an
    # extension beyond what the current schema supports without an
    # `indirect_chain` flag on EvidenceAssertion.
    # Modeled here: parse extracts BOTH literal (CD133+→CASP9) AND a
    # pass-through assertion (HSPB1→CASP9). Adjudicator: matching binding
    # found → match → correct.
    OracleCase(
        rid=6, label="loss",
        claim=ClaimCommitment(
            stmt_type="Inhibition", subject="HSPB1", objects=("CASP9",),
            axis="activity", sign="negative",
        ),
        evidence=_ev(
            EvidenceAssertion(
                agents=("HSPB1", "Hsp27"), targets=("CASP9", "caspase 9"),
                axis="activity", sign="negative", perturbation="none",
            ),
        ),
        groundings=(_gv("HSPB1", "mentioned"), _gv("CASP9", "mentioned")),
        aliases={"HSPB1": frozenset({"HSPB1", "Hsp27"}),
                 "CASP9": frozenset({"CASP9", "caspase 9"})},
        gold_polarity="correct",
        invest="parse_evidence (indirect-chain extraction policy)",
        note="Indirect chain HSP27→CD133+→CASP9; perfect parse should extract pass-through.",
    ),

    # [#7] NAT8/PROM1 Complex, gold=hypothesis.
    # Evidence: "we turned to MYTH assay to gain evidence for an interaction
    # between CD133 (PROM1) and ATase1/ATase2 (NAT8)."
    # Hedged: they're SEEKING evidence, not asserting. Task #52 ships the
    # `hedged` field on EvidenceAssertion + parse_evidence prompt rule;
    # adjudicator emits hedging_hypothesis on a matching hedged assertion.
    # Verdict=incorrect with structured attribution.
    OracleCase(
        rid=7, label="loss",
        claim=ClaimCommitment(
            stmt_type="Complex", subject="NAT8", objects=("PROM1",),
            axis="binding", sign="neutral",
        ),
        evidence=_ev(
            EvidenceAssertion(
                agents=("CD133", "PROM1"), targets=("ATase1", "ATase2", "NAT8"),
                axis="binding", sign="neutral", hedged=True,
            ),
        ),
        groundings=(_gv("NAT8", "mentioned"), _gv("PROM1", "mentioned")),
        aliases={"NAT8": frozenset({"NAT8", "ATase1", "ATase2"}),
                 "PROM1": frozenset({"PROM1", "CD133"})},
        gold_polarity="incorrect",
        invest="parse_evidence (hedge detection) — task #52",
        note="Perfect parse marks hedged=True; adjudicator emits hedging_hypothesis (#52 wired).",
    ),

    # [#9] GEMIN4/PTGS2 Deubiquitination, gold=no_relation.
    # Evidence: "suppression of p97 increased ubiquitination of COX-2"
    # COX-2=PTGS2, p97 is a different gene. GEMIN4 was likely resolved via
    # a low-confidence alias. Perfect grounder: GEMIN4 → not_present.
    # Adjudicator: grounding gate hits not_present → grounding_gap →
    # incorrect. Matches gold=incorrect.
    OracleCase(
        rid=9, label="loss",
        claim=ClaimCommitment(
            stmt_type="Deubiquitination", subject="GEMIN4", objects=("PTGS2",),
            axis="modification", sign="negative",
        ),
        evidence=_ev(
            EvidenceAssertion(
                agents=("p97",), targets=("COX-2", "PTGS2"),
                axis="modification", sign="positive",
                perturbation="loss_of_function",  # "suppression of p97"
            ),
        ),
        groundings=(_gv("GEMIN4", "not_present"), _gv("PTGS2", "mentioned")),
        aliases={"GEMIN4": frozenset({"GEMIN4"}),
                 "PTGS2": frozenset({"PTGS2", "COX-2"})},
        gold_polarity="incorrect",
        invest="grounder calibration (ambiguous-alias)",
        note="Perfect grounder flags GEMIN4 not_present; adjudicator emits grounding_gap.",
    ),

    # [#18] Histone/JDP2 Complex, gold=entity_boundaries.
    # Evidence: "JDP2... recruits a histone deacetylase 3 complex"
    # "Histone" is a generic family term. "Histone deacetylase" is HDAC, not
    # a histone protein. Perfect grounder: "Histone" → not_present.
    # Adjudicator: grounding_gap → incorrect.
    OracleCase(
        rid=18, label="loss",
        claim=ClaimCommitment(
            stmt_type="Complex", subject="Histone", objects=("JDP2",),
            axis="binding", sign="neutral",
        ),
        evidence=_ev(
            EvidenceAssertion(
                agents=("JDP2",), targets=("histone deacetylase 3", "HDAC3"),
                axis="binding", sign="neutral",
            ),
        ),
        groundings=(_gv("Histone", "not_present"), _gv("JDP2", "mentioned")),
        aliases={"Histone": frozenset({"Histone"}),
                 "JDP2": frozenset({"JDP2"})},
        gold_polarity="incorrect",
        invest="grounder calibration (family-name detection)",
        note="Perfect grounder flags 'Histone' as generic family → not_present.",
    ),

    # [#20] NKX2-5/MYH IncreaseAmount, gold=act_vs_amt (really role_swap).
    # Evidence: "MEF2C expression initiated cardiomyogenesis, resulting in
    # the up-regulation of... Nkx2-5, GATA-4, ... myosin heavy chain"
    # NKX2-5 is being UP-REGULATED here, not the up-regulator of MYH.
    # The agent is MEF2C; both NKX2-5 and MYH are targets.
    # Perfect parse: assertion(amount, sign=positive, agents={MEF2C},
    # targets={NKX2-5, MYH, GATA-4, ...}). Adjudicator: claim subject NKX2-5
    # is in TARGETS not agents → no matching assertion → check role_swap →
    # NKX2-5 in targets, MYH (object) in targets too → role_swap requires
    # claim.subject in a.targets AND claim.objects in a.agents. MYH is in
    # targets, so role_swap fails. Falls to absent_relationship → incorrect.
    OracleCase(
        rid=20, label="loss",
        claim=ClaimCommitment(
            stmt_type="IncreaseAmount", subject="NKX2-5", objects=("MYH",),
            axis="amount", sign="positive",
        ),
        evidence=_ev(
            EvidenceAssertion(
                agents=("MEF2C",),
                targets=("Brachyury T", "BMP-4", "Nkx2-5", "NKX2-5",
                         "GATA-4", "cardiac alpha-actin",
                         "myosin heavy chain", "MYH"),
                axis="amount", sign="positive", perturbation="none",
            ),
        ),
        groundings=(_gv("NKX2-5", "mentioned"), _gv("MYH", "mentioned")),
        aliases={"NKX2-5": frozenset({"NKX2-5", "Nkx2-5"}),
                 "MYH": frozenset({"MYH", "myosin heavy chain"})},
        gold_polarity="incorrect",
        invest="parse_evidence (multi-target listing extraction)",
        note="Currently abstains on zero assertions; perfect parse → absent_relationship.",
    ),

    # === SPOT-CHECK WINS (verify currently-correct verdicts hold) ===

    # [#2] ATM/BCL10 Ubiquitination, gold=correct, dec=correct.
    # Evidence: "ATM dependent phosphorylation and RNF8 mediated
    # ubiquitination of BCL10". Indirect-chain attribution: gold counts ATM
    # as ubiquitinator because ATM phosphorylation triggers RNF8
    # ubiquitination. Perfect parse extracts BOTH assertions.
    # Adjudicator: claim Ubiq(ATM, BCL10). RNF8 is the literal ubiquitinase.
    # ATM appears as agent of phosphorylation (different axis).
    # Strict perfect parse → no matching ubiq assertion with ATM as agent
    # → absent_relationship → incorrect. DOES NOT match gold=correct.
    # This is a coincidentally-correct case in the actual run.
    OracleCase(
        rid=2, label="win-control",
        claim=ClaimCommitment(
            stmt_type="Ubiquitination", subject="ATM", objects=("BCL10",),
            axis="modification", sign="positive",
        ),
        evidence=_ev(
            EvidenceAssertion(
                agents=("ATM",), targets=("BCL10",),
                axis="modification", sign="positive",  # phosphorylation
            ),
            EvidenceAssertion(
                agents=("RNF8",), targets=("BCL10",),
                axis="modification", sign="positive",  # ubiquitination
            ),
        ),
        groundings=(_gv("ATM", "mentioned"), _gv("BCL10", "mentioned")),
        aliases={"ATM": frozenset({"ATM"}), "BCL10": frozenset({"BCL10"})},
        gold_polarity="correct",
        invest="(architectural — indirect-chain attribution rule)",
        note="Currently correct via lucky parse; strict perfect parse would flip to incorrect.",
    ),

    # [#11] TIGAR/RB1 Dephos, gold=correct, dec=correct (with uncertain).
    # Verify perfect sub-calls maintain match.
    OracleCase(
        rid=11, label="win-control",
        claim=ClaimCommitment(
            stmt_type="Dephosphorylation", subject="TIGAR", objects=("RB1",),
            axis="modification", sign="negative",
        ),
        evidence=_ev(
            EvidenceAssertion(
                agents=("TIGAR",), targets=("RB", "RB1"),
                axis="modification", sign="negative",
            ),
        ),
        groundings=(_gv("TIGAR", "mentioned"), _gv("RB1", "equivalent")),
        aliases={"TIGAR": frozenset({"TIGAR"}),
                 "RB1": frozenset({"RB1", "RB"})},
        gold_polarity="correct",
        invest="(none)",
        note="Should hold under perfect sub-calls.",
    ),

    # [#12] LRIG1/BCL2 DecAmt, gold=correct, dec=correct.
    OracleCase(
        rid=12, label="win-control",
        claim=ClaimCommitment(
            stmt_type="DecreaseAmount", subject="LRIG1", objects=("BCL2",),
            axis="amount", sign="negative",
        ),
        evidence=_ev(
            EvidenceAssertion(
                agents=("LRIG1",), targets=("Bcl-2", "BCL2"),
                axis="amount", sign="negative",
            ),
        ),
        groundings=(_gv("LRIG1", "mentioned"), _gv("BCL2", "equivalent")),
        aliases={"LRIG1": frozenset({"LRIG1"}),
                 "BCL2": frozenset({"BCL2", "Bcl-2"})},
        gold_polarity="correct",
        invest="(none)",
        note="Should hold under perfect sub-calls.",
    ),
]


def main() -> None:
    print("=" * 78)
    print("CEILING ORACLE — perfect-sub-call adjudicator verdicts")
    print("=" * 78)

    losses_fixed = 0
    losses_total = 0
    wins_held = 0
    wins_total = 0

    for c in CASES:
        adj = adjudicate(c.claim, c.evidence, c.groundings,
                         entity_aliases=c.aliases)
        result_polarity = adj.verdict if adj.verdict in ("correct", "incorrect") else "abstain"
        match = result_polarity == c.gold_polarity
        marker = "✓" if match else "✗"

        print(f"\n[#{c.rid:>2d}] ({c.label}) → adjudicator verdict: "
              f"{adj.verdict}/{adj.confidence} reasons={list(adj.reasons)}")
        print(f"      gold polarity: {c.gold_polarity}    "
              f"oracle gives: {result_polarity}    {marker}")
        print(f"      investment: {c.invest}")
        print(f"      note: {c.note}")
        if adj.rationale:
            print(f"      rationale: {adj.rationale}")

        if c.label == "loss":
            losses_total += 1
            if match:
                losses_fixed += 1
        elif c.label == "win-control":
            wins_total += 1
            if match:
                wins_held += 1

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  Losses with perfect sub-calls: {losses_fixed}/{losses_total} "
          f"would now match gold polarity")
    print(f"  Wins held under perfect sub-calls: {wins_held}/{wins_total}")
    print()

    # Project to dual-run scale
    n = 24
    actual_dec_correct = 16  # from the v5 run
    flips = losses_fixed - (wins_total - wins_held)
    projected_dec = actual_dec_correct + flips
    print(f"  v5 dual-run: dec was {actual_dec_correct}/{n} "
          f"= {actual_dec_correct/n:.1%}")
    print(f"  Net flips with perfect sub-calls: +{losses_fixed} fixed, "
          f"-{wins_total - wins_held} broken (within sampled wins)")
    print(f"  PROJECTED dec ceiling on n={n}: "
          f"{projected_dec}/{n} = {projected_dec/n:.1%}")
    print(f"  Mono on same n: 20/{n} = {20/n:.1%}")


if __name__ == "__main__":
    main()

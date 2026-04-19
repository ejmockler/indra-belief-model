"""System prompt, contrastive examples, and verdict parsing for the scorer.

This module is *data* plus minimal rendering/parsing — no model client, no
scoring logic. The active scorer in `scorer.py` imports from here.
"""
from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# System prompt — the scoring contract
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You judge whether a biomedical text-mining extraction is correct.

You are given:
- A CLAIM: SUBJECT [TYPE] OBJECT, optionally with @residue+position
- Optionally, an "Entities:" line listing canonical names and aliases for
  connecting claim names to text mentions (includes family membership notes)
- Optionally, an "Extraction provenance:" block when grounding is uncertain,
  showing what the NLP reader actually pulled from the sentence and how it
  was mapped. Treat MISMATCH and LOW_CONFIDENCE entries as strong grounding
  signals (probably incorrect unless evidence independently supports the claim)
- EVIDENCE text from the source paper

Key rules:
1. If the claim includes @residue+position, verify it matches the evidence.
   "S51A" is a mutation (Ser→Ala), NOT a phosphorylation site.
2. Use entity aliases to connect claim entities to text mentions.
   Family-level claims: if the claim names a protein family (e.g., JUN, AKT,
   ERK, CK2, MAPK, PKA), evidence about any specific family member
   (e.g., c-Jun, AKT1, ERK1/2, CSNK2A1, MAPK1) SUPPORTS the claim.
   This is not a grounding error — family-level claims are deliberately
   less specific than member-level evidence.
3. "Activation" = activity state change. "Inhibition" = direct activity suppression.
   NOT expression/production/degradation (those are IncreaseAmount/DecreaseAmount).
   When a transcription factor increases a gene's promoter activity, reporter
   activity, or transcription, that is IncreaseAmount, not Activation — even if the
   text uses the word "activate" about the promoter.
   miRNA exception: when the claim SUBJECT is a microRNA (starts with "MIR" or
   "let-", e.g. MIR101, MIRLET7A), evidence that the miRNA reduces target mRNA or
   protein is Inhibition by INDRA convention. This exception is ONLY for miRNA
   subjects — siRNA, shRNA, and knockdown of a protein all remain IncreaseAmount/
   DecreaseAmount, not Inhibition.
4. Hedging scope: "may/could/might" on the RELATIONSHIP ITSELF = hypothesis.
   Hedging on a CONSEQUENCE while the relationship is stated = correct.
5. Sentence structure: epithets in negative contexts are background, not evidence.
   "Kinase-dead mutant was unable to..." = negative result, not positive evidence.
6. GROUNDING: If an "Extraction provenance:" block shows MISMATCH or
   LOW CONFIDENCE for an entity, the NLP reader's extracted text does not
   reliably map to the claim entity. Example: "Aβ" → APP is valid (Aβ is an
   APP fragment); "HNF-4alpha" → RXR is not (different proteins). Flagged
   grounding is a strong signal of an incorrect claim unless the evidence
   independently names the claim entity.

Output JSON: {"verdict": "correct" or "incorrect", "confidence": "high" | "medium" | "low"}\
"""


# ---------------------------------------------------------------------------
# Contrastive examples — teach discrimination at the boundaries
# ---------------------------------------------------------------------------

CONTRASTIVE_EXAMPLES = [
    # --- Pair 1: Complex with explicit signal ---
    {
        "claim": "Actin [Complex] CDK9",
        "evidence": "Actin was found to interact with Cdk9, a catalytic subunit of P-TEFb, in elongation complexes.",
        "verdict": "correct", "confidence": "high",
        "reason": "Evidence says 'in elongation complexes' — Complex relationship supported.",
    },
    {
        "claim": "AKT [Complex] CASP3",
        "evidence": "Akt and caspase-3 expression interact to regulate proliferation and apoptosis.",
        "verdict": "incorrect", "confidence": "high",
        "reason": "Text says 'interact' metaphorically about signaling pathways, without complex formation.",
    },

    # --- Pair 2: Activity vs amount ---
    {
        "claim": "TGFB1 [Activation] ADAM17",
        "evidence": "TGF-beta1 induced a rapid activation of the tumour necrosis factor-alpha-converting enzyme (TACE and ADAM (a disintegrin and metalloprotease) 17).",
        "verdict": "correct", "confidence": "high",
        "reason": "Text describes functional activation of the ADAM17 enzyme — activity state change.",
    },
    {
        "claim": "TGFB1 [Activation] ADAM17",
        "evidence": "Furthermore, ADAM17 mRNA and protein expression were up-regulated by TGF-beta1.",
        "verdict": "incorrect", "confidence": "high",
        "reason": "Text describes mRNA/protein expression increase — that is IncreaseAmount, not Activation.",
    },

    # --- Pair 3: Logical inversion ---
    {
        "claim": "AGER [Activation] MMP2",
        "evidence": "RAGE blockade reduced MMP-2 activity to control level.",
        "verdict": "correct", "confidence": "high",
        "reason": "Logical inversion: blocking RAGE reduces MMP-2 activity, so RAGE activates MMP-2.",
    },
    {
        "claim": "TP53 [Inhibition] MDM2",
        "evidence": "TP53 knockdown increased MDM2 protein levels in these cells.",
        "verdict": "correct", "confidence": "high",
        "reason": "Logical inversion: knockdown of TP53 increases MDM2, so TP53 normally decreases MDM2.",
    },

    # --- Pair 4: Hedging scope — same word "could", opposite scope ---
    {
        "claim": "MYB [Complex] PPID",
        "evidence": "However, we found that the cyclophilin Cyp-40 could interact with c-Myb to inhibit its DNA binding activity.",
        "verdict": "correct", "confidence": "high",
        "reason": "'we found that...could interact' reports a discovered result. 'Could' scopes over the consequence (inhibiting DNA binding), not the interaction itself.",
    },
    {
        "claim": "MYB [Complex] PPID",
        "evidence": "A binding assay was used to test whether c-Myb and Cyp-40 could interact directly with one another in vitro.",
        "verdict": "incorrect", "confidence": "high",
        "reason": "'to test whether...could interact' is an experimental question. 'Could' scopes over the relationship itself — the interaction is what's being tested, not confirmed.",
    },

    # --- Pair 5: Discourse/co-occurrence trap — activation verb + negation ---
    {
        "claim": "IFNA [Activation] NFkappaB",
        "evidence": "As illustrated in Fig. 2 A, IFN-α activated NF-κB in a time-dependent manner reaching threefold increase at 120 min.",
        "verdict": "correct", "confidence": "high",
        "reason": "Direct: 'IFN-α activated NF-κB' — subject directly acts on object, with quantitative result.",
    },
    {
        "claim": "TNFSF10 [Activation] CASP8",
        "evidence": "These findings suggest that TRAIL activates a pathway dependent on Bid, but largely independent of FADD and caspase-8, in U2OS cells.",
        "verdict": "incorrect", "confidence": "medium",
        "reason": "'independent of caspase-8' negates the TRAIL→CASP8 link despite co-occurrence with 'activates'.",
    },

    # --- Pair 6: Modification site verification ---
    {
        "claim": "AURKB [Phosphorylation] ATXN10 @S12",
        "evidence": "Our findings suggest that Aurora B phosphorylates Ataxin-10 at S12, which colocalizes with the midbody.",
        "verdict": "correct", "confidence": "high",
        "reason": "Claim says @S12, evidence says 'at S12' — site matches.",
    },
    {
        "claim": "AURKB [Phosphorylation] ATXN10 @S77",
        "evidence": "Our findings suggest that Aurora B phosphorylates Ataxin-10 at S12, which colocalizes with the midbody.",
        "verdict": "incorrect", "confidence": "high",
        "reason": "Claim says @S77 but evidence says S12 — wrong modification site.",
    },

    # --- Pair 7: Direct statement vs indirect chain ---
    {
        "claim": "MTOR [Activation] RPS6KB1",
        "evidence": "mTOR phosphorylates and activates S6K1, leading to increased ribosomal biogenesis.",
        "verdict": "correct", "confidence": "high",
        "reason": "Direct activity change: 'mTOR activates S6K1'.",
    },
    {
        "claim": "P70S6K [Activation] RPS6",
        "evidence": "Ghrelin strongly activated mTOR, P70S6K, and S6 in parallel.",
        "verdict": "incorrect", "confidence": "medium",
        "reason": "Text shows ghrelin activating multiple targets in parallel, not P70S6K acting on RPS6.",
    },

    # --- Pair 8: Concessive clause — established fact vs hedged speculation ---
    {
        "claim": "TNFSF15 [Activation] TNF",
        "evidence": "In this study, we show that TL1A in combination with IL-12, IL-15 and IL-18 directly induces antigen-independent IL-6 and TNF-alpha from monocyte-depleted PBMCs.",
        "verdict": "correct", "confidence": "high",
        "reason": "Direct statement: 'we show that TL1A...directly induces TNF-alpha'. TL1A is TNFSF15. Unhedged experimental result.",
    },
    {
        "claim": "TNFSF15 [Activation] TNF",
        "evidence": "It is tempting to speculate that VEGI/TL1A may serve as a critical mediator linking TNF-alpha signaling to endothelial apoptosis.",
        "verdict": "incorrect", "confidence": "high",
        "reason": "'Tempting to speculate' + 'may serve' = pure hypothesis. No experimental confirmation. The relationship direction is also unclear.",
    },

    # --- Pair 9: Degradation-as-mechanism vs direct inhibition ---
    {
        "claim": "GRB10 [Inhibition] DOK1",
        "evidence": "The insulin-stimulated tyrosine phosphorylation of the GAP-associated protein p62(dok) is inhibited by Grb10, an adaptor protein that binds directly to the kinase domain of the IR.",
        "verdict": "correct", "confidence": "high",
        "reason": "Direct activity inhibition: Grb10 inhibits phosphorylation of DOK1. Demonstrated in vitro with direct binding evidence.",
    },
    {
        "claim": "CDKN1B [Inhibition] CDK2",
        "evidence": "The degradation of p27 stimulates the activity of Cdk2 and cyclin E and Cdk2 and cyclin A to promote cell proliferation.",
        "verdict": "incorrect", "confidence": "high",
        "reason": "Text describes degradation of p27 (DecreaseAmount), not direct inhibition of CDK2. Degradation removes the protein; it doesn't suppress CDK2 activity directly.",
    },
]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_example(ex: dict) -> tuple[str, str]:
    """Render a contrastive example as (user_message, assistant_message)."""
    user = f"CLAIM: {ex['claim']}\nEVIDENCE: \"{ex['evidence']}\""
    assistant = (
        f"Reason: {ex['reason']}\n"
        f'{{"verdict": "{ex["verdict"]}", "confidence": "{ex["confidence"]}"}}'
    )
    return user, assistant


# ---------------------------------------------------------------------------
# Verdict parsing and score mapping — canonical implementations.
# Multi-strategy parser: tries strict JSON first, then alternate ordering,
# then phrase-level extraction from reasoning text. This is the single
# source of truth; model_client does not duplicate verdict parsing.
# ---------------------------------------------------------------------------

_JSON_VERDICT = re.compile(
    r'\{[^{}]*?"verdict"\s*:\s*"(correct|incorrect)"[^{}]*?"confidence"\s*:\s*"(high|medium|low)"[^{}]*?\}',
    re.IGNORECASE,
)
_JSON_VERDICT_REV = re.compile(
    r'\{[^{}]*?"confidence"\s*:\s*"(high|medium|low)"[^{}]*?"verdict"\s*:\s*"(correct|incorrect)"[^{}]*?\}',
    re.IGNORECASE,
)

_VERDICT_PHRASE_PATTERNS = [
    re.compile(r'"verdict"\s*:\s*"(correct|incorrect)"', re.IGNORECASE),
    re.compile(r'(?:final\s+)?(?:verdict|decision|conclusion)[^a-z]*?:[^a-z]*?(?:["\'\*]*)(correct|incorrect)', re.IGNORECASE),
    re.compile(r'\b(?:verdict|decision|answer)\s+(?:is|should be|would be|=)\s*[:"\'\*]*\s*(correct|incorrect)', re.IGNORECASE),
]
_CONFIDENCE_PHRASE_PATTERNS = [
    re.compile(r'"confidence"\s*:\s*"(high|medium|low)"', re.IGNORECASE),
    re.compile(r'confidence[^a-z]*?:[^a-z]*?(?:["\'\*]*)(high|medium|low)', re.IGNORECASE),
    re.compile(r'confidence\s+(?:is|level)?[^a-z]*?(high|medium|low)', re.IGNORECASE),
    re.compile(r'with\s+(high|medium|low)\s+confidence', re.IGNORECASE),
]


def extract_verdict(text: str) -> tuple[str | None, str | None]:
    """Extract (verdict, confidence) from model output.

    Returns (None, None) when no parseable verdict is found. Tries, in order:
      1. Strict JSON with verdict before confidence.
      2. Strict JSON with confidence before verdict.
      3. Phrase-level extraction — verdict keyword + confidence keyword.
    """
    if not text:
        return None, None

    matches = _JSON_VERDICT.findall(text)
    if matches:
        v, c = matches[-1]
        return v.lower(), c.lower()

    matches = _JSON_VERDICT_REV.findall(text)
    if matches:
        c, v = matches[-1]
        return v.lower(), c.lower()

    verdict = None
    for pat in _VERDICT_PHRASE_PATTERNS:
        m = pat.findall(text)
        if m:
            verdict = m[-1].lower()
            break
    if not verdict:
        return None, None

    confidence = "medium"
    for pat in _CONFIDENCE_PHRASE_PATTERNS:
        m = pat.findall(text)
        if m:
            confidence = m[-1].lower()
            break
    return verdict, confidence


_SCORE_GRID = {
    ("correct", "high"): 0.95,
    ("correct", "medium"): 0.80,
    ("correct", "low"): 0.65,
    ("incorrect", "low"): 0.35,
    ("incorrect", "medium"): 0.20,
    ("incorrect", "high"): 0.05,
}


def verdict_to_score(verdict: str | None, confidence: str | None) -> float:
    """Convert (verdict, confidence) to a probability score in [0, 1]."""
    if verdict is None:
        return 0.5
    return _SCORE_GRID.get((verdict, confidence or "medium"), 0.50)

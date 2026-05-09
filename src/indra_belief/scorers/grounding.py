"""Per-entity grounding sub-verifier.

Given a claim entity and the evidence text, judge whether the evidence
refers to this entity (directly, via alias, via family membership, or not
at all). Different input SIGNATURE from the monolithic tool-use path —
the monolithic path injected Gilda lookups into a compound-judgment
prompt and regressed to 80% vs all-match's 83% because the model's
attention was diluted across compound tasks. This sub-verifier asks ONLY
the grounding question, carrying Gilda's structured context as typed
input rather than prose injection.

Always returns a GroundingVerdict (never None). Transport failures,
malformed JSON, or out-of-vocabulary status values degrade to
status="uncertain" with the error captured in `rationale` — the
adjudicator branches on status, not on None.
"""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from indra_belief.scorers.commitments import (
    GroundingVerdict,
    _VALID_GROUNDING,
)

if TYPE_CHECKING:
    from indra_belief.data.entity import GroundedEntity
    from indra_belief.model_client import ModelClient


log = logging.getLogger(__name__)


_SYSTEM_PROMPT = """\
You judge whether an evidence sentence refers to a specific biomedical \
entity. Return one of four statuses.

Statuses:
  mentioned     the evidence names this entity directly (exact symbol or \
close spelling variant)
  equivalent    the evidence names an alias, family member, or fragment \
that maps to this entity (e.g., "ERK" → MAPK1; "Aβ" → APP; "c-Jun" → JUN; \
for family claims, any member counts)
  not_present   the evidence does not refer to this entity at all
  uncertain     the evidence is too ambiguous or malformed to decide

RULES:
1. Family claims: if the claim entity is a family (e.g., AKT, ERK, CK2, \
MAPK, PKA) and the evidence names any family member (AKT1, ERK1/2, \
CSNK2A1, MAPK1, PRKACA), emit "equivalent". A family claim is also \
"mentioned" when the evidence uses the family name itself ("HDAC \
inhibitors", "histone deacetylases", "MAPK signaling").
2. Fragments / processed forms: "Aβ" (APP fragment), "cleaved caspase-3" \
(CASP3), and similar processed forms of the claim entity → "equivalent".
3. Pseudogenes: when the claim entity is flagged is_pseudogene=true:
   - If the evidence names the pseudogene by its DISTINCT symbol (e.g., \
"PTENP1", "BRAFP1") or explicitly describes a pseudogene / lncRNA / \
processed-transcript role, emit "mentioned" or "equivalent" as \
appropriate.
   - If the evidence only uses the parent-gene symbol (e.g., "PTEN" for \
a PTENP1 claim), emit "not_present" — a parent-symbol collision is NOT \
evidence for the pseudogene.
4. Unrelated synonym collisions: a symbol that happens to match \
(e.g., "AR" for both androgen receptor and amphiregulin) without \
context-consistent evidence → "uncertain" or "not_present" per context.
5. Generic class nouns: when the claim entity is a generic biochemical \
class word (Histone, Phosphatase, Kinase, Protease, Receptor, \
Transcription factor, Channel, Ligase) — typically signaled by no db \
grounding — apply the role-aware rule:
   - mentioned: the evidence uses the class word (singular OR plural) \
as the ACTOR or TARGET of the relation. The class itself is participating \
in the relationship. Examples:
     * claim="Histone", evidence "deacetylation of histones by HDACs" → \
mentioned (histones is the target of the deacetylation event).
     * claim="HDAC", evidence "HDACs deacetylate histones" → mentioned \
(HDACs is the actor; the family-claim already has equivalence under \
Rule 1, but the bare class name here also counts).
     * claim="Phosphatase", evidence "phosphatase activity is increased" \
→ mentioned (the class is the entity whose activity changes).
   - not_present: the evidence uses a COMPOUND TERM that names a specific \
different entity, and the claim is the bare class itself. Compound terms \
like "histone deacetylase HDAC3", "TBK1 phosphatase", "TGF-beta receptor" \
name an enzyme/receptor (HDAC3/TBK1/TGFBR1), not the class word. \
Examples:
     * claim="Histone", evidence "Histone deacetylase HDAC3 deacetylates \
p53" → not_present (HDAC3 is the actor, not a histone).
     * claim="Phosphatase", evidence "PPM1B is a TBK1 phosphatase" → \
not_present (PPM1B is the specific phosphatase).
   The discriminator: is the class word standing alone as a participant \
(actor or target), or is it part of a compound that names something else?
6. Cross-gene grounding collisions: when the evidence's symbol resolves \
to a DIFFERENT specific gene than the claim entity (e.g., evidence says \
"p97" which is VCP, but claim entity is GEMIN4), emit "not_present" — \
NOT uncertain. The two genes are unrelated; the alias collision is a \
grounding bug, not genuine ambiguity. Use "uncertain" (Rule 4) only when \
the identity is genuinely unclear from context.
7. Case sensitivity for short symbols: short uppercase gene symbols \
(2-4 chars: FAS, MYC, JUN, RAS, INS) are case-distinct from lowercase \
or mixed-case occurrences in the evidence. "FAs" (focal adhesions, \
plural) is NOT a match for the FAS gene. Apply case sensitivity \
strictly for short symbols where the lowercase/plural form has a \
different biomedical referent.
8. Anaphora and bridging references: when the evidence uses a noun \
phrase that REFERS to the claim entity via context, count the \
anaphoric mention. Examples:
   - claim="ARR3", evidence "this short arrestin-3-derived peptide \
binds ASK1 and MKK4/7" → "equivalent" (the noun phrase derives the \
entity from context).
   - claim="HER2", evidence "the receptor was phosphorylated" — only \
"equivalent" if the surrounding sentence(s) already establish the \
referent. If the antecedent is unclear, "uncertain".
   - claim="MEK1", evidence "MEK1/2 was activated by Raf" → \
"equivalent" (compound symbol "MEK1/2" includes the claim instance).
9. Greek-letter and special-character aliases: Greek letters and \
typographic variants are common biomedical aliases. Match them as \
"equivalent":
   - β-catenin / beta-catenin / β-Cat → CTNNB1
   - p38 / p38α / p38 MAPK → MAPK14 (or MAPK family)
   - ERβ / ER-beta / ERβ1 → ESR2
   - PI3-K / PI3'-kinase → PI3K
   - α/β/γ subunits → corresponding gene symbols
10. Reverse family-instance bridging: when the claim is an INSTANCE \
(specific gene like MEK1, ERK2, AKT1) and the evidence names only the \
FAMILY (MEK, ERK, AKT) without an instance specifier, emit "equivalent" \
when the evidence text suggests inclusivity (e.g., "MEK signaling" as \
a process implies MEK family members). Emit "uncertain" when the \
family mention is too generic to attribute to the specific instance.
   - claim="MEK1", evidence "MEK signaling activates ERK" → "equivalent"
   - claim="MEK1", evidence "RAF activates the MEK family" → "uncertain" \
(family membership is mentioned, but no specific instance is invoked).
11. Bidirectional binding: when the evidence asserts a binding \
relation symmetrically ("X and Y interact", "Mutually Exclusive \
Binding of X and Y to Z", "X-Y complex"), match the claim entity even \
when it appears in the second slot. The claim entity does NOT need \
to be in the syntactic subject position.

Output JSON only, no prose outside:
{"status": <status>, "rationale": "<short reason, one sentence>"}
"""


def _build_user_message(claim_entity: "GroundedEntity", evidence_text: str) -> str:
    """Format the claim-entity context + evidence sentence."""
    name = claim_entity.name or "?"
    parts = [f"Claim entity: {name}"]
    if claim_entity.db and claim_entity.db_id:
        parts.append(f"Grounding: {claim_entity.db}:{claim_entity.db_id}")
    else:
        # No DB grounding suggests the claim entity is a generic class
        # noun (Phosphatase, Kinase) rather than a specific gene. The
        # adjudicator-side check is downstream; the prompt's Rule 5
        # tells the model to treat compound mentions like "TBK1
        # phosphatase" as a different specific entity, not as the class.
        parts.append("Grounding: <none — possibly a generic class noun>")
    if claim_entity.is_family:
        members = claim_entity.family_members[:6]
        if members:
            parts.append(f"Family members: {', '.join(members)}")
        else:
            parts.append("Family: yes (members unknown)")
    if claim_entity.is_pseudogene:
        parts.append("is_pseudogene: true — parent-gene symbol matches do NOT count")
    aliases = [a for a in (claim_entity.aliases or []) if a and a != name]
    if aliases:
        parts.append(f"Aliases: {', '.join(aliases[:8])}")
    if claim_entity.is_low_confidence and claim_entity.gilda_score is not None:
        parts.append(f"Gilda score (low confidence): {claim_entity.gilda_score:.2f}")

    parts.append("")
    parts.append(f'Evidence: "{evidence_text}"')
    parts.append("")
    parts.append("Does the evidence reference this entity?")
    return "\n".join(parts)


def _extract_json_object(text: str) -> dict | None:
    """Return the last valid top-level JSON object in `text`, or None."""
    if not text:
        return None
    decoder = json.JSONDecoder()
    results: list[dict] = []
    i = 0
    n = len(text)
    while i < n:
        idx = text.find("{", i)
        if idx < 0:
            break
        try:
            obj, end = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            i = idx + 1
            continue
        if isinstance(obj, dict):
            results.append(obj)
        i = idx + end
    return results[-1] if results else None


def _verdict_from_entity(
    claim_entity: "GroundedEntity",
    status: str,
    rationale: str,
) -> GroundingVerdict:
    """Construct a GroundingVerdict carrying Gilda context forward."""
    return GroundingVerdict(
        claim_entity=claim_entity.name or "?",
        status=status,  # type: ignore[arg-type]
        db_ns=claim_entity.db,
        db_id=claim_entity.db_id,
        gilda_score=claim_entity.gilda_score,
        is_family=bool(claim_entity.is_family),
        is_pseudogene=bool(claim_entity.is_pseudogene),
        rationale=rationale,
    )


def _uncertain(claim_entity: "GroundedEntity", reason: str) -> GroundingVerdict:
    return _verdict_from_entity(claim_entity, "uncertain", reason)


def verify_grounding(
    claim_entity: "GroundedEntity",
    evidence_text: str,
    client: "ModelClient",
    *,
    max_tokens: int = 2000,
    temperature: float = 0.1,
) -> GroundingVerdict:
    """Judge whether `claim_entity` is referenced in `evidence_text`.

    ALWAYS returns a GroundingVerdict — on transport error, malformed
    output, or out-of-vocabulary status, the verdict is status=uncertain
    with the error captured in `rationale`. Gilda structured context is
    preserved regardless of outcome so the audit trail survives.
    """
    if claim_entity is None:
        # Defensive — should be unreachable from the pipeline, but callable
        # from tests.
        return GroundingVerdict(claim_entity="?", status="uncertain",
                                rationale="no claim entity provided")
    if not evidence_text or not evidence_text.strip():
        return _uncertain(claim_entity, "empty evidence text")

    user_msg = _build_user_message(claim_entity, evidence_text)

    try:
        response = client.call(
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
            max_tokens=max_tokens,
            temperature=temperature,
            response_format={"type": "json_object"},
            reasoning_effort="none",
            kind="verify_grounding",
        )
    except Exception as e:
        log.warning("verify_grounding: client.call() failed: %s", e)
        return _uncertain(claim_entity, f"transport error: {type(e).__name__}")

    if getattr(response, "finish_reason", None) == "length":
        log.warning("verify_grounding: response truncated (finish_reason=length)")

    obj = _extract_json_object(response.content)
    if obj is None:
        obj = _extract_json_object(response.raw_text)
    if obj is None:
        log.debug("verify_grounding: no JSON extracted from response")
        return _uncertain(claim_entity, "malformed JSON from model")

    status = obj.get("status")
    rationale_raw = obj.get("rationale", "")
    rationale = rationale_raw if isinstance(rationale_raw, str) else ""

    if not isinstance(status, str) or status not in _VALID_GROUNDING:
        return _uncertain(
            claim_entity,
            f"out-of-vocabulary status: {status!r}",
        )

    try:
        return _verdict_from_entity(claim_entity, status, rationale)
    except ValueError as e:
        # __post_init__ validator caught something unexpected.
        log.debug("verify_grounding: verdict construction failed: %s", e)
        return _uncertain(claim_entity, f"validation error: {e}")

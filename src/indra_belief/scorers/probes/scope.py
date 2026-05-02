"""scope probe — is the relation between subject and object asserted,
hedged, or negated in the evidence?

Closed answer set: asserted, hedged, negated, abstain.

Substrate fast-path covers explicit M10 hedge markers and verb-negator
proximity hits between subject and object positions. The LLM handles
softer cases — "may have", "putative", "appears to", "we tested",
"it remains unclear", and rhetorical-not constructions.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from indra_belief.scorers.probes._llm import llm_classify
from indra_belief.scorers.probes.types import ProbeRequest, ProbeResponse

if TYPE_CHECKING:
    from indra_belief.model_client import ModelClient


_ANSWER_SET = frozenset({"asserted", "hedged", "negated", "abstain"})


_SYSTEM_PROMPT = """\
You classify the EPISTEMIC SCOPE of the relation between two named \
entities in an evidence sentence.

The CLAIM names a relation between subject and object. Your job is to \
classify how the sentence FRAMES that relation — independent of \
whether it's true.

Answer ONE of:
  asserted — the sentence directly affirms the relation. Includes \
    declarative ("X activates Y"), passive ("Y is activated by X"), \
    and presupposed framings ("the X-Y interaction is required for...").
  hedged   — the sentence proposes the relation as exploratory, \
    hypothetical, or putative. Includes "may", "might", "could", \
    "we hypothesize", "appears to", "putative", "is thought to", \
    "we tested whether", "it remains unclear if".
  negated  — the sentence explicitly denies the relation. Includes \
    "X did not activate Y", "Y was not phosphorylated by X", \
    "no effect of X on Y was observed".
  abstain  — the sentence does not commit to any of the above for \
    THIS relation (relation not directly described, or the framing \
    is genuinely ambiguous).

CRITICAL: focus on the CLAIM RELATION specifically. Hedging or \
negation that governs a DIFFERENT proposition in the same sentence \
does NOT propagate. "X activates Y, but Z was not affected" → asserted \
for X→Y; the negation governs Z.

Output ONE JSON object: {"answer": <one_of_above>, "rationale": <short_phrase>}.
The "rationale" is a 5-15 word phrase quoting the relevant words from \
the evidence. NO prose outside the JSON."""


_FEW_SHOTS: list[tuple[str, str]] = [
    (
        "CLAIM: relation between MAPK1 and JUN\n"
        "EVIDENCE: MAPK1 activates JUN in stimulated cells.",
        '{"answer": "asserted", '
        '"rationale": "direct affirmation: MAPK1 activates JUN"}',
    ),
    (
        "CLAIM: relation between CCR7 and AKT\n"
        "EVIDENCE: CCR7 may activate Akt in T-cells, but this remains "
        "to be confirmed.",
        '{"answer": "hedged", '
        '"rationale": "may activate ... remains to be confirmed"}',
    ),
    (
        "CLAIM: relation between MAPK1 and JUN\n"
        "EVIDENCE: MAPK1 did not activate JUN under any tested condition.",
        '{"answer": "negated", '
        '"rationale": "MAPK1 did not activate JUN — explicit denial"}',
    ),
    (
        "CLAIM: relation between MAPK1 and JUN\n"
        "EVIDENCE: MAPK1 activates JUN robustly, but ELK1 was not "
        "affected by the treatment.",
        '{"answer": "asserted", '
        '"rationale": "negation governs ELK1, not the MAPK1-JUN relation"}',
    ),
    (
        "CLAIM: relation between MAPK1 and JUN\n"
        "EVIDENCE: We characterized MAPK1 substrates in cycling cells.",
        '{"answer": "abstain", '
        '"rationale": "MAPK1-JUN relation not described"}',
    ),
]


def answer(
    request: ProbeRequest, client: "ModelClient",
) -> ProbeResponse:
    """Resolve a scope probe via LLM closed-set classification."""
    if request.kind != "scope":
        raise ValueError(
            f"scope.answer received kind={request.kind!r}"
        )
    user_msg_parts = [
        f"CLAIM: {request.claim_component}",
        f"EVIDENCE: {request.evidence_text.strip()}",
    ]
    if request.substrate_hint:
        user_msg_parts.append(f"SUBSTRATE HINT: {request.substrate_hint}")
    user_msg = "\n".join(user_msg_parts)

    answer_value, rationale, succeeded = llm_classify(
        system_prompt=_SYSTEM_PROMPT,
        few_shots=_FEW_SHOTS,
        user_message=user_msg,
        answer_set=_ANSWER_SET,
        kind="scope",
        client=client,
    )
    return ProbeResponse(
        kind="scope",
        answer=answer_value,
        source="llm" if succeeded else "abstain",
        confidence="medium" if succeeded else "low",
        rationale=rationale,
    )

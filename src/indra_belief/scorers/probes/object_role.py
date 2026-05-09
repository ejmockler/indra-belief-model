"""object_role probe — what role does the claim's object play in evidence?

Mirror of subject_role.py with target/agent inversion. The few-shots
exemplify the same five answer values from the OBJECT side.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from indra_belief.scorers.probes._llm import llm_classify
from indra_belief.scorers.probes.types import ProbeRequest, ProbeResponse

if TYPE_CHECKING:
    from indra_belief.model_client import ModelClient


_ANSWER_SET = frozenset({
    "present_as_subject",
    "present_as_object",
    "present_as_mediator",
    "present_as_decoy",
    "absent",
})


_SYSTEM_PROMPT = """\
You classify the role of a single biomedical entity (the CLAIM OBJECT) \
in an evidence sentence.

Answer ONE of:
  present_as_object  — the entity is the TARGET/OBJECT of the main \
    relation (the action is done TO it). The expected role for a \
    well-aligned claim object.
  present_as_subject — the entity acts as the SUBJECT/AGENT of the main \
    relation (it does the action). This catches role-swap.
  present_as_mediator — the entity sits in the middle of a chain \
    (X → ENTITY → Y), neither the upstream initiator nor the final \
    target.
  present_as_decoy   — the entity is mentioned but in a relation other \
    than the claim's (control, co-treatment, bystander, adjacent claim).
  absent             — the entity is not mentioned in the evidence.

Output ONE JSON object: {"answer": <one_of_above>, "rationale": <short_phrase>}.
The "rationale" is a 5-15 word phrase quoting the relevant words from \
the evidence. NO prose outside the JSON."""


_FEW_SHOTS: list[tuple[str, str]] = [
    (
        "CLAIM OBJECT: JUN\n"
        "EVIDENCE: MAPK1 phosphorylates JUN at Ser63 in stimulated cells.",
        '{"answer": "present_as_object", '
        '"rationale": "phosphorylates JUN — JUN is target"}',
    ),
    (
        "CLAIM OBJECT: CytokineY\n"
        "EVIDENCE: ReceptorX-induced release of CytokineW, CytokineY, "
        "CytokineZ and CytokineV into the supernatants was measured.",
        '{"answer": "present_as_object", '
        '"rationale": "release of ... CytokineY — target in coordinated list"}',
    ),
    (
        "CLAIM OBJECT: MAPK1\n"
        "EVIDENCE: MAPK1 in turn phosphorylates RSK and several other "
        "downstream kinases.",
        '{"answer": "present_as_subject", '
        '"rationale": "MAPK1 phosphorylates — MAPK1 acts (role-swap)"}',
    ),
    (
        "CLAIM OBJECT: AKT\n"
        "EVIDENCE: PI3K activates PDK1 which phosphorylates AKT to drive "
        "downstream survival signaling.",
        '{"answer": "present_as_mediator", '
        '"rationale": "AKT mid-chain: PI3K -> PDK1 -> AKT -> survival"}',
    ),
    (
        "CLAIM OBJECT: GAPDH\n"
        "EVIDENCE: MAPK1 phosphorylates JUN; GAPDH was used as a loading "
        "control on the Western blot.",
        '{"answer": "present_as_decoy", '
        '"rationale": "GAPDH is a loading control, not the target"}',
    ),
    (
        "CLAIM OBJECT: TP53\n"
        "EVIDENCE: MAPK1 phosphorylates JUN at Ser63 in stimulated cells.",
        '{"answer": "absent", '
        '"rationale": "TP53 not mentioned in evidence"}',
    ),
]


def answer(
    request: ProbeRequest, client: "ModelClient",
    *, reasoning_effort: str = "none",
) -> ProbeResponse:
    """Resolve an object_role probe via LLM closed-set classification.

    U3 selective reasoning: pass reasoning_effort="medium" to escalate
    on hard cases.
    """
    if request.kind != "object_role":
        raise ValueError(
            f"object_role.answer received kind={request.kind!r}"
        )
    user_msg_parts = [
        f"CLAIM OBJECT: {request.claim_component}",
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
        kind="object_role",
        client=client,
        reasoning_effort=reasoning_effort,
    )
    return ProbeResponse(
        kind="object_role",
        answer=answer_value if succeeded else "absent",
        source="llm" if succeeded else "abstain",
        confidence="medium" if succeeded else "low",
        rationale=rationale,
    )

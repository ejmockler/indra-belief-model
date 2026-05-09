"""Deterministic claim-parse sub-call.

Maps an INDRA Statement directly to a ClaimCommitment. No LLM call — the
INDRA type system already canonicalizes (axis, sign), so this projection
is a pure function. The miRNA subject rule is the one documented case
where subject identity modulates the mapping: Inhibition by a miRNA
subject projects to the amount axis because the evidence describes target
mRNA/protein reduction (per INDRA convention; see scorers/_prompts.py
Key Rule 3 for the shipped monolithic articulation).

Keeping this deterministic is a load-bearing design choice. At small-model
scale, every LLM call is a reliability tax; this step has no text to read
and no ambiguity to resolve, so it pays no tax.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from indra_belief.scorers.commitments import ClaimCommitment

if TYPE_CHECKING:
    from indra.statements import Statement


def _is_mirna_name(name: str) -> bool:
    """HGNC convention: miRNA canonical symbols start with 'MIR' (MIR101,
    MIRLET7A) or 'let-' (let-7a). This is ONLY for miRNA subjects — siRNA,
    shRNA, and protein-knockdown constructs do NOT trigger the rule."""
    if not name:
        return False
    upper = name.upper()
    return upper.startswith("MIR") or upper.startswith("LET-")


def parse_claim(statement: "Statement") -> ClaimCommitment:
    """Map an INDRA Statement to a ClaimCommitment. Pure function."""
    # Local imports — avoids eager indra dependency at module load.
    from indra.statements import (
        Activation,
        AddModification,
        Complex,
        Conversion,
        DecreaseAmount,
        Gap,
        Gef,
        GtpActivation,
        IncreaseAmount,
        Inhibition,
        RemoveModification,
        SelfModification,
        Translocation,
    )

    stmt_type = type(statement).__name__

    # Subject + objects. Complex uses members; everything else uses agent_list.
    if isinstance(statement, Complex):
        members = [m for m in statement.members if m is not None]
        subject = members[0].name if members else "?"
        objects = tuple(m.name for m in members[1:])
    else:
        agents = [a for a in statement.agent_list() if a is not None]
        subject = agents[0].name if agents else "?"
        if isinstance(statement, SelfModification):
            # Same entity is both agent and substrate.
            objects = (subject,)
        else:
            objects = tuple(a.name for a in agents[1:])

    # Axis + sign from INDRA class hierarchy. Order matters — GtpActivation
    # is a subclass of Activation, so it must be matched first.
    if isinstance(statement, GtpActivation):
        axis, sign = "gtp_state", "positive"
    elif isinstance(statement, Gef):
        axis, sign = "gtp_state", "positive"
    elif isinstance(statement, Gap):
        axis, sign = "gtp_state", "negative"
    elif isinstance(statement, Activation):
        axis, sign = "activity", "positive"
    elif isinstance(statement, Inhibition):
        axis, sign = "activity", "negative"
    elif isinstance(statement, IncreaseAmount):
        axis, sign = "amount", "positive"
    elif isinstance(statement, DecreaseAmount):
        axis, sign = "amount", "negative"
    elif isinstance(statement, Complex):
        axis, sign = "binding", "neutral"
    elif isinstance(statement, Translocation):
        axis, sign = "localization", "neutral"
    elif isinstance(statement, Conversion):
        axis, sign = "conversion", "neutral"
    elif isinstance(statement, (AddModification, SelfModification)):
        axis, sign = "modification", "positive"
    elif isinstance(statement, RemoveModification):
        axis, sign = "modification", "negative"
    else:
        axis, sign = "unclear", "neutral"

    # Modification site (Phos/Dephos/Acetyl/etc. carry residue + position).
    site: str | None = None
    residue = getattr(statement, "residue", None)
    position = getattr(statement, "position", None)
    if residue or position:
        combined = (residue or "") + (position or "")
        site = combined or None

    # Translocation endpoints.
    location_from: str | None = None
    location_to: str | None = None
    if isinstance(statement, Translocation):
        location_from = statement.from_location
        location_to = statement.to_location

    # miRNA subject rule: Inhibition-by-miRNA projects to the amount axis.
    subject_is_mirna = _is_mirna_name(subject)
    if subject_is_mirna and isinstance(statement, Inhibition):
        axis = "amount"

    return ClaimCommitment(
        stmt_type=stmt_type,
        subject=subject,
        objects=objects,
        axis=axis,
        sign=sign,
        site=site,
        location_from=location_from,
        location_to=location_to,
        subject_is_mirna=subject_is_mirna,
    )

"""ScoringRecord — native INDRA Statement + Evidence wrapper for scoring.

Wraps an INDRA statement + evidence pair, resolving entities once at
construction and carrying all derived metadata through the pipeline. Owns
the Tier-1 auto-reject policy and the user-message rendering.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from indra_belief.data.entity import GroundedEntity

if TYPE_CHECKING:
    from indra.statements import Statement, Evidence


@dataclass
class ScoringRecord:
    """Everything the scorer needs for one extraction, resolved once."""

    # INDRA native objects
    statement: Statement
    evidence: Evidence

    # Curator annotation (from benchmark, not present in production)
    tag: str | None = None
    curator_note: str | None = None

    # Resolved entities (populated by resolve())
    subject_entity: GroundedEntity | None = None
    object_entity: GroundedEntity | None = None

    # Cached derived fields
    _claim: str | None = field(default=None, repr=False)

    def __post_init__(self):
        self.resolve_entities()

    # --- Core properties from INDRA objects ---

    @property
    def stmt_type(self) -> str:
        return type(self.statement).__name__

    @property
    def subject(self) -> str:
        agents = self.statement.agent_list()
        return agents[0].name if agents and agents[0] else "?"

    @property
    def object(self) -> str:
        """Second-agent name, or the first agent for SelfModification
        statements (where subject and object are the same entity). For
        Complex with >=2 members, returns member[1]. Returns '?' only when
        no second agent is available."""
        from indra.statements import SelfModification
        agents = self.statement.agent_list()
        if isinstance(self.statement, SelfModification):
            return agents[0].name if agents and agents[0] else "?"
        if len(agents) > 1 and agents[1]:
            return agents[1].name
        return "?"

    @property
    def evidence_text(self) -> str:
        return self.evidence.text or ""

    @property
    def source_hash(self) -> int:
        return self.evidence.get_source_hash()

    @property
    def source_api(self) -> str:
        return self.evidence.source_api or ""

    @property
    def found_by(self) -> str:
        return self.evidence.annotations.get("found_by") or ""

    @property
    def is_direct(self) -> bool | None:
        return self.evidence.epistemics.get("direct")

    @property
    def raw_text(self) -> list[str | None]:
        return self.evidence.annotations.get("agents", {}).get("raw_text") or []

    @property
    def raw_grounding(self) -> list[dict]:
        return self.evidence.annotations.get("agents", {}).get("raw_grounding") or []

    @property
    def pmid(self) -> str | None:
        return self.evidence.pmid

    # --- Modification-specific (Phosphorylation, etc.) ---

    @property
    def residue(self) -> str | None:
        return getattr(self.statement, "residue", None)

    @property
    def position(self) -> str | None:
        return getattr(self.statement, "position", None)

    # --- Agent details ---

    @property
    def agents(self) -> list:
        """Native INDRA Agent objects."""
        return [a for a in self.statement.agent_list() if a is not None]

    def agent_db_refs(self, index: int) -> dict:
        """Get db_refs for agent at index (statement-level grounding)."""
        agents = self.statement.agent_list()
        if index < len(agents) and agents[index]:
            return agents[index].db_refs
        return {}

    def agent_mods(self, index: int) -> list:
        agents = self.statement.agent_list()
        if index < len(agents) and agents[index]:
            return agents[index].mods
        return []

    def agent_mutations(self, index: int) -> list:
        agents = self.statement.agent_list()
        if index < len(agents) and agents[index]:
            return agents[index].mutations
        return []

    def agent_bound_conditions(self, index: int) -> list:
        agents = self.statement.agent_list()
        if index < len(agents) and agents[index]:
            return agents[index].bound_conditions
        return []

    # --- Entity resolution ---

    def resolve_entities(self) -> None:
        """Resolve both entities via gilda. Called once at construction."""
        clean_rt = [r for r in self.raw_text if r is not None]
        subj_rt = clean_rt[0] if len(clean_rt) > 0 else None
        obj_rt = clean_rt[1] if len(clean_rt) > 1 else None

        self.subject_entity = GroundedEntity.resolve(self.subject, subj_rt)
        self.object_entity = GroundedEntity.resolve(self.object, obj_rt)

    # --- Formatting for LLM prompt ---

    def format_claim(self) -> str:
        """Render the claim string in a statement-type-aware form.

        Shapes:
          - Binary types (Phosphorylation, Activation, ...): `A [Type] B @site`
          - Complex (>=2 members): `A + B + C [Complex]`
          - SelfModification (Auto/Transphosphorylation): `A [Type] A @site`
          - Translocation: `A [Translocation] from X to Y`
        """
        from indra.statements import Complex, SelfModification, Translocation

        stmt = self.statement
        stype = self.stmt_type

        if isinstance(stmt, Complex):
            names = [m.name for m in stmt.members if m]
            if len(names) >= 2:
                return f"{' + '.join(names)} [{stype}]"
            # Fallback to binary rendering if Complex has <2 members
            # (malformed input — not expected but defensive).

        if isinstance(stmt, SelfModification):
            agents = stmt.agent_list()
            name = agents[0].name if agents and agents[0] else "?"
            ann = self._format_agent_annotations(0)
            claim = f"{name}{ann} [{stype}] {name}"
            if self.residue or self.position:
                site = "@" + (self.residue or "") + (self.position or "")
                claim += f" {site}"
            return claim

        if isinstance(stmt, Translocation):
            agents = stmt.agent_list()
            name = agents[0].name if agents and agents[0] else "?"
            ann = self._format_agent_annotations(0)
            from_loc = stmt.from_location or "?"
            to_loc = stmt.to_location or "?"
            return f"{name}{ann} [{stype}] from {from_loc} to {to_loc}"

        # Binary default
        subj_ann = self._format_agent_annotations(0)
        obj_ann = self._format_agent_annotations(1)
        claim = f"{self.subject}{subj_ann} [{stype}] {self.object}{obj_ann}"
        if self.residue or self.position:
            site = "@" + (self.residue or "") + (self.position or "")
            claim += f" {site}"
        return claim

    def _format_agent_annotations(self, index: int) -> str:
        """Format activity, mutations, bound conditions for an agent."""
        parts = []
        agents = self.statement.agent_list()
        if index < len(agents) and agents[index]:
            agent = agents[index]
            if agent.activity:
                # agent.activity stringifies as "(activity)" — extract the label
                act_str = str(agent.activity).strip("()")
                parts.append(act_str)
        for mut in self.agent_mutations(index):
            res_from = mut.residue_from or ""
            res_to = mut.residue_to or ""
            pos = mut.position or ""
            label = f"{res_from}{pos}{res_to}".strip()
            if label:
                parts.append(f"mutation: {label}")
        for bc in self.agent_bound_conditions(index):
            if bc.agent and bc.is_bound:
                parts.append(f"bound to {bc.agent.name}")
        if not parts:
            return ""
        return f" ({'; '.join(parts)})"

    def format_entity_context(self) -> str:
        """Entity alias context line for the LLM prompt."""
        subj_ctx = self.subject_entity.format_alias_context() if self.subject_entity else ""
        obj_ctx = self.object_entity.format_alias_context() if self.object_entity else ""

        if self.subject == self.object:
            parts = [subj_ctx] if subj_ctx else []
        else:
            parts = [p for p in (subj_ctx, obj_ctx) if p]

        if not parts:
            base = ""
        else:
            base = "Entities: " + " | ".join(parts)

        # Add warnings
        warnings = []
        for entity in (self.subject_entity, self.object_entity):
            if entity:
                w = entity.format_warning()
                if w:
                    warnings.append(w)
        if warnings:
            warning_block = "\n".join(warnings)
            return base + "\n" + warning_block if base else warning_block

        return base

    def format_provenance(self) -> str:
        """Structured provenance context — only when MISMATCH signal exists."""
        entities = [
            (self.subject_entity, "Subject"),
            (self.object_entity, "Object"),
        ]

        lines = []
        has_strong_signal = False

        for entity, role in entities:
            if not entity or not entity.raw_text:
                continue

            if entity.raw_text.lower() == entity.name.lower():
                lines.append((role, entity.raw_text, entity.name, "exact", None))
                continue

            if entity.verification_status == "MISMATCH":
                # Skip descriptive names (>15 chars with spaces) that resolve to non-HGNC
                if " " in entity.raw_text and len(entity.raw_text) > 15:
                    lines.append((role, entity.raw_text, entity.name, "alias", None))
                else:
                    has_strong_signal = True
                    lines.append((role, entity.raw_text, entity.name, "MISMATCH", entity.gilda_score))
            elif entity.verification_status == "MATCH" and entity.is_low_confidence:
                has_strong_signal = True
                lines.append((role, entity.raw_text, entity.name, "LOW_CONFIDENCE", entity.gilda_score))
            else:
                lines.append((role, entity.raw_text, entity.name, "alias", None))

        if not has_strong_signal:
            return ""

        parts = ["Extraction provenance:"]
        for role, rt, ce, status, score in lines:
            score_str = f", gilda: {score:.2f}" if score is not None else ""
            if status == "exact":
                parts.append(f'  {role}: NLP extracted "{rt}" → {ce} (exact match)')
            elif status == "alias":
                parts.append(f'  {role}: NLP extracted "{rt}" → {ce} (confirmed alias{score_str})')
            elif status == "MISMATCH":
                parts.append(f'  {role}: NLP extracted "{rt}" → mapped to {ce} (MISMATCH — "{rt}" is a DIFFERENT entity{score_str})')
            elif status == "LOW_CONFIDENCE":
                parts.append(f'  {role}: NLP extracted "{rt}" → mapped to {ce} (LOW CONFIDENCE{score_str} — not a confirmed alias)')

        if self.found_by:
            parts.append(f"  Reader: {self.source_api}, pattern: {self.found_by}")

        return "\n".join(parts)

    def format_user_message(self) -> str:
        """Build the complete user message for Tier 2 LLM scoring."""
        parts = [f"CLAIM: {self.format_claim()}"]

        entity_ctx = self.format_entity_context()
        if entity_ctx:
            parts.append(entity_ctx)

        # Provenance: only inject when grounding is flagged (has_grounding_signal).
        # Full-population provenance hurt accuracy by 6.7pp (72.2% vs 78.9% on
        # 3754 records), but the flagged-grounding subset (n=361) is already at
        # 71.2% — provenance may help that specific slice without harming the rest.
        has_flagged_grounding = any(
            e.has_grounding_signal
            for e in (self.subject_entity, self.object_entity)
            if e
        )
        if has_flagged_grounding:
            provenance = self.format_provenance()
            if provenance:
                parts.append(provenance)

        parts.append(f'EVIDENCE: "{self.evidence_text}"')

        return "\n".join(parts)

    # --- Tier 1 deterministic checks ---

    def tier1_auto_reject(self) -> dict | None:
        """Run deterministic grounding checks. Returns result dict or None."""
        for entity in (self.subject_entity, self.object_entity):
            if not entity:
                continue
            reject, reason = entity.should_auto_reject(self.evidence_text)
            if reject:
                tier = "deterministic_mismatch"
                if entity.is_pseudogene:
                    tier = "deterministic_pseudogene"
                return {
                    "score": 0.05,
                    "verdict": "incorrect",
                    "confidence": "high",
                    "raw_text": f"[TIER 1 AUTO-REJECT] {reason}",
                    "tokens": 0,
                    "tier": tier,
                    "grounding_status": entity.verification_status or "MISMATCH",
                    "provenance_triggered": False,
                }
        return None

    # --- Construction helpers ---

    @classmethod
    def from_holdout(
        cls,
        holdout_record: dict,
        stmt: Statement,
        evidence: Evidence,
    ) -> ScoringRecord:
        """Build from a holdout record + matched INDRA objects."""
        return cls(
            statement=stmt,
            evidence=evidence,
            tag=holdout_record.get("tag"),
            curator_note=holdout_record.get("curator_note"),
        )

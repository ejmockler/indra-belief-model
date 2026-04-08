"""Grounded entity resolution — resolve once, use everywhere.

Replaces scattered gilda.ground() / get_names() / verify_mapping calls
with a single resolution step per entity. The GroundedEntity carries all
identity information through the scoring pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any


@dataclass
class GroundedEntity:
    """An entity resolved through gilda with all identity metadata."""

    # Claim-level identity
    name: str                          # claim entity name (e.g., "S100A8")
    raw_text: str | None = None        # what the NLP reader extracted (e.g., "CagA")

    # Gilda resolution of the claim entity name
    canonical: str | None = None       # gilda top-hit entry_name
    db: str | None = None              # namespace (HGNC, FPLX, CHEBI, MESH)
    db_id: str | None = None
    aliases: list[str] = field(default_factory=list)
    all_names: list[str] = field(default_factory=list)
    is_family: bool = False
    family_members: list[str] = field(default_factory=list)
    description: str = ""
    is_pseudogene: bool = False

    # Verification: raw_text → claim entity mapping
    verification_status: str | None = None  # MATCH, MISMATCH, AMBIGUOUS, UNRESOLVABLE
    verification_note: str = ""
    gilda_score: float | None = None        # gilda score for raw_text → claim mapping
    is_low_confidence: bool = False
    is_known_alias: bool = False
    competing_candidates: list[dict] = field(default_factory=list)
    text_top_name: str | None = None        # what raw_text resolves to (if different)

    @classmethod
    def resolve(cls, name: str, raw_text: str | None = None) -> GroundedEntity:
        """Resolve an entity name via gilda. One call, all metadata."""
        import gilda

        entity = cls(name=name, raw_text=raw_text)

        if not name or name == "?":
            return entity

        # Ground the claim entity
        matches = _cached_ground(name)
        if not matches:
            return entity

        top = matches[0]
        entity.canonical = top.term.entry_name
        entity.db = top.term.db
        entity.db_id = str(top.term.id)

        # Get names/aliases for HGNC
        if entity.db == "HGNC":
            entity.all_names = _cached_get_names("HGNC", entity.db_id)
            entity.aliases = _filter_aliases(entity.all_names, name, entity.canonical)
            entity.description, entity.is_pseudogene = _cached_get_desc(
                entity.db, entity.db_id
            )

        # Family detection for FPLX
        elif entity.db == "FPLX":
            entity.is_family = True
            entity.family_members = _get_fplx_members(entity.db_id)

        # Verify raw_text → claim mapping (if raw_text differs)
        if raw_text and raw_text.lower() != name.lower():
            entity._verify_raw_text(raw_text)

        return entity

    def _verify_raw_text(self, raw_text: str) -> None:
        """Check whether raw_text maps to the same entity as the claim."""
        from indra_belief.tools.grounding_verifier import LOW_CONFIDENCE_THRESHOLD

        text_results = _cached_ground(raw_text)
        if not text_results:
            self.verification_status = "UNRESOLVABLE"
            self.verification_note = f'"{raw_text}" not found in gene databases'
            return

        if not self.db:
            self.verification_status = "UNRESOLVABLE"
            self.verification_note = f'"{self.name}" not found in gene databases'
            return

        text_top = text_results[0].term
        self.gilda_score = text_results[0].score
        self.text_top_name = text_top.entry_name
        self.is_low_confidence = self.gilda_score <= LOW_CONFIDENCE_THRESHOLD

        # Same (db, id)?
        if text_top.db == self.db and str(text_top.id) == self.db_id:
            self.verification_status = "MATCH"
            self.is_known_alias = any(
                n.lower() == raw_text.lower() for n in self.all_names
            ) if self.db == "HGNC" else False
            self.verification_note = (
                f'"{raw_text}" resolves to {text_top.entry_name} = {self.name}'
            )
            return

        # Check if claim entity is in lower-ranked candidates
        claim_in_candidates = any(
            r.term.db == self.db and str(r.term.id) == self.db_id
            for r in text_results[:5]
        )

        if claim_in_candidates:
            self.verification_status = "AMBIGUOUS"
            for r in text_results[:5]:
                desc, pseudo = _cached_get_desc(r.term.db, str(r.term.id))
                self.competing_candidates.append({
                    "name": r.term.entry_name,
                    "db": r.term.db,
                    "id": str(r.term.id),
                    "score": r.score,
                    "description": desc,
                    "is_pseudogene": pseudo,
                })
            self.verification_note = (
                f'"{raw_text}" most likely refers to {text_top.entry_name}, '
                f'not {self.name}'
            )
        else:
            self.verification_status = "MISMATCH"
            self.verification_note = (
                f'"{raw_text}" resolves to {text_top.entry_name}, '
                f'NOT {self.name}'
            )

    # --- Formatting helpers ---

    def format_alias_context(self) -> str:
        """Format alias context for the LLM prompt."""
        if not self.db:
            return ""

        if self.db == "HGNC":
            if not self.aliases and self.canonical == self.name:
                return ""
            parts = f"{self.name} (HGNC: {self.canonical}"
            if self.aliases:
                parts += f", aliases: {', '.join(self.aliases)}"
            parts += ")"
            return parts

        if self.is_family:
            parts = f"{self.name} (family: {self.canonical}"
            if self.family_members:
                parts += f", members: {', '.join(self.family_members[:6])}"
            parts += " — if text names a specific member, the claim should use that member, not the family)"
            return parts

        return ""

    def format_warning(self) -> str:
        """Format graduated warning (only PSEUDOGENE and LOW_CONFIDENCE)."""
        warnings = []
        if self.is_pseudogene and self.verification_status in ("AMBIGUOUS", None):
            warnings.append(
                f"{self.name} is a PSEUDOGENE (does not encode functional protein)"
            )
        if self.is_low_confidence and self.verification_status == "MATCH" and not self.is_known_alias:
            warnings.append(
                f'"{self.raw_text}" mapped to {self.name} '
                f'(gilda score: {self.gilda_score:.2f}, LOW CONFIDENCE — '
                f'not a known alias for {self.name})'
            )
        return " | ".join(f"⚠ {w}" for w in warnings)

    @property
    def has_grounding_signal(self) -> bool:
        """Whether this entity has a non-trivial grounding result."""
        if self.verification_status in ("MISMATCH", "AMBIGUOUS"):
            return True
        if self.verification_status == "MATCH" and self.is_low_confidence:
            return True
        if self.is_pseudogene:
            return True
        return False

    def should_auto_reject(self, evidence_text: str) -> tuple[bool, str]:
        """Whether Tier 1 should auto-reject based on this entity.

        Returns (should_reject, reason). Includes safety check against
        evidence text to prevent false rejections.
        """
        if self.verification_status == "MISMATCH":
            if not self._entity_in_evidence(evidence_text):
                return True, f"Grounding mismatch: {self.verification_note}"

        if self.verification_status == "MATCH" and self.is_low_confidence:
            # Safety check: if the claim entity name (not the raw_text alias)
            # appears in the evidence, the mapping may be valid.
            # Exclude the raw_text from alias matching — it's the very name
            # that triggered low confidence. e.g., "CagA" matches HGNC alias
            # "CAGA" for S100A8 but CagA is actually an H. pylori protein.
            if not self._entity_in_evidence(evidence_text, exclude_raw_text=True):
                return True, (
                    f'Low-confidence grounding: "{self.raw_text}" mapped to '
                    f'{self.name} (gilda score: {self.gilda_score:.3f} '
                    f'— below confidence threshold)'
                )

        if self.verification_status == "AMBIGUOUS" and self.is_pseudogene:
            return True, f"Pseudogene mapping: {self.name} is a pseudogene. {self.verification_note}"

        return False, ""

    def _entity_in_evidence(self, evidence_text: str, exclude_raw_text: bool = False) -> bool:
        """Check if the claim entity (or aliases) appears in evidence text.

        When exclude_raw_text=True, skip aliases that match the raw_text —
        prevents circular matching where the raw_text that triggered
        LOW_CONFIDENCE also appears as an HGNC alias (e.g., CagA = CAGA).
        """
        ev_lower = evidence_text.lower()
        ev_collapsed = ev_lower.replace("-", "").replace(" ", "")
        ce_low = self.name.lower()

        if ce_low in ev_lower or ce_low.replace("-", "").replace(" ", "") in ev_collapsed:
            return True

        # Check aliases
        rt_low = self.raw_text.lower() if self.raw_text and exclude_raw_text else None
        for alias in self.all_names:
            if len(alias) >= 3:
                a_low = alias.lower()
                # Skip aliases that match the raw_text (circular match)
                if rt_low and a_low == rt_low:
                    continue
                if a_low in ev_lower or a_low.replace("-", "").replace(" ", "") in ev_collapsed:
                    return True

        # Check descriptive name overlap with raw_text
        if self.raw_text:
            rt_words = set(self.raw_text.lower().replace("-", " ").split())
            for alias in self.all_names:
                if len(alias) > 15:
                    n_words = set(alias.lower().replace("-", " ").split())
                    shared = rt_words & n_words - {"protein", "factor", "the", "of", "and", "a"}
                    if len(shared) >= 2:
                        return True

        return False


# --- Cached helpers (shared across all entities) ---

@lru_cache(maxsize=4096)
def _cached_ground(name: str):
    import gilda
    try:
        return gilda.ground(name) or []
    except Exception:
        return []


@lru_cache(maxsize=4096)
def _cached_get_names(db: str, db_id: str) -> list[str]:
    import gilda
    try:
        return gilda.get_names(db, db_id)
    except Exception:
        return []


@lru_cache(maxsize=4096)
def _cached_get_desc(db: str, db_id: str) -> tuple[str, bool]:
    if db != "HGNC":
        return "", False
    names = _cached_get_names(db, db_id)
    descs = sorted(
        [n for n in names if len(n) > 12 and n[0].isupper()],
        key=len, reverse=True,
    )
    desc = descs[0] if descs else ""
    pseudo = any("pseudogene" in n.lower() for n in names)
    return desc, pseudo


def _filter_aliases(aliases: list[str], entity_name: str, canonical: str) -> list[str]:
    """Filter aliases to keep only informative, unambiguous ones."""
    _AMBIGUOUS = {
        "AF-1", "AF1", "AF-2", "AF2", "CD", "PI", "HR", "NR", "AD",
        "BD", "KD", "TF", "Receptor", "Receptors", "Protein", "Ligand",
    }
    candidates = []
    for a in aliases:
        if a == canonical or a == entity_name:
            continue
        if a in _AMBIGUOUS or len(a) <= 1:
            continue
        a_lower = a.lower()
        if a_lower in ("antigen", "protein", "receptor", "ligand", "factor",
                        "kinase", "enzyme", "inhibitor", "substrate"):
            continue
        if a.count(" ") >= 2 or len(a) > 20:
            continue
        is_symbol = len(a) <= 10 and a.count(" ") == 0
        score = (100 - len(a)) if is_symbol else (50 - len(a))
        candidates.append((score, a))
    candidates.sort(key=lambda x: -x[0])
    return [a for _, a in candidates[:6]]


def _get_fplx_members(fplx_id: str) -> list[str]:
    try:
        from indra.ontology.bio import bio_ontology
        children = bio_ontology.get_children("FPLX", fplx_id)
        names = []
        for child_db, child_id in children:
            name = bio_ontology.get_name(child_db, child_id)
            if name:
                names.append(name)
        return sorted(names)
    except Exception:
        return []

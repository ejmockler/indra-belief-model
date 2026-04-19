"""Grounded entity resolution — resolve once, use everywhere.

Replaces scattered gilda.ground() / get_names() / verify_mapping calls
with a single resolution step per entity. The GroundedEntity carries all
identity information through the scoring pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache


# Gilda score threshold below which a text→entity mapping is treated as
# low-confidence. Calibrated empirically: catches CagA→S100A8 (0.521),
# ActA→ACTA1 (0.521) while allowing known aliases through.
LOW_CONFIDENCE_THRESHOLD = 0.53


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

        # Equivalence check 1a: family ancestry (claim is member of text's family).
        # Family-level evidence supports member-level claim.
        if _is_descendant(self.db, self.db_id, text_top.db, str(text_top.id)):
            self.verification_status = "MATCH"
            self.verification_note = (
                f'"{raw_text}" resolves to family {text_top.entry_name}; '
                f'{self.name} is a member (family-level evidence)'
            )
            return

        # Equivalence check 1b: family ancestry (text is member of claim's family).
        # Member-level evidence supports family-level claim. Mirrors the prompt
        # rule "a family-level claim is supported by evidence about any specific
        # family member" and format_alias_context for FPLX entities.
        if _is_descendant(text_top.db, str(text_top.id), self.db, self.db_id):
            self.verification_status = "MATCH"
            self.verification_note = (
                f'"{raw_text}" resolves to {text_top.entry_name}, a member '
                f'of family {self.name} (member-level evidence for family claim)'
            )
            return

        # Equivalence check 2: alias substring match.
        # raw_text may be a descriptive alias of the claim entity (e.g.,
        # "collagenase 1" is an alias of MMP1, "beta-arrestin 2" of ARRB2).
        # Requires at least one specific (non-generic) token overlap to
        # avoid accepting ultra-generic terms like "protein phosphatase".
        if self.db == "HGNC" and _alias_substring_match(raw_text, self.all_names):
            self.verification_status = "MATCH"
            self.verification_note = (
                f'"{raw_text}" matches an alias of {self.name} '
                f'(specific-term overlap)'
            )
            return

        # Equivalence check 3: display-name alignment.
        # If gilda grounded the claim to an entity whose canonical name differs
        # from the claim (ambiguous grounding), but text_top's display name
        # matches the claim name, the raw_text extraction is a better match.
        if (self.canonical
                and self.canonical.lower() != self.name.lower()
                and text_top.entry_name.lower() == self.name.lower()):
            self.verification_status = "MATCH"
            self.verification_note = (
                f'"{raw_text}" resolves to {text_top.entry_name} '
                f'(claim name {self.name} is ambiguous; text extraction agrees)'
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
            parts = f"{self.name} (protein family"
            if self.family_members:
                parts += f", includes {', '.join(self.family_members[:6])}"
            parts += (
                " — a family-level claim is supported by evidence about any"
                " specific family member)"
            )
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

        # LOW_CONFIDENCE auto-reject disabled: 53.6% precision at scale
        # (37 true / 32 false rejections on 3754 records). The LLM handles
        # these records at ~79% accuracy, so letting them through is net positive.
        # The low_confidence signal is still available in format_warning() for
        # the LLM to consider as context if needed.

        if self.verification_status == "AMBIGUOUS" and self.is_pseudogene:
            # Per prompt rule: pseudogene claims are likely wrong UNLESS evidence
            # explicitly describes pseudogene transcripts / lncRNAs. Skip auto-
            # reject when the evidence text discusses pseudogene biology.
            ev_low = evidence_text.lower()
            if any(kw in ev_low for kw in ("pseudogene", "lncrna", "lnc-rna",
                                            "non-coding rna", "noncoding rna")):
                return False, ""
            return True, f"Pseudogene mapping: {self.name} is a pseudogene. {self.verification_note}"

        return False, ""

    def _entity_in_evidence(self, evidence_text: str, exclude_raw_text: bool = False) -> bool:
        """Check if the claim entity (or aliases) appears in evidence text.

        Uses word-boundary matching for short claim names (≤4 chars) to avoid
        false-matches like "AR" hitting "erased" or "MET" hitting "method".
        Longer names use substring match (tolerant to hyphen/space variants).

        When exclude_raw_text=True, skip aliases that match the raw_text —
        prevents circular matching where the raw_text that triggered
        LOW_CONFIDENCE also appears as an HGNC alias (e.g., CagA = CAGA).
        """
        import re

        ev_lower = evidence_text.lower()
        ev_collapsed = ev_lower.replace("-", "").replace(" ", "")
        ce_low = self.name.lower()

        if _text_contains(ce_low, ev_lower, ev_collapsed):
            return True

        # Check aliases
        rt_low = self.raw_text.lower() if self.raw_text and exclude_raw_text else None
        for alias in self.all_names:
            if len(alias) >= 3:
                a_low = alias.lower()
                # Skip aliases that match the raw_text (circular match)
                if rt_low and a_low == rt_low:
                    continue
                if _text_contains(a_low, ev_lower, ev_collapsed):
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


@lru_cache(maxsize=4096)
def _is_descendant(child_db: str | None, child_id: str | None,
                    parent_db: str | None, parent_id: str | None) -> bool:
    """Whether (child_db, child_id) is a descendant of (parent_db, parent_id)
    in the INDRA bio ontology. Used to accept family-level evidence for
    member-level claims (e.g., claim ERK, evidence MAPK → MATCH).
    """
    if not all((child_db, child_id, parent_db, parent_id)):
        return False
    if (child_db, child_id) == (parent_db, parent_id):
        return False  # handled by caller
    try:
        from indra.ontology.bio import bio_ontology
        children = bio_ontology.get_children(parent_db, parent_id)
        for cdb, cid in children:
            if cdb == child_db and str(cid) == str(child_id):
                return True
        # Transitive check: check grandchildren via get_parents on the child
        parents = bio_ontology.get_parents(child_db, child_id)
        for pdb, pid in parents:
            if pdb == parent_db and str(pid) == str(parent_id):
                return True
    except Exception:
        pass
    return False


# Short claim names (≤4 chars) are prone to false-match via unbounded substring
# (e.g., "AR" hits "erased", "MET" hits "method"). Use word-boundary regex for
# these; longer names use fast substring match.
_SHORT_SYMBOL_LEN = 4


def _text_contains(needle: str, haystack: str, haystack_collapsed: str) -> bool:
    """Membership check with letter-boundary enforcement for short symbols.

    Short (≤4 char) needles require LETTER boundaries (neighbor must not be
    a lowercase letter). This keeps "AKT" matching "AKT1" (T→1 = letter→digit,
    not a letter neighbor) and rejects "AR" matching "erased" (a→r = letter
    neighbor). Digit-prefix/suffix variants stay matched.

    Known limitation: prefixed-letter abbreviations like "pAKT", "cMET" will
    be MISSED by the short path. That's a FN in `_entity_in_evidence`, so
    the tier-1 safety check fails and auto-reject fires. Accepted trade —
    the old unbounded substring had worse FP behavior for short symbols.

    Longer needles use substring + collapsed-hyphen variant.
    """
    import re

    if not needle:
        return False
    if len(needle) <= _SHORT_SYMBOL_LEN:
        # Letter boundaries: allow digits, hyphens, parens, whitespace as
        # neighbors. Reject only when a lowercase letter is adjacent.
        # Haystack is pre-lowercased by the caller, so `[a-z]` covers all
        # alphabetic neighbors.
        pattern = r'(?<![a-z])' + re.escape(needle) + r'(?![a-z])'
        return bool(re.search(pattern, haystack))
    if needle in haystack:
        return True
    collapsed = needle.replace("-", "").replace(" ", "")
    return collapsed in haystack_collapsed


_GENERIC_TOKENS = {
    "protein", "proteins", "factor", "factors", "receptor", "receptors",
    "ligand", "ligands", "phosphatase", "phosphatases", "kinase", "kinases",
    "enzyme", "enzymes", "subunit", "family", "complex", "domain",
    "binding", "inhibitor", "activator", "substrate", "type",
    "alpha", "beta", "gamma", "delta", "1", "2", "3", "4", "5",
    "the", "of", "and", "a", "an", "for", "to", "by", "or",
}


def _tokenize(text: str) -> set[str]:
    """Normalize to lowercase tokens, splitting on non-alphanumeric AND
    on letter/digit boundaries (so 'collagenase1' → {'collagenase', '1'})."""
    import re
    # Replace Greek letters first
    greek_map = {
        'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta',
        'κ': 'kappa', 'ε': 'epsilon', 'ζ': 'zeta', 'η': 'eta',
    }
    lowered = text.lower()
    for greek, latin in greek_map.items():
        lowered = lowered.replace(greek, latin)
    # Extract contiguous letter-only or digit-only runs.
    tokens = re.findall(r'[a-z]+|\d+', lowered)
    return {t for t in tokens if t}


def _alias_substring_match(raw_text: str, all_names: list[str]) -> bool:
    """Whether raw_text overlaps with claim's alias list on a specific
    (non-generic) token. Requires at least one meaningful shared token.

    Accepts:
      'collagenase1' vs 'Interstitial Collagenase' (shared: 'collagenase')
      'beta-arrestin 2' vs 'Beta-arrestin-2'       (shared: 'arrestin')
    Rejects:
      'protein phosphatase' vs 'protein phosphatase 7, catalytic...'
        (only generic tokens shared)
    """
    rt_tokens = _tokenize(raw_text)
    specific_rt = rt_tokens - _GENERIC_TOKENS
    if not specific_rt:
        return False

    for alias in all_names or []:
        if len(alias) < 3:
            continue
        alias_tokens = _tokenize(alias)
        specific_alias = alias_tokens - _GENERIC_TOKENS
        if specific_rt & specific_alias:
            return True
    return False

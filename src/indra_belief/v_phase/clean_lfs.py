"""V6b — clean (non-substrate) labeling functions.

Per research/v5r_data_prep_doctrine.md §3 (LF tables tagged [clean]),
§10 (no holdout-tuned features), §11 (no curator gold leakage).

Each LF takes (statement, evidence) — or, for verify_grounding, (entity,
evidence) — and returns an integer class index ∈ {0..K-1} for the probe's
class enum or -1 (ABSTAIN). Snorkel's LabelModel consumes the integer
column directly.

Constraints honored (V6b task brief):
  - LFs return -1 (ABSTAIN) on the no-vote path; never default to a class.
  - No `mode` parameter — these LFs have no tuned/baseline distinction.
  - NO consumption of context_builder._CYTOKINE_LIGAND_HGNC, _SITE_DENYLIST,
    _HEDGE_MARKERS, _LOF_PATTERNS, CATALOG, or _binding_admissible_for.
    Substrate-tuned constants are routed through V6a (substrate_lfs.py).
  - Lexicons (hedge cues, amount cues, chain markers, negation patterns)
    are inline from open-source lexicons (LingScope-derived hedge cues,
    BioScope-derived negation cues, BIOMED stopwords) NOT from holdout
    audit.
  - Class index maps imported from substrate_lfs.py — same convention.
"""
from __future__ import annotations

import re
from typing import Any

from indra_belief.v_phase.substrate_lfs import (
    ABSTAIN,
    GROUNDING_CLASSES,
    RELATION_AXIS_CLASSES,
    ROLE_CLASSES,
    SCOPE_CLASSES,
    _claim_axis_sign,
    _claim_stmt_type,
    _claim_subject_object,
    _evidence_text,
)


# ---------------------------------------------------------------------------
# Inline lexicons (open-source provenance — see V5r §3.3, §3 doctrine)
# ---------------------------------------------------------------------------

# LingScope-derived speculative/hedge cues. Compact subset (~30 cues) that
# survives in cross-corpus hedge-detection benchmarks. Standalone — does NOT
# import _HEDGE_MARKERS from substrate.
_HEDGE_CUES_CLEAN: tuple[str, ...] = (
    "may", "might", "could", "would",
    "suggest", "suggests", "suggested", "suggesting",
    "propose", "proposes", "proposed",
    "hypothesize", "hypothesized", "hypothesise",
    "putative", "potential", "potentially",
    "appear", "appears", "appeared",
    "seem", "seems", "seemed", "seemingly",
    "likely", "possibly", "perhaps", "presumably",
    "indicate", "indicates", "indicated",
    "imply", "implies", "implied",
    "consistent\\s+with", "in\\s+keeping\\s+with",
)

_HEDGE_RE = re.compile(
    r"\b(?:" + "|".join(_HEDGE_CUES_CLEAN) + r")\b",
    re.IGNORECASE,
)

# BioScope-derived clean negation cue list (§3.3 doctrine wording).
_NEGATION_RE_CLEAN = re.compile(
    r"\b(?:no|not|never|failed?\s+to|did\s+not|does\s+not|do\s+not|"
    r"cannot|can\s+not|"
    r"lack(?:ed|s)?\s+(?:of|to)|"
    r"absence\s+of|absent|"
    r"without)\b",
    re.IGNORECASE,
)

# Conditional clause regex (§3.3). Open-source phrasing.
_CONDITIONAL_RE_CLEAN = re.compile(
    r"\b(?:in|under|upon|during|after|with)\s+"
    r"(?:wild[\s-]type|mutant|presence\s+of|absence\s+of|"
    r"knockout|knockdown|the\s+presence\s+of|the\s+absence\s+of)\b",
    re.IGNORECASE,
)

# Amount-axis lexicon (§3.1 doctrine wording verbatim).
_AMOUNT_LEXICON: tuple[str, ...] = (
    "expression", "abundance", "level", "levels",
    "protein levels", "transcript", "mRNA",
    "upregulat", "downregulat",
)
_AMOUNT_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(k).replace("\\ ", "\\s+")
                         for k in _AMOUNT_LEXICON) + r")",
    re.IGNORECASE,
)

# Activity-axis lexicon used by amount_keyword_negative (§3.1).
_ACTIVITY_LEXICON: tuple[str, ...] = (
    "phosphorylat", "activat", "binds", "bind to",
)
_ACTIVITY_RE = re.compile(
    r"(?:" + "|".join(re.escape(k).replace("\\ ", "\\s+")
                       for k in _ACTIVITY_LEXICON) + r")",
    re.IGNORECASE,
)

# Chain markers (§3.1) — open-source phrasing.
_CHAIN_MARKERS_CLEAN: tuple[str, ...] = (
    "thereby", "which then", "mediated by", "via",
    "through", "leading to",
)
_CHAIN_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(m).replace("\\ ", "\\s+")
                         for m in _CHAIN_MARKERS_CLEAN) + r")\b",
    re.IGNORECASE,
)

# DNA-binding-element lexicon (§3.1, partner_dna_lexical).
_DNA_BINDING_LEXICON: tuple[str, ...] = (
    "promoter", "enhancer", "binding site", "motif",
    "consensus sequence", "DNA-binding element",
    "DNA binding element", "response element",
)
_DNA_BINDING_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(k).replace("\\ ", "\\s+")
                         for k in _DNA_BINDING_LEXICON) + r")",
    re.IGNORECASE,
)

# Boilerplate patterns for low_information_evidence (§3.3).
_BOILERPLATE_PATTERNS: tuple[str, ...] = (
    r"this\s+is\s+consistent\s+with",
    r"previous\s+studies\s+suggest",
    r"future\s+work\s+will",
    r"further\s+studies?\s+are?\s+(?:needed|required)",
)
_BOILERPLATE_RE = re.compile(
    "|".join(_BOILERPLATE_PATTERNS), re.IGNORECASE,
)

# Curated databases per §3.1 — direct by convention.
_CURATED_DB_APIS: frozenset[str] = frozenset({
    "hprd", "biopax", "signor", "bel", "trrust",
})

# REACH `found_by` axis prefixes — pattern-IDs encode axis + sign as
# "Positive_activation_syntax_…" / "Phosphorylation_syntax_…".
# axis name → (canonical_axis, sign | None when sign is in prefix)
_REACH_AXIS_TOKENS: dict[str, tuple[str, str | None]] = {
    "activation":          ("activity", None),       # sign in Positive_/Negative_ prefix
    "phosphorylation":     ("modification", "positive"),
    "dephosphorylation":   ("modification", "negative"),
    "acetylation":         ("modification", "positive"),
    "deacetylation":       ("modification", "negative"),
    "methylation":         ("modification", "positive"),
    "demethylation":       ("modification", "negative"),
    "ubiquitination":      ("modification", "positive"),
    "deubiquitination":    ("modification", "negative"),
    "transcription":       ("amount", "positive"),
    "translation":         ("amount", "positive"),
    "translocation":       ("localization", "neutral"),
    "binding":             ("binding", "neutral"),
    "complex_assembly":    ("binding", "neutral"),
    "regulation":          ("activity", None),
    "positive_regulation": ("activity", "positive"),
    "negative_regulation": ("activity", "negative"),
    "expression":          ("amount", "positive"),
    "amount":              ("amount", None),
}

_REACH_FOUND_BY_RE = re.compile(
    r"^(?P<sign>Positive_|Negative_)?(?P<axis>[a-z_]+?)_syntax",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ev_get(evidence: Any, key: str, default: Any = None) -> Any:
    """Tolerant getter — accepts dict or INDRA Evidence."""
    if isinstance(evidence, dict):
        return evidence.get(key, default)
    return getattr(evidence, key, default) or default


def _annotations(evidence: Any) -> dict:
    """Return annotations dict, never None."""
    a = _ev_get(evidence, "annotations", None)
    return a if isinstance(a, dict) else {}


def _epistemics(evidence: Any) -> dict:
    """Return epistemics dict, never None."""
    e = _ev_get(evidence, "epistemics", None)
    return e if isinstance(e, dict) else {}


def _source_api(evidence: Any, statement: Any = None) -> str:
    """Return source_api as lowercase string, falling back to the statement
    record (V5r training shape carries source_api at statement level)."""
    api = _ev_get(evidence, "source_api", None)
    if api:
        return api.lower()
    if isinstance(statement, dict):
        return (statement.get("source_api") or "").lower()
    return ""


def _evidence_tokens(text: str) -> list[str]:
    """Whitespace tokenization — used for token-window thresholds."""
    return text.split()


def _word_re(name: str) -> re.Pattern[str]:
    """Word-boundary regex for short or long names."""
    if not name:
        return re.compile(r"$x")  # never matches
    return re.compile(r"\b" + re.escape(name) + r"\b", re.IGNORECASE)


def _entity_in_text(name: str, text: str) -> bool:
    """Word-boundary membership test."""
    if not name or not text:
        return False
    return bool(_word_re(name).search(text))


def _gilda_aliases(name: str) -> list[str]:
    """Look up the FULL alias list for a name — used by absence LFs to
    confirm the entity isn't referenced under any synonym.

    V6e Fix 4: previously returned just `ge.all_names` (HGNC primary
    list). Now also pulls UniProt synonyms when an HGNC→UniProt mapping
    is available, so e.g. APP includes 'A4' / 'amyloid precursor protein
    A4', IL6 includes 'interleukin-6 (interferon, beta 2)'. The
    enriched list reduces false `absent` votes when curators grounded
    the claim entity via a non-HGNC alias.
    """
    if not name:
        return []
    try:
        from indra_belief.data.entity import GroundedEntity
        ge = GroundedEntity.resolve(name)
        out: list[str] = list(ge.all_names or [])
        seen = {a.lower() for a in out if isinstance(a, str)}
        # UniProt synonym path: HGNC db_id → UniProt id via the bio
        # ontology (when available), then gilda.get_names("UP", up_id).
        if ge.db == "HGNC" and ge.db_id:
            try:
                from indra.ontology.bio import bio_ontology
                xrefs = bio_ontology.get_mappings("HGNC", ge.db_id) or []
                up_ids = [xid for (xdb, xid) in xrefs if xdb == "UP"]
            except Exception:
                up_ids = []
            for up_id in up_ids[:3]:
                try:
                    import gilda
                    up_names = gilda.get_names("UP", up_id) or []
                except Exception:
                    up_names = []
                for n in up_names:
                    if not isinstance(n, str) or len(n) < 2:
                        continue
                    if n.lower() in seen:
                        continue
                    out.append(n)
                    seen.add(n.lower())
        return out
    except Exception:
        return []


def _entity_in_text_tolerant(name: str, text: str) -> bool:
    """V6e Fix 4: tolerant membership test for short-symbol absence LFs.

    Strict `_entity_in_text` uses word-boundary regex (`\\b{name}\\b`).
    That misses hyphen/space variants, e.g., `'Plk-1'` vs `'Plk 1'` vs
    `'Plk1'`. This helper additionally checks a normalized-form match:
    strip non-alphanumerics from both sides and compare with letter
    boundaries when the symbol is short.
    """
    if not name or not text:
        return False
    if _entity_in_text(name, text):
        return True
    # Normalized: drop hyphens/spaces, compare on collapsed form.
    norm_name = re.sub(r"[\s\-_]+", "", name).lower()
    if len(norm_name) < 3:
        # Too short to safely substring-match on collapsed form.
        return False
    norm_text = re.sub(r"[\s\-_]+", "", text).lower()
    if norm_name in norm_text:
        # Letter-boundary safety check: avoid e.g. "PLK1" matching "PLK10".
        # Pattern: in the collapsed text, neighbor must not be a letter.
        idx = 0
        while True:
            j = norm_text.find(norm_name, idx)
            if j < 0:
                break
            before = norm_text[j - 1] if j > 0 else ""
            after = norm_text[j + len(norm_name)] if j + len(norm_name) < len(norm_text) else ""
            if not (before.isalpha() or after.isalpha()):
                return True
            idx = j + 1
    return False


# ---------------------------------------------------------------------------
# relation_axis LFs
# ---------------------------------------------------------------------------

def lf_curated_db_axis(statement, evidence) -> int:
    """Vote on the relation axis when source_api is a curated database.

    Per V5r §3.1: curated DBs (HPRD, BioPAX, SIGNOR, BEL, TRRUST) encode
    direct interactions by curation convention. V6f refinement:
      - ABSTAIN when evidence text is empty (a non-trivial fraction of
        curated-DB entries are claim-level annotations with no curatable
        text; those records are labeled `no_relation` by curators because
        there is no evidence span to evaluate).
      - For amount-axis stmts (IncreaseAmount / DecreaseAmount), vote
        `direct_amount_match` (1) — SIGNOR and TRRUST carry expression
        regulation as amount axis.
      - For all other stmt_types, vote `direct_sign_match` (0).
      - ABSTAIN when stmt_type is unknown.
    """
    api = _source_api(evidence, statement)
    if not api:
        return ABSTAIN
    if api not in _CURATED_DB_APIS:
        return ABSTAIN
    # V6f guard: empty evidence → ABSTAIN.
    text = _evidence_text(evidence, statement)
    if not text or not text.strip():
        return ABSTAIN
    stmt_type = _claim_stmt_type(statement)
    if stmt_type in ("IncreaseAmount", "DecreaseAmount"):
        return RELATION_AXIS_CLASSES["direct_amount_match"]
    if stmt_type is None:
        return ABSTAIN
    return RELATION_AXIS_CLASSES["direct_sign_match"]


def _parse_reach_found_by(found_by: str) -> tuple[str | None, str | None]:
    """Parse REACH `found_by` pattern_id → (canonical_axis, sign).

    Pattern format examples:
      `Positive_activation_syntax_1_verb` → ("activity", "positive")
      `Negative_activation_syntax_1_verb` → ("activity", "negative")
      `Phosphorylation_syntax_1a_noun`    → ("modification", "positive")
      `Acetylation_syntax_1a_noun`        → ("modification", "positive")
    """
    if not isinstance(found_by, str) or not found_by:
        return None, None
    m = _REACH_FOUND_BY_RE.match(found_by)
    if m:
        sign_prefix = (m.group("sign") or "").lower().rstrip("_")
        axis_token = (m.group("axis") or "").lower()
        spec = _REACH_AXIS_TOKENS.get(axis_token)
        if spec is None:
            return None, None
        canon_axis, fixed_sign = spec
        if fixed_sign is not None:
            return canon_axis, fixed_sign
        # Sign comes from the Positive_/Negative_ prefix.
        if sign_prefix == "positive":
            return canon_axis, "positive"
        if sign_prefix == "negative":
            return canon_axis, "negative"
        return canon_axis, None
    # Some REACH IDs lead with the axis without the syntax suffix.
    head = found_by.split("_", 1)[0].lower()
    if head in _REACH_AXIS_TOKENS:
        spec = _REACH_AXIS_TOKENS[head]
        return spec[0], spec[1]
    return None, None


def lf_reach_found_by_axis_match(statement, evidence) -> int:
    """Vote `direct_sign_match` (0) when REACH `annotations.found_by`
    parses to the claim's stmt_type axis AND sign matches.

    Vote `direct_sign_mismatch` (2) when axis matches but sign opposite.

    Per V5r §3.1. ABSTAIN when source_api != reach, found_by missing, or
    parse yields a different axis (handled by axis_mismatch LF below).
    """
    if _source_api(evidence, statement) != "reach":
        return ABSTAIN
    found_by = _annotations(evidence).get("found_by", "")
    parsed_axis, parsed_sign = _parse_reach_found_by(found_by)
    if parsed_axis is None:
        return ABSTAIN
    claim_axis, claim_sign = _claim_axis_sign(statement)
    if claim_axis is None or parsed_axis != claim_axis:
        return ABSTAIN
    # Axis matches — decide on sign.
    if parsed_sign is None or claim_sign is None:
        return ABSTAIN
    # neutral-axis claims (binding/localization) take any parse as match.
    if claim_sign == "neutral" or parsed_sign == "neutral":
        return RELATION_AXIS_CLASSES["direct_sign_match"]
    if parsed_sign == claim_sign:
        return RELATION_AXIS_CLASSES["direct_sign_match"]
    return RELATION_AXIS_CLASSES["direct_sign_mismatch"]


def lf_reach_found_by_axis_mismatch(statement, evidence) -> int:
    """Vote `direct_axis_mismatch` (3) when REACH `found_by` parses to a
    different axis than the claim's stmt_type axis.

    Per V5r §3.1: e.g., Translocation parse on Activation claim.
    """
    if _source_api(evidence, statement) != "reach":
        return ABSTAIN
    found_by = _annotations(evidence).get("found_by", "")
    parsed_axis, _parsed_sign = _parse_reach_found_by(found_by)
    if parsed_axis is None:
        return ABSTAIN
    claim_axis, _ = _claim_axis_sign(statement)
    if claim_axis is None:
        return ABSTAIN
    if parsed_axis != claim_axis:
        return RELATION_AXIS_CLASSES["direct_axis_mismatch"]
    return ABSTAIN


def lf_epistemics_direct_true(statement, evidence) -> int:
    """Vote `direct_sign_match` (0) when `epistemics.direct == True`.

    Per V5r §3.1.
    """
    if _epistemics(evidence).get("direct") is True:
        return RELATION_AXIS_CLASSES["direct_sign_match"]
    return ABSTAIN


# V6f: dropped lf_epistemics_direct_false (V7b 0% on 13 fires).
# medscan epistemics.direct=False is unreliable as a via_mediator signal;
# the field is set on directly-extracted relations more often than not.


def lf_multi_extractor_axis_agreement(statement, evidence) -> int:
    """Vote `direct_sign_match` (0) when statement has ≥2 distinct
    source_apis from its evidences (indicating multi-extractor agreement
    on the claim type).

    Per V5r §3.1. The signal is statement-level (`source_counts` carries
    per-API evidence counts); we apply per-evidence by reading
    `statement['source_counts']` if present.
    """
    if not isinstance(statement, dict):
        return ABSTAIN
    counts = statement.get("source_counts") or {}
    if not isinstance(counts, dict):
        return ABSTAIN
    distinct_apis = sum(1 for v in counts.values() if isinstance(v, int) and v > 0)
    if distinct_apis >= 2:
        return RELATION_AXIS_CLASSES["direct_sign_match"]
    return ABSTAIN


def _within_chars(text: str, target_pos: int, name: str, max_dist: int) -> bool:
    """Whether `name` appears in `text` within `max_dist` chars of position
    `target_pos`. Word-boundary."""
    if not name or not text:
        return False
    for m in _word_re(name).finditer(text):
        if abs(m.start() - target_pos) <= max_dist:
            return True
    return False


def lf_amount_lexical(statement, evidence) -> int:
    """Vote `direct_amount_match` (1) when the claim is Activation/
    Inhibition AND the evidence text contains amount-axis lexicon
    within 50 chars of either claim entity.

    Per V5r §3.1.
    """
    stmt_type = _claim_stmt_type(statement)
    if stmt_type not in ("Activation", "Inhibition"):
        return ABSTAIN
    text = _evidence_text(evidence, statement)
    if not text:
        return ABSTAIN
    subj, obj = _claim_subject_object(statement)
    if not subj and not obj:
        return ABSTAIN
    for am in _AMOUNT_RE.finditer(text):
        am_pos = am.start()
        for name in (subj, obj):
            if name and _within_chars(text, am_pos, name, 50):
                return RELATION_AXIS_CLASSES["direct_amount_match"]
    return ABSTAIN


def lf_amount_keyword_negative(statement, evidence) -> int:
    """Vote `direct_axis_mismatch` (3) when the claim is IncreaseAmount/
    DecreaseAmount AND the evidence text contains activity-axis lexicon
    AND does NOT contain amount-axis lexicon.

    Per V5r §3.1: amount claim where evidence describes activity instead.
    """
    stmt_type = _claim_stmt_type(statement)
    if stmt_type not in ("IncreaseAmount", "DecreaseAmount"):
        return ABSTAIN
    text = _evidence_text(evidence, statement)
    if not text:
        return ABSTAIN
    if not _ACTIVITY_RE.search(text):
        return ABSTAIN
    if _AMOUNT_RE.search(text):
        return ABSTAIN
    return RELATION_AXIS_CLASSES["direct_axis_mismatch"]


def lf_chain_with_named_intermediate(statement, evidence) -> int:
    """Vote `via_mediator` (5) when the lexical pattern
    `<claim_subj> ... <X> ... <chain_marker> ... <claim_obj>` appears
    where X is an HGNC-shaped symbol AND X is neither claim entity.

    Per V5r §3.1.
    """
    text = _evidence_text(evidence, statement)
    if not text:
        return ABSTAIN
    subj, obj = _claim_subject_object(statement)
    if not subj or not obj:
        return ABSTAIN
    s_match = _word_re(subj).search(text)
    o_match = _word_re(obj).search(text)
    if not s_match or not o_match:
        return ABSTAIN
    chain_match = _CHAIN_RE.search(text)
    if not chain_match:
        return ABSTAIN
    # Ordering: subj must come before chain marker, obj after.
    if not (s_match.start() < chain_match.start() < o_match.end()):
        return ABSTAIN
    # Look for an HGNC-shaped symbol between subj and obj that is NOT
    # one of the claim entities. HGNC symbols are 2-10 alnum chars,
    # uppercase first letter or all-caps.
    span_lo = s_match.end()
    span_hi = o_match.start()
    if span_hi <= span_lo:
        return ABSTAIN
    middle = text[span_lo:span_hi]
    for m in re.finditer(r"\b([A-Z][A-Z0-9-]{1,9})\b", middle):
        candidate = m.group(1)
        if candidate.upper() in (subj.upper(), obj.upper()):
            continue
        if len(candidate) < 2:
            continue
        return RELATION_AXIS_CLASSES["via_mediator"]
    return ABSTAIN


_VERB_RE = re.compile(
    r"\b(?:phosphorylates?|activates?|inhibits?|binds?|"
    r"interacts?|associates?|recruits?|induces?|"
    r"upregulates?|downregulates?|increases?|decreases?|"
    r"stimulates?|suppresses?|represses?|enhances?|"
    r"forms?|complex(?:es)?|expresses?|methylates?|acetylates?|"
    r"ubiquitinates?|catalyz(?:es|e)|cleaves?|degrades?)\b",
    re.IGNORECASE,
)


def lf_no_entity_overlap(statement, evidence) -> int:
    """Vote `no_relation` (7) when neither claim entity appears in
    evidence text within ±20 tokens of any verb.

    Per V5r §3.1: post-Gilda alias resolution. We check the claim entity
    name plus any aliases via Gilda lookup.
    """
    text = _evidence_text(evidence, statement)
    if not text:
        return ABSTAIN
    subj, obj = _claim_subject_object(statement)
    if not subj or not obj:
        return ABSTAIN
    tokens = _evidence_tokens(text)
    if not tokens:
        return ABSTAIN
    # Verb token positions.
    verb_token_idxs: list[int] = []
    for i, tok in enumerate(tokens):
        if _VERB_RE.fullmatch(tok.strip(".,;:!?'\"")):
            verb_token_idxs.append(i)
    if not verb_token_idxs:
        return ABSTAIN
    # Gather aliases for both entities (cheap, lru-cached).
    candidates: set[str] = set()
    for name in (subj, obj):
        candidates.add(name)
        for a in _gilda_aliases(name):
            if isinstance(a, str) and 2 <= len(a) <= 30:
                candidates.add(a)
    # Token positions where any alias appears.
    alias_token_idxs: set[int] = set()
    for i, tok in enumerate(tokens):
        clean = tok.strip(".,;:!?'\"()")
        for cand in candidates:
            if not cand:
                continue
            # Case-insensitive equality on alphanumeric core.
            if clean.lower() == cand.lower():
                alias_token_idxs.add(i)
                break
    # If any alias is within ±20 tokens of any verb, this LF cannot vote.
    for vi in verb_token_idxs:
        for ai in alias_token_idxs:
            if abs(ai - vi) <= 20:
                return ABSTAIN
    # No alias near any verb → vote no_relation.
    return RELATION_AXIS_CLASSES["no_relation"]


def lf_partner_dna_lexical(statement, evidence) -> int:
    """Vote `direct_partner_mismatch` (4) when the claim is Complex
    AND evidence shows binding to DNA-binding-element lexicon on the
    claim subject.

    Per V5r §3.1: signals that the binding partner is DNA, not the
    claim's protein object.
    """
    stmt_type = _claim_stmt_type(statement)
    if stmt_type != "Complex":
        return ABSTAIN
    text = _evidence_text(evidence, statement)
    if not text:
        return ABSTAIN
    subj, _obj = _claim_subject_object(statement)
    if not subj:
        return ABSTAIN
    if not _DNA_BINDING_RE.search(text):
        return ABSTAIN
    s_match = _word_re(subj).search(text)
    if not s_match:
        return ABSTAIN
    # DNA element within 80 chars of the subject mention → partner mismatch.
    for dm in _DNA_BINDING_RE.finditer(text):
        if abs(dm.start() - s_match.start()) <= 80:
            return RELATION_AXIS_CLASSES["direct_partner_mismatch"]
    return ABSTAIN


# ---------------------------------------------------------------------------
# subject_role / object_role LFs
# ---------------------------------------------------------------------------

# Active verb regex used for role-swap detection (claim subject appearing in
# grammatical-object position).
_ACTIVE_VERB_RE = re.compile(
    r"\b(?:phosphorylates?|activates?|inhibits?|binds?|interacts?|"
    r"recruits?|induces?|stimulates?|suppresses?|"
    r"increases?|decreases?|methylates?|acetylates?|cleaves?)\b",
    re.IGNORECASE,
)


def _resolve_role(statement: Any, role: str) -> str | None:
    """Resolve the named role to its claim entity name.

    `role` ∈ {"subject", "object"}. Returns the entity name string or None
    when the role isn't present in the statement.
    """
    subj, obj = _claim_subject_object(statement)
    if role == "subject":
        return subj
    if role == "object":
        return obj
    return None


def lf_position_subject(statement, evidence, *, role: str) -> int:
    """Vote `present_as_subject` (0) when the entity for the named role
    matches the source-API extracted statement's subject position.

    Per V5r §3.2: the statement's subject IS the entity in subject role
    if `_claim_subject_object()` puts that entity at index 0.
    """
    target = _resolve_role(statement, role)
    if not target:
        return ABSTAIN
    subj, _obj = _claim_subject_object(statement)
    if not subj:
        return ABSTAIN
    if target.lower() == subj.lower():
        return ROLE_CLASSES["present_as_subject"]
    return ABSTAIN


def lf_position_object(statement, evidence, *, role: str) -> int:
    """Vote `present_as_object` (1) when the entity for the named role
    matches the statement's object position.

    Per V5r §3.2.
    """
    target = _resolve_role(statement, role)
    if not target:
        return ABSTAIN
    _subj, obj = _claim_subject_object(statement)
    if not obj:
        return ABSTAIN
    if target.lower() == obj.lower():
        return ROLE_CLASSES["present_as_object"]
    return ABSTAIN


def lf_role_swap_lexical(statement, evidence, *, role: str) -> int:
    """Vote `present_as_object` (1) for the claim subject, or
    `present_as_subject` (0) for the claim object, when lexical pattern
    detects the entity in grammatically-swapped position.

    Pattern: `<claim_obj> <active-verb> <claim_subj>` — claim subject
    grammatically the object of evidence verb. Per V5r §3.2.
    """
    text = _evidence_text(evidence, statement)
    if not text:
        return ABSTAIN
    subj, obj = _claim_subject_object(statement)
    target = _resolve_role(statement, role)
    if not (subj and obj and target):
        return ABSTAIN
    s_match = _word_re(subj).search(text)
    o_match = _word_re(obj).search(text)
    v_match = _ACTIVE_VERB_RE.search(text)
    if not (s_match and o_match and v_match):
        return ABSTAIN
    # swap detected: object position before verb, subject after verb.
    if not (o_match.end() <= v_match.start() < v_match.end() <= s_match.start()):
        return ABSTAIN
    if role == "subject":
        return ROLE_CLASSES["present_as_object"]
    if role == "object":
        return ROLE_CLASSES["present_as_subject"]
    return ABSTAIN


_CHAIN_POS_PATTERNS: tuple[str, ...] = (
    r"\bvia\s+{name}\b",
    r"\bthrough\s+{name}\b",
    r"\bmediated\s+by\s+{name}\b",
    r"\b{name}-dependent\b",
    r"\b{name}\s+dependent\b",
)


def lf_chain_position_lexical(statement, evidence, *, role: str) -> int:
    """Vote `present_as_mediator` (2) when the entity for the named role
    appears in `via X`, `through X`, `mediated by X`, `X-dependent`
    patterns AND the evidence has non-empty A and B context (i.e., the
    text includes both an upstream and a downstream entity reference).

    Per V5r §3.2.
    """
    text = _evidence_text(evidence, statement)
    if not text:
        return ABSTAIN
    target = _resolve_role(statement, role)
    if not target:
        return ABSTAIN
    name_re = re.escape(target)
    for tmpl in _CHAIN_POS_PATTERNS:
        pat = tmpl.format(name=name_re)
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            # Need at least some text before and after the match (context).
            before = text[:m.start()].strip()
            after = text[m.end():].strip()
            if len(before) >= 5 and len(after) >= 5:
                return ROLE_CLASSES["present_as_mediator"]
    return ABSTAIN


def lf_decoy_lexical(statement, evidence, *, role: str) -> int:
    """Vote `present_as_decoy` (3) when the entity is present in evidence
    but in a relation pattern that does NOT match the claim's stmt_type
    axis.

    Per V5r §3.2: e.g., claim is Phosphorylation, but evidence shows
    entity in a binding/expression pattern.
    """
    text = _evidence_text(evidence, statement)
    if not text:
        return ABSTAIN
    target = _resolve_role(statement, role)
    if not target:
        return ABSTAIN
    if not _entity_in_text(target, text):
        return ABSTAIN
    claim_axis, _ = _claim_axis_sign(statement)
    if claim_axis is None:
        return ABSTAIN
    # Token-level axis detection from a small inline lexicon.
    axis_lex_map = {
        "modification": (r"\bphosphorylat", r"\bacetylat",
                          r"\bmethylat", r"\bubiquitinat"),
        "activity":     (r"\bactivat", r"\binhibit", r"\bstimulat",
                          r"\bsuppress"),
        "amount":       (r"\bexpression", r"\babundance",
                          r"\blevel", r"\btranscript", r"\bmRNA",
                          r"\bupregulat", r"\bdownregulat"),
        "binding":      (r"\bbind", r"\binteract", r"\bcomplex"),
        "localization": (r"\btranslocat", r"\blocali[sz]", r"\bnuclear"),
    }
    axes_observed: set[str] = set()
    for ax, patterns in axis_lex_map.items():
        for p in patterns:
            if re.search(p, text, re.IGNORECASE):
                axes_observed.add(ax)
                break
    if not axes_observed:
        return ABSTAIN
    # If only non-claim-axis patterns observed → decoy.
    if claim_axis not in axes_observed and axes_observed:
        return ROLE_CLASSES["present_as_decoy"]
    return ABSTAIN


def lf_no_grounded_match(statement, evidence, *, role: str) -> int:
    """Vote `absent` (4) when Gilda finds no match for the entity name in
    evidence text.

    V6e Fix 4: previous version used the strict word-boundary
    `_entity_in_text` for both the official name and aliases, missing
    hyphen/space/case variants ('Plk-1' vs 'PLK1', 'IL 6' vs 'IL6')
    and missing UniProt synonyms not in HGNC's all_names. Now uses the
    tolerant `_entity_in_text_tolerant` (collapsed-form match for
    longer names) AND the broadened alias list (HGNC all_names +
    UniProt synonyms via `_gilda_aliases`).
    """
    text = _evidence_text(evidence, statement)
    if not text:
        return ABSTAIN
    target = _resolve_role(statement, role)
    if not target:
        return ABSTAIN
    if _entity_in_text_tolerant(target, text):
        return ABSTAIN
    # No literal match — confirm via Gilda alias check (broadened to
    # include HGNC alias_symbol + UniProt synonyms).
    aliases = _gilda_aliases(target)
    for a in aliases:
        if isinstance(a, str) and len(a) >= 2 and _entity_in_text_tolerant(a, text):
            return ABSTAIN
    return ROLE_CLASSES["absent"]


def lf_absent_alias_check(statement, evidence, *, role: str) -> int:
    """Vote `absent` (4) when the claim entity name + ALL aliases
    (HGNC + UniProt + commonly-used variants) are not found in evidence
    text via tolerant matching.

    V6e Fix 4: same broadening as `lf_no_grounded_match`. The two LFs
    differ in framing (no_grounded_match expresses "Gilda lookup fails"
    intent; absent_alias_check expresses "deterministic exhaustive
    check") but share the underlying alias list and tolerant match.
    """
    text = _evidence_text(evidence, statement)
    if not text:
        return ABSTAIN
    target = _resolve_role(statement, role)
    if not target:
        return ABSTAIN
    # Entity name itself.
    if _entity_in_text_tolerant(target, text):
        return ABSTAIN
    # All aliases (HGNC all_names + UniProt synonyms).
    aliases = _gilda_aliases(target)
    for a in aliases:
        if not isinstance(a, str) or len(a) < 2:
            continue
        if _entity_in_text_tolerant(a, text):
            return ABSTAIN
    return ROLE_CLASSES["absent"]


# ---------------------------------------------------------------------------
# scope LFs
# ---------------------------------------------------------------------------

def _claim_verb_position(text: str, statement: Any) -> int | None:
    """Approximate position of the claim verb in evidence — returns the
    position of any active verb between the claim subject and object,
    falling back to the first active verb in text. None if no verb."""
    subj, obj = _claim_subject_object(statement)
    if subj and obj:
        s = _word_re(subj).search(text)
        o = _word_re(obj).search(text)
        if s and o:
            lo, hi = sorted((s.start(), o.end()))
            span = text[lo:hi]
            vm = _ACTIVE_VERB_RE.search(span)
            if vm:
                return lo + vm.start()
    vm = _ACTIVE_VERB_RE.search(text)
    if vm:
        return vm.start()
    return None


# V6e Fix 1: claim-verb cue map. Each stmt_type maps to a regex stem that
# the LF anchors on for hedge/negation scoping. The cue must match
# the claim's actual verb, NOT any active verb in evidence text.
_CLAIM_VERB_CUE_MAP: dict[str, str] = {
    "Phosphorylation":     r"phosphorylat",
    "Dephosphorylation":   r"dephosphorylat",
    "Acetylation":         r"acetylat",
    "Deacetylation":       r"deacetylat",
    "Methylation":         r"methylat",
    "Demethylation":       r"demethylat",
    "Ubiquitination":      r"ubiquitinat",
    "Deubiquitination":    r"deubiquitinat",
    "Autophosphorylation": r"phosphorylat",
    "Activation":          r"activat|stimulat|induc|enhanc|promot",
    "Inhibition":          r"inhibit|suppress|repress|block|abolish|abrogat|attenuat",
    "IncreaseAmount":      r"upregulat|increas|express|induc|elevat",
    "DecreaseAmount":      r"downregulat|decreas|reduc|diminish|loss",
    "Complex":             r"bind|interact|associat|complex|recruit",
    "Translocation":       r"translocat|locali[sz]|transport|import|export|shuttl",
    "Gef":                 r"gef|exchange|activat",
    "Gap":                 r"gap|hydrolys|inactivat",
}


def _claim_verb_cue_positions(text: str, statement: Any) -> list[int]:
    """Return all character positions in `text` where the claim's
    stmt-type verb cue appears (case-insensitive). Empty list if
    stmt_type is unknown or no cue matches.

    Used by V6e-fixed scope LFs to anchor hedge/negation cues on the
    CLAIM verb specifically, not any active verb in evidence text.
    """
    stmt_type = _claim_stmt_type(statement)
    if not stmt_type:
        return []
    cue = _CLAIM_VERB_CUE_MAP.get(stmt_type)
    if not cue:
        return []
    pat = re.compile(r"\b(?:" + cue + r")", re.IGNORECASE)
    return [m.start() for m in pat.finditer(text)]


def _within_token_window(text: str, target_pos: int,
                          pattern: re.Pattern[str], window_tokens: int) -> bool:
    """Whether `pattern` matches within `window_tokens` tokens of
    `target_pos` (character position)."""
    # Convert character window using avg 6 chars/token as a fallback;
    # we measure tokens by walking backward and forward.
    # Simpler: take a substring around target_pos covering ~window_tokens
    # tokens on each side.
    char_window = window_tokens * 8  # ~8 chars per token incl. spaces
    lo = max(0, target_pos - char_window)
    hi = min(len(text), target_pos + char_window)
    return bool(pattern.search(text[lo:hi]))


def _cue_within_window_of_claim_verb(
    text: str, statement: Any, pattern: re.Pattern[str],
    window_chars: int = 50,
) -> bool:
    """Whether `pattern` matches within ±`window_chars` of the CLAIM's
    stmt-type verb cue (NOT any active verb).

    V6e Fix 1: previous code anchored on `_claim_verb_position` which
    falls back to any active verb in evidence; the result is hedge/neg
    cues firing on context surrounding (but not modifying) the claim
    verb. This helper restricts to the claim's verb specifically.
    """
    positions = _claim_verb_cue_positions(text, statement)
    if not positions:
        return False
    for pos in positions:
        lo = max(0, pos - window_chars)
        hi = min(len(text), pos + window_chars)
        if pattern.search(text[lo:hi]):
            return True
    return False


def lf_hedge_lexical(statement, evidence) -> int:
    """Vote `hedged` (1) when LingScope-derived hedge cues appear within
    ~50 chars (≈8 tokens) of the CLAIM stmt-type verb cue.

    V6e Fix 1: previously the LF fell back to any active verb in
    evidence and then to a flat scan. Both paths produced FPs on records
    where hedge cues live in surrounding context (not modifying the
    claim verb). Now ABSTAINs when claim verb cue absent or no hedge
    cue near it.
    """
    text = _evidence_text(evidence, statement)
    if not text:
        return ABSTAIN
    if _cue_within_window_of_claim_verb(text, statement, _HEDGE_RE,
                                         window_chars=50):
        return SCOPE_CLASSES["hedged"]
    return ABSTAIN


# V6f: dropped lf_negation_lexical (V7a 23%, V7b 18%).
# V6f: dropped lf_conditional_lexical (V7a 0%, V7b 20%).
# Replacement: lf_scope_negated_anchored (defined below) requires the
# negation cue to sit IMMEDIATELY BEFORE the claim verb cue (not just
# within a 50-char window).


def lf_clean_assertion(statement, evidence) -> int:
    """Vote `asserted` (0) when none of the hedge/negation/condition
    lexical LFs fire AND claim entities are resolved AND a verb is
    present.

    Per V5r §3.3: clean-assertion default.
    """
    text = _evidence_text(evidence, statement)
    if not text:
        return ABSTAIN
    subj, obj = _claim_subject_object(statement)
    if not subj or not obj:
        return ABSTAIN
    if not (_entity_in_text(subj, text) or _entity_in_text(obj, text)):
        return ABSTAIN
    pos = _claim_verb_position(text, statement)
    if pos is None:
        return ABSTAIN
    if _HEDGE_RE.search(text):
        return ABSTAIN
    if _NEGATION_RE_CLEAN.search(text):
        return ABSTAIN
    if _CONDITIONAL_RE_CLEAN.search(text):
        return ABSTAIN
    return SCOPE_CLASSES["asserted"]


def lf_low_information_evidence(statement, evidence) -> int:
    """Vote `abstain` (4) when evidence text < 12 tokens OR matches
    boilerplate patterns.

    Per V5r §3.3.
    """
    text = _evidence_text(evidence, statement)
    if not text:
        return SCOPE_CLASSES["abstain"]
    tokens = _evidence_tokens(text)
    if len(tokens) < 12:
        return SCOPE_CLASSES["abstain"]
    if _BOILERPLATE_RE.search(text):
        return SCOPE_CLASSES["abstain"]
    return ABSTAIN


def lf_text_too_short(statement, evidence) -> int:
    """Vote `abstain` (4) when evidence text < 6 tokens after split.

    Per V5r §3.3 — companion to low_information_evidence with stricter
    threshold for catching extractor fragments.
    """
    text = _evidence_text(evidence, statement)
    if not text:
        return SCOPE_CLASSES["abstain"]
    if len(_evidence_tokens(text)) < 6:
        return SCOPE_CLASSES["abstain"]
    return ABSTAIN


# ---------------------------------------------------------------------------
# verify_grounding LFs (entity-level)
# ---------------------------------------------------------------------------

def _entity_dict(entity: Any) -> dict:
    """Coerce entity argument to a dict view exposing 'name' and
    optionally 'symbol'/'aliases'."""
    if isinstance(entity, dict):
        return entity
    # GroundedEntity-like: project to dict.
    out: dict = {}
    for k in ("name", "canonical", "all_names", "aliases", "is_family",
              "family_members", "db", "db_id"):
        v = getattr(entity, k, None)
        if v is not None:
            out[k] = v
    return out


def _entity_evidence_text(evidence: Any) -> str:
    if isinstance(evidence, dict):
        t = evidence.get("text") or evidence.get("evidence_text")
        if t:
            return t
    if hasattr(evidence, "text") and evidence.text:
        return evidence.text
    return ""


def _entity_official_symbol(ent: dict) -> str | None:
    """Best official symbol — prefer canonical, fall back to name."""
    canon = ent.get("canonical")
    if isinstance(canon, str) and canon:
        return canon
    name = ent.get("name")
    if isinstance(name, str) and name:
        return name
    return None


def _entity_aliases(ent: dict) -> list[str]:
    aliases = ent.get("all_names") or ent.get("aliases") or []
    if not isinstance(aliases, (list, tuple)):
        return []
    # Lazy resolution — if no aliases on the dict, do a fresh Gilda lookup.
    if not aliases and ent.get("name"):
        aliases = _gilda_aliases(ent["name"])
    return [a for a in aliases if isinstance(a, str) and len(a) >= 2]


def lf_gilda_exact_symbol(entity, evidence) -> int:
    """Vote `mentioned` (0) when entity's HGNC/UniProt official symbol
    appears literally in evidence (regex word-boundary match).

    Per V5r §3.4.
    """
    ent = _entity_dict(entity)
    text = _entity_evidence_text(evidence)
    if not text:
        return ABSTAIN
    sym = _entity_official_symbol(ent)
    if not sym:
        return ABSTAIN
    if _entity_in_text(sym, text):
        return GROUNDING_CLASSES["mentioned"]
    return ABSTAIN


def lf_evidence_contains_official_symbol(entity, evidence) -> int:
    """Vote `mentioned` (0) when official symbol appears in evidence.

    Alternate path; complementary to `lf_gilda_exact_symbol`. The
    distinction is that this LF accepts ent['symbol'] / ent['name']
    without requiring a successful Gilda canonicalization.

    Per V5r §3.4.
    """
    ent = _entity_dict(entity)
    text = _entity_evidence_text(evidence)
    if not text:
        return ABSTAIN
    name = ent.get("name") or ent.get("symbol")
    if not isinstance(name, str) or not name:
        return ABSTAIN
    if _entity_in_text(name, text):
        return GROUNDING_CLASSES["mentioned"]
    return ABSTAIN


def lf_gilda_alias(entity, evidence) -> int:
    """Vote `equivalent` (1) when a known alias (not equal to the
    official symbol) appears in evidence AND that alias text grounds
    BACK to the claim entity (top Gilda match), not a different entity.

    V6e Fix 2: previous version voted `equivalent` whenever ANY alias of
    the claim entity appeared in evidence, even if that alias text would
    Gilda-resolve to a different entity. Result: 0% accuracy on 328
    fires. Now requires:
      - Alias text appears in evidence
      - Alias is not identical to the claim's official symbol
      - `gilda.ground(alias_text)` top match is the claim entity (by
        canonical name OR HGNC db_id)
    """
    ent = _entity_dict(entity)
    text = _entity_evidence_text(evidence)
    if not text:
        return ABSTAIN
    sym = _entity_official_symbol(ent) or ""
    name = ent.get("name") or sym
    if not name:
        return ABSTAIN
    aliases = _entity_aliases(ent)
    if not aliases:
        return ABSTAIN
    try:
        import gilda
    except Exception:
        return ABSTAIN
    claim_db = ent.get("db")
    claim_db_id = ent.get("db_id")
    claim_canonical = (ent.get("canonical") or sym or name).lower()
    for a in aliases:
        if not isinstance(a, str) or len(a) < 2:
            continue
        if a.lower() == sym.lower() or a.lower() == name.lower():
            continue
        if not _entity_in_text(a, text):
            continue
        # Verify the alias text grounds back to the claim entity.
        try:
            matches = gilda.ground(a) or []
        except Exception:
            continue
        if not matches:
            continue
        top = matches[0]
        # Match by HGNC db_id when available (most precise).
        if claim_db and claim_db_id:
            if (getattr(top.term, "db", None) == claim_db
                    and str(getattr(top.term, "id", "")) == str(claim_db_id)):
                return GROUNDING_CLASSES["equivalent"]
        # Fallback: top.entry_name == claim canonical (case-insensitive).
        top_name = getattr(top.term, "entry_name", "") or ""
        if top_name and top_name.lower() == claim_canonical:
            return GROUNDING_CLASSES["equivalent"]
    return ABSTAIN


def lf_gilda_family_member(entity, evidence) -> int:
    """Vote `equivalent` (1) when the entity is a family head AND the
    evidence names a member of that family.

    Per V5r §3.4.
    """
    ent = _entity_dict(entity)
    text = _entity_evidence_text(evidence)
    if not text:
        return ABSTAIN
    if not ent.get("is_family"):
        return ABSTAIN
    members = ent.get("family_members") or []
    if not members and ent.get("name"):
        # Fresh resolve to populate members.
        try:
            from indra_belief.data.entity import GroundedEntity
            ge = GroundedEntity.resolve(ent["name"])
            if ge.is_family:
                members = ge.family_members or []
        except Exception:
            members = []
    for m in members:
        if isinstance(m, str) and len(m) >= 2 and _entity_in_text(m, text):
            return GROUNDING_CLASSES["equivalent"]
    return ABSTAIN


def lf_gilda_no_match(entity, evidence) -> int:
    """Vote `not_present` (2) when no symbol or alias appears in
    evidence.

    Per V5r §3.4.
    """
    ent = _entity_dict(entity)
    text = _entity_evidence_text(evidence)
    if not text:
        return ABSTAIN
    sym = _entity_official_symbol(ent)
    if sym and _entity_in_text(sym, text):
        return ABSTAIN
    name = ent.get("name")
    if name and _entity_in_text(name, text):
        return ABSTAIN
    for a in _entity_aliases(ent):
        if _entity_in_text(a, text):
            return ABSTAIN
    return GROUNDING_CLASSES["not_present"]


def lf_evidence_too_short_grounding(entity, evidence) -> int:
    """Vote `uncertain` (3) when evidence < 8 tokens.

    Per V5r §3.4: companion threshold to scope's `lf_text_too_short`.
    """
    text = _entity_evidence_text(evidence)
    if len(_evidence_tokens(text)) < 8:
        return GROUNDING_CLASSES["uncertain"]
    return ABSTAIN


def _gilda_match_is_claim(m: Any, claim_db: Any, claim_db_id: Any,
                            claim_canon: str) -> bool:
    """Whether a Gilda ScoredMatch points at the claim entity (HGNC id
    or canonical name match)."""
    if m is None or m.term is None:
        return False
    if claim_db and claim_db_id:
        if (getattr(m.term, "db", None) == claim_db
                and str(getattr(m.term, "id", "")) == str(claim_db_id)):
            return True
    name = (getattr(m.term, "entry_name", "") or "").lower()
    return bool(name) and name == claim_canon


def lf_ambiguous_grounding(entity, evidence) -> int:
    """Vote `uncertain` (3) when Gilda returns ≥2 matches for the claim
    entity name, top-1 vs top-2 score gap < 0.05, AND the top match is
    NOT the claim entity (claim is one of the also-rans).

    V6e Fix 3: previous LF voted `uncertain` whenever multiple
    upper-cased tokens in evidence yielded similar Gilda scores. Result:
    0% accuracy on 295 fires. Now grounds the CLAIM entity name and
    inspects Gilda's own top-2 score gap.
    """
    ent = _entity_dict(entity)
    text = _entity_evidence_text(evidence)
    if not text or not ent.get("name"):
        return ABSTAIN
    name = ent["name"]
    try:
        import gilda
    except Exception:
        return ABSTAIN
    try:
        matches = gilda.ground(name, organisms=["9606"]) or []
    except Exception:
        try:
            matches = gilda.ground(name) or []
        except Exception:
            return ABSTAIN
    if len(matches) < 2:
        return ABSTAIN
    top_score = float(getattr(matches[0], "score", 0.0) or 0.0)
    runner_score = float(getattr(matches[1], "score", 0.0) or 0.0)
    if (top_score - runner_score) >= 0.05:
        return ABSTAIN
    claim_db = ent.get("db")
    claim_db_id = ent.get("db_id")
    claim_canon = (ent.get("canonical") or name).lower()
    if _gilda_match_is_claim(matches[0], claim_db, claim_db_id, claim_canon):
        return ABSTAIN
    for m in matches[1:5]:
        if _gilda_match_is_claim(m, claim_db, claim_db_id, claim_canon):
            return GROUNDING_CLASSES["uncertain"]
    return ABSTAIN


# ---------------------------------------------------------------------------
# LF index — companion to V6a's LF_INDEX
# ---------------------------------------------------------------------------

# Each entry: (probe_kind, lf_name, lf_callable, default_kwargs).
# Clean LF inventory; V6a covers substrate-tuned counterparts.
LF_INDEX_CLEAN: tuple[tuple[str, str, callable, dict], ...] = (
    # relation_axis
    ("relation_axis", "lf_curated_db_axis", lf_curated_db_axis, {}),
    ("relation_axis", "lf_reach_found_by_axis_match",
     lf_reach_found_by_axis_match, {}),
    ("relation_axis", "lf_reach_found_by_axis_mismatch",
     lf_reach_found_by_axis_mismatch, {}),
    ("relation_axis", "lf_epistemics_direct_true",
     lf_epistemics_direct_true, {}),
    ("relation_axis", "lf_epistemics_direct_false",
     lf_epistemics_direct_false, {}),
    ("relation_axis", "lf_multi_extractor_axis_agreement",
     lf_multi_extractor_axis_agreement, {}),
    ("relation_axis", "lf_amount_lexical", lf_amount_lexical, {}),
    ("relation_axis", "lf_amount_keyword_negative",
     lf_amount_keyword_negative, {}),
    ("relation_axis", "lf_chain_with_named_intermediate",
     lf_chain_with_named_intermediate, {}),
    ("relation_axis", "lf_no_entity_overlap", lf_no_entity_overlap, {}),
    ("relation_axis", "lf_partner_dna_lexical", lf_partner_dna_lexical, {}),
    # subject_role
    ("subject_role", "lf_position_subject_subj",
     lf_position_subject, {"role": "subject"}),
    ("subject_role", "lf_position_object_subj",
     lf_position_object, {"role": "subject"}),
    ("subject_role", "lf_role_swap_lexical_subj",
     lf_role_swap_lexical, {"role": "subject"}),
    ("subject_role", "lf_chain_position_lexical_subj",
     lf_chain_position_lexical, {"role": "subject"}),
    ("subject_role", "lf_decoy_lexical_subj",
     lf_decoy_lexical, {"role": "subject"}),
    ("subject_role", "lf_no_grounded_match_subj",
     lf_no_grounded_match, {"role": "subject"}),
    ("subject_role", "lf_absent_alias_check_subj",
     lf_absent_alias_check, {"role": "subject"}),
    # object_role
    ("object_role", "lf_position_subject_obj",
     lf_position_subject, {"role": "object"}),
    ("object_role", "lf_position_object_obj",
     lf_position_object, {"role": "object"}),
    ("object_role", "lf_role_swap_lexical_obj",
     lf_role_swap_lexical, {"role": "object"}),
    ("object_role", "lf_chain_position_lexical_obj",
     lf_chain_position_lexical, {"role": "object"}),
    ("object_role", "lf_decoy_lexical_obj",
     lf_decoy_lexical, {"role": "object"}),
    ("object_role", "lf_no_grounded_match_obj",
     lf_no_grounded_match, {"role": "object"}),
    ("object_role", "lf_absent_alias_check_obj",
     lf_absent_alias_check, {"role": "object"}),
    # scope
    ("scope", "lf_hedge_lexical", lf_hedge_lexical, {}),
    ("scope", "lf_negation_lexical", lf_negation_lexical, {}),
    ("scope", "lf_conditional_lexical", lf_conditional_lexical, {}),
    ("scope", "lf_clean_assertion", lf_clean_assertion, {}),
    ("scope", "lf_low_information_evidence",
     lf_low_information_evidence, {}),
    ("scope", "lf_text_too_short", lf_text_too_short, {}),
    # verify_grounding (entity-level)
    ("verify_grounding", "lf_gilda_exact_symbol",
     lf_gilda_exact_symbol, {}),
    ("verify_grounding", "lf_evidence_contains_official_symbol",
     lf_evidence_contains_official_symbol, {}),
    ("verify_grounding", "lf_gilda_alias", lf_gilda_alias, {}),
    ("verify_grounding", "lf_gilda_family_member",
     lf_gilda_family_member, {}),
    ("verify_grounding", "lf_gilda_no_match", lf_gilda_no_match, {}),
    ("verify_grounding", "lf_evidence_too_short_grounding",
     lf_evidence_too_short_grounding, {}),
    ("verify_grounding", "lf_ambiguous_grounding",
     lf_ambiguous_grounding, {}),
)


def all_clean_lf_votes(statement, evidence) -> dict:
    """Run every (statement, evidence) clean LF in LF_INDEX_CLEAN and
    return a name→vote dict. Entity-keyed grounding LFs are excluded
    here — call `all_clean_grounding_lf_votes(entity, evidence)`."""
    out: dict[str, int] = {}
    for kind, name, fn, kwargs in LF_INDEX_CLEAN:
        if kind == "verify_grounding":
            continue
        out[name] = fn(statement, evidence, **kwargs)
    return out


def all_clean_grounding_lf_votes(entity, evidence) -> dict:
    """Run every entity-level grounding LF and return a name→vote dict."""
    out: dict[str, int] = {}
    for kind, name, fn, kwargs in LF_INDEX_CLEAN:
        if kind != "verify_grounding":
            continue
        out[name] = fn(entity, evidence, **kwargs)
    return out

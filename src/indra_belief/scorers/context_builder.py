"""Builder for EvidenceContext — pure deterministic Python.

Sole entrypoint: `build_context(stmt, evidence) -> EvidenceContext`.
Called once per (Statement, Evidence) at the top of
`score_evidence_decomposed`, before any LLM sub-call.

Resolution sources (all cached):
  - GroundedEntity.resolve (Gilda) for aliases / families / pseudogene
  - Deterministic clause split for complex sentences (J5)
  - INDRA Statement type → BindingDomain admissibility (J3)
  - statement.residue / statement.position → acceptable_sites set
  - L-phase regex detectors:
      L1 chain markers + intermediate candidates
      L4 nominalization patterns
      L5 modification site enumeration
  - L-phase Gilda-derived classifiers:
      L2 subject/object class (cytokine/ligand/mirna/protein/...)
      L3 precision class (specific/family/ambiguous_alias)

L-phase doctrine: every detector is regex- or Gilda-bounded; no LLM
call introduced. See context.py module docstring for the full doctrine
and field inventory.
"""
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from indra_belief.scorers.context import DetectedRelation, EvidenceContext
from indra_belief.scorers.relation_patterns import CATALOG

if TYPE_CHECKING:
    from indra.statements import Evidence, Statement


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Statement-type → binding admissibility
# ---------------------------------------------------------------------------
# Complex demands protein-protein binding. Promoter binding (DNA) and
# coordinated DNA-site binding (the MYB/MYBL2-share-Myb-binding-site FP)
# both surface as `X binds Y` in raw text but are NOT INDRA Complex.
# Catalytic / regulatory statements don't gate on binding domain — for
# them, ctx.binding_admissible is empty (no gate).
_COMPLEX_TYPES = frozenset({"Complex"})
_TRANSLOCATION_TYPES = frozenset({"Translocation"})

# Modification statements where binding-to-substrate is implicit (not
# the gating concept). For these, ctx.binding_admissible is empty.
_MODIFICATION_TYPES = frozenset({
    "Phosphorylation", "Dephosphorylation",
    "Acetylation", "Deacetylation",
    "Methylation", "Demethylation",
    "Ubiquitination", "Deubiquitination",
    "Sumoylation", "Desumoylation",
    "Hydroxylation", "Dehydroxylation",
    "Glycosylation", "Deglycosylation",
    "Ribosylation", "Deribosylation",
    "Farnesylation", "Defarnesylation",
    "Palmitoylation", "Depalmitoylation",
    "Geranylgeranylation", "Degeranylgeranylation",
    "Myristoylation", "Demyristoylation",
    "Phosphorylation_serine", "Phosphorylation_threonine",
    "Phosphorylation_tyrosine",
    "Autophosphorylation", "Transphosphorylation",
})


def _binding_admissible_for(stmt_type: str) -> frozenset[str]:
    """Per-INDRA-type binding domain policy.

    Complex → {"protein"} (rejects DNA/RNA/membrane "binding" surface forms).
    Translocation → {"membrane", "complex"} (the destination is a compartment
    or partner complex, not a free protein).
    Everything else → empty (no gate; assertion's binding_partner_type is
    informational, not gating).
    """
    if stmt_type in _COMPLEX_TYPES:
        return frozenset({"protein"})
    if stmt_type in _TRANSLOCATION_TYPES:
        return frozenset({"membrane", "complex"})
    return frozenset()


# ---------------------------------------------------------------------------
# Subject + object extraction (mirrors parse_claim's logic; kept local so
# the builder doesn't import parse_claim — avoids circular dependency risk
# and lets the context build before parse_claim runs if reordering ever
# becomes desirable).
# ---------------------------------------------------------------------------
def _agent_names(stmt) -> list[str]:
    """Extract canonical agent names from an INDRA Statement.

    Complex uses .members (multi-way binding); everything else uses
    .agent_list(). None agents (missing INDRA grounding) are dropped —
    they would just produce '?' which contributes no aliases.
    """
    from indra.statements import Complex, SelfModification

    if isinstance(stmt, Complex):
        members = [m for m in stmt.members if m is not None]
        return [m.name for m in members if getattr(m, "name", None)]
    agents = [a for a in stmt.agent_list() if a is not None]
    names = [a.name for a in agents if getattr(a, "name", None)]
    if isinstance(stmt, SelfModification) and names:
        # Self-modification: agent == target; only one unique name.
        return [names[0]]
    return names


def _raw_text_for(name: str, evidence) -> str | None:
    """Pull the evidence-side raw_text for a claim entity (same logic
    as decomposed.py:_raw_text_for; duplicated to avoid a cycle)."""
    try:
        agents = evidence.annotations.get("agents", {})
    except AttributeError:
        return None
    names = agents.get("agent_list") or []
    raws = agents.get("raw_text") or []
    for n, rt in zip(names, raws):
        if n == name and rt:
            return rt
    return None


# ---------------------------------------------------------------------------
# Alias-map construction
# ---------------------------------------------------------------------------
# M5: FPLX manual backfill for under-expanded families.
# ---------------------------------------------------------------------------
# Gilda's FPLX rosters are sparse for some families: GPCR collapses to
# only ~25 members despite ~800 documented receptors; 14-3-3 / TCF_LEF
# / IKB return only the canonical name with no member roster. The L8
# diagnosis traced 13+ FNs to these under-expanded families.
#
# This backfill is a curated static patch — each entry includes the
# canonical name and a roster of accepted member surface forms. Entries
# are merged into the entity's alias set after Gilda resolution; both
# the parser-bind path and the M3 substrate-fallback see the expanded
# aliases.
#
# Validation discipline: every entry must be auditable against the
# FPLX yaml (https://github.com/sorgerlab/famplex). When updating,
# verify against the upstream roster — over-expansion causes precision
# loss in the bilateral_ambiguity guard (L3) which already handles
# family-family confidence downgrades, so the cost is bounded but real.
#
# When a family member set IS produced via Gilda, the backfill UNIONS
# rather than replaces — Gilda's coverage of well-mapped families
# (NFkappaB, ERK, MEK, p38) stays authoritative.
_FPLX_BACKFILL: dict[str, frozenset[str]] = {
    # Chemokine + prostaglandin + histamine + adrenergic GPCRs —
    # the chemokine receptor subfamily was the FN driver (GPCR→CCL3).
    "GPCR": frozenset({
        # CC-chemokine receptors
        "CCR1", "CCR2", "CCR3", "CCR4", "CCR5", "CCR6", "CCR7",
        "CCR8", "CCR9", "CCR10", "CCRL2",
        # CXC chemokine receptors
        "CXCR1", "CXCR2", "CXCR3", "CXCR4", "CXCR5", "CXCR6", "CXCR7",
        # CX3C and XC chemokine receptors
        "CX3CR1", "XCR1",
        # Prostaglandin receptors
        "PTGER1", "PTGER2", "PTGER3", "PTGER4", "PTGFR", "PTGIR",
        "PTGDR", "PTGDR2", "TBXA2R",
        # Histamine receptors
        "HRH1", "HRH2", "HRH3", "HRH4",
        # Common surface forms used in literature
        "G-protein coupled receptor", "G-protein coupled receptors",
        "G protein coupled receptor", "G protein-coupled receptor",
        "GPCRs",
    }),
    # 14-3-3 family — YWHA + SFN (stratifin)
    "p14_3_3": frozenset({
        "YWHAB", "YWHAE", "YWHAG", "YWHAH", "YWHAQ", "YWHAZ", "SFN",
        "14-3-3", "14_3_3", "14 3 3",
        "14-3-3 epsilon", "14-3-3 zeta", "14-3-3 beta", "14-3-3 gamma",
        "14-3-3 sigma", "14-3-3 theta", "14-3-3 eta",
        "14-3-3 protein", "14-3-3 proteins",
    }),
    # TCF/LEF transcription factors
    "TCF_LEF": frozenset({
        "TCF7", "TCF7L1", "TCF7L2", "LEF1",
        "TCF/LEF", "TCF-LEF", "TCF/Lef", "TCF-Lef",
        "TCF/LEF proteins", "TCF/Lef proteins",
        "TCF", "LEF",  # standalone usage
    }),
    # IκB inhibitor family — NFKBI
    "IKB": frozenset({
        "NFKBIA", "NFKBIB", "NFKBID", "NFKBIE", "NFKBIZ",
        "IkappaB", "IκB", "I-kappa-B", "I-kappaB", "IkB",
        "IkBalpha", "IkBbeta", "IkBepsilon", "IkBzeta",
        "IκBα", "IκBβ", "IκBε", "IκBζ",
    }),
    # N1: PKC family — Gilda gives the PRKC* member roster but misses
    # the spelled-out written form. Trace-1 miss (PKC→EIF4E):
    # evidence "phosphorylation of eIF-4E ... by protein kinase C"
    # parsed agents=["protein kinase c"]; alias bind missed because the
    # spelled form wasn't in aliases["PKC"]. The members are kept here
    # too so the backfill is self-contained when Gilda is offline.
    "PKC": frozenset({
        "PRKCA", "PRKCB", "PRKCD", "PRKCE", "PRKCG", "PRKCH",
        "PRKCI", "PRKCQ", "PRKCZ",
        "protein kinase C", "Protein Kinase C", "protein kinase c",
        "PKCs", "PKC family", "PKC isoforms", "PKC isozymes",
        "classical PKC", "novel PKC", "atypical PKC",
    }),
    # PKA family — same shape as PKC. "cAMP-dependent protein kinase"
    # is the alternate written form authors use interchangeably with PKA.
    "PKA": frozenset({
        "PRKACA", "PRKACB", "PRKACG",
        "protein kinase A", "Protein Kinase A", "protein kinase a",
        "PKAs", "PKA family",
        "cAMP-dependent protein kinase", "cAMP dependent protein kinase",
    }),
    # PKG family — cGMP-dependent protein kinase.
    "PKG": frozenset({
        "PRKG1", "PRKG2",
        "protein kinase G", "Protein Kinase G", "protein kinase g",
        "cGMP-dependent protein kinase", "cGMP dependent protein kinase",
    }),
    # HDAC family — written form "histone deacetylase(s)" is the
    # alternate of "HDAC". Trace-class includes HDAC→AR records where
    # evidence says "histone deacetylase (HDAC) inhibitors" — the
    # parenthetical written form is the common pattern. Member roster
    # included so substrate matches member-specific claims (HDAC1→…)
    # against generic "histone deacetylase" evidence at the L3-downgrade
    # confidence cap.
    "HDAC": frozenset({
        "HDAC1", "HDAC2", "HDAC3", "HDAC4", "HDAC5", "HDAC6",
        "HDAC7", "HDAC8", "HDAC9", "HDAC10", "HDAC11",
        "histone deacetylase", "histone deacetylases", "HDACs",
        "histone deacetylase family",
    }),
}


def _apply_fplx_backfill(name: str, syns: frozenset[str]) -> frozenset[str]:
    """Union the manual FPLX backfill into Gilda-derived aliases.

    Returns the input unchanged when no backfill exists for `name`.
    Defense-in-depth: short tokens (<2 chars) are dropped from BOTH
    the input and the backfill, in case a caller bypassed
    _expand_synonyms's own filter.
    """
    backfill = _FPLX_BACKFILL.get(name)
    if not backfill:
        return syns
    merged = {s for s in (syns | backfill) if isinstance(s, str) and len(s) >= 2}
    return frozenset(merged)


def _expand_synonyms(grounded_entity) -> frozenset[str]:
    """Collect all accepted synonyms for a resolved entity.

    Includes: name, canonical, aliases, all_names, family_members,
    plus M5 FPLX backfill for under-expanded families. Synonyms
    shorter than 2 chars are dropped (over-match risk against
    single-letter tokens). Order is not preserved (frozenset).
    """
    name = getattr(grounded_entity, "name", None)
    if not name or name == "?":
        return frozenset()
    syns: set[str] = {name}
    canonical = getattr(grounded_entity, "canonical", None)
    if canonical:
        syns.add(canonical)
    syns.update(getattr(grounded_entity, "aliases", []) or [])
    syns.update(getattr(grounded_entity, "all_names", []) or [])
    syns.update(getattr(grounded_entity, "family_members", []) or [])
    base = frozenset(s for s in syns if isinstance(s, str) and len(s) >= 2)
    return _apply_fplx_backfill(name, base)


# ---------------------------------------------------------------------------
# Complexity-driven clause split
# ---------------------------------------------------------------------------
# Rationale: long multi-clause sentences ("our favored model is that X
# phosphorylates Y at S1045 after detecting a block in replication, that
# phosphorylated Y is then translocated...") choke single-shot
# parse_evidence — the v1 FN profile had ~5 cases of this shape (ATR→
# FANCM, CASP9→APAF1, MAP3K5→PPP5C, INS→PI3K, TGFB1→AKT) where the
# relation IS in the text but parse abstained.
#
# J5 splits when the deterministic heuristic fires:
#   1. Multi-sentence Evidence.text — split on sentence boundaries
#      ('. ' or '? ' or '! '). Per-sentence parse is cheaper than
#      multi-sentence reasoning at single-shot.
#   2. Long-single-sentence with multiple subordinators — left to the
#      single-shot path. We do NOT regex-split on subordinators here;
#      that was tried (#44) and was fragile on pronouns / coordinated
#      subjects. The O-phase deprecation also removed the LLM-side
#      decomposition retry that previously covered this case; the
#      M-phase relation catalog (regex substrate) plus adjudicate's
#      M3 substrate-fallback bind now handle the residual class.
#
# Cap on clauses: 4. Above that, the cost (4× LLM calls) dominates the
# value; the few really-long passages with 5+ clauses need per-document
# analysis, not per-sentence.
_MAX_CLAUSES = 4


def _split_into_clauses(text: str) -> tuple[str, ...]:
    """Deterministic sentence-level split. Returns the trimmed clause
    list when 2 or more sentences are detected; otherwise returns
    `(text,)`."""
    if not text or not text.strip():
        return ()

    # Sentence boundaries: '. ', '? ', '! ' followed by whitespace.
    # We don't split on '.' alone because abbreviations ('et al.',
    # 'i.e.') would over-fire. Requiring the trailing space scopes the
    # heuristic to actual sentence ends in evidence text.
    import re
    parts = re.split(r"(?<=[.?!])\s+(?=[A-Z(])", text.strip())
    parts = [p.strip() for p in parts if p and p.strip()]

    # Filter trivial parts (< 12 chars) — these are usually citation
    # markers ("[ ].", "(Smith, 2020).") that aren't real clauses.
    parts = [p for p in parts if len(p) >= 12]

    if len(parts) < 2:
        return (text,)
    return tuple(parts[:_MAX_CLAUSES])


# ---------------------------------------------------------------------------
# L1: Chain-signal detector
# ---------------------------------------------------------------------------
# High-precision markers for indirect causal chains. The LLM is unreliable
# at discovering chains under attention pressure on long passages; regex
# encoding of the canonical lexical markers is more reliable. The signal
# is fed forward as a conditional prompt nudge AND backward as an
# informational adjudicate reason when intermediates were missed.
_CHAIN_MARKERS = re.compile(
    r"\b("
    r"thereby|"
    r"leads?\s+to|"
    r"results?\s+in|"
    r"which\s+then|"
    r"is\s+mediated\s+by|"
    r"is\s+catalyzed\s+by|"
    r"by\s+way\s+of|"
    r"through\s+(?:its|the)?|"
    r"via\s+\w+|"
    r"through\s+\w+(?:\s+\w+)?\s*(?:to|→|->)"
    r")\b",
    re.IGNORECASE,
)

# Captures intermediate entity candidates from explicit chain forms
# "X → Y → Z" or "X via Y" or "X-mediated Y by Z". The output is a
# best-effort list of capitalized tokens flanking the chain markers;
# parse_evidence still confirms semantic role.
_CHAIN_VIA = re.compile(
    r"\bvia\s+([A-Z][\w-]{1,30})\b",
    re.IGNORECASE,
)
_CHAIN_ARROW = re.compile(
    r"([A-Z][\w-]{1,30})\s*(?:→|->)\s*([A-Z][\w-]{1,30})\s*(?:→|->)\s*([A-Z][\w-]{1,30})",
)
_CHAIN_MEDIATED_BY = re.compile(
    r"is\s+mediated\s+by\s+([A-Z][\w-]{1,30})",
    re.IGNORECASE,
)


def _detect_subject_upstream_anchor(
    text: str, subject_aliases: frozenset[str]
) -> bool:
    """L7-fix for C2: detect explicit X-as-upstream-actor construction.

    Returns True iff the evidence text contains
        "X-induced/mediated/driven/dependent Y"  or
        "induced/mediated/driven by X"
    where X is any of the subject's aliases (case-insensitive).

    Used to gate the cytokine/miRNA bypass on indirect chains. The
    class label "this entity is a cytokine" is necessary but not
    sufficient — the cytokine could appear as an in-chain mediator
    rather than the upstream actor. The deterministic anchor check
    distinguishes the two.
    """
    if not text or not subject_aliases:
        return False
    # Build alias group (escaped, alternation)
    alias_pattern = "|".join(re.escape(a) for a in subject_aliases if a)
    if not alias_pattern:
        return False
    # X-induced/mediated/driven/dependent Y
    pattern_a = re.compile(
        rf"\b(?:{alias_pattern})[-\s]?(?:induced|mediated|driven|"
        rf"dependent|catalyzed|stimulated)\b",
        re.IGNORECASE,
    )
    if pattern_a.search(text):
        return True
    # induced/mediated/driven by X
    pattern_b = re.compile(
        rf"\b(?:induced|mediated|driven|catalyzed|stimulated)\s+by\s+"
        rf"(?:{alias_pattern})\b",
        re.IGNORECASE,
    )
    if pattern_b.search(text):
        return True
    return False


def _detect_chain_signal(text: str) -> tuple[bool, tuple[str, ...]]:
    """Return (has_chain_signal, intermediate_candidates).

    has_chain_signal is True iff at least one chain marker OR an
    explicit arrow chain ("X → Y → Z") is present. Candidates are a
    deduplicated, ordered list of capitalized tokens captured by
    chain-form patterns. Empty when no signal.
    """
    if not text:
        return False, ()
    has_marker = bool(_CHAIN_MARKERS.search(text))
    has_arrow = bool(_CHAIN_ARROW.search(text))
    if not (has_marker or has_arrow):
        return False, ()

    candidates: list[str] = []
    seen: set[str] = set()
    for match in _CHAIN_VIA.finditer(text):
        name = match.group(1)
        if name and name not in seen:
            candidates.append(name)
            seen.add(name)
    for match in _CHAIN_ARROW.finditer(text):
        for name in match.groups():
            if name and name not in seen:
                candidates.append(name)
                seen.add(name)
    for match in _CHAIN_MEDIATED_BY.finditer(text):
        name = match.group(1)
        if name and name not in seen:
            candidates.append(name)
            seen.add(name)
    return True, tuple(candidates)


# ---------------------------------------------------------------------------
# L2: Subject/object class (cytokine/ligand/miRNA detection)
# ---------------------------------------------------------------------------
# Conservative starting set. Each entry is a canonical HGNC symbol whose
# product is a cytokine, secreted ligand, or growth factor that INDRA
# curators treat as a valid upstream-attribution agent. Expanding this
# set is L9-gated: only after the holdout shows no FPs from the current
# set do we add more.
_CYTOKINE_LIGAND_HGNC = frozenset({
    # Interleukins (IL-1 through IL-36 — most common)
    "IL1A", "IL1B", "IL2", "IL4", "IL5", "IL6", "IL7", "IL8",
    "IL9", "IL10", "IL11", "IL12A", "IL12B", "IL13", "IL15",
    "IL17A", "IL17B", "IL18", "IL21", "IL22", "IL23A", "IL27",
    "IL33", "IL36A", "IL36B", "IL36G",
    # Interferons
    "IFNA1", "IFNA2", "IFNB1", "IFNG",
    # TNF superfamily
    "TNF", "TNFSF10", "TNFSF11",
    # TGF-beta family
    "TGFB1", "TGFB2", "TGFB3",
    # Insulin / IGF / EGF / FGF / VEGF / PDGF
    "INS", "IGF1", "IGF2",
    "EGF", "TGFA", "HBEGF",
    "FGF1", "FGF2", "FGF7", "FGF10",
    "VEGFA", "VEGFB", "VEGFC",
    "PDGFA", "PDGFB",
    # Hedgehog / Wnt (cytokine-like ligands)
    "SHH", "IHH", "WNT1", "WNT3A", "WNT5A", "WNT11",
    # CCL/CXCL chemokines (top 12 most common)
    "CCL2", "CCL3", "CCL4", "CCL5", "CCL11", "CCL19", "CCL21",
    "CXCL1", "CXCL8", "CXCL9", "CXCL10", "CXCL12",
    # BMP / GDF
    "BMP2", "BMP4", "BMP7", "GDF15",
})


def _classify_subject(grounded_entity) -> str:
    """Map a GroundedEntity to a semantic class.

    Returns one of: "protein", "cytokine_or_ligand", "mirna", "family",
    "small_molecule", "ambiguous", "unknown".
    """
    db = getattr(grounded_entity, "db", None)
    canonical = getattr(grounded_entity, "canonical", None)
    name = getattr(grounded_entity, "name", None) or ""
    score = getattr(grounded_entity, "gilda_score", None)

    # Family from FPLX
    if db == "FPLX":
        return "family"

    # miRNA name convention
    upper_canonical = (canonical or "").upper()
    upper_name = name.upper()
    if (upper_canonical.startswith("MIR") or upper_canonical.startswith("LET-")
            or upper_name.startswith("MIR") or upper_name.startswith("LET-")):
        return "mirna"

    # Cytokine/ligand from curated HGNC set
    if db == "HGNC" and canonical and canonical in _CYTOKINE_LIGAND_HGNC:
        return "cytokine_or_ligand"
    # Also match by name when canonical missing
    if db == "HGNC" and name in _CYTOKINE_LIGAND_HGNC:
        return "cytokine_or_ligand"

    # Small molecule
    if db == "CHEBI":
        return "small_molecule"

    # Low-score Gilda match → ambiguous
    if score is not None and score < 0.6:
        return "ambiguous"

    # HGNC protein default
    if db == "HGNC":
        return "protein"

    # Other namespace or unresolved
    if db is None:
        return "unknown"
    return "protein"  # MESH/GO/etc default to protein-like


# ---------------------------------------------------------------------------
# L3: Precision class for bilateral-ambiguity guard
# ---------------------------------------------------------------------------
# "specific" — HGNC-grounded with high-confidence Gilda match
# "family" — FPLX (multiple isoforms collapsed to a family entry)
# "ambiguous_alias" — Gilda found a match but with low confidence or
#                     multiple competing top-K hits
# "unknown" — Gilda failed to resolve at all
def _classify_precision(grounded_entity) -> str:
    db = getattr(grounded_entity, "db", None)
    score = getattr(grounded_entity, "gilda_score", None)
    competing = getattr(grounded_entity, "competing_candidates", None) or []
    is_low_conf = getattr(grounded_entity, "is_low_confidence", False)

    if db is None:
        return "unknown"
    if db == "FPLX":
        return "family"
    if is_low_conf or (score is not None and score < 0.6):
        return "ambiguous_alias"
    if len(competing) >= 2:
        return "ambiguous_alias"
    if db == "HGNC":
        return "specific"
    return "specific"  # other namespaces default to specific


# ---------------------------------------------------------------------------
# L4: Nominalization detector
# ---------------------------------------------------------------------------
# Captures "X-induced Y of Z" / "X-mediated Y of Z" / "X-driven Y of Z" /
# "X-dependent Y of Z" patterns where Y is a nominalized verb
# (phosphorylation, activation, inhibition, methylation, ...) and Z is
# the substrate. Returns short normalized phrases parse_evidence can
# use as hints to emit standard verbal assertions.
_NOMINAL_VERBS = (
    "phosphorylation", "dephosphorylation",
    "acetylation", "deacetylation",
    "methylation", "demethylation",
    "ubiquitination", "deubiquitination",
    "sumoylation", "activation", "inhibition",
    "expression", "secretion", "release",
    "translocation", "stabilization", "degradation",
    "cleavage", "binding", "interaction",
    "induction", "suppression",
    "upregulation", "downregulation",
)
_NOMINAL_VERB_PATTERN = "|".join(_NOMINAL_VERBS)
_NOMINALIZATION = re.compile(
    rf"\b([A-Z][\w-]*?)-?(induced|mediated|driven|dependent|catalyzed|stimulated)\s+"
    rf"({_NOMINAL_VERB_PATTERN})\s+(?:of|on|for|in)\s+"
    rf"([A-Z][\w-]*)",
    re.IGNORECASE,
)


def _detect_nominalizations(text: str) -> tuple[str, ...]:
    """Return short pre-extracted nominalized-relation strings.

    Each string is a compact human-readable form ("INS-induced
    activation of PI3K") parse_evidence can quote in its prompt nudge.
    Capped at 6 to keep prompt overhead bounded.
    """
    if not text:
        return ()
    seen: set[str] = set()
    results: list[str] = []
    for m in _NOMINALIZATION.finditer(text):
        actor, modifier, verb, target = m.groups()
        phrase = f"{actor}-{modifier} {verb} of {target}"
        norm = phrase.lower()
        if norm in seen:
            continue
        seen.add(norm)
        results.append(phrase)
        if len(results) >= 6:
            break
    return tuple(results)


# ---------------------------------------------------------------------------
# L5: Site detection regex
# ---------------------------------------------------------------------------
# Captures all standard modification site notations and normalizes to
# the canonical single-letter form (S102, T461, Y732). Used by
# adjudicate's site_check as a fallback union when the parser drops
# sites past the first under coordination pressure.
_SITE_LETTER_FORM = re.compile(
    r"\b([STY])(?:-|\s)?(\d{1,4})\b",
)
_SITE_LONG_FORM = re.compile(
    r"\b(serine|threonine|tyrosine|ser|thr|tyr)[-\s]+(\d{1,4})\b",
    re.IGNORECASE,
)
# Negative anchors: avoid figure callouts ("Fig. 4"), table refs
# ("Table 5"), reference markers ("S1", "T2" used as supplementary
# table indices). Approach: drop matches where preceding token
# matches one of these markers.
_SITE_NEGATIVE_PRECEDERS = re.compile(
    r"\b(fig\.?|figure|table|tab\.?|supplementary|suppl?\.?|fig)\s*$",
    re.IGNORECASE,
)

# L7-fix for C3b: known protein-family / non-site tokens that match
# the letter-form regex but are NOT modification sites. Conservative
# starting set — expand as holdout shows new false-fires.
#   S100 / S100A* — calcium-binding S100 family
#   T7 — phage / nomenclature
#   Y2 / Y4 / Y5 — neuropeptide Y receptors (Y1, Y2, Y4, Y5, Y6)
#   S6 — ribosomal protein S6 (and other S\d ribosomal proteins);
#         genuine S6 site contexts have a phosphorylation anchor
#         which we additionally require below
_SITE_DENYLIST = frozenset({
    "S100", "T7", "Y1", "Y2", "Y4", "Y5", "Y6",
})

# Tokens that follow a candidate site and indicate the digit is part
# of an entity name, not a position. "S100 protein" / "T7 phage" /
# "Y2 receptor".
_SITE_FOLLOWS_REJECT = re.compile(
    r"^\s*(?:protein|family|receptor|phage|polymerase|kinase|"
    r"phosphatase|complex|domain|element|expression|levels|"
    r"transcription|translation|isoform|paralog|orthologue|orthologs?)\b",
    re.IGNORECASE,
)

# Modification anchors required within ~50 chars BEFORE the site for
# letter-form matches. Long-form ("serine 102") is self-anchoring; the
# short letter-digit form needs context. Without this, "S100" in
# "S100 expression was elevated" matches as site S100.
_SITE_MOD_ANCHORS = re.compile(
    r"\b(?:phospho(?:rylat\w*)?|de-?phospho\w*|acetylat\w*|"
    r"methylat\w*|ubiquitin\w*|sumoylat\w*|residue|site|"
    r"position|amino\s+acid|modified|catalyz\w*)\b",
    re.IGNORECASE,
)


def _normalize_site(letter_or_word: str, position: str) -> str | None:
    """Canonical form: 'S102', 'T461', 'Y732'."""
    word = letter_or_word.lower()
    letter_map = {
        "serine": "S", "ser": "S", "s": "S",
        "threonine": "T", "thr": "T", "t": "T",
        "tyrosine": "Y", "tyr": "Y", "y": "Y",
    }
    letter = letter_map.get(word)
    if not letter:
        return None
    try:
        n = int(position)
    except (ValueError, TypeError):
        return None
    if n <= 0 or n > 9999:
        return None
    return f"{letter}{n}"


def _detect_sites(text: str) -> frozenset[str]:
    """Extract all canonical site forms from evidence text.

    Letter-form ("S102") matches require a modification anchor within
    50 chars before the match (phosphorylation/acetylation/residue/
    site/position/...) to avoid capturing protein-name digits like
    "S100 protein" or "T7 polymerase". Long-form ("serine 102") is
    self-anchoring and bypasses the modification-anchor requirement.
    """
    if not text:
        return frozenset()
    results: set[str] = set()

    for m in _SITE_LETTER_FORM.finditer(text):
        normalized = _normalize_site(m.group(1), m.group(2))
        if not normalized:
            continue
        # Check preceding context for figure/table negative anchors.
        start = max(0, m.start() - 20)
        prefix = text[start:m.start()]
        if _SITE_NEGATIVE_PRECEDERS.search(prefix):
            continue
        # Denylist: known protein-family / non-site tokens
        if normalized in _SITE_DENYLIST:
            continue
        # Following-token reject: "S100 protein", "T7 phage", "Y2 receptor"
        end = m.end()
        suffix = text[end:end + 30]
        if _SITE_FOLLOWS_REJECT.match(suffix):
            continue
        # Letter-form requires a modification anchor within ~50 chars
        # in either direction. The anchor can precede ("phosphorylation
        # at S102") or follow ("S102 phosphorylation by SRC"). Without
        # this, "S100 was elevated" matches as site S100.
        anchor_start = max(0, m.start() - 50)
        anchor_end = min(len(text), m.end() + 50)
        anchor_window = text[anchor_start:m.start()] + " " + text[m.end():anchor_end]
        if not _SITE_MOD_ANCHORS.search(anchor_window):
            continue
        results.add(normalized)

    for m in _SITE_LONG_FORM.finditer(text):
        normalized = _normalize_site(m.group(1), m.group(2))
        if normalized:
            results.add(normalized)

    return frozenset(results)


# ---------------------------------------------------------------------------
# Acceptable site set
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# M10: explicit hedge-marker detection
# ---------------------------------------------------------------------------
# Diagnosis FP: CCR7→AKT — "CCR7 may activate Akt" is a hypothesis-level
# claim, not asserted. The parser emitted claim_status='asserted' under
# attention pressure; the substrate detects 'may' anchored to CCR7 and
# downgrades to 'hedged' at the adjudicator.
#
# Markers are anchored to claim entity proximity to avoid generic
# hedging in unrelated parts of the sentence ('It may be that the
# experiment is preliminary' shouldn't trigger a hedge on an unrelated
# entity-claim).
_HEDGE_MARKERS = (
    "may", "could", "might", "would",
    "is thought to", "are thought to",
    "is believed to", "are believed to",
    "appears to", "appear to", "appears as",
    "we hypothesize", "we propose", "we suggest", "we postulate",
    "likely", "possibly", "putatively",
    "presumably", "presumably to",
)
_HEDGE_PROXIMITY_CHARS = 60  # window around entity name to count as anchored

# N2: clause-break punctuation. When ANY of these sits between a hedge
# marker and an entity mention, the proximity check rejects — the
# marker is in a different clause and shouldn't anchor to the entity.
# Diagnosis: M13 surfaced 9 hedging_hypothesis regressions where the
# 60-char window crossed semicolons, sentence boundaries, or em-dash
# inserts. Strong separators only: `;`, `:`, em-dash, `--`, sentence-end
# period (`. ` followed by capital). Commas are intentionally excluded
# (most sentences have many; conjunction clauses share scope).
_HEDGE_CLAUSE_BREAK_RE = re.compile(r"[;:—]|--|\.\s+[A-Z(]")


def _has_clause_break_between(text: str, lo: int, hi: int) -> bool:
    """True if a strong clause-break punctuation sits in text[lo:hi+1].

    The slice extends one byte past `hi` so that sentence-end period
    matching can consume the leading capital after the period without
    falling off the slice (the regex consumes [A-Z(] rather than
    using a lookahead, since lookahead-at-slice-end fails)."""
    if lo >= hi:
        return False
    return bool(_HEDGE_CLAUSE_BREAK_RE.search(text[lo:hi + 1]))


def _detect_hedge_markers(
    text: str,
    claim_alias_sets: list[frozenset[str]],
) -> frozenset[str]:
    """Return the set of detected hedge markers anchored within 60
    chars of any claim-entity alias mention AND in the same clause.

    `claim_alias_sets` is a list of alias frozensets (one per claim
    entity). A marker is included only when:
      (1) it appears within _HEDGE_PROXIMITY_CHARS of an alias mention, AND
      (2) no strong clause-break punctuation sits between the marker
          and the entity (N2 fix — reject cross-clause anchoring).
    """
    if not text:
        return frozenset()
    text_lower = text.lower()
    # Find all alias mention spans (case-insensitive).
    entity_spans: list[tuple[int, int]] = []
    for aliases in claim_alias_sets:
        for alias in aliases:
            if not isinstance(alias, str) or len(alias) < 2:
                continue
            pat = re.compile(r"\b" + re.escape(alias.lower()) + r"\b")
            for m in pat.finditer(text_lower):
                entity_spans.append(m.span())
    if not entity_spans:
        return frozenset()
    # For each marker, check proximity to any entity span AND no
    # clause-break between them.
    detected: set[str] = set()
    for marker in _HEDGE_MARKERS:
        # Word-boundary match on lowercased text.
        marker_pat = re.compile(r"\b" + re.escape(marker) + r"\b")
        for m in marker_pat.finditer(text_lower):
            ms, me = m.span()
            for es, ee in entity_spans:
                # Distance: gap between marker and entity spans.
                if ms <= ee + _HEDGE_PROXIMITY_CHARS and me >= es - _HEDGE_PROXIMITY_CHARS:
                    # N2: clause-break gate. Compute the byte span
                    # between marker and entity (whichever is earlier
                    # to whichever is later) and reject if a strong
                    # clause-break sits inside.
                    lo, hi = (me, es) if me <= es else (ee, ms)
                    if _has_clause_break_between(text, lo, hi):
                        continue
                    detected.add(marker)
                    break
    return frozenset(detected)


# ---------------------------------------------------------------------------
# M9: deterministic perturbation-marker detection
# ---------------------------------------------------------------------------
# The diagnosis traced ~20 FNs and several FPs to perturbation language
# the parser missed:
#   FN MEK→MAPK1: "inhibiting MEK ... blocked ERK phosphorylation"
#       — MEK LOF flips sign; effective = "MEK phosphorylates ERK"
#   FP HDAC→AR: "HDAC inhibitors induced AR acetylation"
#       — HDAC LOF flips sign; effective = "HDAC deacetylates AR"
#         which contradicts the claim "HDAC acetylates AR"
#
# Marker patterns are entity-anchored: a generic "inhibitor" mention
# without entity context doesn't fire. The detector checks if any
# marker construct names ANY surface form of the claim subject (or
# object). When found, ctx.subject_perturbation_marker is set.
_LOF_PATTERNS = (
    # "<X> inhibitor", "<X> inhibitors", "<X> blocker", "<X> antagonist"
    # Optionally allow a parenthetical acronym/expansion between entity
    # and marker: "histone deacetylase (HDAC) inhibitors".
    r"\b(?P<X>{name})(?:\s*\([^)]*\))?\s+"
    r"(?:inhibitor[s]?|blocker[s]?|antagonist[s]?)\b",
    # Catch the parenthetical-inner case too: "(HDAC) inhibitors"
    # binds to HDAC even when the "X (Y) inhibitor" outer form is
    # captured by the same regex above (X=histone deacetylase). The
    # inner pattern adds redundant coverage for X=HDAC to prevent
    # missing the marker when ctx.aliases doesn't include the
    # spelled-out form.
    r"\((?P<X>{name})\)\s+"
    r"(?:inhibitor[s]?|blocker[s]?|antagonist[s]?)\b",
    # "inhibitor of <X>", "inhibition of <X>", "blockade of <X>"
    r"\b(?:inhibitor[s]?|inhibition|blockade)\s+of\s+(?P<X>{name})\b",
    # "<X> knockdown", "<X> KO", "<X> siRNA", "<X> shRNA"
    # R5: extended with "silencing" — entity-first form. Q-phase
    # regression VHL-VIM ("VHL silencing increased vimentin") missed
    # the perturbation flag because entity-first "silencing" wasn't
    # in the alternation; only verb-first "silencing of X" was.
    r"\b(?P<X>{name})\s+(?:knockdown|knock[\- ]?down|knockout|KO|"
    r"siRNA|shRNA|silencing|null|deficient|deficiency|depletion)\b",
    # "knockdown of <X>", "silencing of <X>", "depletion of <X>",
    # "inhibiting <X>", "blocking <X>", "blockade of <X>"
    r"\b(?:knockdown|knock[\- ]?down|silencing|depletion|"
    r"inhibiting|blocking|disruption)\s+of\s+(?P<X>{name})\b",
    # "inhibiting <X>", "blocking <X>" (without "of")
    r"\b(?:inhibiting|blocking|antagonizing|abrogating)\s+(?P<X>{name})\b",
)
_GOF_PATTERNS = (
    # "overexpression of <X>", "ectopic expression of <X>"
    r"\b(?:overexpression|over-expression|ectopic\s+expression)\s+of\s+"
    r"(?P<X>{name})\b",
    # "<X> overexpression", "<X> agonist"
    r"\b(?P<X>{name})\s+(?:overexpression|over-expression|agonist[s]?)\b",
    # "constitutively active <X>", "dominant-active <X>"
    r"\bconstitutively\s+active\s+(?P<X>{name})\b",
    r"\b(?:dominant[\- ]active|gain[\- ]of[\- ]function)\s+(?P<X>{name})\b",
    # "<X>-CA" (constitutively active form)
    r"\b(?P<X>{name})[\- ]CA\b",
)


def _detect_perturbation_for(
    text: str,
    name: str,
    aliases: frozenset[str],
) -> str | None:
    """Return 'loss_of_function' / 'gain_of_function' / None for `name`.

    Scans `text` for LOF/GOF marker constructs anchored to `name` or
    any alias surface form. LOF takes precedence on conflict (more
    common; safer fail-mode is to assume LOF when both detected).

    The {name} placeholder in pattern strings is filled per-alias to
    bound the regex to the actual entity. Aliases are escaped to
    prevent regex injection.
    """
    if not text or not name:
        return None
    surface_forms = list(aliases) + [name]
    # De-dup case-insensitively while preserving longest-first preference
    # so "TGF-beta-1" wins over "TGF" in compile-order.
    seen: set[str] = set()
    forms: list[str] = []
    for s in sorted(surface_forms, key=lambda x: -len(x)):
        if not isinstance(s, str) or len(s) < 2:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        forms.append(s)
    # Build alternation; escape regex metacharacters in entity names.
    alt = "|".join(re.escape(s) for s in forms)
    name_pat = f"(?:{alt})"

    has_lof = False
    has_gof = False
    for tmpl in _LOF_PATTERNS:
        if re.search(tmpl.format(name=name_pat), text, re.IGNORECASE):
            has_lof = True
            break
    if not has_lof:
        for tmpl in _GOF_PATTERNS:
            if re.search(tmpl.format(name=name_pat), text, re.IGNORECASE):
                has_gof = True
                break
    if has_lof:
        return "loss_of_function"
    if has_gof:
        return "gain_of_function"
    return None


# ---------------------------------------------------------------------------
# M7: cascade-terminal detection
# ---------------------------------------------------------------------------
# Pathway-listing surface forms encode a directional cascade. The LAST
# element is treated as the direct upstream of any downstream effect
# described nearby. Diagnosis case: "LCN2 expression is upregulated by
# HER2/PI3K/AKT/NFkappaB pathway" — claim NFkappaB→LCN2 is FN because
# the parser doesn't decompose pathway listings; the regex catches it.
#
# Patterns:
#   - "X/Y/Z pathway", "X/Y/Z signaling", "X/Y/Z cascade", "X/Y/Z axis"
#   - "X-Y-Z pathway" / "X-Y-Z signaling" (hyphen-separated)
# Direction inference: the last element is treated as the immediate
# upstream regulator. Reasonable for the standard "upstream/downstream"
# directional convention; over-aggressive detection is mitigated by
# the consumer requiring claim.subject ∈ cascade_terminals AND claim
# object mentioned in evidence.
_CASCADE_PATTERNS = (
    # "A/B/C pathway" — minimum 3 elements separated by /, followed by
    # a pathway nominative. Permissive on element token form to allow
    # hyphenated and Greek-bearing names.
    re.compile(
        r"\b(?P<chain>[A-Za-z][\w-]+(?:/[A-Za-z][\w-]+){2,})\s+"
        r"(?:pathway|signaling|signalling|cascade|axis|module)\b",
        re.IGNORECASE,
    ),
    # "A-B-C pathway" — three or more uppercase-prefixed tokens
    # separated by hyphens. Hyphens are ambiguous (used inside
    # entity names like "HNP-1"), so require ≥2 hyphens AND each
    # element starts with an uppercase letter.
    re.compile(
        r"\b(?P<chain>[A-Z]\w*(?:-[A-Z]\w*){2,})\s+"
        r"(?:pathway|signaling|signalling|cascade|axis|module)\b",
    ),
)


def _detect_cascade_terminals(
    text: str,
    aliases: dict[str, frozenset[str]],
) -> frozenset[str]:
    """Return claim entities that appear as the cascade terminal in
    any pathway listing in `text`.

    A cascade pattern's LAST element (after splitting on / or -) is
    alias-validated against the claim alias map; if it binds to a
    canonical claim entity, that canonical name is included.
    """
    if not text or not aliases:
        return frozenset()
    out: set[str] = set()
    for pat in _CASCADE_PATTERNS:
        for m in pat.finditer(text):
            chain = m.group("chain")
            # Split on either / or -
            elements = re.split(r"[/-]", chain)
            if len(elements) < 3:
                continue
            last = elements[-1].strip()
            canonical = _bind_to_claim_canonical(last, aliases)
            if canonical:
                out.add(canonical)
    return frozenset(out)


def _bind_to_claim_canonical(token: str,
                              aliases: dict[str, frozenset[str]]) -> str | None:
    """Return the canonical claim entity name `token` matches, or None.

    `token` is a regex capture from the relation-pattern catalog (an
    entity surface form in evidence text). `aliases` is the claim-
    keyed alias map. Match is normalized via M6: case-insensitive,
    Greek↔Latin shortform, hyphen/apostrophe/dot stripped. The first
    canonical name whose alias set contains a normalize-equivalent
    form of `token` wins.

    Why not fuzzy/substring: fuzzy matching is a precision hazard at
    this layer (we'd over-bind 'CDK' to 'CDK1', 'CDK2', ...). Exact
    normalized membership keeps the substrate auditable.
    Multi-word entity names that single-token capture under-collects
    (e.g., 'protein kinase C') are handled by M5's FPLX backfill
    (adding the spelled-out name as an alias).
    """
    if not token:
        return None
    t_norm = _norm_alias(token)
    if not t_norm:
        return None
    for canonical, syns in aliases.items():
        if _norm_alias(canonical) == t_norm:
            return canonical
        for s in syns:
            if _norm_alias(s) == t_norm:
                return canonical
    return None


# Greek letter forms (lowercase + capitalized) → Latin shortform.
# Used by _norm_alias to normalize "p38α" / "PI3Kbeta" / "alpha-actinin"
# variants to a single canonical form for substrate alias matching.
_GREEK_TO_LATIN = {
    "alpha": "a", "α": "a", "Α": "a",
    "beta": "b", "β": "b", "Β": "b",
    "gamma": "g", "γ": "g", "Γ": "g",
    "delta": "d", "δ": "d", "Δ": "d",
    "epsilon": "e", "ε": "e", "Ε": "e",
    "zeta": "z", "ζ": "z", "Ζ": "z",
    "eta": "h", "η": "h", "Η": "h",
    "theta": "q", "θ": "q", "Θ": "q",
    "iota": "i", "ι": "i", "Ι": "i",
    "kappa": "k", "κ": "k", "Κ": "k",
    "lambda": "l", "λ": "l", "Λ": "l",
    "mu": "m", "μ": "m", "Μ": "m",
    "nu": "n", "ν": "n", "Ν": "n",
    "omicron": "o", "ο": "o", "Ο": "o",
    "pi": "p", "π": "p", "Π": "p",
    "rho": "r", "ρ": "r", "Ρ": "r",
    "sigma": "s", "σ": "s", "Σ": "s",
    "tau": "t", "τ": "t", "Τ": "t",
    "phi": "f", "φ": "f", "Φ": "f",
    "chi": "c", "χ": "c", "Χ": "c",
    "psi": "y", "ψ": "y", "Ψ": "y",
    "omega": "w", "ω": "w", "Ω": "w",
}


def _norm_alias(s: str | None) -> str | None:
    """Normalize an entity surface form for substrate alias matching.

    Lowercase, strip punctuation (hyphen, apostrophe, dot), and replace
    Greek letter words with their Latin shortform (alpha → a, beta → b).
    Returns None for empty inputs to bypass match attempts.
    """
    if not s:
        return None
    out = s.lower()
    for greek, latin in _GREEK_TO_LATIN.items():
        out = out.replace(greek.lower(), latin)
    # Strip non-alphanumerics (covers hyphens, apostrophes, dots, slashes).
    out = "".join(ch for ch in out if ch.isalnum())
    return out or None


def _detect_relations(
    text: str,
    aliases: dict[str, frozenset[str]],
) -> tuple:
    """Run the relation_patterns catalog over evidence text; return
    alias-validated DetectedRelation tuple.

    Yields one DetectedRelation per (pattern, match) where BOTH the
    agent and target captures alias-bind to a claim entity. Captures
    where only one side binds are dropped — they aren't actionable
    in the adjudicator's bind-check.

    Site captures are normalized to the canonical Sxx form via the
    same _normalize_site helper used for L5 detected_sites.
    """
    if not text or not aliases:
        return ()
    out: list[DetectedRelation] = []
    seen: set[tuple] = set()
    for pat in CATALOG:
        for m in pat.regex.finditer(text):
            x_text = m.group("X")
            y_text = m.group("Y")
            agent_canon = _bind_to_claim_canonical(x_text, aliases)
            target_canon = _bind_to_claim_canonical(y_text, aliases)
            if agent_canon is None or target_canon is None:
                continue
            if agent_canon == target_canon:
                # Self-relation captures (e.g., regex matches "X X")
                # would survive unless we drop. Self-modification is
                # legitimate but should come from parser, not regex
                # over-capture. Drop here.
                continue
            site = None
            if "site" in pat.regex.groupindex:
                site_text = m.group("site")
                if site_text:
                    site = _normalize_site_freeform(site_text)
            key = (pat.axis, pat.sign, agent_canon, target_canon, site)
            if key in seen:
                continue
            seen.add(key)
            out.append(DetectedRelation(
                axis=pat.axis,
                sign=pat.sign,
                agent_canonical=agent_canon,
                target_canonical=target_canon,
                site=site,
                pattern_id=pat.pattern_id,
                span=m.span(),
            ))
    return tuple(out)


def _normalize_site_freeform(s: str) -> str | None:
    """Map a regex-captured site surface form to canonical Sxx.

    Accepts: 'S102', 'Ser-102', 'Ser102', 'serine 102', 'S-102'.
    Returns: 'S102' (single-letter prefix + position).

    Reuses _normalize_site for the common case; falls through to a
    direct-letter+digits parse for pre-stripped forms.
    """
    s = s.strip().lower()
    # Direct: s102, t461, y732 (with or without dash)
    m = re.match(r"^([sty])-?(\d+)$", s)
    if m:
        return _normalize_site(m.group(1), m.group(2))
    # Word forms: serine 102, threonine 461
    m = re.match(r"^(serine|threonine|tyrosine)\s+(\d+)$", s)
    if m:
        word = m.group(1)
        return _normalize_site(word, m.group(2))
    # Ser102 / Ser-102 / Thr461 / Tyr-732
    m = re.match(r"^(ser|thr|tyr)-?(\d+)$", s)
    if m:
        return _normalize_site(m.group(1), m.group(2))
    return None


def _acceptable_sites_from(stmt) -> frozenset[str]:
    """Build the set of sites the claim accepts as a match.

    Currently a single residue+position pair on the statement; J5 may
    extend this if multi-site claims become a thing in INDRA.
    """
    residue = getattr(stmt, "residue", None)
    position = getattr(stmt, "position", None)
    if not residue and not position:
        return frozenset()
    combined = (residue or "") + (position or "")
    return frozenset({combined}) if combined else frozenset()


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------
def build_context(stmt, evidence) -> EvidenceContext:
    """Build an EvidenceContext from an INDRA Statement and Evidence.

    Pure deterministic Python — no LLM calls. Gilda lookups are cached.
    Failure-tolerant: any per-entity Gilda failure is logged at WARNING
    and the entity is skipped (no aliases recorded for it). The pipeline
    degrades gracefully — an empty alias map for one entity just means
    the parse_evidence prompt doesn't list synonyms for that entity,
    matching pre-J0 behavior.
    """
    from indra_belief.data.entity import GroundedEntity

    stmt_type = type(stmt).__name__
    evidence_text = (getattr(evidence, "text", "") or "").strip()

    # Resolve every distinct claim entity via Gilda. Track first
    # subject + object resolutions for L2/L3 classification.
    aliases: dict[str, frozenset[str]] = {}
    families: dict[str, frozenset[str]] = {}
    pseudogene_set: set[str] = set()
    resolved_entities: list = []

    seen: set[str] = set()
    for n in _agent_names(stmt):
        if not n or n == "?" or n in seen:
            continue
        seen.add(n)
        try:
            ge = GroundedEntity.resolve(n, _raw_text_for(n, evidence))
        except Exception as e:
            log.warning("context_builder: GroundedEntity.resolve(%r) failed: %s", n, e)
            continue
        resolved_entities.append(ge)
        syns = _expand_synonyms(ge)
        if syns:
            aliases[n] = syns
        if getattr(ge, "is_family", False):
            members = getattr(ge, "family_members", []) or []
            family_set = frozenset(m for m in members if isinstance(m, str) and len(m) >= 2)
            if family_set:
                families[n] = family_set
        if getattr(ge, "is_pseudogene", False):
            pseudogene_set.add(n)

    # L2 + L3: subject/object semantic class and precision class. The
    # first resolved entity is the subject; second (when present) is
    # the object. Self-modification has only one entity → object_class
    # mirrors subject_class.
    subject_class = "unknown"
    object_class = "unknown"
    subject_precision = "unknown"
    object_precision = "unknown"
    if resolved_entities:
        subject_class = _classify_subject(resolved_entities[0])
        subject_precision = _classify_precision(resolved_entities[0])
        if len(resolved_entities) > 1:
            object_class = _classify_subject(resolved_entities[1])
            object_precision = _classify_precision(resolved_entities[1])
        else:
            object_class = subject_class
            object_precision = subject_precision

    # Clause split (J5): multi-sentence Evidence.text is split into
    # per-sentence clauses; single-sentence text passes through as
    # `(text,)`. Long single-sentence evidence is left to the
    # single-shot path. The O-phase removed the LLM-side decomposition
    # retry that previously rescued this case; the substrate (M-phase
    # relation catalog + adjudicate substrate-fallback bind) now
    # handles the residual class.
    clauses: tuple[str, ...] = _split_into_clauses(evidence_text)

    # L1: chain-signal regex on the full evidence text (not per-clause
    # — chain markers may span sentence boundaries).
    has_chain_signal, chain_candidates = _detect_chain_signal(evidence_text)

    # L7-fix C2: subject-anchored upstream-attribution detection.
    # Required gate for the L2 cytokine bypass: only bypass when the
    # text explicitly nominates the subject as the upstream actor,
    # not just because the subject's class is "cytokine_or_ligand".
    subject_has_upstream_anchor = False
    if resolved_entities and aliases:
        first_name = getattr(resolved_entities[0], "name", None)
        if first_name and first_name in aliases:
            subject_has_upstream_anchor = _detect_subject_upstream_anchor(
                evidence_text, aliases[first_name]
            )

    # L4: nominalization detection.
    nominalizations = _detect_nominalizations(evidence_text)

    # L5: regex-detected modification sites.
    detected_sites = _detect_sites(evidence_text)

    # M1/M2: catalog-detected canonical relation surface forms,
    # alias-validated against the claim entity set built above.
    # Empty when aliases is empty (no claim entities resolved →
    # nothing to bind captures against).
    detected_relations = _detect_relations(evidence_text, aliases)

    # M7: pathway-listing terminal detection.
    cascade_terminals = _detect_cascade_terminals(evidence_text, aliases)

    # M9: per-side perturbation marker detection. Subject is the first
    # resolved entity; object is the second (matching L2/L3
    # convention). NB: do NOT iterate `seen` — it's a set with
    # non-deterministic order. Use resolved_entities (ordered list).
    subject_perturbation_marker = None
    object_perturbation_marker = None
    if resolved_entities and aliases:
        first_name = getattr(resolved_entities[0], "name", None)
        if first_name and first_name in aliases:
            subject_perturbation_marker = _detect_perturbation_for(
                evidence_text, first_name, aliases[first_name]
            )
        if len(resolved_entities) > 1:
            second_name = getattr(resolved_entities[1], "name", None)
            if second_name and second_name in aliases:
                object_perturbation_marker = _detect_perturbation_for(
                    evidence_text, second_name, aliases[second_name]
                )

    # M10: hedge-marker detection anchored to any claim entity.
    explicit_hedge_markers = _detect_hedge_markers(
        evidence_text, list(aliases.values())
    ) if aliases else frozenset()

    # N6: claim entities for TOPICS OF INTEREST hint in parse_evidence.
    # Source: _agent_names(stmt) — same ordered list parse_claim uses
    # to derive (subject, objects). This avoids re-implementing the
    # Complex-vs-binary projection here (small dup risk, but the doctrine
    # is "context_builder doesn't import parse_claim to dodge cycles").
    # SelfModification yields a single name; we still expose it as both
    # subject and the sole object so the topic hint is symmetric.
    _claim_names = _agent_names(stmt)
    claim_subject_name = _claim_names[0] if _claim_names else ""
    claim_object_names = (
        tuple(_claim_names[1:]) if len(_claim_names) > 1
        else ((_claim_names[0],) if _claim_names else ())
    )
    # SelfModification: subject == object semantically. _agent_names
    # returns just one name in that case; we already mirrored it above.

    return EvidenceContext(
        aliases=aliases,
        families=families,
        is_pseudogene=frozenset(pseudogene_set),
        clauses=clauses,
        binding_admissible=_binding_admissible_for(stmt_type),
        acceptable_sites=_acceptable_sites_from(stmt),
        stmt_type=stmt_type,
        is_complex=stmt_type in _COMPLEX_TYPES,
        is_modification=stmt_type in _MODIFICATION_TYPES,
        is_translocation=stmt_type in _TRANSLOCATION_TYPES,
        # L1
        has_chain_signal=has_chain_signal,
        chain_intermediate_candidates=chain_candidates,
        subject_has_upstream_anchor=subject_has_upstream_anchor,
        # L2
        subject_class=subject_class,
        object_class=object_class,
        # L3
        subject_precision=subject_precision,
        object_precision=object_precision,
        # L4
        nominalized_relations=nominalizations,
        # L5
        detected_sites=detected_sites,
        # M1/M2
        detected_relations=detected_relations,
        # M7
        cascade_terminals=cascade_terminals,
        # M9
        subject_perturbation_marker=subject_perturbation_marker,
        object_perturbation_marker=object_perturbation_marker,
        # M10
        explicit_hedge_markers=explicit_hedge_markers,
        # N6
        claim_subject=claim_subject_name,
        claim_objects=claim_object_names,
    )

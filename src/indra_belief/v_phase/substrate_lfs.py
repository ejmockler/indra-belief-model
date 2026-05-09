"""V6a — substrate-tuned labeling functions in dual mode.

Per research/v5r_data_prep_doctrine.md §3 (LF tables tagged
[substrate-tuned]), §7c (baseline-mode definition), §10 (substrate-tuning
audit).

Each LF is a thin wrapper around an existing detector in
`indra_belief.scorers.context_builder` and
`indra_belief.scorers.relation_patterns`. LFs return an integer class
index ∈ {0..K-1} for the probe's class enum or -1 (ABSTAIN). Snorkel's
LabelModel consumes the integer column directly.

Dual-mode is implemented as a context manager `substrate_baseline_mode()`
that monkey-patches the seven holdout-tuned constants/structures listed
in V5r §7c, runs whatever code is inside the `with` block, then restores
the original state. The baseline state is REPRODUCIBLE FROM CODE — the
context manager is the canonical "M-baseline" definition.

Constraints honored (from the V6a task brief):
  - LFs return -1 (ABSTAIN) when they cannot judge — never default to a class.
  - Existing substrate code is NOT modified — swap is via `setattr` on
    the imported module objects.
  - Detectors are reused, not reimplemented, where possible.
  - Tests run on CPU without llama-server.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Iterator

from indra_belief.scorers import context_builder as _cb
from indra_belief.scorers import relation_patterns as _rp


# ---------------------------------------------------------------------------
# Probe class index maps — must match probes/types.py and grounding.py exactly.
# Snorkel's Λ matrix expects integer class indices, not string class names.
# ---------------------------------------------------------------------------

RELATION_AXIS_CLASSES: dict[str, int] = {
    "direct_sign_match":       0,
    "direct_amount_match":     1,
    "direct_sign_mismatch":    2,
    "direct_axis_mismatch":    3,
    "direct_partner_mismatch": 4,
    "via_mediator":            5,
    "via_mediator_partial":    6,
    "no_relation":             7,
}

ROLE_CLASSES: dict[str, int] = {
    "present_as_subject":  0,
    "present_as_object":   1,
    "present_as_mediator": 2,
    "present_as_decoy":    3,
    "absent":              4,
}

SCOPE_CLASSES: dict[str, int] = {
    "asserted":               0,
    "hedged":                 1,
    "asserted_with_condition": 2,
    "negated":                3,
    "abstain":                4,
}

GROUNDING_CLASSES: dict[str, int] = {
    "mentioned":   0,
    "equivalent":  1,
    "not_present": 2,
    "uncertain":   3,
}

ABSTAIN: int = -1


# ---------------------------------------------------------------------------
# V5r §7c — baseline-substrate state. Code-recoverable enumeration (NOT a
# git commit), so the doctrine is anchored to a deterministic value set.
# ---------------------------------------------------------------------------

# CATALOG entries whose docstring cites a holdout-derived diagnosis (Q-phase
# regression, R2/R5 entries, M11 gap-fix, U-phase additions). Audited from
# `inspect.getsource(relation_patterns)` against V5r §7c criterion.
# Re-derive with: scripts/v6a_audit_catalog_tuned.py if patterns evolve.
_TUNED_PATTERN_IDS: frozenset[str] = frozenset({
    # modification — Q-phase / R2 additions
    "mod_pos.target_compound_by",
    "mod_pos.x_increases_phos_of_y",
    "mod_pos.relative_clause",
    # binding — M11 gap-fix + Q-phase / R2
    "bind.gerund_to",
    "bind.possessive_interaction",
    "bind.forms_complex_with",
    "bind.coord_complex",
    "bind.binding_affinity_for",
    "bind.bound_to",
    "bind.passive_associated",
    "bind.coord_bound_to",
    "bind.gerund_modifier_to",
    # activity — Q-phase / R2 (act_pos.induced_adj base + coord_last are core)
    "act_pos.x_induced_act_of_y",
    "act_pos.x_induced_release_of",
    # amount — Q-phase / R2
    "amt_pos.x_induces_expression_of_y",
    "amt_pos.coord_first",
})

# Universal hedge cues per V5r §7c — restrict to what survives in baseline.
_BASELINE_HEDGE_MARKERS: tuple[str, ...] = (
    "may", "might", "suggest", "propose", "hypothesize",
)

# Default M13 hedging-window; V5r §7c: revert from 60 to 30.
_BASELINE_HEDGE_PROXIMITY: int = 30


# ---------------------------------------------------------------------------
# Dual-mode swap primitive
# ---------------------------------------------------------------------------

@dataclass
class _Snapshot:
    """Captures the original state of the seven swapped attributes."""
    cb_cytokine_ligand_hgnc: frozenset
    cb_site_denylist: frozenset
    cb_hedge_markers: tuple
    cb_hedge_proximity: int
    cb_lof_patterns: tuple
    rp_catalog: tuple
    rp_binding_admissible_for: object


def _baseline_binding_admissible_for(stmt_type: str) -> frozenset:
    """Core binding gate per V5r §7c: retain core (Complex/Translocation)
    rules; drop the diagnosis-tuned _MODIFICATION_TYPES refinements.

    The CURRENT _binding_admissible_for already only encodes Complex
    and Translocation — modification gates are a no-op (return empty).
    The 'diagnosis-tuned' refinement called out in §7c is the
    enumerated _MODIFICATION_TYPES set used elsewhere in the builder
    (modification statements emit empty binding_admissible regardless).
    Baseline binding gate is therefore the same shape, but without
    the Translocation refinement (which V5r §7c categorizes as a
    diagnosis-derived refinement). Translocation defaults to empty.
    """
    if stmt_type == "Complex":
        return frozenset({"protein"})
    return frozenset()


@contextlib.contextmanager
def substrate_baseline_mode() -> Iterator[None]:
    """Context manager: swap holdout-tuned substrate state for baseline.

    Inside the `with` block, the seven attributes named in V5r §7c are
    set to their baseline values. On exit (success OR exception), the
    original values are restored. The swap is global (module-level
    `setattr`); concurrent use across threads is unsafe by design — V7c
    runs LFs sequentially.

    Usage:
        with substrate_baseline_mode():
            vote_baseline = lf_substrate_catalog_match(stmt, ev)
        vote_tuned = lf_substrate_catalog_match(stmt, ev)
    """
    snap = _Snapshot(
        cb_cytokine_ligand_hgnc=_cb._CYTOKINE_LIGAND_HGNC,
        cb_site_denylist=_cb._SITE_DENYLIST,
        cb_hedge_markers=_cb._HEDGE_MARKERS,
        cb_hedge_proximity=_cb._HEDGE_PROXIMITY_CHARS,
        cb_lof_patterns=_cb._LOF_PATTERNS,
        rp_catalog=_rp.CATALOG,
        rp_binding_admissible_for=_cb._binding_admissible_for,
    )
    baseline_catalog = tuple(p for p in _rp.CATALOG
                             if p.pattern_id not in _TUNED_PATTERN_IDS)
    try:
        _cb._CYTOKINE_LIGAND_HGNC = frozenset()
        _cb._SITE_DENYLIST = frozenset()
        _cb._HEDGE_MARKERS = _BASELINE_HEDGE_MARKERS
        _cb._HEDGE_PROXIMITY_CHARS = _BASELINE_HEDGE_PROXIMITY
        _cb._LOF_PATTERNS = ()
        _rp.CATALOG = baseline_catalog
        _cb._binding_admissible_for = _baseline_binding_admissible_for
        yield
    finally:
        _cb._CYTOKINE_LIGAND_HGNC = snap.cb_cytokine_ligand_hgnc
        _cb._SITE_DENYLIST = snap.cb_site_denylist
        _cb._HEDGE_MARKERS = snap.cb_hedge_markers
        _cb._HEDGE_PROXIMITY_CHARS = snap.cb_hedge_proximity
        _cb._LOF_PATTERNS = snap.cb_lof_patterns
        _rp.CATALOG = snap.rp_catalog
        _cb._binding_admissible_for = snap.rp_binding_admissible_for


# ---------------------------------------------------------------------------
# (statement, evidence) input adapter
# ---------------------------------------------------------------------------
# V5r training corpus carries dict-shaped records (data/benchmark/*.jsonl);
# substrate detectors expect INDRA Statement+Evidence. Build INDRA objects
# lazily from a dict, or accept INDRA objects directly.

_STMT_TYPE_AXIS: dict[str, tuple[str, str]] = {
    # (axis, sign) per parse_claim.py
    "Activation":     ("activity", "positive"),
    "Inhibition":     ("activity", "negative"),
    "Phosphorylation": ("modification", "positive"),
    "Dephosphorylation": ("modification", "negative"),
    "Acetylation":    ("modification", "positive"),
    "Deacetylation":  ("modification", "negative"),
    "Methylation":    ("modification", "positive"),
    "Demethylation":  ("modification", "negative"),
    "Ubiquitination": ("modification", "positive"),
    "Deubiquitination": ("modification", "negative"),
    "IncreaseAmount": ("amount",      "positive"),
    "DecreaseAmount": ("amount",      "negative"),
    "Complex":        ("binding",     "neutral"),
    "Translocation":  ("localization", "neutral"),
    "Autophosphorylation": ("modification", "positive"),
    "Gef":            ("gtp_state",   "positive"),
    "Gap":            ("gtp_state",   "negative"),
}


def _to_indra_pair(statement, evidence):
    """Coerce (statement, evidence) into INDRA objects. Accepts dicts
    (V5r training shape) or INDRA objects.

    Returns (stmt, ev) or (None, None) if coercion fails (caller treats
    as ABSTAIN).
    """
    # If already INDRA-shaped (has .agent_list or .members), pass through.
    if hasattr(statement, "agent_list") or hasattr(statement, "members"):
        return statement, evidence

    # Dict-shaped: build INDRA objects.
    from indra.statements import (
        Activation, Inhibition, Phosphorylation, Dephosphorylation,
        Acetylation, Deacetylation, Methylation, Demethylation,
        Ubiquitination, Deubiquitination, IncreaseAmount, DecreaseAmount,
        Complex, Translocation, Autophosphorylation, Gef, Gap,
        Agent, Evidence as _Ev,
    )
    type_map = {
        "Activation": Activation, "Inhibition": Inhibition,
        "Phosphorylation": Phosphorylation, "Dephosphorylation": Dephosphorylation,
        "Acetylation": Acetylation, "Deacetylation": Deacetylation,
        "Methylation": Methylation, "Demethylation": Demethylation,
        "Ubiquitination": Ubiquitination, "Deubiquitination": Deubiquitination,
        "IncreaseAmount": IncreaseAmount, "DecreaseAmount": DecreaseAmount,
        "Complex": Complex, "Translocation": Translocation,
        "Autophosphorylation": Autophosphorylation,
        "Gef": Gef, "Gap": Gap,
    }
    cls = type_map.get(statement.get("stmt_type"))
    subj = statement.get("subject")
    obj = statement.get("object")
    if cls is None or not subj:
        return None, None

    text = (
        evidence.get("text")
        if isinstance(evidence, dict) and evidence.get("text")
        else (statement.get("evidence_text") or "")
    )
    pmid = (evidence.get("pmid") if isinstance(evidence, dict)
            else statement.get("pmid"))
    source_api = (evidence.get("source_api") if isinstance(evidence, dict)
                  else statement.get("source_api", "reach"))
    ev = _Ev(source_api=source_api or "reach", pmid=pmid, text=text or "")
    if cls is Complex:
        if not obj:
            return None, None
        stmt = Complex([Agent(subj), Agent(obj)])
    elif cls is Autophosphorylation:
        stmt = Autophosphorylation(Agent(subj))
    elif cls is Translocation:
        stmt = Translocation(Agent(subj), None, obj)
    else:
        if not obj:
            return None, None
        stmt = cls(Agent(subj), Agent(obj))
    return stmt, ev


def _build_context_safe(statement, evidence):
    """Build EvidenceContext, returning None if coercion or build fails."""
    stmt, ev = _to_indra_pair(statement, evidence)
    if stmt is None:
        return None
    try:
        return _cb.build_context(stmt, ev)
    except Exception:
        return None


def _claim_axis_sign(statement) -> tuple[str | None, str | None]:
    """Return (axis, sign) for a dict or INDRA statement. None on unknown."""
    if hasattr(statement, "agent_list") or hasattr(statement, "members"):
        stmt_type = type(statement).__name__
    elif isinstance(statement, dict):
        stmt_type = statement.get("stmt_type")
    else:
        return None, None
    return _STMT_TYPE_AXIS.get(stmt_type, (None, None))


def _claim_subject_object(statement) -> tuple[str | None, str | None]:
    """Return (subject_canonical, object_canonical) string names."""
    if isinstance(statement, dict):
        return statement.get("subject"), statement.get("object")
    if hasattr(statement, "members"):  # Complex
        members = [m for m in statement.members if m is not None]
        if len(members) >= 2:
            return members[0].name, members[1].name
        return (members[0].name if members else None), None
    if hasattr(statement, "agent_list"):
        agents = [a for a in statement.agent_list() if a is not None]
        return (agents[0].name if agents else None), (
            agents[1].name if len(agents) > 1 else None
        )
    return None, None


def _claim_stmt_type(statement) -> str | None:
    if isinstance(statement, dict):
        return statement.get("stmt_type")
    return type(statement).__name__ if statement is not None else None


def _evidence_text(evidence, statement=None) -> str:
    if isinstance(evidence, dict):
        t = evidence.get("text")
        if t:
            return t
    if hasattr(evidence, "text") and evidence.text:
        return evidence.text
    if isinstance(statement, dict):
        return statement.get("evidence_text") or ""
    return ""


# ---------------------------------------------------------------------------
# relation_axis LFs
# ---------------------------------------------------------------------------

def lf_substrate_catalog_match(statement, evidence, mode: str = "tuned") -> int:
    """Vote `direct_sign_match` (0) when CATALOG verb regex anchored on
    claim entities matches evidence on the SAME axis as the claim AND
    sign aligns. Per V5r §3.1.

    Returns ABSTAIN when no aligned match, when claim entities don't
    resolve, or when context build fails.
    """
    if mode == "baseline":
        with substrate_baseline_mode():
            return _lf_substrate_catalog_match_inner(statement, evidence)
    return _lf_substrate_catalog_match_inner(statement, evidence)


def _lf_substrate_catalog_match_inner(statement, evidence) -> int:
    ctx = _build_context_safe(statement, evidence)
    if ctx is None:
        return ABSTAIN
    claim_axis, claim_sign = _claim_axis_sign(statement)
    subj, obj = _claim_subject_object(statement)
    if not subj or claim_axis is None:
        return ABSTAIN
    # Find aligned (subject→object on claim axis with matching sign).
    for dr in ctx.detected_relations:
        # Probe router normalizes "translocation" → "localization"; mirror.
        dr_axis = "localization" if dr.axis == "translocation" else dr.axis
        if dr_axis != claim_axis:
            continue
        if dr.agent_canonical != subj:
            # Binding is order-symmetric.
            if claim_axis == "binding" and dr.target_canonical == subj \
                    and dr.agent_canonical == obj:
                return RELATION_AXIS_CLASSES["direct_sign_match"]
            continue
        if dr.target_canonical != obj:
            continue
        if dr.sign == claim_sign or claim_axis == "binding":
            return RELATION_AXIS_CLASSES["direct_sign_match"]
    return ABSTAIN


# V6f: dropped lf_substrate_negation_regex (V7a 12.5%, V7b 0%).
# V6f: dropped lf_chain_no_terminal (V7a 0%, V7b 0%).
# V6f: dropped lf_partner_substrate_gate (V7b 0%; LF voted on wrong proposition).


# ---------------------------------------------------------------------------
# subject_role / object_role LFs
# ---------------------------------------------------------------------------

def lf_substrate_chain_position(statement, evidence, *,
                                 entity: str = "subject",
                                 mode: str = "tuned") -> int:
    """Vote `present_as_mediator` (2) when the entity is detected as a
    middle node in a chain.

    Per V5r §3.2: substrate detects entity as middle node (path
    A→entity→B). Implementation: ctx.has_chain_signal AND entity name
    appears in ctx.chain_intermediate_candidates AND BOTH a candidate
    upstream and a candidate downstream entity exist. Conservative —
    requires the upstream/downstream actually present in evidence.
    """
    if mode == "baseline":
        with substrate_baseline_mode():
            return _lf_substrate_chain_position_inner(statement, evidence, entity)
    return _lf_substrate_chain_position_inner(statement, evidence, entity)


def _lf_substrate_chain_position_inner(statement, evidence, entity: str) -> int:
    ctx = _build_context_safe(statement, evidence)
    if ctx is None:
        return ABSTAIN
    subj, obj = _claim_subject_object(statement)
    target = subj if entity == "subject" else obj
    if not target:
        return ABSTAIN
    if not ctx.has_chain_signal:
        return ABSTAIN
    candidates = ctx.chain_intermediate_candidates or ()
    if target not in candidates:
        return ABSTAIN
    if len(candidates) < 2:
        return ABSTAIN
    return ROLE_CLASSES["present_as_mediator"]


def lf_substrate_decoy(statement, evidence, *,
                       entity: str = "subject",
                       mode: str = "tuned") -> int:
    """Vote `present_as_decoy` (3) when the entity appears in a relation
    pattern that does NOT match the claim's relation.

    Per V5r §3.2: substrate detects entity in unrelated relation
    pattern (different verb, different partner). Implementation: claim
    axis + entity participates in CATALOG match where (axis !=
    claim.axis) AND the partner is NOT the claim's other entity.
    Conservative — fires only on observed CATALOG hits.
    """
    if mode == "baseline":
        with substrate_baseline_mode():
            return _lf_substrate_decoy_inner(statement, evidence, entity)
    return _lf_substrate_decoy_inner(statement, evidence, entity)


def _lf_substrate_decoy_inner(statement, evidence, entity: str) -> int:
    ctx = _build_context_safe(statement, evidence)
    if ctx is None:
        return ABSTAIN
    claim_axis, _ = _claim_axis_sign(statement)
    subj, obj = _claim_subject_object(statement)
    target = subj if entity == "subject" else obj
    counterpart = obj if entity == "subject" else subj
    if not target or claim_axis is None:
        return ABSTAIN
    for dr in ctx.detected_relations:
        if target not in (dr.agent_canonical, dr.target_canonical):
            continue
        partner = (dr.target_canonical if dr.agent_canonical == target
                   else dr.agent_canonical)
        dr_axis = "localization" if dr.axis == "translocation" else dr.axis
        if dr_axis != claim_axis and partner != counterpart:
            return ROLE_CLASSES["present_as_decoy"]
    return ABSTAIN


# ---------------------------------------------------------------------------
# scope LFs
# ---------------------------------------------------------------------------

def lf_substrate_hedge_marker(statement, evidence, mode: str = "tuned") -> int:
    """Vote `hedged` (1) when M10 hedge detector fires.

    Per V5r §3.3: M10 hedge detector (may, might, suggest, propose,
    hypothesize, putative, etc.) anchored to claim entity proximity.
    Reuses `_detect_hedge_markers` via `build_context.explicit_hedge_markers`.
    """
    if mode == "baseline":
        with substrate_baseline_mode():
            return _lf_substrate_hedge_marker_inner(statement, evidence)
    return _lf_substrate_hedge_marker_inner(statement, evidence)


# V6e Fix 1: claim-verb cue map for substrate scope LFs. Mirrors
# clean_lfs._CLAIM_VERB_CUE_MAP but kept local so substrate_lfs has no
# import-time dependency on clean_lfs.
_CLAIM_VERB_CUE_MAP_SUB: dict[str, str] = {
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

_CLAIM_VERB_WINDOW_CHARS = 50


def _claim_verb_cue_positions_sub(text: str, statement) -> list[int]:
    """Positions of the claim's stmt-type verb cue in evidence text."""
    stmt_type = _claim_stmt_type(statement)
    if not stmt_type:
        return []
    cue = _CLAIM_VERB_CUE_MAP_SUB.get(stmt_type)
    if not cue:
        return []
    pat = _re.compile(r"\b(?:" + cue + r")", _re.IGNORECASE)
    return [m.start() for m in pat.finditer(text)]


def _hedge_or_neg_within_claim_verb_window(
    text: str, statement, marker_pattern: _re.Pattern[str],
    window_chars: int = _CLAIM_VERB_WINDOW_CHARS,
) -> bool:
    """Whether `marker_pattern` matches within ±window_chars of the
    CLAIM stmt-type verb cue (NOT any verb in evidence)."""
    positions = _claim_verb_cue_positions_sub(text, statement)
    if not positions:
        return False
    for pos in positions:
        lo = max(0, pos - window_chars)
        hi = min(len(text), pos + window_chars)
        if marker_pattern.search(text[lo:hi]):
            return True
    return False


def _lf_substrate_hedge_marker_inner(statement, evidence) -> int:
    """V6e Fix 1: substrate hedge marker now requires the marker to fall
    within the claim-verb window. Previously voted `hedged` whenever
    `ctx.explicit_hedge_markers` was non-empty — but that detector
    anchors on entity proximity (60-char window), not on the claim verb,
    producing FPs when hedging language sits in surrounding context."""
    ctx = _build_context_safe(statement, evidence)
    if ctx is None:
        return ABSTAIN
    if not ctx.explicit_hedge_markers:
        return ABSTAIN
    text = _evidence_text(evidence, statement)
    if not text:
        return ABSTAIN
    # Build a regex over the markers the substrate detected; require any
    # of them to fall within the claim-verb window.
    markers = sorted({m for m in ctx.explicit_hedge_markers if m})
    if not markers:
        return ABSTAIN
    pat = _re.compile(
        r"\b(?:" + "|".join(_re.escape(m) for m in markers) + r")\b",
        _re.IGNORECASE,
    )
    if _hedge_or_neg_within_claim_verb_window(text, statement, pat):
        return SCOPE_CLASSES["hedged"]
    return ABSTAIN


# Negation cue regex shared with router — kept inline so the LF doesn't
# depend on `probes/router.py` (router pulls in LLM probe machinery).
import re as _re

_NEGATION_RE = _re.compile(
    r"\b(?:not|cannot|did\s+not|does\s+not|do\s+not|"
    r"is\s+not|are\s+not|was\s+not|were\s+not|"
    r"failed\s+to|fails\s+to|fail\s+to|never)\b",
    _re.IGNORECASE,
)
_NEG_WINDOW_CHARS = 50


def lf_substrate_negation_explicit(statement, evidence,
                                    mode: str = "tuned") -> int:
    """Vote `negated` (3) when M9 negation pattern anchored on claim
    verb fires.

    Per V5r §3.3: M9 negation (no/not/never/fail to/did not/lack of)
    anchored on claim verb. Implementation: explicit verb-negator
    within _NEG_WINDOW_CHARS of the claim subject's alias mention AND
    sitting between the subject and object positions.
    """
    if mode == "baseline":
        with substrate_baseline_mode():
            return _lf_substrate_negation_explicit_inner(statement, evidence)
    return _lf_substrate_negation_explicit_inner(statement, evidence)


def _lf_substrate_negation_explicit_inner(statement, evidence) -> int:
    """V6e Fix 1: anchor negation cue on the CLAIM stmt-type verb, AND
    require the cue to sit between the claim subject and object
    mentions. Previous version anchored on subject/object alias span,
    which fired on negation language anywhere between the entities even
    when that language modified a different verb (e.g., `PLK1, which
    does not bind ATP under stress, phosphorylates CDC20`)."""
    ctx = _build_context_safe(statement, evidence)
    if ctx is None:
        return ABSTAIN
    text = _evidence_text(evidence, statement)
    if not text:
        return ABSTAIN
    subj, obj = _claim_subject_object(statement)
    if not subj or not obj:
        return ABSTAIN
    subj_aliases = ctx.aliases.get(subj, frozenset()) | {subj}
    obj_aliases = ctx.aliases.get(obj, frozenset()) | {obj}

    def _positions(aliases: frozenset[str]) -> list[int]:
        out: list[int] = []
        for a in aliases:
            if not isinstance(a, str) or len(a) < 2:
                continue
            for m in _re.finditer(r"\b" + _re.escape(a) + r"\b",
                                   text, _re.IGNORECASE):
                out.append(m.start())
        return out

    s_pos = _positions(subj_aliases)
    o_pos = _positions(obj_aliases)
    if not s_pos or not o_pos:
        return ABSTAIN
    # V6e: require negation within the claim-verb window.
    if not _hedge_or_neg_within_claim_verb_window(
            text, statement, _NEGATION_RE,
            window_chars=_CLAIM_VERB_WINDOW_CHARS):
        return ABSTAIN
    # AND between subject and object mentions (preserves the prior gate
    # so negation in unrelated clauses doesn't fire).
    span_lo = min(min(s_pos), min(o_pos))
    span_hi = max(max(s_pos), max(o_pos))
    for m in _NEGATION_RE.finditer(text):
        ns = m.start()
        if span_lo <= ns <= span_hi:
            return SCOPE_CLASSES["negated"]
    return ABSTAIN


_CONDITIONAL_RE = _re.compile(
    # Two shapes:
    #   (a) preposition + condition word: "in <condition>"
    #   (b) bare wild-type / mutant phrase as adjective: "X binds wild-type Y"
    r"\b(?:in|under|upon|during|after|with|without)\s+"
    r"(?:wild[\s-]type|mutant|presence\s+of|absence\s+of|"
    r"knockout|knockdown|the\s+presence\s+of|the\s+absence\s+of)\b"
    r"|\bwild[\s-]?type\b|\bmutant\b",
    _re.IGNORECASE,
)


def lf_conditional_clause_substrate(statement, evidence,
                                     mode: str = "tuned") -> int:
    """Vote `asserted_with_condition` (2) when conditional clause fires
    near a claim entity.

    Per V5r §3.3: substrate detects "in <condition>", "wild-type vs
    mutant", "in the presence of X" pattern. Substrate-tagged because
    the wild-type/mutant lexicon was tuned during S-phase Intervention E.
    """
    if mode == "baseline":
        with substrate_baseline_mode():
            return _lf_conditional_clause_substrate_inner(statement, evidence)
    return _lf_conditional_clause_substrate_inner(statement, evidence)


def _lf_conditional_clause_substrate_inner(statement, evidence) -> int:
    text = _evidence_text(evidence, statement)
    if not text:
        return ABSTAIN
    if _CONDITIONAL_RE.search(text):
        return SCOPE_CLASSES["asserted_with_condition"]
    return ABSTAIN


# ---------------------------------------------------------------------------
# verify_grounding LFs
# ---------------------------------------------------------------------------

# Processed-form indicator lexicon — V5r §3.4 flags this as substrate-tuned
# (the Aβ→APP entry was added during S-phase audit). Each indicator is a
# regex that, when matched alongside a claim-entity alias, votes "equivalent".
_PROCESSED_FORM_INDICATORS: tuple[str, ...] = (
    r"\bcleaved\s+",
    r"\bphosphorylated\s+",
    r"\bprocessed\s+",
    r"\b(?:[A-Z][\w-]+)\s+peptide\b",
    r"\bA(?:β|beta)\b",       # Aβ — APP fragment, holdout-derived addition
    r"-CTD\b",                 # X-CTD (C-terminal domain)
    r"-NTD\b",                 # X-NTD
    r"-CTF\b",
)
_PROCESSED_RE_TUNED = _re.compile(
    "|".join(_PROCESSED_FORM_INDICATORS), _re.IGNORECASE
)
# Baseline drops the holdout-derived Aβ entry and the NTD/CTF refinements.
_PROCESSED_RE_BASELINE = _re.compile(
    r"\bcleaved\s+|\bphosphorylated\s+|\bprocessed\s+|"
    r"\b(?:[A-Z][\w-]+)\s+peptide\b|-CTD\b",
    _re.IGNORECASE,
)


def lf_fragment_processed_form(statement, evidence, *,
                                entity: str = "subject",
                                mode: str = "tuned") -> int:
    """Vote `equivalent` (1) when a processed-form indicator + claim
    entity alias matches in evidence.

    Per V5r §3.4: lexicon includes holdout-derived entries (Aβ→APP from
    S-phase audit). Subject to V7c contamination measurement.
    """
    if mode == "baseline":
        with substrate_baseline_mode():
            return _lf_fragment_processed_form_inner(
                statement, evidence, entity, _PROCESSED_RE_BASELINE
            )
    return _lf_fragment_processed_form_inner(
        statement, evidence, entity, _PROCESSED_RE_TUNED
    )


def _lf_fragment_processed_form_inner(statement, evidence, entity, regex) -> int:
    text = _evidence_text(evidence, statement)
    if not text:
        return ABSTAIN
    ctx = _build_context_safe(statement, evidence)
    if ctx is None:
        return ABSTAIN
    subj, obj = _claim_subject_object(statement)
    target = subj if entity == "subject" else obj
    if not target:
        return ABSTAIN
    # Either: a processed-form indicator anywhere in evidence (Aβ-style),
    # OR an indicator within 30 chars of a claim entity alias.
    if not regex.search(text):
        return ABSTAIN
    aliases = ctx.aliases.get(target, frozenset()) | {target}
    for a in aliases:
        if not isinstance(a, str) or len(a) < 2:
            continue
        for m in _re.finditer(r"\b" + _re.escape(a) + r"\b",
                               text, _re.IGNORECASE):
            window = text[max(0, m.start() - 30):m.end() + 30]
            if regex.search(window):
                return GROUNDING_CLASSES["equivalent"]
    # Aβ-style: indicator without a literal alias mention also counts
    # equivalent IF the indicator is one of the holdout-flagged tokens
    # (Aβ/-CTD/-NTD). For the baseline regex we don't have those, so
    # this branch is naturally narrower in baseline mode.
    standalone = _re.search(r"\bA(?:β|beta)\b|-CTD\b|-NTD\b", text)
    if standalone and regex is _PROCESSED_RE_TUNED:
        return GROUNDING_CLASSES["equivalent"]
    return ABSTAIN


# ---------------------------------------------------------------------------
# LF index — convenience accessor for V6c and V7c
# ---------------------------------------------------------------------------

# Each entry: (probe_kind, lf_callable, default_kwargs).
# Substrate-tuned LF inventory; V6b adds the [clean] LFs separately.
LF_INDEX: tuple[tuple[str, str, callable, dict], ...] = (
    ("relation_axis", "lf_substrate_catalog_match",
     lf_substrate_catalog_match, {}),
    # V6f: dropped lf_substrate_negation_regex, lf_chain_no_terminal,
    # lf_partner_substrate_gate. See research/v6f_redesign_log.md.
    ("subject_role", "lf_substrate_chain_position_subject",
     lf_substrate_chain_position, {"entity": "subject"}),
    ("subject_role", "lf_substrate_decoy_subject",
     lf_substrate_decoy, {"entity": "subject"}),
    ("object_role", "lf_substrate_chain_position_object",
     lf_substrate_chain_position, {"entity": "object"}),
    ("object_role", "lf_substrate_decoy_object",
     lf_substrate_decoy, {"entity": "object"}),
    ("scope", "lf_substrate_hedge_marker",
     lf_substrate_hedge_marker, {}),
    ("scope", "lf_substrate_negation_explicit",
     lf_substrate_negation_explicit, {}),
    ("scope", "lf_conditional_clause_substrate",
     lf_conditional_clause_substrate, {}),
    ("verify_grounding", "lf_fragment_processed_form_subject",
     lf_fragment_processed_form, {"entity": "subject"}),
    ("verify_grounding", "lf_fragment_processed_form_object",
     lf_fragment_processed_form, {"entity": "object"}),
)


def all_lf_votes(statement, evidence, mode: str = "tuned") -> dict:
    """Run every LF in LF_INDEX in `mode` and return a name→vote dict."""
    out: dict[str, int] = {}
    for _kind, name, fn, kwargs in LF_INDEX:
        out[name] = fn(statement, evidence, mode=mode, **kwargs)
    return out

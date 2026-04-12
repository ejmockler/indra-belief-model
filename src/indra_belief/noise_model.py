"""Parametric belief scoring from INDRA source metadata.

Implements the INDRA noise model (SimpleScorer formula):

    P(incorrect) = Product_source [syst(s) + Product_evidence rand(s, j)]
    belief = 1 - P(incorrect)

This is the ADDITIVE formula from INDRA's actual SimpleScorer
(indra/belief/__init__.py), NOT the conditional form
syst + (1-syst) * rand^n from the Gyori et al. 2017 paper.
The priors are calibrated to this additive formula:
    rand = 1 - accuracy - syst  (see BayesianScorer derivation)

Default error priors from INDRA's calibration.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# INDRA default error priors: {source: (rand, syst)}
# Source: indra/resources/default_belief_probs.json
INDRA_PRIORS: dict[str, tuple[float, float]] = {
    "reach": (0.30, 0.05),
    "sparser": (0.30, 0.05),
    "trips": (0.30, 0.05),
    "rlimsp": (0.20, 0.05),
    "medscan": (0.30, 0.05),
    "eidos": (0.30, 0.05),
    "cwms": (0.30, 0.05),
    "signor": (0.049, 0.01),
    "biogrid": (0.01, 0.01),
    "phosphosite": (0.01, 0.01),
    "cbn": (0.01, 0.01),
    "trrust": (0.10, 0.01),
    "tas": (0.01, 0.01),
    "hprd": (0.01, 0.01),
    "pid": (0.01, 0.01),
    "reactome": (0.01, 0.01),
    "kegg": (0.01, 0.01),
    "drugbank": (0.01, 0.01),
}

# Fallback for unknown sources
_DEFAULT_PRIOR = (0.30, 0.10)


def compute_edge_reliability(
    sources: list[str],
    evidence_count: int,
    priors: dict[str, tuple[float, float]] | None = None,
) -> float:
    """Compute edge reliability from source metadata.

    Uses the INDRA additive formula: for each source s with n_s evidence,
    the source incorrectness factor is syst(s) + rand(s)^n_s.

    When per-source evidence counts are not available, distributes
    evidence_count across sources: 1 per source, remainder to the
    first source (conservative — assumes least informative source
    gets the bulk).

    Args:
        sources: List of source API names (e.g., ['reach', 'signor']).
        evidence_count: Total evidence count across all sources.
        priors: Optional custom {source: (rand, syst)} priors.

    Returns:
        Reliability score in [0, 1].
    """
    if not sources or evidence_count <= 0:
        return 0.0

    if priors is None:
        priors = INDRA_PRIORS

    # Distribute evidence across sources
    unique_sources = sorted(set(sources))
    n_sources = len(unique_sources)

    if n_sources == 1:
        per_source = {unique_sources[0]: evidence_count}
    else:
        per_source = {s: 1 for s in unique_sources}
        remainder = evidence_count - n_sources
        if remainder > 0:
            per_source[unique_sources[0]] += remainder

    # INDRA additive formula: P(incorrect) = Product [syst + rand^n]
    p_incorrect = 1.0
    for source, n in per_source.items():
        rand, syst = priors.get(source.lower(), _DEFAULT_PRIOR)
        p_incorrect *= syst + rand ** n

    return max(0.0, min(1.0, 1.0 - p_incorrect))


def compute_edge_reliability_with_contradiction(
    edges: list[dict],
    priors: dict[str, tuple[float, float]] | None = None,
) -> tuple[float, str, bool]:
    """Compute aggregated reliability for a gene pair with contradictory evidence.

    Groups edges by regulation_type, computes reliability for each direction,
    then applies the contradictory penalty:
        reliability = reliability_dominant * (1 - reliability_opposing)

    Args:
        edges: List of edge metadata dicts, each with 'regulation_type',
            'sources', and 'evidence_count'.
        priors: Optional custom priors.

    Returns:
        (aggregated_reliability, dominant_direction, is_contradictory)
    """
    if not edges:
        return 0.0, "unknown", False

    by_direction: dict[str, list[dict]] = {}
    for e in edges:
        d = e.get("regulation_type", "unknown")
        by_direction.setdefault(d, []).append(e)

    dir_reliabilities: dict[str, float] = {}
    for direction, dir_edges in by_direction.items():
        all_sources = []
        total_evidence = 0
        for e in dir_edges:
            all_sources.extend(e.get("sources", []))
            total_evidence += e.get("evidence_count", 0)
        dir_reliabilities[direction] = compute_edge_reliability(
            all_sources, total_evidence, priors,
        )

    if not dir_reliabilities:
        return 0.0, "unknown", False

    dominant_dir = max(dir_reliabilities, key=dir_reliabilities.get)
    dominant_reliability = dir_reliabilities[dominant_dir]

    opposing_dirs = {
        d: b for d, b in dir_reliabilities.items()
        if d != dominant_dir and d != "unknown"
    }
    if opposing_dirs:
        opposing_reliability = max(opposing_dirs.values())
        return (
            dominant_reliability * (1.0 - opposing_reliability),
            dominant_dir,
            True,
        )

    return dominant_reliability, dominant_dir, False


# ---------------------------------------------------------------------------
# Gated belief: evidence filtered by LLM verdicts
# ---------------------------------------------------------------------------

@dataclass
class SourceBreakdown:
    """Per-source contribution to a gated belief score."""
    source: str
    n_total: int
    n_surviving: int
    rand: float
    syst: float
    incorrectness_factor: float  # syst + rand^n_surviving, or None if removed


@dataclass
class GatedBeliefResult:
    """Result of computing belief with LLM-gated evidence."""
    belief: float
    parametric_only: float  # belief without any gating (all evidence counts)
    n_total_evidence: int
    n_surviving_evidence: int
    n_gated: int
    per_source: list[SourceBreakdown]


def compute_gated_belief(
    evidence: list[dict],
    priors: dict[str, tuple[float, float]] | None = None,
) -> GatedBeliefResult:
    """Compute belief with per-evidence gating from LLM verdicts.

    Each evidence dict must have:
        - source_api: str (e.g., 'reach', 'signor')
        - included: bool (True = LLM approved or unscored, False = gated out)

    Under INDRA's additive formula, setting rand_j=1.0 for gated evidence
    is INVALID (syst + 1.0 > 1.0). Instead, when ALL evidence from a source
    is gated out, that source is removed entirely from the product — it
    contributes nothing, as if it never reported the edge.

    Args:
        evidence: List of evidence dicts with 'source_api' and 'included'.
        priors: Optional custom priors.

    Returns:
        GatedBeliefResult with belief, parametric-only, and per-source breakdown.
    """
    if priors is None:
        priors = INDRA_PRIORS

    if not evidence:
        return GatedBeliefResult(
            belief=0.0, parametric_only=0.0,
            n_total_evidence=0, n_surviving_evidence=0, n_gated=0,
            per_source=[],
        )

    # Group evidence by source
    by_source: dict[str, dict] = {}  # {source: {total: int, surviving: int}}
    for ev in evidence:
        src = ev["source_api"].lower()
        if src not in by_source:
            by_source[src] = {"total": 0, "surviving": 0}
        by_source[src]["total"] += 1
        if ev.get("included", True):
            by_source[src]["surviving"] += 1

    # Compute parametric-only (no gating)
    p_incorrect_parametric = 1.0
    for src, counts in by_source.items():
        rand, syst = priors.get(src, _DEFAULT_PRIOR)
        p_incorrect_parametric *= syst + rand ** counts["total"]

    parametric_only = max(0.0, min(1.0, 1.0 - p_incorrect_parametric))

    # Compute gated belief (sources with 0 surviving evidence are removed)
    p_incorrect_gated = 1.0
    breakdowns = []
    n_total = 0
    n_surviving = 0

    for src, counts in sorted(by_source.items()):
        rand, syst = priors.get(src, _DEFAULT_PRIOR)
        n_total += counts["total"]
        n_surviving += counts["surviving"]

        if counts["surviving"] == 0:
            # Source removed — contributes nothing to the product
            breakdowns.append(SourceBreakdown(
                source=src, n_total=counts["total"],
                n_surviving=0, rand=rand, syst=syst,
                incorrectness_factor=1.0,  # neutral (removed)
            ))
        else:
            factor = syst + rand ** counts["surviving"]
            p_incorrect_gated *= factor
            breakdowns.append(SourceBreakdown(
                source=src, n_total=counts["total"],
                n_surviving=counts["surviving"],
                rand=rand, syst=syst,
                incorrectness_factor=factor,
            ))

    belief = max(0.0, min(1.0, 1.0 - p_incorrect_gated))

    return GatedBeliefResult(
        belief=belief,
        parametric_only=parametric_only,
        n_total_evidence=n_total,
        n_surviving_evidence=n_surviving,
        n_gated=n_total - n_surviving,
        per_source=breakdowns,
    )

"""U-phase U4: INDRA KG signal as confidence modifier.

Queries INDRA's curated benchmark corpus (894K statements) for whether
a (claim_subject, claim_object, claim_axis) triple has curator support.
Used as a CONFIDENCE MODIFIER ONLY — never as a verdict override.

Q-phase failure mode reminder (project_q_phase_outcome): using KG as a
verdict source produced a -2.65pp regression because curator coverage
gaps were treated as "claim is wrong" rather than "we don't know". The
U-phase contract is strict: KG presence boosts confidence; KG absence
is a no-op (silent, neutral).

Public API:
    get_signal(subj: str, obj: str, claim_axis: str) -> dict | None
    preload()  — eagerly build the pair-index (call from holdout runners)

Lazy initialization: the pair-index is built on first call. Cost is
~30s and ~50MB memory (one-time). Subsequent calls are O(1) dict lookups.

For tests/contexts where the corpus shouldn't be loaded, set the env
var INDRA_BELIEF_DISABLE_KG=1 — get_signal returns None unconditionally.
"""
from __future__ import annotations

import gzip
import json
import logging
import os
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# File path: src/indra_belief/scorers/kg_signal.py
# parents: [0]=scorers, [1]=indra_belief, [2]=src, [3]=project root
_DEFAULT_CORPUS = (
    Path(__file__).resolve().parents[3]
    / "data" / "benchmark" / "indra_benchmark_corpus.json.gz"
)

# stmt_type → axis mapping. Mirrors the canonical map in commitments.py
# (parse_claim builds ClaimCommitment.axis from stmt_type via this same
# logic; replicating here keeps kg_signal a leaf module without imports
# into commitments).
_STMT_TO_AXIS = {
    'Activation': 'activity',
    'Inhibition': 'activity',
    'IncreaseAmount': 'amount',
    'DecreaseAmount': 'amount',
    'Phosphorylation': 'modification',
    'Dephosphorylation': 'modification',
    'Methylation': 'modification',
    'Demethylation': 'modification',
    'Acetylation': 'modification',
    'Deacetylation': 'modification',
    'Ubiquitination': 'modification',
    'Deubiquitination': 'modification',
    'Sumoylation': 'modification',
    'Desumoylation': 'modification',
    'Hydroxylation': 'modification',
    'Ribosylation': 'modification',
    'Glycosylation': 'modification',
    'Farnesylation': 'modification',
    'Palmitoylation': 'modification',
    'Myristoylation': 'modification',
    'Geranylgeranylation': 'modification',
    'Autophosphorylation': 'modification',
    'Transphosphorylation': 'modification',
    'Complex': 'binding',
    'Translocation': 'localization',
    'Conversion': 'conversion',
    'GtpActivation': 'gtp_state',
    'Gef': 'gtp_state',
    'Gap': 'gtp_state',
}

# Module-level cached pair-index. None = not yet loaded; {} = loaded but empty
# (empty corpus or env-disabled).
_PAIR_INDEX: Optional[dict] = None
_LOAD_FAILED: bool = False


def _disabled() -> bool:
    return bool(os.environ.get("INDRA_BELIEF_DISABLE_KG"))


def _ensure_loaded(corpus_path: Path = _DEFAULT_CORPUS) -> None:
    """Build the pair-index lazily. Idempotent; safe to call repeatedly."""
    global _PAIR_INDEX, _LOAD_FAILED
    if _PAIR_INDEX is not None:
        return
    if _LOAD_FAILED:
        return
    if _disabled():
        log.info("kg_signal: disabled via INDRA_BELIEF_DISABLE_KG")
        _PAIR_INDEX = {}
        return
    if not corpus_path.exists():
        log.warning("kg_signal: corpus not found at %s; KG disabled", corpus_path)
        _PAIR_INDEX = {}
        _LOAD_FAILED = True
        return

    log.info("kg_signal: loading corpus from %s", corpus_path)
    try:
        with gzip.open(corpus_path, "rt", encoding="utf-8") as fh:
            corpus = json.load(fh)
    except Exception as e:
        log.error("kg_signal: failed to load corpus: %s", e)
        _PAIR_INDEX = {}
        _LOAD_FAILED = True
        return

    pair_index: dict = {}
    skipped = 0
    for stmt_json in corpus:
        stmt_type = stmt_json.get("type")
        if not stmt_type:
            continue
        agents = []
        for k in ("subj", "obj", "enz", "sub", "agent"):
            if stmt_json.get(k):
                agents.append(stmt_json[k])
        if "members" in stmt_json and isinstance(stmt_json["members"], list):
            agents.extend(stmt_json["members"])

        names = [a.get("name", "").lower() for a in agents
                 if isinstance(a, dict) and a.get("name")]
        if len(names) < 2:
            skipped += 1
            continue
        # Pair: first two unique names, sorted (symmetric).
        seen = []
        for n in names:
            if n not in seen:
                seen.append(n)
            if len(seen) >= 2:
                break
        if len(seen) < 2:
            continue
        pair = tuple(sorted(seen[:2]))
        belief = float(stmt_json.get("belief", 0.0) or 0.0)
        n_ev = len(stmt_json.get("evidence", []))
        pair_index.setdefault(pair, []).append({
            "stmt_type": stmt_type,
            "belief": belief,
            "n_evidence": n_ev,
        })

    _PAIR_INDEX = pair_index
    log.info(
        "kg_signal: indexed %d pairs from %d statements (skipped %d)",
        len(pair_index), len(corpus), skipped,
    )


def preload(corpus_path: Path = _DEFAULT_CORPUS) -> None:
    """Eager preload. Call from holdout runners to amortize the cost
    out of per-record latency."""
    _ensure_loaded(corpus_path)


def get_signal(subj: str, obj: str, claim_axis: str) -> Optional[dict]:
    """Return KG signal for the (subj, obj, claim_axis) triple, or None.

    Returns:
      None — corpus not loaded, no curated triples for this pair, or KG
        disabled. Treated as no-signal (no confidence modification).
      {"kind": "same_axis", "max_belief": float, "count": int,
       "total_evidence": int}
        — at least one curated statement on the claim's axis.
        Adjudicator boosts confidence one tier on a correct verdict
        when max_belief >= 0.5.
      {"kind": "diff_axis", "count": int}
        — curated statements exist between these entities but on a
        different axis. Informational; no verdict modification (the LLM
        probe should already pick this up via direct_axis_mismatch).

    Never used to override a verdict. This is a STRICT contract — see
    Q-phase failure mode in project_q_phase_outcome memory.
    """
    if not subj or not obj:
        return None
    _ensure_loaded()
    if not _PAIR_INDEX:
        return None
    pair = tuple(sorted([subj.lower().strip(), obj.lower().strip()]))
    matches = _PAIR_INDEX.get(pair)
    if not matches:
        return None
    same_axis = [m for m in matches
                 if _STMT_TO_AXIS.get(m["stmt_type"]) == claim_axis]
    if same_axis:
        return {
            "kind": "same_axis",
            "max_belief": max(m["belief"] for m in same_axis),
            "count": len(same_axis),
            "total_evidence": sum(m["n_evidence"] for m in same_axis),
        }
    return {
        "kind": "diff_axis",
        "count": len(matches),
    }


def reset() -> None:
    """Reset module state (for tests). Drops the cached index."""
    global _PAIR_INDEX, _LOAD_FAILED
    _PAIR_INDEX = None
    _LOAD_FAILED = False

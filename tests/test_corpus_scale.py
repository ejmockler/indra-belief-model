"""Scale test: synthetic 8.7K-stmt corpus exercising the full pipeline.

Skip-by-default: set `INDRA_BELIEF_SCALE_TEST=1` to run. Adds ~10-30s
depending on machine. Validates that the pipeline does NOT degrade at
rasmachine-equivalent scale even without the live JSON.

Skipping rationale: keeps `pytest -q` under a few seconds for the dev
loop. Surface the scale-test on demand or in nightly CI.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import duckdb
import pytest
from indra.statements import (
    Activation,
    Agent,
    Evidence,
    Inhibition,
    Phosphorylation,
)

from indra_belief.corpus import (
    apply_schema,
    estimate_cost,
    export_beliefs,
    ingest_statements,
    score_corpus,
)


pytestmark = pytest.mark.skipif(
    os.environ.get("INDRA_BELIEF_SCALE_TEST") != "1",
    reason="scale test (set INDRA_BELIEF_SCALE_TEST=1 to run)",
)


def _generate_synthetic_corpus(n_stmts: int = 8724):
    """Generate a synthetic INDRA Statement list resembling rasmachine.

    Distribution roughly mirrors rasmachine: mostly Phosphorylation +
    Activation/Inhibition, varied source_api, varied epistemics.
    """
    sources = ["reach", "biopax", "biogrid", "signor", "trips"]
    agents = [
        Agent(f"PROT{i}", db_refs={"HGNC": str(1000 + i)}) for i in range(50)
    ]

    stmts = []
    for i in range(n_stmts):
        a = agents[i % 50]
        b = agents[(i + 1) % 50]
        n_ev = 1 + (i % 4)  # 1, 2, 3, or 4 evidences
        evs = []
        for j in range(n_ev):
            src = sources[j % len(sources)]
            epi = {"direct": (i + j) % 3 != 0, "curated": (i + j) % 5 == 0}
            evs.append(Evidence(source_api=src, pmid=str(100000 + i * 4 + j),
                                text=f"PROT{i % 50} → PROT{(i + 1) % 50} stmt {i} ev {j}",
                                epistemics=epi))
        cls = (Phosphorylation, Activation, Inhibition)[i % 3]
        if cls is Phosphorylation:
            stmts.append(cls(a, b, residue="T", position=str(200 + i % 100), evidence=evs))
        else:
            stmts.append(cls(a, b, evidence=evs))
    return stmts


def _mock_scorer(statement, evidence, client):
    """Deterministic mock; varies score by epistemics."""
    epi = getattr(evidence, "epistemics", {}) or {}
    if epi.get("curated"):
        return {"score": 0.92, "verdict": "correct", "confidence": "high",
                "reasons": ["match", "curated"], "call_log": []}
    if epi.get("direct"):
        return {"score": 0.75, "verdict": "correct", "confidence": "medium",
                "reasons": ["match"], "call_log": []}
    return {"score": 0.50, "verdict": "abstain", "confidence": "low",
            "reasons": ["hedged"], "call_log": []}


def test_scale_pipeline_holds_at_8724_stmts(tmp_path: Path):
    """Full pipeline against rasmachine-scale synthetic corpus."""
    n_stmts = 8724
    stmts = _generate_synthetic_corpus(n_stmts)

    con = duckdb.connect(":memory:")
    apply_schema(con)

    # 1. Ingest
    t0 = time.perf_counter()
    counters = ingest_statements(con, stmts, source_dump_id="scale_synthetic")
    t_ingest = time.perf_counter() - t0
    assert counters["n_statements"] == n_stmts
    assert counters["n_evidences"] >= n_stmts  # at least 1 per stmt

    # 2. Cost estimate (no actual spend; just a budget projection)
    est = estimate_cost(stmts, model_id="claude-sonnet-4-6")
    assert est["n_stmts"] == n_stmts
    # Sonnet projection should be in the $200-350 range per task graph 0.1
    assert 200 < est["cost_usd"] < 400

    # 3. Score (mock; no LLM cost)
    t0 = time.perf_counter()
    run_id = score_corpus(
        con, stmts,
        scorer_version="scale-test",
        score_evidence=_mock_scorer,
        decompose=True,
        with_validity=True,
    )
    t_score = time.perf_counter() - t0
    assert run_id

    # 4. Verify rows persisted
    n_aggregate = con.execute(
        "SELECT COUNT(*) FROM scorer_step WHERE step_kind = 'aggregate'"
    ).fetchone()[0]
    assert n_aggregate == counters["n_evidences"]

    n_decomposed = con.execute(
        "SELECT COUNT(*) FROM scorer_step WHERE step_kind != 'aggregate'"
    ).fetchone()[0]
    # parse_claim + build_context + substrate_route per evidence + per-probe
    # rows when substrate resolves; at minimum 3× n_evidences
    assert n_decomposed >= 3 * counters["n_evidences"]

    # 5. Validity metrics computed
    n_metrics = con.execute(
        "SELECT COUNT(*) FROM metric WHERE run_id = ?", [run_id]
    ).fetchone()[0]
    assert n_metrics > 0

    # 6. Export
    t0 = time.perf_counter()
    out_path = tmp_path / "scale_export.json"
    export_beliefs(con, run_id, out_path)
    t_export = time.perf_counter() - t0
    assert out_path.exists()

    # Print timing for visibility (pytest -s reveals it)
    print(f"\n  ingest:  {t_ingest:.2f}s  ({n_stmts:,} stmts → "
          f"{counters['n_evidences']:,} evidences)")
    print(f"  score:   {t_score:.2f}s  ({n_aggregate:,} aggregate + "
          f"{n_decomposed:,} decomposed steps)")
    print(f"  export:  {t_export:.2f}s")

    # Soft performance assertions: catch regressions if something blows up.
    # Tighten/loosen if your machine differs significantly.
    assert t_ingest < 60.0, f"ingest too slow: {t_ingest:.1f}s"
    assert t_score < 120.0, f"score too slow: {t_score:.1f}s"
    assert t_export < 30.0, f"export too slow: {t_export:.1f}s"

    con.close()

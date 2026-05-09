"""V6c labeling-pipeline tests.

Verifies:
  1. Holdout-exclusion logic catches source_hash, matches_hash, and
     (pmid, entity_pair) overlaps and lets clean records through.
  2. `build_label_matrices` produces per-probe Λ matrices with the
     expected (n_records × n_LFs_for_probe) shape.
  3. Snorkel LabelModel fits cleanly on a 100-record fixture corpus and
     `predict_proba` is finite.
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pytest

# INDRA emits DeprecationWarnings on import.
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from indra_belief.v_phase.labeling import (
    _build_probe_lf_index,
    _make_ev_dict,
    _make_stmt_dict,
    _statement_subj_obj,
    _stream_corpus_records,
    build_label_matrices,
    diagnose_lf_correlations,
    fit_label_model,
    is_holdout_excluded,
    iter_pairs,
    load_holdout_exclusion,
    per_class_vote_distribution,
    per_lf_firing_rates,
    predict_proba_safe,
    write_parquet,
    ABSTAIN,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _holdout_jsonl(tmp_path: Path) -> Path:
    """Build a tiny holdout JSONL with one of each exclusion-key kind."""
    fpath = tmp_path / "holdout.jsonl"
    rows = [
        {"source_hash": 1234, "matches_hash": "0000", "pmid": "PMA",
         "subject": "TP53", "object": "MDM2"},
        {"source_hash": 9999, "matches_hash": "MH-A", "pmid": "PMA",
         "subject": "TP53", "object": "MDM2"},
        {"source_hash": 5555, "matches_hash": "MH-B", "pmid": "PMB",
         "subject": "AKT1", "object": "FOXO3"},
    ]
    with fpath.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return fpath


def _sample_records(n: int = 100) -> list[dict]:
    """Synthesize n INDRA-shape statement dicts with simple `evidence` lists.

    These records carry no holdout markers (source_hash=0, matches_hash="X").
    """
    out = []
    types = ["Phosphorylation", "Activation", "Inhibition",
             "IncreaseAmount", "Complex"]
    for i in range(n):
        t = types[i % len(types)]
        if t == "Complex":
            rec = {
                "type": "Complex",
                "members": [
                    {"name": f"GENE{i}A"},
                    {"name": f"GENE{i}B"},
                ],
                "matches_hash": f"MH-{i}",
                "evidence": [{
                    "source_api": "reach",
                    "pmid": str(20000000 + i),
                    "text": f"GENE{i}A binds GENE{i}B in cells.",
                    "annotations": {"found_by": "Binding_syntax_1_noun"},
                    "epistemics": {"direct": True},
                    "source_hash": 100000 + i,
                }],
            }
        else:
            rec = {
                "type": t,
                "enz" if t in ("Phosphorylation", "Activation",
                                "Inhibition") else "subj":
                    {"name": f"GENE{i}A"},
                "sub" if t in ("Phosphorylation",) else "obj":
                    {"name": f"GENE{i}B"},
                "matches_hash": f"MH-{i}",
                "evidence": [{
                    "source_api": "reach",
                    "pmid": str(20000000 + i),
                    "text": (
                        f"GENE{i}A phosphorylates GENE{i}B at S123 in cells "
                        f"and may regulate downstream signaling pathway."
                    ),
                    "annotations": {"found_by": "Phosphorylation_syntax_1_noun"},
                    "epistemics": {"direct": True},
                    "source_hash": 100000 + i,
                }],
            }
        out.append(rec)
    return out


def _corpus_gz(tmp_path: Path, records: list[dict]) -> Path:
    """Write a gzipped JSON list of `records` for `_stream_corpus_records`."""
    out = tmp_path / "corpus.json.gz"
    with gzip.open(out, "wt", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_holdout_exclusion_catches_three_keys(tmp_path: Path):
    """All three V5r §5 keys (source_hash, matches_hash, pmid+entity_pair)
    are detected; clean records are not flagged."""
    hpath = _holdout_jsonl(tmp_path)
    exc = load_holdout_exclusion(hpath)
    assert 1234 in exc["source_hashes"]
    assert "MH-A" in exc["matches_hashes"]
    assert ("PMA", frozenset({"TP53", "MDM2"})) in exc["pmid_pairs"]

    # Match by source_hash:
    s1 = {"matches_hash": "OTHER", "subject": "X", "object": "Y", "pmid": "Q"}
    e1 = {"source_hash": 1234, "pmid": "Q"}
    assert is_holdout_excluded(s1, e1, exc) is True

    # Match by matches_hash:
    s2 = {"matches_hash": "MH-A", "subject": "X", "object": "Y", "pmid": "Q"}
    e2 = {"source_hash": 1, "pmid": "Q"}
    assert is_holdout_excluded(s2, e2, exc) is True

    # Match by (pmid, entity_pair):
    s3 = {"matches_hash": "OTHER", "subject": "TP53", "object": "MDM2",
           "pmid": "PMA"}
    e3 = {"source_hash": 7, "pmid": "PMA"}
    assert is_holdout_excluded(s3, e3, exc) is True

    # Match (pmid, entity_pair) order-independent:
    s3b = {"matches_hash": "OTHER", "subject": "MDM2", "object": "TP53",
            "pmid": "PMA"}
    e3b = {"source_hash": 7, "pmid": "PMA"}
    assert is_holdout_excluded(s3b, e3b, exc) is True

    # Clean: PMID-only overlap (pmid yes, but entity_pair NOT in holdout):
    s4 = {"matches_hash": "OTHER", "subject": "OTHER1", "object": "OTHER2",
           "pmid": "PMA"}
    e4 = {"source_hash": 7, "pmid": "PMA"}
    assert is_holdout_excluded(s4, e4, exc) is False

    # Clean: nothing overlaps.
    s5 = {"matches_hash": "OTHER", "subject": "X", "object": "Y", "pmid": "Q"}
    e5 = {"source_hash": 7, "pmid": "Q"}
    assert is_holdout_excluded(s5, e5, exc) is False


def test_corpus_walker_yields_pairs(tmp_path: Path):
    """`_stream_corpus_records` round-trips a 5-record gzipped JSON list."""
    recs = _sample_records(5)
    cpath = _corpus_gz(tmp_path, recs)
    parsed = list(_stream_corpus_records(cpath))
    assert len(parsed) == 5
    assert parsed[0]["type"] == "Phosphorylation"
    assert parsed[1]["type"] == "Activation"


def test_iter_pairs_excludes_holdout(tmp_path: Path):
    """`iter_pairs` drops records whose (pmid, entity_pair) is in holdout."""
    # Build holdout that captures the third record's entity pair.
    rec3_subj = "GENE2A"
    rec3_obj = "GENE2B"
    rec3_pmid = "20000002"
    holdout_rows = [{
        "source_hash": -1,
        "matches_hash": "X",
        "pmid": rec3_pmid,
        "subject": rec3_subj,
        "object": rec3_obj,
    }]
    hpath = tmp_path / "h.jsonl"
    with hpath.open("w") as f:
        for r in holdout_rows:
            f.write(json.dumps(r) + "\n")

    recs = _sample_records(5)
    cpath = _corpus_gz(tmp_path, recs)
    exc = load_holdout_exclusion(hpath)
    pairs = list(iter_pairs(corpus_path=cpath, exclusion=exc))
    # We seeded 5 records, each 1 evidence; record index 2 is the
    # Inhibition GENE2A/GENE2B at pmid 20000002.
    assert len(pairs) == 4
    # The dropped pair would be pmid 20000002.
    pmids_kept = [p[2].get("pmid") for p in pairs]
    assert "20000002" not in pmids_kept


def test_build_label_matrices_shape(tmp_path: Path):
    """For 100 records, Λ shape per probe matches:
        - per-stmt-ev probes: (n_records, n_LFs_for_probe)
        - verify_grounding: (n_records × avg_agents, n_LFs_for_probe)
    """
    recs = _sample_records(100)
    cpath = _corpus_gz(tmp_path, recs)
    pairs = list(iter_pairs(corpus_path=cpath, exclusion={"source_hashes": set(),
                                                              "matches_hashes": set(),
                                                              "pmid_pairs": set()}))
    pair_inputs = [(s, e, a) for _r, s, e, a in pairs]
    matrices = build_label_matrices(pair_inputs)

    # Probe LF count is fixed; assert against the LF index.
    probe_lf_index = _build_probe_lf_index()
    for probe in ("relation_axis", "subject_role", "object_role", "scope"):
        n_lfs = len(probe_lf_index[probe])
        L = matrices[probe]["L"]
        assert L.shape == (100, n_lfs), (
            f"{probe} expected (100, {n_lfs}); got {L.shape}"
        )
        assert L.dtype == np.int8

    vg = matrices["verify_grounding"]
    n_lfs_vg = len(probe_lf_index["verify_grounding"])
    # Each fixture record has 2 named agents → 200 verify_grounding rows.
    assert vg["L"].shape[1] == n_lfs_vg
    assert vg["L"].shape[0] >= 100  # at least 100; 200 if all records had 2 agents


def test_label_model_fits_clean(tmp_path: Path):
    """`fit_label_model` on a 100-record fixture returns finite predict_proba.
    Probe = scope (5 classes); LFs include `lf_text_too_short` and
    `lf_low_information_evidence` which fire predictably on short text.
    """
    recs = _sample_records(100)
    # Add 10 records with short evidence to ensure abstain class fires.
    for i in range(10):
        recs[i]["evidence"][0]["text"] = "short"  # < 6 tokens
    cpath = _corpus_gz(tmp_path, recs)
    pairs = list(iter_pairs(corpus_path=cpath,
                              exclusion={"source_hashes": set(),
                                          "matches_hashes": set(),
                                          "pmid_pairs": set()}))
    pair_inputs = [(s, e, a) for _r, s, e, a in pairs]
    matrices = build_label_matrices(pair_inputs)
    info = matrices["scope"]
    L = info["L"]
    K = info["n_classes"]
    # At least one non-ABSTAIN vote per class isn't guaranteed on a
    # synthetic corpus, so we can't insist on full identifiability — but we
    # CAN insist that LabelModel.fit doesn't crash and predict_proba is finite.
    lm = fit_label_model(L, K, verbose=False, seed=0)
    P = predict_proba_safe(lm, L)
    assert np.isfinite(P).all()
    assert P.shape == (L.shape[0], K)
    # Probabilities sum to ~1 per row.
    row_sums = P.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-3)


def test_correlation_diagnosis_returns_pairs():
    """Construct a synthetic Λ where two LFs perfectly co-fire and verify
    `diagnose_lf_correlations` flags the pair."""
    n = 100
    # 3 LFs: col 0 + col 1 perfectly correlated; col 2 random.
    rng = np.random.default_rng(42)
    base = rng.integers(0, 3, size=n)
    col0 = base.copy().astype(np.int8)
    col1 = base.copy().astype(np.int8)
    col2 = rng.integers(0, 3, size=n).astype(np.int8)
    # Sprinkle a few ABSTAINs.
    abstain_mask = rng.random(n) < 0.1
    col0[abstain_mask] = ABSTAIN
    col1[abstain_mask] = ABSTAIN
    L = np.stack([col0, col1, col2], axis=1)
    flagged = diagnose_lf_correlations(L, ["lf_a", "lf_b", "lf_c"],
                                        min_overlap=20, threshold=0.5)
    names = {(a, b) for a, b, _, _ in flagged}
    assert ("lf_a", "lf_b") in names
    # lf_c should NOT correlate strongly with either.
    for a, b, _n, r in flagged:
        if "lf_c" in (a, b):
            pytest.fail(f"unexpected flag: {a}<>{b} r={r}")


def test_per_lf_firing_and_class_distribution():
    """Sanity: per_lf_firing_rates and per_class_vote_distribution count
    correctly on a tiny fixture."""
    L = np.array([
        [-1, 0, 1],
        [0, 0, -1],
        [-1, -1, -1],
    ], dtype=np.int8)
    rates = per_lf_firing_rates(L, ["a", "b", "c"])
    assert rates == [("a", 1, 1/3), ("b", 2, 2/3), ("c", 1, 1/3)]

    counts = per_class_vote_distribution(L, n_classes=3)
    # class 0: appears at L[0,1], L[1,0], L[1,1] = 3 times
    # class 1: appears at L[0,2] = 1 time
    # class 2: 0 times
    assert counts == [3, 1, 0]


def test_write_parquet_roundtrip(tmp_path: Path):
    """`write_parquet` round-trips through pandas+pyarrow and produces the
    expected columns."""
    import pandas as pd
    record_ids = ["a", "b", "c"]
    P = np.array([
        [0.7, 0.2, 0.1],
        [0.1, 0.4, 0.5],
        [0.3, 0.3, 0.4],
    ], dtype=np.float32)
    out_path = write_parquet("test_probe", "_t", record_ids, P,
                              ["x", "y", "z"], out_dir=tmp_path)
    df = pd.read_parquet(out_path)
    assert list(df.columns) == [
        "record_id",
        "class_proba_x", "class_proba_y", "class_proba_z",
        "argmax_class", "max_proba", "kept_for_training",
    ]
    assert df["record_id"].tolist() == ["a", "b", "c"]
    assert df["argmax_class"].tolist() == ["x", "z", "z"]
    # max_proba ≥ 0.5 only for row 0 (0.7) and row 1 (0.5)
    assert df["kept_for_training"].tolist() == [True, True, False]

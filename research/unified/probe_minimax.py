"""Probe minimax-local with the actual unified-call pipeline on a few
records. Measures end-to-end TPS and validates JSON-schema handling.

Usage:
  PYTHONPATH=src python scripts/probe_minimax.py
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from indra_belief.data.corpus import CorpusIndex
from indra_belief.model_client import ModelClient
from indra_belief.scorers.unified import score_unified


def main() -> None:
    logging.basicConfig(level=logging.WARNING)

    index = CorpusIndex()
    records = index.build_records(
        str(ROOT / "data/benchmark/calibration_dec_fn.jsonl"))

    # Build pattern → first-record map
    pat_for: dict[int, str] = {}
    with open(ROOT / "data/benchmark/calibration_dec_fn.jsonl") as f:
        for line in f:
            d = json.loads(line)
            pat_for[d["source_hash"]] = d["pattern"]

    targets = []
    seen = set()
    for r in records:
        pat = pat_for.get(r.source_hash, "?")
        if pat in ("nominalized", "indirect_chain", "negative_test_should_pass") \
                and pat not in seen:
            seen.add(pat)
            targets.append((r, pat))
        if len(targets) >= 3:
            break

    print(f"Probing minimax-local with {len(targets)} records.")
    client = ModelClient("minimax-local")

    for rec, pat in targets:
        print(f"\n=== {pat}: {rec.subject} [{rec.stmt_type}] {rec.object} ===")
        t0 = time.time()
        try:
            result = score_unified(rec.statement, rec.evidence, client)
            elapsed = time.time() - t0
            tps = result["tokens"] / elapsed if elapsed > 0 else 0
            print(f"  verdict={result['verdict']}  conf={result['confidence']}  "
                  f"score={result['score']}  tokens={result['tokens']}  "
                  f"elapsed={elapsed:.1f}s  TPS={tps:.1f}")
            if result.get("raw_text"):
                j = json.loads(result["raw_text"])
                print(f"  axis={j['axis_check']}  sign={j['sign_check']}  "
                      f"stance={j['stance_check']}  indirect={j['indirect_chain_present']}")
                print(f"  reason={j['reason_code']}  secondary={j['secondary_reasons']}")
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {str(e)[:200]}")


if __name__ == "__main__":
    main()

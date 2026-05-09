#!/usr/bin/env python3
"""T7: stratified 50-record probe for T-phase fixes A+B+C.

Runs the 50 stratified records (built from data/benchmark/probe_t_phase_stratified.jsonl)
through the T-phase scorer, then computes per-fix-category accounting:
  - Fix A targets: did relation_axis="abstain" project as expected?
  - Fix B targets: did Phosphorylation+via_mediator flip to correct/low?
  - Fix C targets: did sign/axis/contradicted detectors stop overshooting?
  - Controls: TP/TN preservation.

Usage:
  scripts/run_t_phase_probe.py [--workers auto]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from indra_belief.data.corpus import CorpusIndex
from indra_belief.model_client import ModelClient, concurrency_hint
from indra_belief.scorers.scorer import score_evidence


PROBE_INPUT = ROOT / "data" / "benchmark" / "probe_t_phase_stratified.jsonl"
OUTPUT = ROOT / "data" / "results" / "dec_t_phase_probe50.jsonl"


def _score_one(rec, category, sphase_actual, sphase_reasons, client) -> dict:
    target = "correct" if rec.tag == "correct" else "incorrect"
    try:
        result = score_evidence(rec.statement, rec.evidence, client)
    except Exception as e:
        return {
            "source_hash": rec.source_hash,
            "subject": rec.subject, "stmt_type": rec.stmt_type,
            "object": rec.object, "tag": rec.tag, "target": target,
            "actual": None, "confidence": None, "reasons": [],
            "tier": "error", "tokens": 0,
            "tphase_category": category,
            "sphase_actual": sphase_actual,
            "sphase_reasons": sphase_reasons,
            "error": f"{type(e).__name__}: {e}",
        }
    n_llm = sum(1 for c in (result.get("call_log") or [])
                if c.get("kind", "").startswith("probe_"))
    return {
        "source_hash": rec.source_hash,
        "subject": rec.subject, "stmt_type": rec.stmt_type,
        "object": rec.object, "tag": rec.tag, "target": target,
        "actual": result.get("verdict"),
        "confidence": result.get("confidence"),
        "reasons": result.get("reasons", []),
        "tier": result.get("tier"),
        "tokens": result.get("tokens", 0),
        "call_log": result.get("call_log", []),
        "n_probe_llm_calls": n_llm,
        "tphase_category": category,
        "sphase_actual": sphase_actual,
        "sphase_reasons": sphase_reasons,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gemma-remote")
    p.add_argument("--input", default=str(PROBE_INPUT))
    p.add_argument("--output", default=str(OUTPUT))
    p.add_argument("--workers", default="auto")
    args = p.parse_args()

    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("t_probe")

    workers = (concurrency_hint(args.model) if args.workers == "auto"
               else max(1, int(args.workers)))

    # Load probe records — preserve sphase fields for diff reporting.
    raw_records = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                raw_records.append(json.loads(line))
    log.warning("loaded %d records (stratified probe)", len(raw_records))

    # Re-build via CorpusIndex to get Statement/Evidence objects.
    index = CorpusIndex()
    full_records = index.build_records(args.input)
    by_hash = {r.source_hash: r for r in full_records}

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_f = open(out_path, "w", buffering=1)
    client = ModelClient(args.model)

    pending = []
    for raw in raw_records:
        rec = by_hash.get(raw["source_hash"])
        if rec is None:
            log.warning("skipping: source_hash %s not in CorpusIndex",
                        raw["source_hash"])
            continue
        pending.append((
            rec, raw.get("tphase_category", "unknown"),
            raw.get("sphase_actual"), raw.get("sphase_reasons"),
        ))

    t0 = time.time()
    n_done = 0
    print(f"Scoring {len(pending)} records (workers={workers})",
          file=sys.stderr)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_score_one, *args_, client): args_
                   for args_ in pending}
        for fut in as_completed(futures):
            row = fut.result()
            out_f.write(json.dumps(row) + "\n")
            out_f.flush()
            n_done += 1
            if n_done % 5 == 0 or n_done == len(pending):
                elapsed = time.time() - t0
                rate = n_done / elapsed if elapsed > 0 else 0
                eta = (len(pending) - n_done) / rate if rate > 0 else 0
                print(f"[{n_done:>3}/{len(pending)}] "
                      f"rate={rate:.2f}/s  ETA={eta/60:.1f}min",
                      file=sys.stderr)

    out_f.close()
    elapsed = time.time() - t0
    print(f"\nDONE: {n_done} records in {elapsed/60:.1f}min",
          file=sys.stderr)


if __name__ == "__main__":
    main()

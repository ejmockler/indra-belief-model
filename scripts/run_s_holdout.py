"""S10 holdout: full 482-record run on the S-phase pipeline.

Adapted from the R-phase run_n9_holdout.py concurrency pattern, threaded
via ThreadPoolExecutor sized by `concurrency_hint(model)` to amortize
the slow gemma-remote endpoint.

Output JSONL is comparable to dec_e3_v18_mphase.jsonl / dec_q_phase.jsonl
/ dec_r_phase.jsonl so S11 can do the four-way comparison.

Usage:
  scripts/run_s_holdout.py [--limit N] [--workers N]
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


HOLDOUT = ROOT / "data" / "benchmark" / "holdout_v15_sample.jsonl"
OUTPUT = ROOT / "data" / "results" / "dec_s_phase.jsonl"


def _score_one(rec, client) -> dict:
    """Score one record via the S-phase pipeline. Returns the
    flattened-row dict in the same shape m13_analyze.py / s11_ship
    expect."""
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
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gemma-remote")
    p.add_argument("--input", default=str(HOLDOUT))
    p.add_argument("--output", default=str(OUTPUT))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", default="auto",
                   help="auto=concurrency_hint(model), or integer")
    p.add_argument("--resume", action="store_true", default=True,
                   help="Skip records already in --output (default on)")
    args = p.parse_args()

    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("s_holdout")

    workers = (concurrency_hint(args.model) if args.workers == "auto"
               else max(1, int(args.workers)))

    index = CorpusIndex()
    records = index.build_records(args.input)
    if args.limit:
        records = records[: args.limit]
    log.warning("loaded %d records, workers=%d", len(records), workers)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    already: set[int] = set()
    if args.resume and out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    already.add(json.loads(line)["source_hash"])
                except (json.JSONDecodeError, KeyError):
                    pass
        log.warning("resume: %d records already scored", len(already))

    pending = [r for r in records if r.source_hash not in already]
    if not pending:
        log.warning("nothing to do — all %d records already scored",
                    len(records))
        return

    out_f = open(out_path, "a", buffering=1)
    client = ModelClient(args.model)

    t0 = time.time()
    n_done = 0
    n_correct = 0
    n_total = 0
    print(f"Scoring {len(pending)} records (workers={workers}, "
          f"endpoint={args.model})", file=sys.stderr)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_score_one, r, client): r for r in pending}
        for fut in as_completed(futures):
            row = fut.result()
            out_f.write(json.dumps(row) + "\n")
            out_f.flush()
            n_done += 1
            if row["actual"] in ("correct", "incorrect"):
                n_total += 1
                if row["actual"] == row["target"]:
                    n_correct += 1
            if n_done % 10 == 0 or n_done == len(pending):
                elapsed = time.time() - t0
                rate = n_done / elapsed if elapsed > 0 else 0
                acc = n_correct / n_total if n_total else 0
                eta = (len(pending) - n_done) / rate if rate > 0 else 0
                print(f"[{n_done:>4}/{len(pending)}] "
                      f"acc={acc:.4f} ({n_correct}/{n_total})  "
                      f"rate={rate:.2f}/s  ETA={eta/60:.1f}min",
                      file=sys.stderr)

    out_f.close()
    elapsed = time.time() - t0
    print(f"\nDONE: {n_done} records in {elapsed/60:.1f}min "
          f"(parsed {n_total}, "
          f"acc={n_correct / n_total if n_total else 0:.4f})",
          file=sys.stderr)


if __name__ == "__main__":
    main()

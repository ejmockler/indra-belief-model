"""Dual-run harness — score a stratified small sample via both the
monolithic and decomposed paths, then report per-path accuracy, agreement,
and crash-freeness against the pre-registered criteria below.

Purpose (task #25): catch integration bugs the unit tests missed, observe
where decomposed disagrees with monolithic, confirm nothing crashes at
real-INDRA scale before committing to the full-holdout run (#27).

=============================================================================
PRE-REGISTERED CRITERIA (locked before seeing any results)
=============================================================================

Small-n sanity gates (this script) — must ALL pass before proceeding to #27:
  (S1) Decomposed must return a non-None verdict for every record.
       Crash-to-abstain is fine; crash-to-exception is not.
  (S2) Decomposed accuracy within 10pp of monolithic on the sample. A
       larger gap means a wiring bug, not a real accuracy tradeoff; the
       ceiling analysis already showed the sub-calls are individually
       competent.
  (S3) Agreement on GOLD-correct records ≥ 80%. Decomposition must not
       arbitrarily flip correct → abstain on clean cases.

Full-holdout criteria (#27) — locked here so #27 cannot move them:
  (F1) Primary ship criterion: decomposed accuracy ≥ monolithic baseline
       (≥ 0.8394 on n=501 holdout_v15_sample).
       Rationale: decomposed's win is INTERPRETABILITY (structured reason
       codes, attributable sub-call failures) — it does not need to beat
       monolithic on accuracy to justify shipping. Matching baseline at
       the same cost is a clear win.
  (F2) Aspirational criterion (non-blocking): decomposed ≥ 0.86.
       Would validate the +5.2pp ceiling analysis prediction.
  (F3) No sub-call regression: parse_evidence joint load-bearing
       accuracy on its v4_clean calibration ≥ 92% (retry-aware; prior
       measurement 96%). Drift below 92% halts release.
  (F4) Crash-freeness: zero uncaught exceptions across all 501 records.
       Every failure must be a typed abstain verdict, not a stack trace.
  (F5) Error-class improvements (directional, not hard gates):
       - Polarity/sign mismatch false-negatives should DROP relative to
         monolithic v16's ~31% on the same records.
       - Family-grounding false-positives should DROP relative to
         monolithic v16's ~42%.
       - Explicit negation false-negatives should DROP relative to v16's
         33–50% range.
       These are directional targets; the two-step inversion and typed
       grounding verifier were designed specifically to attack them.

=============================================================================

Usage:
    .venv/bin/python scripts/dual_run.py \\
        --model gemma-google-moe \\
        --sample-size 25 \\
        --holdout data/benchmark/holdout_v15_sample.jsonl \\
        --output data/results/dual_run_small.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from indra_belief.data.corpus import CorpusIndex                        # noqa: E402
from indra_belief.data.scoring_record import ScoringRecord              # noqa: E402
from indra_belief.model_client import ModelClient, concurrency_hint     # noqa: E402
from indra_belief.scorers.scorer import score_evidence                  # noqa: E402


# --- Pre-registered thresholds ---
SMALL_N_ACC_GAP_PP = 10.0          # S2
SMALL_N_CORRECT_AGREEMENT = 0.80   # S3
# (F*) thresholds live in the module docstring; they guide #27 tooling, not this script.


def _stratified_sample(records: list[ScoringRecord], n: int, seed: int) -> list[ScoringRecord]:
    """Pick `n` records balanced across (stmt_type, tag). Deterministic
    given `seed`. Buckets smaller than their target share shrink the
    effective sample — we never upsample with replacement."""
    if len(records) <= n:
        return list(records)

    buckets: dict[tuple[str, str], list[ScoringRecord]] = defaultdict(list)
    for r in records:
        key = (type(r.statement).__name__, (r.tag or "?").lower())
        buckets[key].append(r)

    rng = random.Random(seed)
    total = sum(len(v) for v in buckets.values())
    # Allocate per-bucket quotas by proportion, with a floor of 1 for every
    # non-empty bucket so rare types always appear in the small sample.
    quotas: dict[tuple[str, str], int] = {}
    for k, rs in buckets.items():
        share = max(1, round(n * len(rs) / total))
        quotas[k] = min(share, len(rs))
    # If rounding overshoots, trim from the largest buckets first.
    while sum(quotas.values()) > n:
        biggest = max(quotas, key=quotas.get)
        if quotas[biggest] <= 1:
            break
        quotas[biggest] -= 1

    picked: list[ScoringRecord] = []
    for k, rs in buckets.items():
        rng.shuffle(rs)
        picked.extend(rs[: quotas[k]])
    rng.shuffle(picked)
    return picked[:n]


def _score_both(client: ModelClient, record: ScoringRecord,
                max_tokens_mono: int, max_tokens_dec: int) -> dict:
    """Score one record via both paths and attribute any exception to the
    path that raised it. Neither path should raise — the contract is
    crash-to-abstain — but we defend against it so a single bad record
    doesn't abort the whole run."""
    mono_err = dec_err = None
    try:
        mono = score_evidence(record.statement, record.evidence, client,
                              max_tokens=max_tokens_mono,
                              use_decomposed=False)
    except Exception as e:
        mono_err = f"{type(e).__name__}: {e}"
        mono = None
    try:
        dec = score_evidence(record.statement, record.evidence, client,
                             use_decomposed=True,
                             max_tokens=max_tokens_dec)
    except Exception as e:
        dec_err = f"{type(e).__name__}: {e}"
        dec = None
    return {
        "monolithic": mono, "mono_error": mono_err,
        "decomposed": dec,  "dec_error": dec_err,
    }


def _verdict_ok(gold: str | None, verdict: str | None) -> bool | None:
    """Accuracy rule: verdict must be 'correct' iff gold is 'correct'.
    Abstain and None don't count either way (ambiguous prediction).
    Returns True (right), False (wrong), or None (no prediction)."""
    if verdict in (None, "abstain"):
        return None
    if gold is None:
        return None
    return (verdict.lower() == "correct") == (gold.lower() == "correct")


def _agent_name(record: ScoringRecord, i: int) -> str | None:
    """Safe indexed agent-name access. `agent_list()[i]` may be None (missing
    NLP extraction) or raise IndexError (monomer statement); both should
    land as None rather than crash the formatter."""
    agents = record.statement.agent_list()
    if i >= len(agents):
        return None
    a = agents[i]
    return a.name if a is not None else None


def _format_row(record: ScoringRecord, gold: str | None, result: dict) -> dict:
    mono = result["monolithic"] or {}
    dec = result["decomposed"] or {}
    return {
        "stmt_type": type(record.statement).__name__,
        "subject": _agent_name(record, 0) or "?",
        "object": _agent_name(record, 1),
        "evidence_text": record.evidence.text,
        "gold": gold,
        "monolithic": {
            "verdict": mono.get("verdict"),
            "confidence": mono.get("confidence"),
            "score": mono.get("score"),
            "tier": mono.get("tier"),
            "error": result["mono_error"],
        },
        "decomposed": {
            "verdict": dec.get("verdict"),
            "confidence": dec.get("confidence"),
            "score": dec.get("score"),
            "reasons": dec.get("reasons", []),
            "rationale": dec.get("rationale", ""),
            "grounding_status": dec.get("grounding_status"),
            "error": result["dec_error"],
        },
        "agreement": (
            (mono.get("verdict") or "").lower()
            == (dec.get("verdict") or "").lower()
            if mono.get("verdict") and dec.get("verdict") else False
        ),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gemma-google-moe")
    p.add_argument("--holdout",
                   default=str(ROOT / "data" / "benchmark"
                               / "holdout_v15_sample.jsonl"))
    p.add_argument("--output",
                   default=str(ROOT / "data" / "results" / "dual_run_small.jsonl"))
    p.add_argument("--sample-size", type=int, default=25)
    p.add_argument("--seed", type=int, default=1729,
                   help="Deterministic stratified sampling.")
    p.add_argument("--max-tokens-mono", type=int, default=12000)
    p.add_argument("--max-tokens-dec", type=int, default=6000)
    p.add_argument("--workers", default="auto")
    p.add_argument("--resume", action="store_true",
                   help="Skip records already present in --output. Matching "
                        "is by (stmt_type, evidence_text). Safe for the "
                        "stratified sample because sampling is deterministic "
                        "under --seed.")
    args = p.parse_args()

    workers = (concurrency_hint(args.model) if args.workers == "auto"
               else max(1, int(args.workers)))

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("dual_run")

    log.info("loading corpus + building records from %s", args.holdout)
    index = CorpusIndex()
    records = index.build_records(args.holdout)
    log.info("loaded %d records from holdout", len(records))

    sample = _stratified_sample(records, args.sample_size, args.seed)
    log.info("stratified sample: %d records (seed=%d, workers=%d)",
             len(sample), args.seed, workers)

    client = ModelClient(args.model)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Resume: load existing rows + build skip-set ---
    existing_rows: list[dict] = []
    skip_keys: set[tuple[str, str]] = set()
    if args.resume and out_path.exists():
        with open(out_path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                existing_rows.append(r)
                skip_keys.add((r["stmt_type"], r["evidence_text"]))
        log.info("resume: %d records already in %s — will skip",
                 len(existing_rows), out_path)

    pending = [r for r in sample
               if (type(r.statement).__name__, r.evidence.text) not in skip_keys]
    log.info("scoring %d new records (skipped %d via --resume)",
             len(pending), len(sample) - len(pending))

    rows: list[dict] = list(existing_rows)
    write_lock = threading.Lock()
    completed = 0
    file_mode = "a" if (args.resume and out_path.exists()) else "w"

    def _process(i_rec):
        i, rec = i_rec
        gold = rec.tag
        result = _score_both(client, rec,
                             args.max_tokens_mono, args.max_tokens_dec)
        return i, _format_row(rec, gold, result)

    with open(out_path, file_mode) as out_fh, \
            ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_process, (i, r)): i
                for i, r in enumerate(pending)}
        for fut in as_completed(futs):
            try:
                i, row = fut.result()
            except Exception as e:
                # Future raised despite _score_both's per-path defenses —
                # this is a formatter or test-harness bug, not a scoring
                # bug. Log and move on so one bad record doesn't kill the
                # rest of the small-n run (a partial report is more useful
                # than none).
                log.error("dual_run: future raised %s: %s", type(e).__name__, e)
                continue
            rows.append(row)
            with write_lock:
                out_fh.write(json.dumps(row) + "\n")
                out_fh.flush()
                completed += 1
            mv = row["monolithic"]["verdict"]
            dv = row["decomposed"]["verdict"]
            log.info("[%d/%d] %s mono=%s dec=%s gold=%s",
                     completed, len(pending),
                     row["stmt_type"], mv, dv, row["gold"])

    _print_summary(rows, args.model, out_path)


def _print_summary(rows: list[dict], model: str, out_path: Path) -> None:
    n = len(rows)
    print("\n" + "=" * 72)
    print(f"DUAL-RUN small-n: n={n}, model={model}")
    print("=" * 72)

    # Crash-freeness (S1)
    mono_errors = sum(1 for r in rows if r["monolithic"]["error"])
    dec_errors = sum(1 for r in rows if r["decomposed"]["error"])
    print(f"\nCrash-freeness (S1): mono exceptions={mono_errors} "
          f"dec exceptions={dec_errors}")
    s1_pass = dec_errors == 0

    # Accuracy per path
    mono_results = [_verdict_ok(r["gold"], r["monolithic"]["verdict"])
                    for r in rows]
    dec_results = [_verdict_ok(r["gold"], r["decomposed"]["verdict"])
                   for r in rows]

    def _acc(xs):
        decided = [x for x in xs if x is not None]
        return (sum(decided) / len(decided)) if decided else 0.0, len(decided)

    mono_acc, mono_n = _acc(mono_results)
    dec_acc, dec_n = _acc(dec_results)
    print(f"\nCoverage-conditional accuracy (accuracy on records the path"
          f" actually decided):")
    print(f"  monolithic: {mono_acc:.1%} ({mono_n} decided / {n})")
    print(f"  decomposed: {dec_acc:.1%} ({dec_n} decided / {n})")
    gap_pp = abs(mono_acc - dec_acc) * 100
    s2_pass = gap_pp <= SMALL_N_ACC_GAP_PP
    print(f"  gap = {gap_pp:.1f}pp  (S2: must be ≤ {SMALL_N_ACC_GAP_PP:.0f}pp)")

    # Abstention rate — first-class metric for the decomposed path.
    # The strategic-review reframe: abstention is the path's calibrated
    # uncertainty signal, not a defect to suppress. Tracked here as a
    # primary number so we can see the coverage/precision trade.
    dec_abstain = sum(1 for r in rows
                      if (r["decomposed"]["verdict"] or "").lower() == "abstain")
    mono_undecided = sum(1 for r in rows
                         if r["monolithic"]["verdict"] in (None, "abstain"))
    print(f"\nAbstention / undecided:")
    print(f"  monolithic: {mono_undecided}/{n} = {mono_undecided/n:.1%}")
    print(f"  decomposed: {dec_abstain}/{n} = {dec_abstain/n:.1%}")

    # Verdict distribution
    def _dist(key):
        c = Counter()
        for r in rows:
            c[r[key]["verdict"] or "None"] += 1
        return dict(c)
    print(f"\nVerdict distribution:")
    print(f"  monolithic: {_dist('monolithic')}")
    print(f"  decomposed: {_dist('decomposed')}")

    # Agreement on correct cases (S3)
    correct_gold = [r for r in rows if (r["gold"] or "").lower() == "correct"]
    if correct_gold:
        agreed = sum(1 for r in correct_gold
                     if (r["monolithic"]["verdict"] or "").lower()
                     == (r["decomposed"]["verdict"] or "").lower())
        corr_agree = agreed / len(correct_gold)
    else:
        corr_agree = 1.0
    s3_pass = corr_agree >= SMALL_N_CORRECT_AGREEMENT
    print(f"\nAgreement on GOLD-correct records: {corr_agree:.1%} "
          f"(S3: must be ≥ {SMALL_N_CORRECT_AGREEMENT:.0%})")

    # Cross-tab (monolithic × decomposed)
    print(f"\nCross-tab (monolithic rows × decomposed cols):")
    ctab: dict[tuple[str, str], int] = Counter()
    for r in rows:
        mv = r["monolithic"]["verdict"] or "None"
        dv = r["decomposed"]["verdict"] or "None"
        ctab[(mv, dv)] += 1
    verdicts = ["correct", "incorrect", "abstain", "None"]
    header = "          " + " ".join(f"{v:>9}" for v in verdicts)
    print(f"  {header}")
    for mv in verdicts:
        row = [str(ctab.get((mv, dv), 0)).rjust(9) for dv in verdicts]
        print(f"  {mv:>8} " + " ".join(row))

    # Decomposed reason-code histogram — useful for error-class attribution
    reason_counts: Counter = Counter()
    for r in rows:
        for rc in r["decomposed"].get("reasons", []) or []:
            reason_counts[rc] += 1
    if reason_counts:
        print(f"\nDecomposed reason-code histogram:")
        for rc, c in reason_counts.most_common():
            print(f"  {c:>3}  {rc}")

    # --- Verdict ---
    print("\n" + "-" * 72)
    gates = {
        "S1 (dec zero crashes)": s1_pass,
        f"S2 (acc gap ≤ {SMALL_N_ACC_GAP_PP:.0f}pp)": s2_pass,
        f"S3 (correct agreement ≥ {SMALL_N_CORRECT_AGREEMENT:.0%})": s3_pass,
    }
    for g, ok in gates.items():
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {g}")
    overall = "PASS — proceed to #27" if all(gates.values()) \
              else "FAIL — diagnose before full-holdout run"
    print(f"\nVERDICT: {overall}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

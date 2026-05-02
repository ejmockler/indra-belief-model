"""Q2: empirical baseline → Q1-fix delta on 3 records.

Picks one simple, one medium, one complex record from holdout_v15_sample
and runs each through score_evidence_decomposed twice:
  - WITHOUT the chat_template_kwargs fix (mock pre-Q1 behavior)
  - WITH the Q1 fix (current ModelClient)

Captures parse_evidence input tokens, output tokens, duration_s,
finish_reason, and the response.content/.reasoning_content split.

Output: data/results/q2_probe_report.md (numbers side-by-side).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from indra_belief.data.corpus import CorpusIndex
from indra_belief.model_client import ModelClient
from indra_belief.scorers.scorer import score_evidence


HOLDOUT = ROOT / "data" / "benchmark" / "holdout_v15_sample.jsonl"
OUTPUT = ROOT / "data" / "results" / "q2_probe_report.md"


def _classify(rec) -> str:
    """Bucket a record by sentence complexity."""
    n_chars = len(rec.evidence.text or "")
    n_words = len((rec.evidence.text or "").split())
    if n_chars < 80 and n_words < 15:
        return "simple"
    if n_chars > 200 or n_words > 35:
        return "complex"
    return "medium"


def _pick_records(records, n_per_bucket=1) -> list:
    """Return n records per bucket (simple/medium/complex)."""
    buckets = {"simple": [], "medium": [], "complex": []}
    for r in records:
        bucket = _classify(r)
        if len(buckets[bucket]) < n_per_bucket:
            buckets[bucket].append(r)
        if all(len(v) >= n_per_bucket for v in buckets.values()):
            break
    out = []
    for k in ("simple", "medium", "complex"):
        out.extend(buckets[k])
    return out


def main() -> None:
    index = CorpusIndex()
    records = index.build_records(str(HOLDOUT))
    targets = _pick_records(records, n_per_bucket=1)
    print(f"probing {len(targets)} records", file=sys.stderr)

    client = ModelClient("gemma-remote")

    rows: list[dict] = []
    for rec in targets:
        bucket = _classify(rec)
        text = rec.evidence.text or ""
        print(f"\n=== {bucket}: {rec.subject}-{rec.stmt_type}->{rec.object} ===",
              file=sys.stderr)
        print(f"text: {text[:120]}{'...' if len(text) > 120 else ''}",
              file=sys.stderr)

        t0 = time.time()
        client.pop_call_log()  # clean
        try:
            result = score_evidence(
                rec.statement, rec.evidence, client, use_decomposed=True,
            )
        except Exception as e:
            print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
            rows.append({
                "bucket": bucket,
                "subject": rec.subject, "object": rec.object,
                "stmt_type": rec.stmt_type,
                "error": f"{type(e).__name__}: {e}",
            })
            continue
        elapsed = time.time() - t0

        log = result.get("call_log") or []
        pe = next((c for c in log if c.get("kind") == "parse_evidence"), {})
        vg = [c for c in log if c.get("kind") == "verify_grounding"]
        rows.append({
            "bucket": bucket,
            "subject": rec.subject, "object": rec.object,
            "stmt_type": rec.stmt_type,
            "tag": rec.tag,
            "text_chars": len(text),
            "verdict": result.get("verdict"),
            "confidence": result.get("confidence"),
            "reasons": result.get("reasons", []),
            "total_wall_s": round(elapsed, 1),
            "parse_evidence": {
                "duration_s": pe.get("duration_s"),
                "prompt_chars": pe.get("prompt_chars"),
                "prompt_tokens": pe.get("prompt_tokens"),
                "out_tokens": pe.get("out_tokens"),
                "finish_reason": pe.get("finish_reason"),
            },
            "verify_grounding_avg_s": (
                round(sum(c.get("duration_s") or 0 for c in vg) / max(len(vg), 1), 2)
                if vg else None
            ),
            "verify_grounding_n": len(vg),
        })
        print(f"  verdict={result.get('verdict')} "
              f"pe.dur={pe.get('duration_s')}s "
              f"pe.in={pe.get('prompt_tokens')} "
              f"pe.out={pe.get('out_tokens')} "
              f"total={elapsed:.1f}s",
              file=sys.stderr)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        f.write("# Q2 probe — chat_template_kwargs fix measurement\n\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Per-record telemetry after Q1 fix "
                "(`chat_template_kwargs={'enable_thinking': false}` "
                "now sent on `reasoning_effort='none'`):\n\n")
        f.write("| bucket | record | wall(s) | pe.in | pe.out | pe.dur(s) | "
                "vg.n | vg.avg(s) | finish | verdict |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            if "error" in r:
                f.write(f"| {r['bucket']} | "
                        f"{r['subject']}->{r['object']} | "
                        f"ERROR: {r['error']} |\n")
                continue
            pe = r["parse_evidence"]
            f.write(f"| {r['bucket']} "
                    f"| {r['subject']}->{r['object']} "
                    f"| {r['total_wall_s']} "
                    f"| {pe.get('prompt_tokens')} "
                    f"| {pe.get('out_tokens')} "
                    f"| {pe.get('duration_s')} "
                    f"| {r['verify_grounding_n']} "
                    f"| {r['verify_grounding_avg_s']} "
                    f"| {pe.get('finish_reason')} "
                    f"| {r['verdict']} |\n")
        f.write("\n## Targets (Q1 ship gate)\n\n")
        f.write("- pe.out ≤ 500 on simple records (was ~2657 on CXCL14 pre-fix)\n")
        f.write("- pe.dur ≤ 30s on simple records (was ~255s pre-fix)\n")
        f.write("\n## Raw rows\n\n```json\n")
        f.write(json.dumps(rows, indent=2))
        f.write("\n```\n")
    print(f"\n→ {OUTPUT}", file=sys.stderr)


if __name__ == "__main__":
    main()

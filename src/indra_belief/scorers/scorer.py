"""Evidence quality scorer using native INDRA objects (S-phase).

The S-phase scorer has exactly one path: parse_claim → substrate_route
→ four probes (LLM where escalated) → ProbeBundle → adjudicate.
The doctrine §7 migration discipline removed:
  - the monolithic prompt + tool-use path (parse-evidence-as-extraction)
  - the use_decomposed flag (no flag, no fallback)
  - the cascade-on-abstain logic (abstention is the calibrated signal)

All scoring routes through `score_via_probes` in
`indra_belief.scorers.probes.orchestrator`.

Run:
    PYTHONPATH=src python -m indra_belief.scorers.scorer
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from indra_belief.model_client import ModelClient
from indra_belief.scorers.probes.orchestrator import score_via_probes


ROOT = Path(__file__).resolve().parent.parent.parent.parent


def score_evidence(statement, evidence, client: ModelClient) -> dict:
    """Score one (Statement, Evidence) pair via the four-probe pipeline.

    Per-sentence comprehension layer. For scoring a whole Statement
    (which carries `statement.evidence` as a list), use
    `score_statement`, which iterates this function.

    Args:
        statement: An `indra.statements.Statement`. Binary types,
            SelfModification, Complex, and Translocation are rendered.
        evidence: An `indra.statements.Evidence`.
        client: A `ModelClient` configured for the chosen backend.

    Returns a dict with keys:
        score                float in [0, 1]
        verdict              "correct" | "incorrect" | "abstain"
        confidence           "high" | "medium" | "low"
        tier                 "decomposed" (S-phase always emits this)
        grounding_status     "all_match" | "flagged"
        provenance_triggered bool (always False in S-phase)
        tokens               completion tokens consumed
        raw_text             decision trace
        reasons              list[ReasonCode]
        rationale            informational human-readable note
        call_log             per-LLM-call telemetry
    """
    return score_via_probes(statement, evidence, client)


def score_statement(statement, client: ModelClient) -> list[dict]:
    """Score an INDRA Statement by scoring each evidence sentence.

    Mirrors INDRA's abstraction. Returns one scoring dict per evidence,
    in the same order as `statement.evidence`. Returns `[]` if the
    statement has no evidence.

    Pair with `indra_belief.composed_scorer.ComposedBeliefScorer` to
    aggregate per-sentence verdicts into edge-level belief.
    """
    evidences = list(statement.evidence or [])
    return [score_evidence(statement, ev, client) for ev in evidences]


def main():
    import argparse
    from indra_belief.data.corpus import CorpusIndex

    parser = argparse.ArgumentParser(
        description="Evidence quality scorer (INDRA native, S-phase)"
    )
    parser.add_argument("--model", default="gemma-remote")
    parser.add_argument("--holdout",
                        default=str(ROOT / "data" / "benchmark" / "holdout.jsonl"))
    parser.add_argument("--output",
                        default=str(ROOT / "data" / "results" / "scorer_output.jsonl"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from existing output file (skip scored records)")
    args = parser.parse_args()

    index = CorpusIndex()
    records = index.build_records(args.holdout)
    if args.limit:
        records = records[: args.limit]

    scored_hashes: set = set()
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            with open(resume_path) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        scored_hashes.add(r.get("source_hash"))
                    except json.JSONDecodeError:
                        pass
            print(f"Resuming: {len(scored_hashes)} records already scored")

    print(f"\nScorer (S-phase): {len(records)} records, model={args.model}")

    client = ModelClient(args.model)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    out_fh = open(output_path, mode)

    correct = 0
    total_parsed = 0
    t_start = time.time()

    for i, record in enumerate(records):
        if record.source_hash in scored_hashes:
            continue

        result = score_evidence(record.statement, record.evidence, client)

        gt_correct = record.tag == "correct"
        verdict = result.get("verdict")
        llm_correct = (verdict == "correct") if verdict else None
        if llm_correct is not None:
            total_parsed += 1
            if llm_correct == gt_correct:
                correct += 1

        result.update({
            "source_hash": record.source_hash,
            "tag": record.tag or "",
            "subject": record.subject,
            "stmt_type": record.stmt_type,
            "object": record.object,
        })

        r_save = {k: v for k, v in result.items() if k != "raw_text"}
        r_save["raw_text_preview"] = (result.get("raw_text") or "")[:500]
        out_fh.write(json.dumps(r_save) + "\n")
        out_fh.flush()

        acc = correct / total_parsed * 100 if total_parsed > 0 else 0
        mark = ("✓" if (llm_correct == gt_correct)
                else ("✗" if llm_correct is not None else "?"))
        print(f"  [{i + 1:3d}/{len(records)}] {mark} "
              f"{record.subject:>10s} [{record.stmt_type:>15s}] "
              f"{record.object:10s} → "
              f"{verdict or 'PARSE':>9s} acc={acc:.1f}%")

    out_fh.close()

    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"RESULTS: {correct}/{total_parsed} = "
          f"{correct / max(total_parsed, 1) * 100:.1f}% in {elapsed/60:.1f}min")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()

"""Evidence quality scorer using native INDRA objects.

Three-tier architecture:
  Tier 1: Deterministic grounding check (GroundedEntity.should_auto_reject)
    - MISMATCH → auto-reject
    - LOW_CONFIDENCE (gilda score ≤ 0.53) → auto-reject
    - PSEUDOGENE + AMBIGUOUS → auto-reject
    - AMBIGUOUS → LLM judges with grounding context
  Tier 2: LLM text comprehension
    - v8 system prompt + Rule 7 (provenance) + contrastive examples
    - Entity context, provenance, directness from ScoringRecord

Run:
    PYTHONPATH=src python -m indra_belief.scorers.scorer
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from indra_belief.data.corpus import CorpusIndex
from indra_belief.data.scoring_record import ScoringRecord
from indra_belief.model_client import ModelClient

from indra_belief.scorers.evidence_scorer import (
    SYSTEM_PROMPT as _V8_SYSTEM_PROMPT,
    CONTRASTIVE_EXAMPLES as _ALL_EXAMPLES,
    _render_example,
    extract_verdict,
    verdict_to_score,
)

ROOT = Path(__file__).resolve().parents[3]

_PROVENANCE_RULE = """
7. PROVENANCE: If "Extraction provenance" is shown, it reveals what the NLP
   reader actually extracted from the sentence and how entities were mapped
   to gene symbols. Pay attention to:
   - MISMATCH: the extracted text refers to a DIFFERENT entity than the claim
   - AMBIGUOUS: the extracted text may refer to a different protein
   - LOW CONFIDENCE: the automated mapping is uncertain; verify carefully
   These signals indicate the text-mining system may have grounded incorrectly.
"""

SYSTEM_PROMPT = _V8_SYSTEM_PROMPT.rstrip() + "\n" + _PROVENANCE_RULE

# --- Adaptive few-shot selection ---
# Core examples (general-purpose, always included): indices into _ALL_EXAMPLES
# Pair 2 (act_vs_amt): indices 2,3
# Pair 3 (logical_inversion): indices 4,5
# Pair 4 (hedging_scope): indices 6,7
# Pair 7 (parallel_activation / third-party agent): indices 12,13
_CORE_INDICES = [2, 3, 4, 5, 6, 7, 12, 13]
_CORE_EXAMPLES = [_ALL_EXAMPLES[i] for i in _CORE_INDICES]

# Type-specific example bank (loaded from JSON)
_EXAMPLE_BANK_PATH = Path(__file__).parent.parent / "data" / "example_bank.json"
_TYPE_EXAMPLES: dict[str, list[dict]] = {}
if _EXAMPLE_BANK_PATH.exists():
    with open(_EXAMPLE_BANK_PATH) as _f:
        _TYPE_EXAMPLES = json.load(_f)


def _select_examples(stmt_type: str) -> list[dict]:
    """Select few-shot examples for a record's statement type.

    Always includes 4 core pairs (8 examples) for general patterns.
    Adds 1 type-matched pair (2 examples) if available.
    Falls back to all 18 v8 examples for types with no bank entry.
    """
    type_pair = _TYPE_EXAMPLES.get(stmt_type, [])
    if type_pair:
        return _CORE_EXAMPLES + type_pair
    # For covered types (Complex, Activation, Phosphorylation), use existing v8 examples
    # that match the type, plus core
    type_specific = [ex for ex in _ALL_EXAMPLES
                     if ex["claim"].split("[")[1].split("]")[0].strip() == stmt_type
                     and _ALL_EXAMPLES.index(ex) not in _CORE_INDICES]
    if type_specific:
        return _CORE_EXAMPLES + type_specific[:2]
    # Fallback: core only
    return _CORE_EXAMPLES


def score(client: ModelClient, record: ScoringRecord, max_tokens: int = 4000) -> dict:
    """Score a single extraction.

    Returns dict with: score, verdict, confidence, raw_text, tokens,
    tier, grounding_status, provenance_triggered.
    """
    # --- Tier 1: Deterministic auto-reject ---
    reject = record.tier1_auto_reject()
    if reject:
        return reject

    # AMBIGUOUS entities go directly to Tier 2 — the intermediate AMBIGUOUS
    # LLM was 64% accurate at scale (barely better than coin flip) and added
    # an extra LLM call that primed the model with grounding-focused evaluation
    # before the comprehension evaluation.

    # --- Tier 2: LLM text comprehension ---
    user_msg = record.format_user_message()
    provenance_triggered = bool(record.format_provenance())

    # Adaptive few-shot: core examples + type-matched pair
    examples = _select_examples(record.stmt_type)

    messages = []
    for ex in examples:
        u, a = _render_example(ex)
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": user_msg})

    response = client.call(
        system=SYSTEM_PROMPT,
        messages=messages,
        max_tokens=max_tokens,
    )
    verdict, confidence = extract_verdict(response.raw_text)

    # Determine grounding status
    if any(e.has_grounding_signal for e in (record.subject_entity, record.object_entity) if e):
        grounding_status = "flagged"
    else:
        grounding_status = "all_match"

    return {
        "score": verdict_to_score(verdict, confidence),
        "verdict": verdict,
        "confidence": confidence,
        "raw_text": f"[TIER 2 LLM]\n{response.raw_text}",
        "tokens": response.tokens,
        "tier": "llm_comprehension",
        "grounding_status": grounding_status,
        "provenance_triggered": provenance_triggered,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evidence quality scorer (INDRA native)")
    parser.add_argument("--model", default="gemma-remote")
    parser.add_argument("--holdout", default=str(ROOT / "data" / "benchmark" / "holdout.jsonl"))
    parser.add_argument("--output", default=str(ROOT / "data" / "results" / "v11_native.jsonl"))
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    # Load corpus and build records
    index = CorpusIndex()
    records = index.build_records(args.holdout)
    if args.limit:
        records = records[:args.limit]

    print(f"\nScorer: {len(records)} records, model={args.model}")

    client = ModelClient(args.model)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_fh = open(output_path, "w")

    correct = 0
    total_parsed = 0
    tier_counts = {}
    t_start = time.time()

    for i, record in enumerate(records):
        result = score(client, record, args.max_tokens)

        gt_correct = record.tag == "correct"
        llm_correct = (result["verdict"] == "correct") if result["verdict"] else None

        if llm_correct is not None:
            total_parsed += 1
            if llm_correct == gt_correct:
                correct += 1

        tier = result.get("tier", "?")
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

        result.update({
            "source_hash": record.source_hash,
            "tag": record.tag or "",
            "subject": record.subject,
            "stmt_type": record.stmt_type,
            "object": record.object,
        })

        r_save = {k: v for k, v in result.items() if k != "raw_text"}
        r_save["raw_text_preview"] = result.get("raw_text", "")[:500]
        out_fh.write(json.dumps(r_save) + "\n")
        out_fh.flush()

        acc = correct / total_parsed * 100 if total_parsed > 0 else 0
        mark = "✓" if (llm_correct == gt_correct) else ("✗" if llm_correct is not None else "?")
        tier_short = {
            "deterministic_mismatch": "T1:MSMATCH",
            "deterministic_low_confidence": "T1:LOWCONF",
            "deterministic_pseudogene": "T1:PSEUDO",
            "ambiguous_rejected": "T1:AMBIG_R",
            "ambiguous_then_llm": "T1→T2",
            "llm_comprehension": "T2:LLM",
        }.get(tier, tier)
        print(f"  [{i+1:3d}/{len(records)}] {mark} {record.subject:>10s} [{record.stmt_type:>15s}] {record.object:10s} "
              f"→ {result['verdict'] or 'PARSE':>9s} [{tier_short:10s}] acc={acc:.1f}%")

    out_fh.close()

    print(f"\n{'='*70}")
    print(f"RESULTS: {correct}/{total_parsed} = {correct/max(total_parsed,1)*100:.1f}%")
    print(f"Tier breakdown: {tier_counts}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()

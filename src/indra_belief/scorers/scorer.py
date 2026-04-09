"""Evidence quality scorer using native INDRA objects.

Two-tier architecture:
  Tier 1: Deterministic grounding check (GroundedEntity.should_auto_reject)
    - MISMATCH → auto-reject
    - PSEUDOGENE + AMBIGUOUS → auto-reject
    - AMBIGUOUS → LLM judges with grounding context
  Tier 2: LLM text comprehension
    - v8 system prompt + adaptive contrastive examples
    - Entity context from ScoringRecord

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

# Provenance rule removed — provenance injection disabled at scale
# (72.2% accuracy when triggered vs 78.9% baseline). The LLM performs
# worse when given extraction provenance signals.
SYSTEM_PROMPT = _V8_SYSTEM_PROMPT

# --- Adaptive few-shot selection ---
# Reduced budget (7 pairs = 14 examples) with better type targeting.
# 9 pairs (66% of tokens) was diluting attention; 7 pairs frees ~20%
# of context for the model's own reasoning while being more relevant.

# Type-specific example bank (loaded from JSON)
# Bank keys can be exact types ("Activation") or sub-keys ("Activation_no_relation")
_EXAMPLE_BANK_PATH = Path(__file__).parent.parent / "data" / "example_bank.json"
_RAW_BANK: dict[str, list[dict]] = {}
if _EXAMPLE_BANK_PATH.exists():
    with open(_EXAMPLE_BANK_PATH) as _f:
        _RAW_BANK = json.load(_f)

# Build type → list of pairs mapping from bank
# Keys like "Activation_no_relation" contribute to "Activation"
_TYPE_BANK: dict[str, list[list[dict]]] = {}
for key, pair in _RAW_BANK.items():
    base_type = key.split("_")[0] if "_" in key else key
    # Handle types that contain underscores in their names
    if base_type in ("IncreaseAmount", "DecreaseAmount"):
        base_type = key  # these ARE the type names
    elif key.startswith("Increase") or key.startswith("Decrease"):
        base_type = key.split("_")[0] + key.split("_")[1] if "_" in key and key.count("_") > 1 else key
    _TYPE_BANK.setdefault(base_type, []).append(pair)

# Map v8 examples into pairs by their statement type
_V8_PAIRS: dict[str, list[list[dict]]] = {}
for i in range(0, len(_ALL_EXAMPLES), 2):
    stype = _ALL_EXAMPLES[i]["claim"].split("[")[1].split("]")[0].strip()
    _V8_PAIRS.setdefault(stype, []).append([_ALL_EXAMPLES[i], _ALL_EXAMPLES[i + 1]])

# Universal pairs — patterns that apply to all statement types
_UNIVERSAL_PAIRS = [
    _ALL_EXAMPLES[4:6],    # Pair 3: logical inversion (AGER/MMP2, TP53/MDM2)
    _ALL_EXAMPLES[6:8],    # Pair 4: hedging scope (MYB/PPID)
]

# Which types are commonly confused with each other?
_TYPE_ADJACENCY = {
    "Phosphorylation": ["Dephosphorylation", "Autophosphorylation"],
    "Dephosphorylation": ["Phosphorylation", "Inhibition"],
    "Activation": ["IncreaseAmount", "Inhibition"],
    "Inhibition": ["DecreaseAmount", "Activation"],
    "IncreaseAmount": ["Activation", "DecreaseAmount"],
    "DecreaseAmount": ["IncreaseAmount", "Inhibition"],
    "Complex": ["Activation"],
    "Autophosphorylation": ["Phosphorylation"],
    "Translocation": [],
    "Ubiquitination": [],
    "Acetylation": ["Deacetylation"],
}

TARGET_PAIRS = 7  # reduced from 9 — frees ~20% attention for reasoning


def _select_examples(stmt_type: str) -> list[dict]:
    """Select 7 contrastive pairs (14 examples) for a record's statement type.

    Priority:
    1. Own type pair(s) — from bank (may have multiple sub-keys) and/or v8
    2. Adjacent type pairs — types commonly confused with this one
    3. Universal patterns — logical inversion, hedging scope
    4. Fill from remaining v8 pairs
    """
    selected: list[list[dict]] = []
    used_claims: set[str] = set()

    def _add_pair(pair: list[dict]) -> bool:
        key = pair[0]["claim"]
        if key in used_claims or len(selected) >= TARGET_PAIRS:
            return False
        selected.append(pair)
        used_claims.add(key)
        return True

    # 1. Own type from bank (may have multiple pairs from sub-keys)
    for pair in _TYPE_BANK.get(stmt_type, []):
        _add_pair(pair)

    # 1b. Own type from v8
    for pair in _V8_PAIRS.get(stmt_type, []):
        _add_pair(pair)

    # 2. Adjacent types
    for adj_type in _TYPE_ADJACENCY.get(stmt_type, []):
        for pair in _TYPE_BANK.get(adj_type, []):
            _add_pair(pair)
        for pair in _V8_PAIRS.get(adj_type, []):
            _add_pair(pair)

    # 3. Universal patterns
    for pair in _UNIVERSAL_PAIRS:
        _add_pair(pair)

    # 4. Fill remaining from v8 pairs
    for i in range(0, len(_ALL_EXAMPLES), 2):
        _add_pair([_ALL_EXAMPLES[i], _ALL_EXAMPLES[i + 1]])

    # Flatten pairs into example list
    examples = []
    for pair in selected[:TARGET_PAIRS]:
        examples.extend(pair)
    return examples


def _score_single(
    client: ModelClient,
    record: ScoringRecord,
    max_tokens: int,
    temperature: float = 0.1,
) -> dict:
    """Single LLM call for Tier 2. Returns result dict."""
    user_msg = record.format_user_message()

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
        temperature=temperature,
    )
    verdict, confidence = extract_verdict(response.raw_text)
    return {
        "verdict": verdict,
        "confidence": confidence,
        "raw_text": response.raw_text,
        "tokens": response.tokens,
    }


def score(
    client: ModelClient,
    record: ScoringRecord,
    max_tokens: int = 4000,
    voting_k: int = 1,
) -> dict:
    """Score a single extraction.

    Args:
        voting_k: Number of independent samples for self-consistency voting.
            k=1 (default) is a single call. k=3 or k=5 uses majority voting
            with temperature=0.6 for diversity.

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

    provenance_triggered = bool(record.format_provenance())

    # --- Tier 2: LLM text comprehension ---
    if voting_k <= 1:
        result = _score_single(client, record, max_tokens)
        verdict = result["verdict"]
        confidence = result["confidence"]
        total_tokens = result["tokens"]
        raw = f"[TIER 2 LLM]\n{result['raw_text']}"
        tier = "llm_comprehension"
    else:
        # Self-consistency voting: k independent samples at higher temperature
        votes = []
        total_tokens = 0
        raw_parts = []
        for i in range(voting_k):
            r = _score_single(client, record, max_tokens, temperature=0.6)
            total_tokens += r["tokens"]
            raw_parts.append(f"[VOTE {i+1}] {r['verdict']}({r['confidence']})")
            if r["verdict"]:
                votes.append(r["verdict"])

        # Majority vote
        correct_votes = sum(1 for v in votes if v == "correct")
        incorrect_votes = sum(1 for v in votes if v == "incorrect")

        if correct_votes > incorrect_votes:
            verdict = "correct"
        elif incorrect_votes > correct_votes:
            verdict = "incorrect"
        else:
            verdict = votes[0] if votes else None  # tie → first vote

        # Confidence from agreement level
        total_votes = correct_votes + incorrect_votes
        if total_votes > 0:
            agreement = max(correct_votes, incorrect_votes) / total_votes
            if agreement >= 0.8:
                confidence = "high"
            elif agreement >= 0.6:
                confidence = "medium"
            else:
                confidence = "low"
        else:
            confidence = None

        raw = " | ".join(raw_parts)
        tier = f"llm_voting_k{voting_k}"

    # Determine grounding status
    if any(e.has_grounding_signal for e in (record.subject_entity, record.object_entity) if e):
        grounding_status = "flagged"
    else:
        grounding_status = "all_match"

    return {
        "score": verdict_to_score(verdict, confidence),
        "verdict": verdict,
        "confidence": confidence,
        "raw_text": raw,
        "tokens": total_tokens,
        "tier": tier,
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
    parser.add_argument("--voting-k", type=int, default=1,
                        help="Self-consistency voting: k independent samples (1=single call)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from existing output file (skip scored records)")
    args = parser.parse_args()

    # Load corpus and build records
    index = CorpusIndex()
    records = index.build_records(args.holdout)
    if args.limit:
        records = records[:args.limit]

    # Resume support: skip already-scored records
    scored_hashes = set()
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

    voting_label = f", voting_k={args.voting_k}" if args.voting_k > 1 else ""
    print(f"\nScorer: {len(records)} records, model={args.model}{voting_label}")

    client = ModelClient(args.model)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    out_fh = open(output_path, mode)

    correct = 0
    total_parsed = 0
    tier_counts = {}
    t_start = time.time()

    for i, record in enumerate(records):
        if record.source_hash in scored_hashes:
            continue

        result = score(client, record, args.max_tokens, voting_k=args.voting_k)

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
            "deterministic_pseudogene": "T1:PSEUDO",
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

"""v10 scorer: deterministic grounding verification + LLM text comprehension.

Two-tier architecture:
  Tier 1: Deterministic grounding check (raw_text vs claim entities via gilda)
    - MISMATCH → auto-reject (no LLM call, 100% precision)
    - AMBIGUOUS → LLM judges with corrected framing + evidence context
    - MATCH/UNRESOLVABLE → pass to Tier 2
  Tier 2: LLM text comprehension (exact v8 prompt + contrastive examples)

Run:
    PYTHONPATH=src python -m indra_belief.scorers.v10_scorer
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

from indra_belief.data.claim_enricher import (
    build_corpus_index_v8,
    enrich_claim,
    format_entity_context,
    get_evidence_directness,
    lookup_evidence_meta,
)
from indra_belief.model_client import ModelClient
from indra_belief.tools.grounding_verifier import check_record

# Tier 2 uses exact v8 prompt and examples
from indra_belief.scorers.evidence_scorer import (
    SYSTEM_PROMPT,
    CONTRASTIVE_EXAMPLES,
    _render_example,
    extract_verdict,
    verdict_to_score,
)


def score_record(
    client: ModelClient,
    subject: str,
    stmt_type: str,
    obj: str,
    evidence_text: str,
    source_hash: int | None = None,
    corpus_index: dict | None = None,
    evidence_meta: dict | None = None,
    max_tokens: int = 4000,
) -> dict:
    """Score with deterministic grounding check + LLM text comprehension.

    Returns dict with: score, verdict, confidence, raw_text, tokens,
    tier (which tier rendered the verdict), grounding_status.
    """
    # --- Tier 1: Deterministic grounding verification ---
    raw_text = None
    grounding_results = []
    if source_hash is not None and evidence_meta is not None:
        meta = lookup_evidence_meta(source_hash, subject, obj, evidence_meta)
        raw_text = meta.get("raw_text")
        grounding_results = check_record(subject, obj, raw_text)

    # Check for MISMATCH — auto-reject ONLY if the claim entity can't be found
    # in the evidence text (safety check against incomplete raw_text)
    mismatches = [(rt, ce, s, n, m) for rt, ce, s, n, m in grounding_results if s == "MISMATCH"]
    if mismatches:
        # Safety: does the claim entity (or any known alias) appear in evidence?
        ev_lower = evidence_text.lower()
        entity_ctx = format_entity_context(subject, obj)
        aliases_lower = entity_ctx.lower() if entity_ctx else ""

        # Normalize evidence text for matching (collapse hyphens, spaces)
        ev_collapsed = ev_lower.replace("-", "").replace(" ", "")

        safe_to_reject = True
        for rt, ce, _, _, _ in mismatches:
            ce_low = ce.lower()
            ce_collapsed = ce_low.replace("-", "").replace(" ", "")
            # Check claim entity (exact and normalized)
            if ce_low in ev_lower or ce_collapsed in ev_collapsed:
                safe_to_reject = False
                break
            # Check gilda aliases (exact and normalized)
            try:
                import gilda
                ce_results = gilda.ground(ce)
                if ce_results and ce_results[0].term.db == "HGNC":
                    names = gilda.get_names("HGNC", str(ce_results[0].term.id))
                    for name in names:
                        if len(name) >= 3:
                            n_low = name.lower()
                            n_collapsed = n_low.replace("-", "").replace(" ", "")
                            if n_low in ev_lower or n_collapsed in ev_collapsed:
                                safe_to_reject = False
                                break
            except Exception:
                pass
            # Also check: does the raw_text share significant words with claim entity's full name?
            # (handles descriptive names like "nucleosome assembly protein-1" → NAP1L1)
            if safe_to_reject:
                try:
                    ce_results = gilda.ground(ce)
                    if ce_results and ce_results[0].term.db == "HGNC":
                        names = gilda.get_names("HGNC", str(ce_results[0].term.id))
                        rt_words = set(rt.lower().replace("-", " ").split())
                        for name in names:
                            if len(name) > 15:
                                n_words = set(name.lower().replace("-", " ").split())
                                shared = rt_words & n_words - {"protein", "factor", "the", "of", "and", "a"}
                                if len(shared) >= 2:
                                    safe_to_reject = False
                                    break
                except Exception:
                    pass
            if not safe_to_reject:
                break

        if safe_to_reject:
            rt, ce, _, note, _ = mismatches[0]
            return {
                "score": 0.05,
                "verdict": "incorrect",
                "confidence": "high",
                "raw_text": f"[TIER 1 AUTO-REJECT] Grounding mismatch: {note}",
                "tokens": 0,
                "tier": "deterministic_mismatch",
                "grounding_status": "MISMATCH",
            }
        # Else: claim entity IS in evidence — raw_text may be incomplete
        # Fall through to Tier 2

    # Check for AMBIGUOUS — LLM judges with corrected framing
    # If LLM says "incorrect" → accept (grounding error caught)
    # If LLM says "correct" or can't parse → fall through to Tier 2
    ambiguous = [(rt, ce, s, n, m) for rt, ce, s, n, m in grounding_results if s == "AMBIGUOUS"]
    ambiguous_rejected = False
    ambiguous_raw = ""
    ambiguous_tokens = 0
    if ambiguous:
        rt, ce, _, note, _ = ambiguous[0]

        claim_str = enrich_claim(subject, stmt_type, obj, source_hash, corpus_index) if corpus_index else f"{subject} [{stmt_type}] {obj}"

        prompt = (
            f"The text-mining system extracted: {claim_str}\n\n"
            f"Grounding verification: AMBIGUOUS — {note}\n\n"
            f'Evidence: "{evidence_text}"\n\n'
            f"Is this extraction correct or incorrect?\n"
            f'Output JSON: {{"verdict": "correct" or "incorrect"}}'
        )

        response = client.call(
            system="You judge whether a text-mining EXTRACTION is correct. Output JSON.",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max(max_tokens, 16384),
        )
        verdict, confidence = extract_verdict(response.raw_text)
        ambiguous_raw = response.raw_text
        ambiguous_tokens = response.tokens

        if verdict == "incorrect":
            # AMBIGUOUS LLM says the grounding is wrong — accept this
            ambiguous_rejected = True
            return {
                "score": verdict_to_score(verdict, confidence),
                "verdict": verdict,
                "confidence": confidence,
                "raw_text": f"[TIER 1 AMBIGUOUS → REJECTED]\n{response.raw_text}",
                "tokens": response.tokens,
                "tier": "ambiguous_rejected",
                "grounding_status": "AMBIGUOUS",
            }
        # Otherwise (correct or parse fail): fall through to Tier 2
        # The mapping is valid or unclear — let text comprehension decide

    # --- Tier 2: LLM text comprehension (exact v8 behavior) ---
    if corpus_index and source_hash:
        claim = enrich_claim(subject, stmt_type, obj, source_hash, corpus_index)
    else:
        claim = f"{subject} [{stmt_type}] {obj}"

    entity_ctx = format_entity_context(subject, obj)

    ev_prefix = ""
    if source_hash is not None and evidence_meta is not None:
        direct = get_evidence_directness(source_hash, evidence_meta)
        if direct is False:
            ev_prefix = "[indirect evidence] "

    parts = [f"CLAIM: {claim}"]
    if entity_ctx:
        parts.append(entity_ctx)
    parts.append(f'EVIDENCE: "{ev_prefix}{evidence_text}"')
    user_msg = "\n".join(parts)

    messages = []
    for ex in CONTRASTIVE_EXAMPLES:
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

    if not grounding_results:
        grounding_status = "no_raw_text"
    elif ambiguous and not ambiguous_rejected:
        grounding_status = "AMBIGUOUS_accepted"
    else:
        grounding_status = "MATCH"

    tier_label = "llm_comprehension"
    raw_prefix = "[TIER 2 LLM]"
    if ambiguous and not ambiguous_rejected:
        tier_label = "ambiguous_then_llm"
        raw_prefix = f"[TIER 1 AMBIGUOUS → accepted → TIER 2 LLM]"

    return {
        "score": verdict_to_score(verdict, confidence),
        "verdict": verdict,
        "confidence": confidence,
        "raw_text": f"{raw_prefix}\n{response.raw_text}",
        "tokens": response.tokens + ambiguous_tokens,
        "tier": tier_label,
        "grounding_status": grounding_status,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="v10 scorer: deterministic grounding + LLM")
    parser.add_argument("--model", default="gemma-moe")
    parser.add_argument("--holdout", default=str(ROOT / "data" / "benchmark" / "holdout.jsonl"))
    parser.add_argument("--output", default=str(ROOT / "data" / "results" / "v10.jsonl"))
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    with open(args.holdout) as f:
        holdout = [json.loads(line) for line in f]
    if args.limit:
        holdout = holdout[:args.limit]

    print(f"v10 scorer: {len(holdout)} records, model={args.model}")

    print("\nBuilding corpus index...")
    t0 = time.time()
    indexes = build_corpus_index_v8()
    corpus_index = indexes["statements"]
    evidence_meta = indexes["evidence_meta"]
    print(f"  Index built in {time.time()-t0:.1f}s")

    client = ModelClient(args.model)
    print(f"\nScoring...")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_fh = open(output_path, "w")

    correct = 0
    total_parsed = 0
    tier_counts = {"deterministic_mismatch": 0, "ambiguous_llm": 0, "llm_comprehension": 0}
    t_start = time.time()

    for i, record in enumerate(holdout):
        result = score_record(
            client=client,
            subject=record["subject"],
            stmt_type=record["stmt_type"],
            obj=record["object"],
            evidence_text=record.get("evidence_text", ""),
            source_hash=record.get("source_hash"),
            corpus_index=corpus_index,
            evidence_meta=evidence_meta,
            max_tokens=args.max_tokens,
        )

        gt_correct = (record["tag"] == "correct")
        llm_correct = (result["verdict"] == "correct") if result["verdict"] else None

        if llm_correct is not None:
            total_parsed += 1
            if llm_correct == gt_correct:
                correct += 1

        tier = result.get("tier", "?")
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

        result.update({
            "source_hash": record.get("source_hash"),
            "tag": record["tag"],
            "subject": record["subject"],
            "stmt_type": record["stmt_type"],
            "object": record["object"],
        })

        r_save = {k: v for k, v in result.items() if k != "raw_text"}
        r_save["raw_text_preview"] = result.get("raw_text", "")[:500]
        out_fh.write(json.dumps(r_save) + "\n")
        out_fh.flush()

        elapsed = time.time() - t_start
        acc = correct / total_parsed * 100 if total_parsed > 0 else 0
        mark = "✓" if (llm_correct == gt_correct) else ("✗" if llm_correct is not None else "?")
        tier_short = {"deterministic_mismatch": "T1:REJECT", "ambiguous_llm": "T1:AMBIG", "llm_comprehension": "T2:LLM"}.get(tier, tier)
        print(f"  [{i+1:3d}/{len(holdout)}] {mark} {record['subject']:>10s} [{record['stmt_type']:>15s}] {record['object']:10s} "
              f"→ {result['verdict'] or 'PARSE':>9s} [{tier_short:9s}] acc={acc:.1f}%")

    out_fh.close()

    print(f"\n{'='*70}")
    print(f"v10 RESULTS: {correct}/{total_parsed} = {correct/max(total_parsed,1)*100:.1f}%")
    print(f"Tier breakdown: {tier_counts}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()

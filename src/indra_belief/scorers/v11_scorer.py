"""v11 scorer: structured provenance + graduated context + decomposed verification.

Three-tier architecture:
  Tier 1: Deterministic grounding check with enhanced metadata
    - MISMATCH → auto-reject (same as v10)
    - LOW_CONFIDENCE MATCH (gilda score ≤ 0.53) → auto-reject
    - PSEUDOGENE with AMBIGUOUS status → auto-reject
    - AMBIGUOUS → LLM judges with corrected framing
  Tier 2: LLM text comprehension with graduated alias context
    - v8 system prompt + contrastive examples
    - Graduated entity warnings (competing candidates, pseudogene, low confidence)
  Tier 3: Decomposed verification (Phase 3, not yet implemented)
    - Extract-then-compare for flagged "correct" verdicts

Run:
    PYTHONPATH=src python -m indra_belief.scorers.v11_scorer
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
    format_provenance_context,
    get_evidence_directness,
    lookup_evidence_meta,
)
from indra_belief.model_client import ModelClient
from indra_belief.tools.grounding_verifier import check_record

# Tier 2 uses v8 prompt + examples with provenance addendum
from indra_belief.scorers.evidence_scorer import (
    SYSTEM_PROMPT as _V8_SYSTEM_PROMPT,
    CONTRASTIVE_EXAMPLES,
    _render_example,
    extract_verdict,
    verdict_to_score,
)

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
    """Score with enhanced grounding check + graduated context + LLM.

    Returns dict with: score, verdict, confidence, raw_text, tokens,
    tier (which tier rendered the verdict), grounding_status.
    """
    # --- Tier 1: Deterministic grounding verification ---
    raw_text = None
    meta = {}
    grounding_results = []
    if source_hash is not None and evidence_meta is not None:
        meta = lookup_evidence_meta(source_hash, subject, obj, evidence_meta)
        raw_text = meta.get("raw_text")
        grounding_results = check_record(subject, obj, raw_text)

    # --- Tier 1a: Auto-reject for MISMATCH (same as v10) ---
    mismatches = [r for r in grounding_results if r[2] == "MISMATCH"]
    if mismatches:
        ev_lower = evidence_text.lower()
        ev_collapsed = ev_lower.replace("-", "").replace(" ", "")

        safe_to_reject = True
        for rt, ce, _, _, _ in mismatches:
            ce_low = ce.lower()
            ce_collapsed = ce_low.replace("-", "").replace(" ", "")
            if ce_low in ev_lower or ce_collapsed in ev_collapsed:
                safe_to_reject = False
                break
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

    # --- Tier 1b: Auto-reject for LOW_CONFIDENCE MATCH (new in v11) ---
    # Catches cross-species mappings like CagA→S100A8 (score 0.521)
    # Note: is_known_alias check removed — name collisions across species
    # (e.g., CagA = HGNC alias for S100A8 = H. pylori virulence factor)
    # make it unreliable. Threshold 0.53 alone safely separates errors
    # from valid aliases on the holdout set.
    low_conf = [r for r in grounding_results
                if r[2] == "MATCH" and r[4].get("is_low_confidence")]
    if low_conf:
        rt, ce, _, note, meta = low_conf[0]
        return {
            "score": 0.05,
            "verdict": "incorrect",
            "confidence": "high",
            "raw_text": (
                f"[TIER 1 AUTO-REJECT] Low-confidence grounding: "
                f'"{rt}" mapped to {ce} (gilda score: {meta["gilda_score"]:.3f} '
                f"— below confidence threshold)"
            ),
            "tokens": 0,
            "tier": "deterministic_low_confidence",
            "grounding_status": "LOW_CONFIDENCE",
        }

    # --- Tier 1c: Auto-reject for PSEUDOGENE with AMBIGUOUS status (new in v11) ---
    pseudo_ambig = [r for r in grounding_results
                    if r[2] == "AMBIGUOUS" and r[4].get("is_pseudogene")]
    if pseudo_ambig:
        rt, ce, _, note, meta = pseudo_ambig[0]
        return {
            "score": 0.05,
            "verdict": "incorrect",
            "confidence": "high",
            "raw_text": (
                f"[TIER 1 AUTO-REJECT] Pseudogene mapping: "
                f'{ce} is a pseudogene. {note}'
            ),
            "tokens": 0,
            "tier": "deterministic_pseudogene",
            "grounding_status": "PSEUDOGENE",
        }

    # --- Tier 1d: AMBIGUOUS → LLM judges (same as v10) ---
    ambiguous = [r for r in grounding_results if r[2] == "AMBIGUOUS"]
    ambiguous_rejected = False
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
        ambiguous_tokens = response.tokens

        if verdict == "incorrect":
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

    # --- Tier 2: LLM text comprehension with graduated context ---
    if corpus_index and source_hash:
        claim = enrich_claim(subject, stmt_type, obj, source_hash, corpus_index)
    else:
        claim = f"{subject} [{stmt_type}] {obj}"

    # Graduated entity context: includes warnings when grounding is suspicious
    entity_ctx = format_entity_context(
        subject, obj,
        grounding_results=grounding_results if grounding_results else None,
        raw_text=raw_text,
    )

    # Structured provenance: shows what the NLP reader extracted (Phase 2)
    provenance_ctx = ""
    if source_hash is not None and evidence_meta is not None:
        provenance_ctx = format_provenance_context(
            subject, obj, source_hash, evidence_meta, grounding_results,
        )

    ev_prefix = ""
    if source_hash is not None and evidence_meta is not None:
        direct = get_evidence_directness(source_hash, evidence_meta, subject, obj)
        if direct is False:
            ev_prefix = "[indirect evidence] "

    parts = [f"CLAIM: {claim}"]
    if entity_ctx:
        parts.append(entity_ctx)
    if provenance_ctx:
        parts.append(provenance_ctx)
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

    # Determine grounding status label
    if not grounding_results:
        grounding_status = "all_match"  # v11: renamed from misleading "no_raw_text"
    elif ambiguous and not ambiguous_rejected:
        grounding_status = "AMBIGUOUS_accepted"
    else:
        grounding_status = "MATCH"

    # Check if any results had flags even though they were MATCH
    flagged = [r for r in grounding_results if r[4].get("is_low_confidence") or r[4].get("is_pseudogene")]
    if flagged and grounding_status not in ("AMBIGUOUS_accepted",):
        grounding_status = "MATCH_flagged"

    tier_label = "llm_comprehension"
    raw_prefix = "[TIER 2 LLM]"
    if ambiguous and not ambiguous_rejected:
        tier_label = "ambiguous_then_llm"
        raw_prefix = "[TIER 1 AMBIGUOUS → accepted → TIER 2 LLM]"

    return {
        "score": verdict_to_score(verdict, confidence),
        "verdict": verdict,
        "confidence": confidence,
        "raw_text": f"{raw_prefix}\n{response.raw_text}",
        "tokens": response.tokens + ambiguous_tokens,
        "tier": tier_label,
        "grounding_status": grounding_status,
        "provenance_triggered": bool(provenance_ctx),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="v11 scorer: graduated context + enhanced grounding")
    parser.add_argument("--model", default="gemma-moe")
    parser.add_argument("--holdout", default=str(ROOT / "data" / "benchmark" / "holdout.jsonl"))
    parser.add_argument("--output", default=str(ROOT / "data" / "results" / "v11.jsonl"))
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    with open(args.holdout) as f:
        holdout = [json.loads(line) for line in f]
    if args.limit:
        holdout = holdout[:args.limit]

    print(f"v11 scorer: {len(holdout)} records, model={args.model}")

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
    tier_counts = {}
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
        print(f"  [{i+1:3d}/{len(holdout)}] {mark} {record['subject']:>10s} [{record['stmt_type']:>15s}] {record['object']:10s} "
              f"→ {result['verdict'] or 'PARSE':>9s} [{tier_short:10s}] acc={acc:.1f}%")

    out_fh.close()

    print(f"\n{'='*70}")
    print(f"v11 RESULTS: {correct}/{total_parsed} = {correct/max(total_parsed,1)*100:.1f}%")
    print(f"Tier breakdown: {tier_counts}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()

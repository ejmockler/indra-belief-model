"""v9 LLM evidence scorer — agentic grounding verification.

Changes from v8:
- NEW: Tool-use via Gemma 4 native function calling (lookup_gene)
- NEW: Shows NLP extraction provenance (raw_text) alongside claim entities
- Model decides when to verify entity mappings via tool call
- Combines text comprehension (v8 rules) + grounding verification (tool use)

Run:
    python -m indra_belief.scorers.agentic_scorer
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
)
from indra_belief.model_client import ModelClient, ModelResponse
from indra_belief.tools.gilda_tools import TOOLS, TOOL_DECLARATIONS


# ---------------------------------------------------------------------------
# System prompt — Pass 1 uses exact v8 prompt (no tool instructions)
# ---------------------------------------------------------------------------

from indra_belief.scorers.evidence_scorer import SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Contrastive examples — 11 pairs covering all error classes
# ---------------------------------------------------------------------------

CONTRASTIVE_EXAMPLES = [
    # --- Pair 1: Complex with explicit signal (from v5) ---
    {
        "claim": "Actin [Complex] CDK9",
        "evidence": "Actin was found to interact with Cdk9, a catalytic subunit of P-TEFb, in elongation complexes.",
        "verdict": "correct", "confidence": "high",
        "reason": "Evidence says 'in elongation complexes' — Complex relationship supported.",
    },
    {
        "claim": "AKT [Complex] CASP3",
        "evidence": "Akt and caspase-3 expression interact to regulate proliferation and apoptosis.",
        "verdict": "incorrect", "confidence": "high",
        "reason": "Text says 'interact' metaphorically about signaling pathways, without complex formation.",
    },

    # --- Pair 2: Activity vs amount (NEW — same entities, opposite verdicts) ---
    {
        "claim": "TGFB1 [Activation] ADAM17",
        "evidence": "TGF-beta1 induced a rapid activation of the tumour necrosis factor-alpha-converting enzyme (TACE and ADAM (a disintegrin and metalloprotease) 17).",
        "verdict": "correct", "confidence": "high",
        "reason": "Text describes functional activation of the ADAM17 enzyme — activity state change.",
    },
    {
        "claim": "TGFB1 [Activation] ADAM17",
        "evidence": "Furthermore, ADAM17 mRNA and protein expression were up-regulated by TGF-beta1.",
        "verdict": "incorrect", "confidence": "high",
        "reason": "Text describes mRNA/protein expression increase — that is IncreaseAmount, not Activation.",
    },

    # --- Pair 3: Logical inversion (from v5) ---
    {
        "claim": "AGER [Activation] MMP2",
        "evidence": "RAGE blockade reduced MMP-2 activity to control level.",
        "verdict": "correct", "confidence": "high",
        "reason": "Logical inversion: blocking RAGE reduces MMP-2 activity, so RAGE activates MMP-2.",
    },
    {
        "claim": "TP53 [Inhibition] MDM2",
        "evidence": "TP53 knockdown increased MDM2 protein levels in these cells.",
        "verdict": "correct", "confidence": "high",
        "reason": "Logical inversion: knockdown of TP53 increases MDM2, so TP53 normally decreases MDM2.",
    },

    # --- Pair 4: Hedging scope — same word "could", opposite scope ---
    {
        "claim": "MYB [Complex] PPID",
        "evidence": "However, we found that the cyclophilin Cyp-40 could interact with c-Myb to inhibit its DNA binding activity.",
        "verdict": "correct", "confidence": "high",
        "reason": "'we found that...could interact' reports a discovered result. 'Could' scopes over the consequence (inhibiting DNA binding), not the interaction itself.",
    },
    {
        "claim": "MYB [Complex] PPID",
        "evidence": "A binding assay was used to test whether c-Myb and Cyp-40 could interact directly with one another in vitro.",
        "verdict": "incorrect", "confidence": "high",
        "reason": "'to test whether...could interact' is an experimental question. 'Could' scopes over the relationship itself — the interaction is what's being tested, not confirmed.",
    },

    # --- Pair 5: Discourse/co-occurrence trap (NEW — activation verb + negation) ---
    {
        "claim": "IFNA [Activation] NFkappaB",
        "evidence": "As illustrated in Fig. 2 A, IFN-α activated NF-κB in a time-dependent manner reaching threefold increase at 120 min.",
        "verdict": "correct", "confidence": "high",
        "reason": "Direct: 'IFN-α activated NF-κB' — subject directly acts on object, with quantitative result.",
    },
    {
        "claim": "TNFSF10 [Activation] CASP8",
        "evidence": "These findings suggest that TRAIL activates a pathway dependent on Bid, but largely independent of FADD and caspase-8, in U2OS cells.",
        "verdict": "incorrect", "confidence": "medium",
        "reason": "'independent of caspase-8' negates the TRAIL→CASP8 link despite co-occurrence with 'activates'.",
    },

    # --- Pair 6: Modification site verification (NEW — enriched claim) ---
    {
        "claim": "AURKB [Phosphorylation] ATXN10 @S12",
        "evidence": "Our findings suggest that Aurora B phosphorylates Ataxin-10 at S12, which colocalizes with the midbody.",
        "verdict": "correct", "confidence": "high",
        "reason": "Claim says @S12, evidence says 'at S12' — site matches.",
    },
    {
        "claim": "AURKB [Phosphorylation] ATXN10 @S77",
        "evidence": "Our findings suggest that Aurora B phosphorylates Ataxin-10 at S12, which colocalizes with the midbody.",
        "verdict": "incorrect", "confidence": "high",
        "reason": "Claim says @S77 but evidence says S12 — wrong modification site.",
    },

    # --- Pair 7: Direct statement vs indirect chain (from v5) ---
    {
        "claim": "MTOR [Activation] RPS6KB1",
        "evidence": "mTOR phosphorylates and activates S6K1, leading to increased ribosomal biogenesis.",
        "verdict": "correct", "confidence": "high",
        "reason": "Direct activity change: 'mTOR activates S6K1'.",
    },
    {
        "claim": "P70S6K [Activation] RPS6",
        "evidence": "Ghrelin strongly activated mTOR, P70S6K, and S6 in parallel.",
        "verdict": "incorrect", "confidence": "medium",
        "reason": "Text shows ghrelin activating multiple targets in parallel, not P70S6K acting on RPS6.",
    },

    # --- Pair 8: Loss-of-function context — epithet vs evidence (NEW v7) ---
    {
        "claim": "HIF1A [Activation] TP53",
        "evidence": "Although HIF-1alpha activates p53, HIF-2alpha has been recently reported to inhibit p53 and ROS production.",
        "verdict": "correct", "confidence": "high",
        "reason": "Direct statement: 'HIF-1alpha activates p53' — stated as established fact in a concessive clause.",
    },
    {
        "claim": "HIF1A [Activation] TP53",
        "evidence": "Additionally, HIF-1alpha expression can be upregulated due to loss of tumor suppressor genes PTEN and p53.",
        "verdict": "incorrect", "confidence": "high",
        "reason": "'loss of p53' upregulates HIF1A — this means p53 suppresses HIF1A, not that HIF1A activates p53. The relationship is inverted.",
    },

    # --- Pair 9: Degradation-as-mechanism vs direct inhibition (NEW v7) ---
    {
        "claim": "ROBO1 [Inhibition] SRC",
        "evidence": "Supporting our hypothesis, we observed that Slit2 and Robo1 inhibited the M-gp120-induced activation of c-Src, Pyk2, paxillin, Rac1 and CDC42.",
        "verdict": "correct", "confidence": "high",
        "reason": "Direct activity inhibition: Robo1 inhibited activation of c-Src. No degradation involved.",
    },
    {
        "claim": "Proteasome [Inhibition] ESR1",
        "evidence": "Our results indicated that the ubiquitin and proteasome mediated degradation of ERalpha was promoted by CDK11 (p58).",
        "verdict": "incorrect", "confidence": "high",
        "reason": "Text describes degradation (DecreaseAmount), not activity inhibition. Proteasome degrades the protein, it doesn't inhibit its activity directly.",
    },
]


# ---------------------------------------------------------------------------
# Verdict extraction — same as v5
# ---------------------------------------------------------------------------

JSON_VERDICT_PATTERN = re.compile(
    r'\{[^{}]*?"verdict"\s*:\s*"(correct|incorrect)"[^{}]*?"confidence"\s*:\s*"(high|medium|low)"[^{}]*?\}',
    re.IGNORECASE,
)


def _render_example(ex: dict) -> tuple[str, str]:
    user = f"CLAIM: {ex['claim']}\nEVIDENCE: \"{ex['evidence']}\""
    assistant = (
        f"Reason: {ex['reason']}\n"
        f'{{"verdict": "{ex["verdict"]}", "confidence": "{ex["confidence"]}"}}'
    )
    return user, assistant


def extract_verdict(text: str) -> tuple[str | None, str | None]:
    matches = JSON_VERDICT_PATTERN.findall(text)
    if matches:
        v, c = matches[-1]
        return v.lower(), c.lower()
    return None, None


def verdict_to_score(verdict: str | None, confidence: str | None) -> float:
    if verdict is None:
        return 0.5
    grid = {
        ("correct", "high"): 0.95,
        ("correct", "medium"): 0.80,
        ("correct", "low"): 0.65,
        ("incorrect", "low"): 0.35,
        ("incorrect", "medium"): 0.20,
        ("incorrect", "high"): 0.05,
    }
    return grid.get((verdict, confidence or "medium"), 0.50)


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

def _format_extraction_provenance(
    subject: str, obj: str, source_hash: int, evidence_meta: dict,
) -> str:
    """Format NLP extraction provenance from raw_text metadata.

    Shows what the NLP system actually extracted, only when it differs
    from the claim entities.
    """
    meta = evidence_meta.get(int(source_hash))
    if meta is None:
        return ""
    raw_text = meta.get("raw_text")
    if not raw_text or not isinstance(raw_text, list):
        return ""
    raw_text = [r for r in raw_text if r is not None]
    if not raw_text:
        return ""

    # Only show when raw_text differs from claim entities
    claim_set = {subject.lower(), obj.lower()}
    raw_set = {r.lower() for r in raw_text[:2]}
    if claim_set == raw_set:
        return ""

    # Format: show what was extracted and what it was mapped to
    parts = []
    claim_entities = [subject, obj]
    for i, rt in enumerate(raw_text[:2]):
        if i >= len(claim_entities):
            break
        ce = claim_entities[i]
        if rt.lower() != ce.lower():
            parts.append(f'"{rt}" → {ce}')
    if not parts:
        return ""
    return "NLP extracted: " + ", ".join(parts)


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
    """Score a single record with agentic grounding verification.

    The model sees NLP extraction provenance and can call lookup_gene()
    to verify entity mappings it finds suspicious.

    Returns dict with: score, verdict, confidence, raw_text, tokens, tool_calls.
    """
    # Layer 1: Enriched claim (residue/position, mutations)
    if source_hash is not None and corpus_index is not None:
        claim = enrich_claim(subject, stmt_type, obj, source_hash, corpus_index)
    else:
        claim = f"{subject} [{stmt_type}] {obj}"

    # Layer 2: NLP extraction provenance (raw_text)
    extraction_prov = ""
    if source_hash is not None and evidence_meta is not None:
        extraction_prov = _format_extraction_provenance(
            subject, obj, source_hash, evidence_meta
        )

    # Layer 3: Entity alias context
    entity_ctx = format_entity_context(subject, obj)

    # Layer 4: Evidence directness
    ev_prefix = ""
    if source_hash is not None and evidence_meta is not None:
        direct = get_evidence_directness(source_hash, evidence_meta)
        if direct is False:
            ev_prefix = "[indirect evidence] "

    # Build user message — Pass 1 is pure text comprehension (no extraction provenance)
    parts = [f"CLAIM: {claim}"]
    if entity_ctx:
        parts.append(entity_ctx)
    parts.append(f'EVIDENCE: "{ev_prefix}{evidence_text}"')
    user_msg = "\n".join(parts)

    # Build messages with contrastive examples
    messages = []
    for ex in CONTRASTIVE_EXAMPLES:
        u, a = _render_example(ex)
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": user_msg})

    try:
        # PASS 1: Standard scoring with few-shot examples (same as v8)
        response = client.call(
            system=SYSTEM_PROMPT,
            messages=messages,
            max_tokens=max_tokens,
        )
        verdict, confidence = extract_verdict(response.raw_text)
        pass1_tokens = response.tokens

        # PASS 2: If extraction provenance shows mismatch AND pass 1 says "correct",
        # run a focused grounding verification with tool calling (no few-shot).
        tool_calls_made = 0
        # Only trigger pass 2 if extraction provenance shows a mismatch
        # that ISN'T already explained by the alias context
        needs_verification = False
        if extraction_prov and verdict == "correct":
            aliases_text = entity_ctx.lower() if entity_ctx else ""
            # Check if each raw_text entity is already in aliases
            meta = evidence_meta.get(int(source_hash), {}) if evidence_meta else {}
            raw_text = [r for r in (meta.get("raw_text") or []) if r]
            for rt in raw_text[:2]:
                rt_low = rt.lower()
                if rt_low not in aliases_text and rt_low not in {subject.lower(), obj.lower()}:
                    needs_verification = True
                    break

        if needs_verification:
            grounding_prompt = (
                "You verify NLP gene name mappings. You have a tool:\n"
                "lookup_gene(name) — returns candidate identities with "
                "descriptions and alias info.\n"
                "To call it: TOOL_CALL: lookup_gene(\"name\")\n\n"
                "Key signals from the tool:\n"
                "- \"is NOT a known alias\" → the mapping is likely wrong\n"
                "- If the top candidate is a different gene than claimed → wrong\n"
                "- [PSEUDOGENE] → not a functional protein\n"
                "- (chemical/MeSH) → not a gene at all\n"
                "- If the tool CONFIRMS the alias → accept the mapping\n\n"
                "Output: {\"verdict\": \"incorrect\"} or {\"verdict\": \"correct\"}"
            )
            verify_msg = (
                f"CLAIM: {claim}\n"
                f"{extraction_prov}\n"
                f"Verify the NLP mapping."
            )
            verify_response = client.call_with_tools(
                system=grounding_prompt,
                messages=[{"role": "user", "content": verify_msg}],
                tools=TOOLS,
                tool_declarations="",
                max_tokens=2000,
                max_tool_rounds=5,
                temperature=0.1,
            )
            tool_calls_made = verify_response.raw_text.count("[Tool:")
            verify_verdict, verify_conf = extract_verdict(verify_response.raw_text)
            if verify_verdict == "incorrect":
                verdict = "incorrect"
                confidence = verify_conf or "medium"
                response = ModelResponse(
                    content=verify_response.content,
                    reasoning=response.raw_text,  # keep pass 1 as context
                    tokens=pass1_tokens + verify_response.tokens,
                    raw_text=response.raw_text + "\n[GROUNDING VERIFICATION]\n" + verify_response.raw_text,
                    finish_reason="stop",
                )

        return {
            "score": verdict_to_score(verdict, confidence),
            "verdict": verdict,
            "confidence": confidence,
            "raw_text": response.raw_text,
            "tokens": response.tokens,
            "tool_calls": tool_calls_made,
        }
    except Exception as e:
        return {
            "score": 0.5,
            "verdict": None,
            "confidence": None,
            "raw_text": f"error: {e}",
            "tokens": 0,
        }


# ---------------------------------------------------------------------------
# Main: run v6 on holdout set
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="v9 LLM scorer — agentic grounding verification")
    parser.add_argument("--model", default="gemma-moe", help="Model name (default: gemma-moe)")
    parser.add_argument("--holdout", default=str(ROOT / "data" / "benchmark" / "holdout.jsonl"))
    parser.add_argument("--output", default=str(ROOT / "data" / "results" / "v9.jsonl"))
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--limit", type=int, default=None, help="Limit records for testing")
    args = parser.parse_args()

    # Load holdout set
    with open(args.holdout) as f:
        holdout = [json.loads(line) for line in f]
    if args.limit:
        holdout = holdout[:args.limit]

    print(f"v9 scorer: {len(holdout)} records, model={args.model}")
    print(f"  System prompt: {len(SYSTEM_PROMPT)} chars")
    print(f"  Examples: {len(CONTRASTIVE_EXAMPLES)} ({len(CONTRASTIVE_EXAMPLES)//2} pairs)")

    # Build v8 corpus index (statements + evidence metadata)
    print("\nBuilding v8 corpus index...")
    t0 = time.time()
    indexes = build_corpus_index_v8()
    corpus_index = indexes["statements"]
    evidence_meta = indexes["evidence_meta"]
    print(f"  Index built in {time.time()-t0:.1f}s")

    # Initialize model client
    client = ModelClient(args.model)
    print(f"\nScoring with {args.model}...")

    results = []
    correct = 0
    total_parsed = 0
    t_start = time.time()

    # Stream results incrementally
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_fh = open(output_path, "w")

    for i, record in enumerate(holdout):
        subject = record["subject"]
        stmt_type = record["stmt_type"]
        obj = record["object"]
        evidence = record.get("evidence_text", "")
        source_hash = record.get("source_hash")

        result = score_record(
            client=client,
            subject=subject,
            stmt_type=stmt_type,
            obj=obj,
            evidence_text=evidence,
            source_hash=source_hash,
            corpus_index=corpus_index,
            evidence_meta=evidence_meta,
            max_tokens=args.max_tokens,
        )

        # Ground truth
        gt_correct = (record["tag"] == "correct")
        llm_correct = (result["verdict"] == "correct") if result["verdict"] else None

        if llm_correct is not None:
            total_parsed += 1
            if llm_correct == gt_correct:
                correct += 1

        result.update({
            "source_hash": source_hash,
            "pa_hash": record.get("pa_hash"),
            "tag": record["tag"],
            "subject": subject,
            "stmt_type": stmt_type,
            "object": obj,
        })
        results.append(result)

        # Write incrementally (truncate raw_text for space)
        r_save = {k: v for k, v in result.items() if k != "raw_text"}
        r_save["raw_text_preview"] = result.get("raw_text", "")[:500]
        out_fh.write(json.dumps(r_save) + "\n")
        out_fh.flush()

        elapsed = time.time() - t_start
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        acc = correct / total_parsed * 100 if total_parsed > 0 else 0
        mark = "✓" if (llm_correct == gt_correct) else ("✗" if llm_correct is not None else "?")
        print(f"  [{i+1:3d}/{len(holdout)}] {mark} {subject:>10s} [{stmt_type:>18s}] {obj:<10s} "
              f"→ {result['verdict'] or 'PARSE_FAIL':>9s} ({result['confidence'] or '?':>6s}) "
              f"score={result['score']:.2f}  acc={acc:.1f}%  {rate:.2f}/s")

    # Summary
    elapsed = time.time() - t_start
    parse_fail = len(holdout) - total_parsed
    print(f"\n{'='*80}")
    print(f"v9 RESULTS: {correct}/{total_parsed} correct ({correct/max(total_parsed,1)*100:.1f}%) "
          f"| {parse_fail} parse failures | {elapsed:.0f}s total")

    # Per-tag breakdown
    from collections import Counter
    tag_correct = Counter()
    tag_total = Counter()
    tag_fp = Counter()
    tag_fn = Counter()

    for r in results:
        tag = r["tag"]
        gt = (tag == "correct")
        v = r.get("verdict")
        if v is None:
            continue
        llm = (v == "correct")
        tag_total[tag] += 1
        if llm == gt:
            tag_correct[tag] += 1
        elif llm and not gt:
            tag_fp[tag] += 1
        elif not llm and gt:
            tag_fn[tag] += 1

    print(f"\nPer-tag accuracy:")
    for tag in sorted(tag_total.keys()):
        n_c = tag_correct[tag]
        n_t = tag_total[tag]
        n_fp = tag_fp[tag]
        n_fn = tag_fn[tag]
        print(f"  {tag:25s}: {n_c}/{n_t} correct  (FP={n_fp}, FN={n_fn})")

    print(f"\nFP total: {sum(tag_fp.values())}  FN total: {sum(tag_fn.values())}")

    # Close streaming output
    out_fh.close()
    print(f"\nResults streamed to {output_path}")


if __name__ == "__main__":
    main()

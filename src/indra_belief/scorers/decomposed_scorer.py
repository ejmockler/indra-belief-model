"""Decomposed evidence scorer — 3 focused calls instead of 1 monolithic call.

Hypothesis: a 26B model does ONE simple task more reliably than FIVE
things at once. Each call gets a minimal prompt focused on one sub-task.

Call 1 (Entity):       "Does this text mention these entities?"
Call 2 (Relationship): "What relationship does this text describe?" (claim-blind)
Call 3 (Match):        "Does extracted relationship match the claimed type?"

Deterministic combination. No cross-task interference.

Run:
    PYTHONPATH=src python -m indra_belief.scorers.decomposed_scorer
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from indra_belief.data.corpus import CorpusIndex
from indra_belief.data.scoring_record import ScoringRecord
from indra_belief.model_client import ModelClient
from indra_belief.scorers.evidence_scorer import extract_verdict, verdict_to_score

ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Call 1: Entity mention verification
# ---------------------------------------------------------------------------

_ENTITY_SYSTEM = """\
You check whether specific biological entities are mentioned in a sentence.
Answer with JSON: {"subject_found": true/false, "object_found": true/false, "notes": "..."}
"""

def _call_entity(client: ModelClient, record: ScoringRecord) -> dict:
    """Check if the claimed entities are mentioned in the evidence."""
    subj_aliases = ", ".join(record.subject_entity.aliases[:4]) if record.subject_entity else ""
    obj_aliases = ", ".join(record.object_entity.aliases[:4]) if record.object_entity else ""

    subj_hint = f' (also known as: {subj_aliases})' if subj_aliases else ""
    obj_hint = f' (also known as: {obj_aliases})' if obj_aliases else ""

    prompt = (
        f'Subject: {record.subject}{subj_hint}\n'
        f'Object: {record.object}{obj_hint}\n\n'
        f'Evidence: "{record.evidence_text}"\n\n'
        f'Are both entities mentioned or referenced in the evidence text?'
    )

    response = client.call(
        system=_ENTITY_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
    )

    # Parse from content (not raw_text which has reasoning prefix)
    text = (response.content or response.raw_text).lower()
    subj_found = '"subject_found": true' in text or '"subject_found":true' in text
    obj_found = '"object_found": true' in text or '"object_found":true' in text

    # Fallback: check for natural language in full output
    if not subj_found and not obj_found:
        full = response.raw_text.lower()
        subj_found = "both" in full and ("mentioned" in full or "found" in full or "present" in full)
        obj_found = subj_found  # "both" implies both found

    return {
        "subject_found": subj_found,
        "object_found": obj_found,
        "raw": response.raw_text,
        "tokens": response.tokens,
    }


# ---------------------------------------------------------------------------
# Call 2: Claim-blind relationship extraction
# ---------------------------------------------------------------------------

_RELATIONSHIP_SYSTEM = """\
You extract biological relationships from scientific text.
Given a sentence, identify what relationship is described:
- Who/what is the AGENT (the entity causing the effect)?
- Who/what is the TARGET (the entity being affected)?
- What TYPE of relationship? (binding/complex, activation, inhibition, \
phosphorylation, dephosphorylation, increase expression, decrease expression, \
translocation, ubiquitination, or other)
- Is this relationship DIRECTLY stated or hypothetical/uncertain?

Answer with JSON:
{"agent": "...", "target": "...", "type": "...", "direct": true/false, "notes": "..."}
"""

_RELATIONSHIP_EXAMPLES = [
    # Teaches logical inversion
    {
        "role": "user",
        "content": 'Evidence: "RAGE blockade reduced MMP-2 activity to control level."',
    },
    {
        "role": "assistant",
        "content": '{"agent": "RAGE", "target": "MMP-2", "type": "activation", "direct": true, "notes": "Logical inversion: blocking RAGE reduces MMP-2, so RAGE normally activates MMP-2."}',
    },
    # Teaches third-party agent
    {
        "role": "user",
        "content": 'Evidence: "Ghrelin strongly activated mTOR, P70S6K, and S6 in parallel."',
    },
    {
        "role": "assistant",
        "content": '{"agent": "Ghrelin", "target": "mTOR, P70S6K, S6", "type": "activation", "direct": true, "notes": "Ghrelin activates all three targets in parallel — they do not activate each other."}',
    },
    # Teaches act_vs_amt
    {
        "role": "user",
        "content": 'Evidence: "ADAM17 mRNA and protein expression were up-regulated by TGF-beta1."',
    },
    {
        "role": "assistant",
        "content": '{"agent": "TGF-beta1", "target": "ADAM17", "type": "increase expression", "direct": true, "notes": "mRNA/protein expression increase = amount change, not activity change."}',
    },
]

def _call_relationship(client: ModelClient, record: ScoringRecord) -> dict:
    """Extract the relationship from evidence WITHOUT seeing the claim."""
    messages = list(_RELATIONSHIP_EXAMPLES)
    messages.append({
        "role": "user",
        "content": f'Evidence: "{record.evidence_text}"',
    })

    response = client.call(
        system=_RELATIONSHIP_SYSTEM,
        messages=messages,
        max_tokens=2000,  # model needs token budget for reasoning + JSON
    )

    # Find the last complete JSON object with "agent" in the output
    # (model embeds JSON candidates in reasoning before the final answer)
    import re
    text = response.raw_text
    json_objects = re.findall(r'\{[^{}]*"agent"[^{}]*\}', text)

    agent = target = rel_type = ""
    direct = False

    if json_objects:
        # Take the last one (closest to the final answer)
        last_json = json_objects[-1]
        agent = _extract_field(last_json, "agent")
        target = _extract_field(last_json, "target")
        rel_type = _extract_field(last_json, "type")
        direct = '"direct": true' in last_json.lower() or '"direct":true' in last_json.lower()
    else:
        # Fallback: search full text with field extraction
        agent = _extract_field(text, "agent")
        target = _extract_field(text, "target")
        rel_type = _extract_field(text, "type")
        direct = '"direct": true' in text.lower()

    return {
        "agent": agent,
        "target": target,
        "type": rel_type,
        "direct": direct,
        "raw": response.raw_text,
        "tokens": response.tokens,
    }


def _extract_field(text: str, field: str) -> str:
    """Extract a JSON field value from potentially messy LLM output.

    Searches from the END of the text since reasoning comes before the
    JSON answer in raw_text (reasoning_content + content).
    """
    import re
    # Find ALL matches and take the LAST one (closest to the answer)
    # Try: "field": "value"
    pattern = rf'"{field}"\s*:\s*"([^"]*)"'
    matches = list(re.finditer(pattern, text, re.IGNORECASE))
    if matches:
        return matches[-1].group(1)
    # Try: "field": value (without quotes)
    pattern = rf'"{field}"\s*:\s*([^,\}}\]]+)'
    matches = list(re.finditer(pattern, text, re.IGNORECASE))
    if matches:
        return matches[-1].group(1).strip().strip('"')
    return ""


# ---------------------------------------------------------------------------
# Call 3: Type match verification
# ---------------------------------------------------------------------------

_MATCH_SYSTEM = """\
You compare a claimed biological relationship with an extracted one.
The text-mining system claims a specific relationship type. An independent
analysis extracted what the evidence actually describes. Do they match?

INDRA relationship types:
- Activation = activity state change (enzyme activated, kinase activity increased)
- Inhibition = direct activity suppression (NOT expression decrease)
- IncreaseAmount = mRNA/protein level increase, expression upregulation
- DecreaseAmount = mRNA/protein level decrease, degradation
- Phosphorylation / Dephosphorylation = adding/removing phosphate groups
- Complex = physical binding / protein-protein interaction
- Autophosphorylation = protein phosphorylates itself
- Translocation = movement between cellular compartments
- Ubiquitination = adding ubiquitin tags

Answer with JSON: {"verdict": "correct" or "incorrect", "confidence": "high" or "medium" or "low", "reason": "..."}
"""

_MATCH_EXAMPLES = [
    {
        "role": "user",
        "content": (
            'Claimed: TGFB1 [Activation] ADAM17\n'
            'Extracted: agent="TGF-beta1", target="ADAM17", type="increase expression"\n'
            'Do these match?'
        ),
    },
    {
        "role": "assistant",
        "content": '{"verdict": "incorrect", "confidence": "high", "reason": "Claim says Activation (activity change) but extracted relationship is expression increase (IncreaseAmount). Different relationship types."}',
    },
    {
        "role": "user",
        "content": (
            'Claimed: AGER [Activation] MMP2\n'
            'Extracted: agent="RAGE", target="MMP-2", type="activation"\n'
            'Do these match?'
        ),
    },
    {
        "role": "assistant",
        "content": '{"verdict": "correct", "confidence": "high", "reason": "RAGE = AGER (alias), MMP-2 = MMP2. Both describe activation. Match."}',
    },
]


def _call_match(
    client: ModelClient,
    record: ScoringRecord,
    extraction: dict,
) -> dict:
    """Compare the claim with the extracted relationship."""
    messages = list(_MATCH_EXAMPLES)

    # Build the comparison prompt
    agent = extraction.get("agent", "?")
    target = extraction.get("target", "?")
    rel_type = extraction.get("type", "?")

    prompt = (
        f'Claimed: {record.format_claim()}\n'
        f'Extracted: agent="{agent}", target="{target}", type="{rel_type}"\n'
        f'Do these match?'
    )
    messages.append({"role": "user", "content": prompt})

    response = client.call(
        system=_MATCH_SYSTEM,
        messages=messages,
        max_tokens=500,
    )

    verdict, confidence = extract_verdict(response.raw_text)
    return {
        "verdict": verdict,
        "confidence": confidence,
        "raw": response.raw_text,
        "tokens": response.tokens,
    }


# ---------------------------------------------------------------------------
# Combined scorer
# ---------------------------------------------------------------------------

def score(client: ModelClient, record: ScoringRecord) -> dict:
    """Score using the 3-call decomposed pipeline.

    Returns dict with: score, verdict, confidence, raw_text, tokens, tier.
    """
    total_tokens = 0

    # --- Tier 0: Deterministic pre-filter (same as monolithic) ---
    reject = record.tier1_auto_reject()
    if reject:
        return reject

    # --- Call 1: Entity mention check ---
    entity_result = _call_entity(client, record)
    total_tokens += entity_result["tokens"]

    if not entity_result["subject_found"] and not entity_result["object_found"]:
        return {
            "score": 0.05,
            "verdict": "incorrect",
            "confidence": "high",
            "raw_text": f"[CALL 1: ENTITIES NOT FOUND]\n{entity_result['raw']}",
            "tokens": total_tokens,
            "tier": "decomposed_entity",
            "grounding_status": "entities_absent",
            "provenance_triggered": False,
        }

    # --- Call 2: Claim-blind relationship extraction ---
    rel_result = _call_relationship(client, record)
    total_tokens += rel_result["tokens"]

    # --- Call 3: Type match ---
    match_result = _call_match(client, record, rel_result)
    total_tokens += match_result["tokens"]

    verdict = match_result.get("verdict")
    confidence = match_result.get("confidence")

    return {
        "score": verdict_to_score(verdict, confidence),
        "verdict": verdict,
        "confidence": confidence,
        "raw_text": (
            f"[CALL 1: ENTITY] subj={entity_result['subject_found']} obj={entity_result['object_found']}\n"
            f"[CALL 2: EXTRACT] agent={rel_result.get('agent')} target={rel_result.get('target')} "
            f"type={rel_result.get('type')} direct={rel_result.get('direct')}\n"
            f"[CALL 3: MATCH]\n{match_result['raw']}"
        ),
        "tokens": total_tokens,
        "tier": "decomposed",
        "grounding_status": "decomposed",
        "provenance_triggered": False,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Decomposed 3-call evidence scorer")
    parser.add_argument("--model", default="gemma-remote")
    parser.add_argument("--holdout", default=str(ROOT / "data" / "benchmark" / "holdout_large.jsonl"))
    parser.add_argument("--output", default=str(ROOT / "data" / "results" / "decomposed_500.jsonl"))
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    index = CorpusIndex()
    records = index.build_records(args.holdout)
    if args.limit:
        records = records[:args.limit]

    print(f"\nDecomposed scorer: {len(records)} records, model={args.model}")

    client = ModelClient(args.model)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_fh = open(output_path, "w")

    correct = 0
    total_parsed = 0
    t_start = time.time()

    for i, record in enumerate(records):
        result = score(client, record)

        gt_correct = record.tag == "correct"
        llm_correct = (result["verdict"] == "correct") if result["verdict"] else None

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
        r_save["raw_text_preview"] = result.get("raw_text", "")[:500]
        out_fh.write(json.dumps(r_save) + "\n")
        out_fh.flush()

        acc = correct / total_parsed * 100 if total_parsed > 0 else 0
        mark = "✓" if (llm_correct == gt_correct) else ("✗" if llm_correct is not None else "?")
        tier_short = result.get("tier", "?")[:12]
        print(f"  [{i+1:3d}/{len(records)}] {mark} {record.subject:>10s} [{record.stmt_type:>15s}] {record.object:10s} "
              f"→ {result['verdict'] or 'PARSE':>9s} [{tier_short:12s}] acc={acc:.1f}%")

    out_fh.close()

    print(f"\n{'='*70}")
    print(f"DECOMPOSED: {correct}/{total_parsed} = {correct/max(total_parsed,1)*100:.1f}%")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()

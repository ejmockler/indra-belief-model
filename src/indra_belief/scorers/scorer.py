"""Evidence quality scorer using native INDRA objects.

Two-tier architecture:
  Tier 1: Deterministic grounding check (GroundedEntity.should_auto_reject)
    - MISMATCH → auto-reject
    - PSEUDOGENE + AMBIGUOUS → auto-reject
    - AMBIGUOUS → LLM judges with grounding context
  Tier 2: LLM text comprehension
    - base system prompt + adaptive contrastive examples
    - Entity context from ScoringRecord

Run:
    PYTHONPATH=src python -m indra_belief.scorers.scorer
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from indra_belief.data.scoring_record import ScoringRecord
from indra_belief.model_client import ModelClient

# CorpusIndex is imported lazily in main() — the benchmark harness is the
# only consumer. Keeping it out of the module-level import chain means
# `from indra_belief import ModelResponse` (for typing) doesn't pay the
# cost of pulling in the INDRA corpus index.

from indra_belief.scorers._prompts import (
    SYSTEM_PROMPT,
    CONTRASTIVE_EXAMPLES as _ALL_EXAMPLES,
    render_example as _render_example,
    extract_verdict,
    verdict_to_score,
)

ROOT = Path(__file__).resolve().parents[3]

# Note: provenance is injected only when grounding is flagged. Full-
# population provenance regresses accuracy — attention dilution from the
# extra context outweighs the disambiguation benefit on records where
# grounding is unambiguous. See scoring_record.format_user_message.

# --- Adaptive few-shot selection ---
# Seven contrastive pairs (14 examples) per record — a balance between
# example coverage and leaving prompt budget for the model's own
# reasoning. More examples dilute attention; fewer lose type coverage.

# Type-specific example bank (loaded from JSON)
# Bank keys can be exact types ("Activation") or sub-keys ("Activation_no_relation")
_EXAMPLE_BANK_PATH = Path(__file__).parent.parent / "data" / "example_bank.json"
_RAW_BANK: dict[str, list[dict]] = {}
if _EXAMPLE_BANK_PATH.exists():
    with open(_EXAMPLE_BANK_PATH) as _f:
        _RAW_BANK = json.load(_f)

# Build type → list of pairs mapping from bank
# Keys like "Activation_no_relation" contribute to "Activation"
# Known INDRA statement types (any legal base_type must be one of these).
# Sub-keys use the pattern "{StmtType}_{errorPattern}" — e.g. "Activation_family".
_KNOWN_TYPES = {
    "Activation", "Inhibition", "Phosphorylation", "Dephosphorylation",
    "Autophosphorylation", "Acetylation", "Deacetylation", "Methylation",
    "Demethylation", "Ubiquitination", "Deubiquitination", "Translocation",
    "Complex", "IncreaseAmount", "DecreaseAmount", "Conversion",
    "GtpActivation", "Gef", "Gap",
}

_TYPE_BANK: dict[str, list[list[dict]]] = {}
for key, pair in _RAW_BANK.items():
    # Match the longest known type that the key starts with (handles
    # IncreaseAmount_foo → IncreaseAmount, Activation_family → Activation).
    base_type = next(
        (t for t in sorted(_KNOWN_TYPES, key=len, reverse=True)
         if key == t or key.startswith(t + "_")),
        key,  # fallback: unrecognized key routes to itself
    )
    _TYPE_BANK.setdefault(base_type, []).append(pair)

# Map base examples into pairs by their statement type
_BASE_PAIRS: dict[str, list[list[dict]]] = {}
for i in range(0, len(_ALL_EXAMPLES), 2):
    stype = _ALL_EXAMPLES[i]["claim"].split("[")[1].split("]")[0].strip()
    _BASE_PAIRS.setdefault(stype, []).append([_ALL_EXAMPLES[i], _ALL_EXAMPLES[i + 1]])

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
    1. Own type pair(s) — from bank (may have multiple sub-keys) and/or base
    2. Adjacent type pairs — types commonly confused with this one
    3. Universal patterns — logical inversion, hedging scope
    4. Fill from remaining base pairs
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

    # 1b. Own type from base
    for pair in _BASE_PAIRS.get(stmt_type, []):
        _add_pair(pair)

    # 2. Adjacent types
    for adj_type in _TYPE_ADJACENCY.get(stmt_type, []):
        for pair in _TYPE_BANK.get(adj_type, []):
            _add_pair(pair)
        for pair in _BASE_PAIRS.get(adj_type, []):
            _add_pair(pair)

    # 3. Universal patterns
    for pair in _UNIVERSAL_PAIRS:
        _add_pair(pair)

    # 4. Fill remaining from base pairs
    for i in range(0, len(_ALL_EXAMPLES), 2):
        _add_pair([_ALL_EXAMPLES[i], _ALL_EXAMPLES[i + 1]])

    # Flatten pairs into example list
    examples = []
    for pair in selected[:TARGET_PAIRS]:
        examples.extend(pair)
    return examples


def _build_messages(record: ScoringRecord) -> list[dict]:
    """Build the contrastive-example + user-message conversation for a record."""
    examples = _select_examples(record.stmt_type)
    messages: list[dict] = []
    for ex in examples:
        u, a = _render_example(ex)
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": record.format_user_message()})
    return messages


def _parse_verdict(response) -> tuple[str | None, str | None]:
    """Extract verdict from a model response.

    Strategy:
      1. Try parsing `response.content` (final assistant message). For
         separate-reasoning models (Gemma-4), this is CoT-free — guards
         against hypothetical JSON in reasoning being picked up by
         extract_verdict's last-match logic.
      2. If content is empty OR yields no verdict, fall back to `raw_text`
         (reasoning + content joined). Recovers truncated responses where
         the verdict is in reasoning and final content is incomplete.

    The two-step fallback is load-bearing: truncation at max_tokens
    (finish_reason="length") can produce a non-empty content that lacks
    the JSON verdict. Without the fallback, such records silently
    collapse to (None, None) → score 0.5.
    """
    if response.content:
        verdict, confidence = extract_verdict(response.content)
        if verdict is not None:
            return verdict, confidence
    # Fall back to full raw_text (includes reasoning).
    return extract_verdict(response.raw_text)


def _score_single(
    client: ModelClient,
    record: ScoringRecord,
    max_tokens: int,
    temperature: float = 0.1,
) -> dict:
    """Single LLM call for Tier 2. Returns result dict."""
    response = client.call(
        system=SYSTEM_PROMPT,
        messages=_build_messages(record),
        max_tokens=max_tokens,
        temperature=temperature,
    )
    verdict, confidence = _parse_verdict(response)
    return {
        "verdict": verdict,
        "confidence": confidence,
        "raw_text": response.raw_text,
        "tokens": response.tokens,
    }


_LOOKUP_GUIDANCE = """

EXTERNAL LOOKUP CONTEXT — an "Entity database lookups:" block is included
below for claim entities that may be ambiguous. It shows the top Gilda
candidates for each entity — a proper gene, a chemical, a protein family,
a MeSH concept, or a pseudogene. Use this to ground the evidence correctly.

How to read it:
- If the top candidate is a gene family (FPLX) and the evidence mentions
  a specific member, that is still a valid claim per rule 2.
- If the top candidate is a chemical, lipid (CHEBI), or MeSH concept, the
  "gene" in the claim may be a conflation with a non-protein entity. If
  the evidence describes actions on that non-gene entity (e.g. a lipid
  receptor extracting as the lipid itself), treat it as a grounding error.
- If the top candidate is a pseudogene, the claim is likely wrong unless
  the evidence explicitly describes pseudogene transcripts.
- If "query is a known alias" is False, and the top candidate is a
  different gene from the claim, this is a grounding error.

Use the lookups to refine your verdict. Do NOT emit TOOL_CALL — the
lookups are already done for you.
"""


def _format_entity_lookups(record: ScoringRecord) -> str:
    """Pre-compute gilda lookups for ambiguous entities. Returns a
    formatted block suitable for prompt injection, or "" when neither
    entity benefits from lookup.

    Looks up `raw_text` (the ambiguous mention the reader extracted), not
    `name` (the already-resolved canonical symbol). Looking up the canonical
    just confirms Gilda's existing decision; the raw_text is what actually
    needs disambiguation.
    """
    import logging
    from indra_belief.tools.gilda_tools import lookup_gene_executor

    log = logging.getLogger(__name__)
    lines: list[str] = []
    seen: set[str] = set()
    for entity in (record.subject_entity, record.object_entity):
        if not entity or not entity.name or entity.name == "?":
            continue
        # Prefer raw_text (the ambiguous mention that triggered the flag).
        # Fall back to name when raw_text is missing or identical.
        lookup_target = entity.raw_text or entity.name
        # Avoid duplicate lookups (autophosphorylation, same text twice)
        if lookup_target in seen:
            continue
        seen.add(lookup_target)
        try:
            result = lookup_gene_executor({"entity_name": lookup_target})
        except Exception as e:
            log.warning("lookup_gene failed for %r: %s", lookup_target, e)
            continue
        lines.append(result)
    if not lines:
        return ""
    return "Entity database lookups:\n" + "\n".join(lines)


def _score_with_tools(
    client: ModelClient,
    record: ScoringRecord,
    max_tokens: int,
) -> dict:
    """Tier 2 with pre-computed entity lookups. For records where grounding
    is flagged or entity symbols are short/ambiguous, gilda lookups are
    executed deterministically and injected into the prompt. The model
    does not need to decide whether to call the tool — the external
    signal is always present.
    """
    lookup_ctx = _format_entity_lookups(record)
    messages = _build_messages(record)
    if lookup_ctx:
        # Augment the user message (last message) with the lookup block.
        augmented = messages[-1]["content"] + "\n\n" + lookup_ctx
        messages[-1] = {"role": "user", "content": augmented}

    response = client.call(
        system=SYSTEM_PROMPT + _LOOKUP_GUIDANCE,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.1,
    )
    verdict, confidence = _parse_verdict(response)
    return {
        "verdict": verdict,
        "confidence": confidence,
        "raw_text": response.raw_text,
        "tokens": response.tokens,
    }


def score(
    client: ModelClient,
    record: ScoringRecord,
    max_tokens: int = 12000,
    voting_k: int = 3,
) -> dict:
    """Score a single extraction.

    Args:
        voting_k: Self-consistency voting — number of independent samples at
            temperature=0.6. k=1 is a single deterministic call (temp=0.1).
            k=3 (default) runs up to 3 samples with early-stop after a
            majority is reached. k=5 for hard cases. Must be odd; even k
            is rejected because it cannot produce strict majorities.

    Under voting_k >= 3 the confidence is derived from agreement:
      - all votes agree (k/k) → high
      - bare majority (e.g. 2/3, 3/5) → medium
      - "low" is effectively unreachable from voting; it appears only
        when single-call (k=1) LLM self-rates low.

    Returns dict with: score, verdict, confidence, raw_text, tokens,
    tier, grounding_status, provenance_triggered.
    """
    if voting_k != 1 and voting_k % 2 == 0:
        raise ValueError(
            f"voting_k must be 1 or odd (got {voting_k}). "
            "Even values cannot produce strict majorities and degenerate "
            "into longer runs of k=1 with higher temperature."
        )
    # --- Tier 1: Deterministic auto-reject ---
    reject = record.tier1_auto_reject()
    if reject:
        return reject

    # AMBIGUOUS entities go directly to Tier 2 — the intermediate AMBIGUOUS
    # LLM was 64% accurate at scale (barely better than coin flip) and added
    # an extra LLM call that primed the model with grounding-focused evaluation
    # before the comprehension evaluation.

    provenance_triggered = bool(record.format_provenance())

    # Determine grounding status now — it picks the Tier-2 path.
    flagged = any(
        e.has_grounding_signal
        for e in (record.subject_entity, record.object_entity)
        if e
    )
    # Pre-computed-lookups only fire for flagged grounding. The short-symbol
    # soft-flag was considered but rejected: Gilda is the same oracle that
    # blessed all_match records, so its lookups return the same misranking
    # that caused the FP. Tool-use can only help where Gilda already
    # flagged a mismatch or ambiguity.
    needs_tool_use = flagged
    grounding_status = "flagged" if flagged else "all_match"

    # --- Tier 2: LLM text comprehension ---
    # Ambiguous-grounding records go through tool-use (single deterministic
    # call with gilda lookup available); external signal substitutes for
    # voting's stochastic diversity. Clean records use voting as configured.
    if needs_tool_use:
        result = _score_with_tools(client, record, max_tokens)
        verdict = result["verdict"]
        confidence = result["confidence"]
        total_tokens = result["tokens"]
        raw = f"[TIER 2 TOOL-USE]\n{result['raw_text']}"
        tier = "llm_tool_use"
    elif voting_k <= 1:
        result = _score_single(client, record, max_tokens)
        verdict = result["verdict"]
        confidence = result["confidence"]
        total_tokens = result["tokens"]
        raw = f"[TIER 2 LLM]\n{result['raw_text']}"
        tier = "llm_comprehension"
    else:
        # Self-consistency voting with early-stop: stop once a majority is
        # guaranteed (e.g., 2/3 agreement after sample 2 wins; sample 3 cannot
        # overturn it).
        majority = voting_k // 2 + 1
        votes: list[str] = []
        total_tokens = 0
        raw_parts = []
        samples_taken = 0
        for i in range(voting_k):
            r = _score_single(client, record, max_tokens, temperature=0.6)
            total_tokens += r["tokens"]
            samples_taken += 1
            raw_parts.append(f"[VOTE {i+1}] {r['verdict']}({r['confidence']})")
            if r["verdict"]:
                votes.append(r["verdict"])
            correct_votes = sum(1 for v in votes if v == "correct")
            incorrect_votes = sum(1 for v in votes if v == "incorrect")
            if correct_votes >= majority or incorrect_votes >= majority:
                break  # majority decided; remaining samples can't overturn

        correct_votes = sum(1 for v in votes if v == "correct")
        incorrect_votes = sum(1 for v in votes if v == "incorrect")
        total_votes = correct_votes + incorrect_votes

        if correct_votes > incorrect_votes:
            verdict = "correct"
        elif incorrect_votes > correct_votes:
            verdict = "incorrect"
        elif total_votes > 0:
            # True tie (only possible if some votes failed to parse).
            # Fall back to None+low rather than order-dependent pick.
            verdict = None
        else:
            verdict = None

        if total_votes > 0 and verdict is not None:
            # Confidence is DERIVED from the voting trajectory. Under early-
            # stop (break at majority), two possible outcomes:
            #
            #   - samples_taken < voting_k: first N samples were unanimous
            #     and the loop broke once a majority was reached. This is
            #     the strongest observable signal from voting — both
            #     independent samples agreed on first try. Map to "high".
            #
            #   - samples_taken == voting_k: the first two samples disagreed
            #     and a third-sample tiebreaker produced a majority. This
            #     is weaker — the model waffled. Map to "medium".
            #
            # "high" via full-k unanimous is mathematically unreachable
            # under early-stop (loop would have broken earlier). Expressing
            # confidence via WHEN the loop stopped aligns calibration with
            # the observed evidence, not with a never-reached ideal.
            agreement = max(correct_votes, incorrect_votes) / total_votes
            early_stop_unanimous = (
                samples_taken < voting_k and agreement == 1.0
            )
            if early_stop_unanimous:
                confidence = "high"
            elif agreement >= 0.6:
                confidence = "medium"
            else:
                confidence = "low"
        else:
            confidence = None

        raw = " | ".join(raw_parts)
        tier = f"llm_voting_k{voting_k}_n{samples_taken}"

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


def score_evidence(
    statement,
    evidence,
    client: ModelClient,
    *,
    voting_k: int = 3,
    max_tokens: int = 12000,
) -> dict:
    """Score one (Statement, Evidence) pair — the atomic per-sentence call.

    This is the per-sentence comprehension layer: a single LLM judgment of
    whether `evidence.text` supports the claim expressed by `statement`.
    For scoring a whole Statement (which carries `statement.evidence` as a
    list), use `score_statement`, which iterates this function.

    Args:
        statement: An `indra.statements.Statement`. Binary types
            (Phosphorylation, Activation, …), SelfModification, Complex
            (any arity), and Translocation are rendered correctly.
        evidence: An `indra.statements.Evidence`. Tier-1 grounding
            verification fires only when `evidence.annotations["agents"]
            ["raw_text"]` is populated (i.e., the evidence came from an
            NLP reader). For hand-built Evidence, verification is skipped
            and scoring is driven entirely by the LLM tier.
        client: A `ModelClient` configured for the chosen backend.
        voting_k: Self-consistency voting samples (odd, or 1). Default 3.
        max_tokens: Per-generation token limit. Default 12000.

    Returns a dict with keys:
        score            float in [0, 1]; 0.95=correct/high … 0.05=incorrect/high.
                         Returns 0.5 when verdict cannot be parsed.
        verdict          "correct" | "incorrect" | None (parse failure)
        confidence       "high" | "medium" | "low" | None
        tier             which scoring path produced the verdict
        grounding_status "all_match" | "flagged"
        provenance_triggered bool
        tokens           completion tokens consumed
        raw_text         decision trace (for debugging)

    Callers must handle `verdict is None` explicitly; it denotes a parse
    failure or a voting tie, not a neutral judgement.
    """
    record = ScoringRecord(statement=statement, evidence=evidence)
    return score(client, record, max_tokens=max_tokens, voting_k=voting_k)


def score_statement(
    statement,
    client: ModelClient,
    *,
    voting_k: int = 3,
    max_tokens: int = 12000,
) -> list[dict]:
    """Score an INDRA Statement by scoring each of its evidence sentences.

    Mirrors INDRA's abstraction: a `Statement` bundles `statement.evidence`
    as a list of `Evidence` objects, and a Statement-level belief is
    composed from per-sentence judgments. Returns one scoring dict per
    evidence, in the same order as `statement.evidence`. Returns `[]` if
    the statement has no evidence.

    Pair with `indra_belief.composed_scorer.ComposedBeliefScorer` to lift
    the per-sentence verdicts into an edge-level belief:

        >>> verdicts = score_statement(stmt, client)
        >>> records = [
        ...     EvidenceRecord(source_api=ev.source_api, verdict=v["verdict"])
        ...     for ev, v in zip(stmt.evidence, verdicts)
        ... ]
        >>> belief = ComposedBeliefScorer().score_edge(records).belief

    To score only one sentence of a Statement — skipping the rest of
    `stmt.evidence` — call `score_evidence(stmt, ev, client)` directly.
    """
    evidences = list(statement.evidence or [])
    return [
        score_evidence(
            statement, ev, client,
            voting_k=voting_k, max_tokens=max_tokens,
        )
        for ev in evidences
    ]


def main():
    import argparse
    from indra_belief.data.corpus import CorpusIndex

    parser = argparse.ArgumentParser(description="Evidence quality scorer (INDRA native)")
    parser.add_argument("--model", default="gemma-remote")
    parser.add_argument("--holdout", default=str(ROOT / "data" / "benchmark" / "holdout.jsonl"))
    parser.add_argument("--output", default=str(ROOT / "data" / "results" / "scorer_output.jsonl"))
    parser.add_argument("--max-tokens", type=int, default=12000)
    parser.add_argument("--voting-k", type=int, default=3,
                        help="Self-consistency voting: k samples with early-stop "
                             "(1=single deterministic call at temp=0.1; 3=default production)")
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

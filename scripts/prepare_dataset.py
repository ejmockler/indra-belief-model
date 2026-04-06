"""Join INDRA benchmark corpus with curations to produce a clean evaluation dataset.

Reads:
  data/benchmark/indra_benchmark_corpus.json.gz  (statements + evidence)
  data/benchmark/indra_assembly_curations.json    (labels)

Writes:
  data/benchmark/belief_benchmark.jsonl           (one record per curated mention)

Join key: evidence-level source_hash (perfect match: 5685/5685).
Note: curations pa_hash does NOT match corpus matches_hash.

Each output record contains:
  - pa_hash: from curations (statement-level identifier in INDRA DB)
  - source_hash: evidence-level identifier (join key)
  - stmt_type: e.g. "Activation", "Phosphorylation"
  - subject, object: agent names
  - evidence_text: the source sentence
  - source_api: e.g. "reach", "sparser"
  - source_counts: {source: count} for the full statement
  - evidence_count: total evidence for this statement
  - belief: INDRA's pre-computed belief score
  - tag: curation label (correct, no_relation, grounding, polarity, ...)
  - curator: who curated this mention
"""
from __future__ import annotations

import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = ROOT / "data" / "benchmark"
CORPUS_PATH = BENCHMARK_DIR / "indra_benchmark_corpus.json.gz"
CURATIONS_PATH = BENCHMARK_DIR / "indra_assembly_curations.json"
OUTPUT_PATH = BENCHMARK_DIR / "belief_benchmark.jsonl"


def extract_agents(stmt: dict) -> tuple[str, str]:
    """Extract subject and object names from a statement JSON."""
    # Different INDRA statement types use different agent keys
    stmt_type = stmt.get("type", "")

    # Modification types: enz + sub
    if "enz" in stmt and "sub" in stmt:
        enz = stmt["enz"] or {}
        sub = stmt["sub"] or {}
        return enz.get("name", "?"), sub.get("name", "?")

    # Regulation/Amount types: subj + obj
    if "subj" in stmt and "obj" in stmt:
        subj = stmt["subj"] or {}
        obj = stmt["obj"] or {}
        return subj.get("name", "?"), obj.get("name", "?")

    # Complex: members
    if "members" in stmt:
        members = stmt["members"]
        if len(members) >= 2:
            return members[0].get("name", "?"), members[1].get("name", "?")
        elif len(members) == 1:
            return members[0].get("name", "?"), "?"

    return "?", "?"


def compute_source_counts(evidences: list[dict]) -> dict[str, int]:
    """Compute per-source evidence counts."""
    counts: dict[str, int] = defaultdict(int)
    for ev in evidences:
        src = ev.get("source_api", "unknown")
        counts[src] += 1
    return dict(counts)


def main():
    # Load curations
    print("Loading curations...")
    with open(CURATIONS_PATH) as f:
        curations = json.load(f)

    # Index curations by source_hash
    cur_by_source_hash: dict[int, list[dict]] = defaultdict(list)
    for c in curations:
        cur_by_source_hash[c["source_hash"]].append(c)

    target_source_hashes = set(cur_by_source_hash.keys())
    print(f"  {len(curations)} curations, {len(target_source_hashes)} unique source_hashes")

    # Load corpus and match
    print(f"Loading corpus from {CORPUS_PATH}...")
    with gzip.open(CORPUS_PATH, "rt") as f:
        corpus = json.load(f)
    print(f"  {len(corpus)} statements loaded")

    # Index corpus evidence by source_hash
    print("Matching evidence to curations...")
    records = []
    matched_hashes = set()

    for stmt in corpus:
        stmt_type = stmt.get("type", "Unknown")
        belief = stmt.get("belief")
        matches_hash = stmt.get("matches_hash")
        evidences = stmt.get("evidence", [])
        source_counts = compute_source_counts(evidences)
        evidence_count = len(evidences)
        subj, obj = extract_agents(stmt)

        for ev in evidences:
            sh = ev.get("source_hash")
            if sh is None or sh not in target_source_hashes:
                continue

            matched_hashes.add(sh)
            ev_text = ev.get("text")
            ev_source = ev.get("source_api", "unknown")
            ev_pmid = ev.get("pmid") or ev.get("text_refs", {}).get("PMID")

            for cur in cur_by_source_hash[sh]:
                records.append({
                    "pa_hash": cur["pa_hash"],
                    "source_hash": sh,
                    "matches_hash": matches_hash,
                    "stmt_type": stmt_type,
                    "subject": subj,
                    "object": obj,
                    "evidence_text": ev_text,
                    "source_api": ev_source,
                    "pmid": ev_pmid,
                    "source_counts": source_counts,
                    "evidence_count": evidence_count,
                    "belief": belief,
                    "tag": cur["tag"],
                    "curator": cur["curator"],
                    "curator_note": cur.get("text"),
                })

    # Free corpus memory
    del corpus

    print(f"\nResults:")
    print(f"  Source hashes matched: {len(matched_hashes)}/{len(target_source_hashes)}")
    print(f"  Output records: {len(records)}")

    # Write output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"  Written to {OUTPUT_PATH}")

    # Sanity stats
    tag_counts = Counter(r["tag"] for r in records)
    print(f"\nTag distribution:")
    for tag, count in tag_counts.most_common():
        print(f"  {tag}: {count}")

    has_text = sum(1 for r in records if r["evidence_text"])
    print(f"\nRecords with evidence text: {has_text}/{len(records)}")

    # Source API distribution
    src_counts = Counter(r["source_api"] for r in records)
    print(f"\nSource API distribution:")
    for src, count in src_counts.most_common():
        correct = sum(1 for r in records if r["source_api"] == src and r["tag"] == "correct")
        print(f"  {src}: {count} ({correct} correct, {100*correct/count:.1f}%)")

    # Sample records with evidence text
    with_text = [r for r in records if r["evidence_text"]]
    if with_text:
        print(f"\nSample record:")
        r = with_text[0]
        print(f"  {r['subject']} --[{r['stmt_type']}]--> {r['object']}")
        print(f"  Source: {r['source_api']}, belief: {r['belief']:.3f}")
        print(f"  Tag: {r['tag']}")
        print(f"  Text: {r['evidence_text'][:200]}")


if __name__ == "__main__":
    main()

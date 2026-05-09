"""Deterministic-only probe: for every FN in L8 result, dump the
evidence text, claim, ctx (alias map, classes, sites, anchors), so we
can see what the LLM was working with WITHOUT re-hitting the endpoint.

Many FNs may already be diagnosable from the substrate alone:
  - if ctx.aliases is empty for an entity → grounding/binding will fail
  - if subject_class is wrong → L2 cytokine bypass won't fire
  - if has_chain_signal is False but evidence is multi-hop → L1 silent
  - if claim is FPLX→FPLX_member → tells us which family the parser
    must have failed to bind
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from indra_belief.data.corpus import CorpusIndex
from indra_belief.scorers.context import EvidenceContext
from indra_belief.scorers.parse_claim import parse_claim
from indra_belief.scorers.context_builder import build_context


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    result_path = ROOT / "data" / "results" / "dec_e3_v17_jklphase.jsonl"
    fns = []
    with open(result_path) as f:
        for line in f:
            r = json.loads(line)
            if r["tag"] == "correct" and r["actual"] == "incorrect":
                fns.append((r["source_hash"], r["reasons"]))

    index = CorpusIndex()
    records = index.build_records(
        str(ROOT / "data" / "benchmark" / "holdout_v15_sample.jsonl")
    )
    by_hash = {r.source_hash: r for r in records}

    out_path = ROOT / "data" / "results" / "fn_probe_l8_det_full.jsonl"
    out_f = open(out_path, "w")
    for h, reasons in fns:
        if h not in by_hash:
            continue
        rec = by_hash[h]
        try:
            claim = parse_claim(rec.statement)
        except Exception as e:
            claim = None
        try:
            ctx = build_context(rec.statement, rec.evidence)
        except Exception as e:
            ctx = EvidenceContext()
        record = {
            "source_hash": h,
            "subject": rec.subject,
            "stmt_type": rec.stmt_type,
            "object": rec.object,
            "tag": rec.tag,
            "fn_reason": reasons,
            "evidence_text": rec.evidence.text,
            "claim": ({
                "stmt_type": claim.stmt_type,
                "subject": claim.subject,
                "objects": list(claim.objects),
                "axis": claim.axis,
                "sign": claim.sign,
                "site": claim.site,
            } if claim else None),
            "ctx": {
                "aliases": {k: list(v)[:10] for k, v in ctx.aliases.items()},
                "families": {k: list(v) for k, v in ctx.families.items()},
                "is_pseudogene": list(ctx.is_pseudogene),
                "binding_admissible": list(ctx.binding_admissible),
                "acceptable_sites": list(ctx.acceptable_sites),
                "subject_class": ctx.subject_class,
                "object_class": ctx.object_class,
                "subject_precision": ctx.subject_precision,
                "object_precision": ctx.object_precision,
                "subject_has_upstream_anchor": ctx.subject_has_upstream_anchor,
                "has_chain_signal": ctx.has_chain_signal,
                "chain_intermediate_candidates": list(ctx.chain_intermediate_candidates),
                "nominalized_relations": list(ctx.nominalized_relations),
                "detected_sites": list(ctx.detected_sites),
                "n_clauses": len(ctx.clauses),
            },
        }
        out_f.write(json.dumps(record) + "\n")
        out_f.flush()
    out_f.close()
    print(f"Wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

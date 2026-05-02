"""Probe FN records from L8 holdout: re-run the dec pipeline with
trace capture, log every parser/grounder/adjudicator decision so we
can see WHY each FN was rejected.

Reads the in-progress L8 result file, picks records where pred=incorrect
but gold=correct, looks them up in the holdout corpus, and re-runs them
with a wrapped client that captures sub-call inputs/outputs.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from indra_belief.data.corpus import CorpusIndex
from indra_belief.model_client import ModelClient
from indra_belief.scorers.commitments import (
    Adjudication,
    EvidenceAssertion,
    EvidenceCommitment,
    GroundingVerdict,
)
from indra_belief.scorers.context import EvidenceContext
from indra_belief.scorers.parse_claim import parse_claim
from indra_belief.scorers.parse_evidence import parse_evidence
from indra_belief.scorers.grounding import verify_grounding
from indra_belief.scorers.adjudicate import adjudicate
from indra_belief.scorers.context_builder import build_context
from indra_belief.scorers.decomposed import _resolve_claim_entities


def probe_one(stmt, evidence, client) -> dict:
    """Re-run the decomposed pipeline with full trace capture."""
    out = {}

    claim = parse_claim(stmt)
    out["claim"] = {
        "stmt_type": claim.stmt_type,
        "subject": claim.subject,
        "objects": list(claim.objects),
        "axis": claim.axis,
        "sign": claim.sign,
        "site": claim.site,
        "perturbation": claim.perturbation,
    }

    try:
        ctx = build_context(stmt, evidence)
    except Exception as e:
        out["ctx_error"] = f"{type(e).__name__}: {e}"
        ctx = EvidenceContext()
    out["ctx"] = {
        "aliases_keys": list(ctx.aliases.keys()),
        "aliases_sample": {k: list(v)[:8] for k, v in ctx.aliases.items()},
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
    }

    ev_text = evidence.text or ""
    ec = parse_evidence(ev_text, client, ctx=ctx)
    out["parse_evidence"] = {
        "ok": ec is not None,
        "agent_candidates": list(ec.agent_candidates) if ec else [],
        "target_candidates": list(ec.target_candidates) if ec else [],
        "n_assertions": len(ec.assertions) if ec else 0,
        "assertions": [
            {
                "axis": a.axis, "sign": a.sign, "negation": a.negation,
                "perturbation": a.perturbation,
                "agents": list(a.agents), "targets": list(a.targets),
                "site": a.site, "claim_status": a.claim_status,
                "binding_partner_type": a.binding_partner_type,
                "intermediates": list(a.intermediates),
            } for a in (ec.assertions if ec else [])
        ],
    }

    entities = _resolve_claim_entities(claim, evidence)
    grounding_list = []
    for e in entities:
        gv = verify_grounding(e, ev_text, client)
        grounding_list.append(gv)
    out["grounding"] = [
        {
            "entity": g.claim_entity, "status": g.status,
            "db_ns": g.db_ns, "db_id": g.db_id,
        } for g in grounding_list
    ]

    adj = adjudicate(claim, ec, tuple(grounding_list),
                     evidence_text=ev_text, ctx=ctx)
    out["adjudicate"] = {
        "verdict": adj.verdict, "confidence": adj.confidence,
        "reasons": list(adj.reasons), "rationale": adj.rationale,
    }
    return out


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    # Read the L8 result file for FN source_hashes.
    result_path = ROOT / "data" / "results" / "dec_e3_v17_jklphase.jsonl"
    fns = []
    with open(result_path) as f:
        for line in f:
            r = json.loads(line)
            if r["tag"] == "correct" and r["actual"] == "incorrect":
                fns.append(r["source_hash"])
    print(f"Found {len(fns)} FNs", file=sys.stderr)

    # Look up in holdout.
    index = CorpusIndex()
    records = index.build_records(str(ROOT / "data" / "benchmark" / "holdout_v15_sample.jsonl"))
    by_hash = {r.source_hash: r for r in records}

    client = ModelClient("gemma-remote")
    out_path = ROOT / "data" / "results" / "fn_probe_l8.jsonl"
    out_f = open(out_path, "w")
    for i, h in enumerate(fns, 1):
        if h not in by_hash:
            print(f"[{i}/{len(fns)}] hash {h} not in holdout — skip", file=sys.stderr)
            continue
        rec = by_hash[h]
        print(f"[{i}/{len(fns)}] {rec.subject} -{rec.stmt_type}-> {rec.object} ({h})", file=sys.stderr)
        try:
            trace = probe_one(rec.statement, rec.evidence, client)
        except Exception as e:
            trace = {"error": f"{type(e).__name__}: {e}"}
        record = {
            "source_hash": h,
            "subject": rec.subject,
            "stmt_type": rec.stmt_type,
            "object": rec.object,
            "tag": rec.tag,
            "evidence_text": rec.evidence.text,
            "trace": trace,
        }
        out_f.write(json.dumps(record) + "\n")
        out_f.flush()
    out_f.close()


if __name__ == "__main__":
    main()

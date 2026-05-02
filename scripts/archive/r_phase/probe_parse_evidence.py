"""Probe parse_evidence on mono-wins to see what dec actually extracts.

For each (statement, evidence) pair where mono got it right and dec
got it wrong, run the dec pipeline with full trace capture and dump:
  - parser-emitted assertions (raw output)
  - alias-validated relations (M-phase substrate)
  - per-assertion bind diagnostic (which side fails)
  - adjudicator's rejection reason

Goal: identify whether dec's failure is at parse, bind, or adjudicate.
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
from indra_belief.scorers.adjudicate import adjudicate, _matches_binding, _names_intersect
from indra_belief.scorers.context_builder import build_context
from indra_belief.scorers.grounding import verify_grounding
from indra_belief.scorers.parse_claim import parse_claim
from indra_belief.scorers.parse_evidence import parse_evidence
from indra_belief.scorers.decomposed import _resolve_claim_entities


def probe_record(rec, client) -> dict:
    out = {
        "subject": rec.subject, "stmt_type": rec.stmt_type, "object": rec.object,
        "evidence": (rec.evidence.text or "")[:300],
    }
    try:
        claim = parse_claim(rec.statement)
        out["claim"] = {"axis": claim.axis, "sign": claim.sign,
                        "subject": claim.subject, "objects": list(claim.objects),
                        "site": claim.site}
    except Exception as e:
        out["err_parse_claim"] = str(e); return out

    try:
        ctx = build_context(rec.statement, rec.evidence)
    except Exception as e:
        out["err_ctx"] = str(e); return out
    out["aliases"] = {k: sorted(v)[:8] for k, v in ctx.aliases.items()}
    out["detected_relations"] = [
        {"axis": r.axis, "sign": r.sign,
         "agent": r.agent_canonical, "target": r.target_canonical,
         "site": r.site, "pattern_id": r.pattern_id}
        for r in ctx.detected_relations
    ]
    out["cascade_terminals"] = list(ctx.cascade_terminals)
    out["subject_pert"] = ctx.subject_perturbation_marker
    out["object_pert"] = ctx.object_perturbation_marker
    out["hedge_markers"] = list(ctx.explicit_hedge_markers)
    out["has_chain_signal"] = ctx.has_chain_signal

    try:
        ec = parse_evidence(rec.evidence.text, client, ctx=ctx)
    except Exception as e:
        out["err_parse_evidence"] = str(e); return out
    if ec is None:
        out["parse_evidence"] = None; return out
    out["parse_evidence"] = {
        "n_assertions": len(ec.assertions),
        "agent_candidates": list(ec.agent_candidates),
        "target_candidates": list(ec.target_candidates),
        "assertions": [
            {"axis": a.axis, "sign": a.sign,
             "agents": list(a.agents), "targets": list(a.targets),
             "perturbation": a.perturbation, "claim_status": a.claim_status,
             "binding_partner_type": a.binding_partner_type,
             "site": a.site, "negation": a.negation,
             "intermediates": list(a.intermediates)}
            for a in ec.assertions
        ],
    }

    binds_check = []
    for i, a in enumerate(ec.assertions):
        binds_check.append({
            "assertion_idx": i,
            "matches_binding": _matches_binding(a, claim, ctx.aliases),
            "subject_in_agents": _names_intersect(claim.subject, a.agents, ctx.aliases),
            "subject_in_targets": _names_intersect(claim.subject, a.targets, ctx.aliases),
            "object_in_agents": (
                _names_intersect(claim.objects[0], a.agents, ctx.aliases)
                if claim.objects else False),
            "object_in_targets": (
                _names_intersect(claim.objects[0], a.targets, ctx.aliases)
                if claim.objects else False),
        })
    out["bind_diagnostic"] = binds_check

    entities = _resolve_claim_entities(claim, rec.evidence)
    groundings = tuple(verify_grounding(e, rec.evidence.text, client) for e in entities)
    adj = adjudicate(claim, ec, groundings,
                     evidence_text=rec.evidence.text, ctx=ctx)
    out["adjudication"] = {
        "verdict": adj.verdict, "confidence": adj.confidence,
        "reasons": list(adj.reasons), "rationale": adj.rationale,
    }
    return out


def main():
    logging.basicConfig(level=logging.WARNING)

    mono = {json.loads(l)['source_hash']: json.loads(l) for l in open('data/results/mono_e1_v15.jsonl')}
    dec = {json.loads(l)['source_hash']: json.loads(l) for l in open('data/results/dec_e3_v18_mphase.jsonl')}
    common = set(mono.keys()) & set(dec.keys())
    target_hashes = []
    for h in common:
        m, d = mono[h], dec[h]
        if m['tag'] == 'correct' and m['actual'] == 'correct' and d['actual'] != 'correct':
            target_hashes.append(h)

    target_hashes = [h for h in target_hashes
                     if 'absent_relationship' in dec[h].get('reasons', [])][:8]

    index = CorpusIndex()
    records = index.build_records('data/benchmark/holdout_v15_sample.jsonl')
    by_hash = {r.source_hash: r for r in records}

    client = ModelClient("gemma-remote")
    out_path = ROOT / "data" / "results" / "monowin_probe.jsonl"
    with open(out_path, "w") as f:
        for i, h in enumerate(target_hashes, 1):
            if h not in by_hash:
                print(f"[{i}/{len(target_hashes)}] hash {h} not found", file=sys.stderr)
                continue
            rec = by_hash[h]
            print(f"[{i}/{len(target_hashes)}] {rec.subject}-{rec.stmt_type}->{rec.object}",
                  file=sys.stderr)
            try:
                trace = probe_record(rec, client)
            except Exception as e:
                trace = {"error": f"{type(e).__name__}: {e}"}
            trace["source_hash"] = h
            f.write(json.dumps(trace) + "\n")
            f.flush()
    print(f"\nWrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

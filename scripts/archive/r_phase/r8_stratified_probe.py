"""R8 stratified probe — 30 hand-picked records targeting the
Q-phase failure classes. Validates the R-phase architecture (substrate
prior + parser verifier; escalation flag diagnostic only) before the
501-record holdout.

Strata (consumed by R9 gates):
  12 of 27 absent_relationship regressions (Complex, Phos, Activation, IncAmt, Inh)
   5 hedging_hypothesis regressions (all)
   3 sign_mismatch regressions (all)
  10 Q-phase gains (sample preservation check)
  ─────
  30 records total

Output: data/results/r8_probe.jsonl + summary stats compared against
M12 + Q-phase verdicts on the same records.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from indra_belief.model_client import ModelClient


ROOT = Path(__file__).resolve().parent.parent
Q = ROOT / "data" / "results" / "dec_q_phase.jsonl"
M = ROOT / "data" / "results" / "dec_e3_v18_mphase.jsonl"
SRC = ROOT / "data" / "benchmark" / "holdout_v15_sample.jsonl"
OUT = ROOT / "data" / "results" / "r8_probe.jsonl"


logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s [%(levelname)s] %(message)s")


def _classify(q, m):
    """Return (regressions, hedge_regs, sign_regs, gains)."""
    common = sorted(set(q) & set(m))
    regs, hedge, sign, gains = [], [], [], []
    for h in common:
        qa, ma = q[h]["actual"], m[h]["actual"]
        tgt = q[h]["target"]
        if qa not in ("correct", "incorrect"):
            continue
        if ma not in ("correct", "incorrect"):
            continue
        if ma == tgt and qa != tgt:
            reasons = q[h].get("reasons") or []
            if "absent_relationship" in reasons:
                regs.append(h)
            if "hedging_hypothesis" in reasons:
                hedge.append(h)
            if "sign_mismatch" in reasons:
                sign.append(h)
        if qa == tgt and ma != tgt:
            gains.append(h)
    return regs, hedge, sign, gains


def _load_holdout_records():
    """Return source_hash → full record dict from holdout_v15_sample."""
    return {r["source_hash"]: r for r in (json.loads(l) for l in open(SRC) if l.strip())}


def _score(stmt, evidence, client) -> dict:
    """Score one (stmt, evidence) and return the result dict."""
    from indra_belief.scorers.decomposed import score_evidence_decomposed
    return score_evidence_decomposed(stmt, evidence, client)


def main() -> None:
    q = {r["source_hash"]: r for r in (json.loads(l) for l in open(Q))}
    m = {r["source_hash"]: r for r in (json.loads(l) for l in open(M))}
    src = _load_holdout_records()

    regs, hedge_regs, sign_regs, gains = _classify(q, m)
    print(f"Population: {len(regs)} ar-regs, {len(hedge_regs)} hedge-regs, "
          f"{len(sign_regs)} sign-regs, {len(gains)} gains")

    # Stratified sample. Hand-pick to cover stmt_type diversity in regs.
    by_type = {}
    for h in regs:
        by_type.setdefault(q[h]["stmt_type"], []).append(h)
    selected_regs = []
    # 12 records — try to span types
    for st, hs in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        for h in hs[:3]:  # up to 3 per stmt_type to spread
            selected_regs.append(h)
            if len(selected_regs) == 12:
                break
        if len(selected_regs) == 12:
            break
    if len(selected_regs) < 12:
        for h in regs:
            if h not in selected_regs:
                selected_regs.append(h)
                if len(selected_regs) == 12:
                    break

    selected = (
        list(selected_regs)
        + list(hedge_regs)
        + list(sign_regs)
        + list(gains[:10])
    )
    print(f"Selected: {len(selected_regs)} regs + {len(hedge_regs)} hedge "
          f"+ {len(sign_regs)} sign + {len(gains[:10])} gains = "
          f"{len(selected)} records\n")

    print("Initializing model client (gemma-remote)...")
    client = ModelClient("gemma-remote")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out_f = OUT.open("w")

    results = []
    t_start = time.time()
    for i, h in enumerate(selected):
        rec = src[h]
        # Build INDRA Statement from the holdout record. The
        # holdout JSONL has matches_hash + stmt_type + subject + object.
        stmt, ev = _stmt_ev_from_record(rec)
        if stmt is None:
            print(f"  [{i+1}/{len(selected)}] SKIP — couldn't rebuild stmt")
            continue
        cls = "REG" if h in selected_regs else (
            "HDG" if h in hedge_regs else (
            "SGN" if h in sign_regs else "GAIN"))
        t0 = time.time()
        try:
            r = _score(stmt, ev, client)
            elapsed = time.time() - t0
        except Exception as e:
            print(f"  [{i+1}/{len(selected)}] {cls} FAIL: {e}")
            continue
        # Compose result row
        row = {
            "source_hash": h,
            "stratum": cls,
            "stmt_type": rec["stmt_type"],
            "subject": rec["subject"],
            "object": rec["object"],
            "target": q[h]["target"],
            "q_actual": q[h]["actual"],
            "m_actual": m[h]["actual"],
            "r_actual": r["verdict"],
            "r_confidence": r["confidence"],
            "r_reasons": r["reasons"],
            "escalation_triggered": r.get("escalation_triggered", False),
            "escalation_reason": r.get("escalation_reason"),
            "elapsed_s": elapsed,
            "tokens": r["tokens"],
            "n_calls": len(r.get("call_log") or []),
        }
        results.append(row)
        out_f.write(json.dumps(row) + "\n")
        out_f.flush()

        # Live status
        m_right = m[h]["actual"] == q[h]["target"]
        q_right = q[h]["actual"] == q[h]["target"]
        r_right = r["verdict"] == q[h]["target"]
        marker = (
            "✓→✗" if (m_right and q_right and not r_right) else
            "✗→✓" if (not m_right and not q_right and r_right) else
            "✓→✓" if (m_right and r_right) else
            "✗→✗" if (not m_right and not r_right) else
            "✗→✓" if (not m_right and r_right) else "✓→✗"
        )
        esc = "+ESC" if r.get("escalation_triggered") else ""
        print(f"  [{i+1}/{len(selected)}] {cls:<4} {rec['subject']:>14}-"
              f"[{rec['stmt_type']:>14}]->{rec['object']:<14}  "
              f"M={m[h]['actual']} Q={q[h]['actual']} R={r['verdict']}  "
              f"{marker} {elapsed:.1f}s {esc}")

    out_f.close()
    total_elapsed = time.time() - t_start
    print(f"\n=== R8 PROBE COMPLETE — {len(results)}/{len(selected)} scored "
          f"in {total_elapsed/60:.1f}min ===\n")
    _summarize(results)


def _summarize(results: list[dict]) -> None:
    by_class = {"REG": [], "HDG": [], "SGN": [], "GAIN": []}
    for r in results:
        by_class[r["stratum"]].append(r)

    for cls, rows in by_class.items():
        if not rows:
            continue
        total = len(rows)
        r_correct = sum(1 for r in rows if r["r_actual"] == r["target"])
        m_correct = sum(1 for r in rows if r["m_actual"] == r["target"])
        q_correct = sum(1 for r in rows if r["q_actual"] == r["target"])
        print(f"{cls}: R={r_correct}/{total}  M12={m_correct}/{total}  "
              f"Q={q_correct}/{total}")

    # Escalation rate
    n_esc = sum(1 for r in results if r["escalation_triggered"])
    print(f"\nEscalation flagged: {n_esc}/{len(results)} ({100*n_esc/max(len(results),1):.0f}%)")
    by_reason = Counter(r["escalation_reason"] for r in results
                        if r["escalation_triggered"])
    for k, v in by_reason.most_common():
        print(f"  {k}: {v}")

    # Latency
    walls = sorted(r["elapsed_s"] for r in results)
    if walls:
        med = walls[len(walls) // 2]
        p99 = walls[int(len(walls) * 0.99)] if len(walls) > 5 else walls[-1]
        mx = walls[-1]
        print(f"\nLatency: median {med:.1f}s p99 {p99:.1f}s max {mx:.1f}s")


def _stmt_ev_from_record(rec):
    """Reconstruct a real INDRA Statement and Evidence from a holdout
    record. Holdout records carry: stmt_type, subject, object,
    evidence_text, source_api, pmid, source_hash, matches_hash, pa_hash,
    belief.

    For most stmt_types we can reconstruct via the type registry and
    Agent factory. Complex needs members; SelfMod uses one agent.
    """
    from indra.statements import (
        Activation, Inhibition, Phosphorylation, Dephosphorylation,
        Acetylation, Deacetylation, Methylation, Demethylation,
        Ubiquitination, Deubiquitination, IncreaseAmount, DecreaseAmount,
        Complex, Translocation, Conversion, Gef, Gap, Autophosphorylation,
        Agent, Evidence,
    )

    type_map = {
        "Activation": Activation, "Inhibition": Inhibition,
        "Phosphorylation": Phosphorylation, "Dephosphorylation": Dephosphorylation,
        "Acetylation": Acetylation, "Deacetylation": Deacetylation,
        "Methylation": Methylation, "Demethylation": Demethylation,
        "Ubiquitination": Ubiquitination, "Deubiquitination": Deubiquitination,
        "IncreaseAmount": IncreaseAmount, "DecreaseAmount": DecreaseAmount,
        "Complex": Complex, "Translocation": Translocation,
        "Conversion": Conversion, "Gef": Gef, "Gap": Gap,
        "Autophosphorylation": Autophosphorylation,
    }
    stmt_type = rec.get("stmt_type")
    cls = type_map.get(stmt_type)
    if cls is None:
        return None, None
    subj = rec.get("subject")
    obj = rec.get("object")
    if not subj or not obj:
        return None, None
    ev = Evidence(
        source_api=rec.get("source_api", "reach"),
        pmid=rec.get("pmid"),
        text=rec.get("evidence_text", ""),
    )
    if cls is Complex:
        stmt = Complex([Agent(subj), Agent(obj)])
    elif cls is Autophosphorylation:
        stmt = Autophosphorylation(Agent(subj))
    elif cls is Translocation:
        stmt = Translocation(Agent(subj), None, obj)
    else:
        stmt = cls(Agent(subj), Agent(obj))
    return stmt, ev


if __name__ == "__main__":
    main()

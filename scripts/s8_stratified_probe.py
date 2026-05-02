"""S8 stratified probe — 30-record evaluation of S-phase architecture.

Uses the same R8 stratum (12 absent_relationship regs, 5 hedge regs,
3 sign regs, 10 gains) so we get like-for-like comparison with R-phase.

Targets (from S8 task description):
  REG ≥ 9/12  (R-phase did 6/12)
  HDG ≥ 4/5
  SGN ≥ 3/3
  GAIN ≥ 9/10  (preserve R-phase wins)

Plus regression-preservation audit on the 4 NEW R-only gains and 3
substrate-fallback gains (verified per-record).

Latency:
  median ≤ 12s, p99 ≤ 60s
  substrate fast-path: ≥ 60% records resolve all 4 probes without LLM

Output: scripts/s8_stratified_probe.py + data/results/s8_probe.jsonl
+ printed summary report.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from indra_belief.model_client import ModelClient
from indra_belief.scorers.scorer import score_evidence


Q_PATH = ROOT / "data" / "results" / "dec_q_phase.jsonl"
M_PATH = ROOT / "data" / "results" / "dec_e3_v18_mphase.jsonl"
SRC_PATH = ROOT / "data" / "benchmark" / "holdout_v15_sample.jsonl"
OUT_PATH = ROOT / "data" / "results" / "s8_probe.jsonl"


logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s [%(levelname)s] %(message)s")


def _classify(q: dict, m: dict) -> tuple[list, list, list, list]:
    """Return (regressions, hedge_regressions, sign_regressions, gains)."""
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


def _build_stmt_ev(rec: dict):
    from indra.statements import (
        Activation, Inhibition, Phosphorylation, Dephosphorylation,
        Acetylation, Deacetylation, Methylation, Demethylation,
        Ubiquitination, Deubiquitination, IncreaseAmount, DecreaseAmount,
        Complex, Translocation, Conversion, Gef, Gap,
        Autophosphorylation, Agent, Evidence,
    )
    type_map = {
        "Activation": Activation, "Inhibition": Inhibition,
        "Phosphorylation": Phosphorylation,
        "Dephosphorylation": Dephosphorylation,
        "Acetylation": Acetylation, "Deacetylation": Deacetylation,
        "Methylation": Methylation, "Demethylation": Demethylation,
        "Ubiquitination": Ubiquitination,
        "Deubiquitination": Deubiquitination,
        "IncreaseAmount": IncreaseAmount,
        "DecreaseAmount": DecreaseAmount,
        "Complex": Complex, "Translocation": Translocation,
        "Conversion": Conversion, "Gef": Gef, "Gap": Gap,
        "Autophosphorylation": Autophosphorylation,
    }
    cls = type_map.get(rec.get("stmt_type"))
    if cls is None:
        return None, None
    subj, obj = rec.get("subject"), rec.get("object")
    if not subj or not obj:
        return None, None
    ev = Evidence(
        source_api=rec.get("source_api", "reach"),
        pmid=rec.get("pmid"),
        text=rec.get("evidence_text", ""),
    )
    if cls is Complex:
        return Complex([Agent(subj), Agent(obj)]), ev
    if cls is Autophosphorylation:
        return Autophosphorylation(Agent(subj)), ev
    if cls is Translocation:
        return Translocation(Agent(subj), None, obj), ev
    return cls(Agent(subj), Agent(obj)), ev


def main() -> None:
    print("Loading reference results...")
    q = {r["source_hash"]: r
         for r in (json.loads(l) for l in open(Q_PATH) if l.strip())}
    m = {r["source_hash"]: r
         for r in (json.loads(l) for l in open(M_PATH) if l.strip())}
    src = {r["source_hash"]: r
           for r in (json.loads(l) for l in open(SRC_PATH) if l.strip())}

    regs, hedge_regs, sign_regs, gains = _classify(q, m)
    print(f"Population: {len(regs)} ar-regs, {len(hedge_regs)} hedge-regs, "
          f"{len(sign_regs)} sign-regs, {len(gains)} gains")

    # Stratify
    by_type: dict[str, list[str]] = {}
    for h in regs:
        by_type.setdefault(q[h]["stmt_type"], []).append(h)
    selected_regs: list[str] = []
    for st, hs in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        for h in hs[:3]:
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
          f"+ {len(sign_regs)} sign + {len(gains[:10])} gains "
          f"= {len(selected)} records\n")

    print("Initializing model client (gemma-remote)...")
    client = ModelClient("gemma-remote")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_f = OUT_PATH.open("w")

    results: list[dict] = []
    t_start = time.time()
    for i, h in enumerate(selected):
        rec = src[h]
        stmt, ev = _build_stmt_ev(rec)
        if stmt is None:
            print(f"  [{i + 1}/{len(selected)}] SKIP — couldn't rebuild stmt")
            continue
        cls = ("REG" if h in selected_regs else
               "HDG" if h in hedge_regs else
               "SGN" if h in sign_regs else "GAIN")
        t0 = time.time()
        try:
            r = score_evidence(stmt, ev, client)
            elapsed = time.time() - t0
        except Exception as e:
            print(f"  [{i + 1}/{len(selected)}] {cls} FAIL: {e}")
            continue

        # n_calls = LLM calls actually invoked (substrate-resolved probes
        # don't appear in the call_log).
        n_calls = sum(
            1 for c in (r.get("call_log") or [])
            if c.get("kind", "").startswith("probe_")
        )

        row = {
            "source_hash": h,
            "stratum": cls,
            "stmt_type": rec["stmt_type"],
            "subject": rec["subject"],
            "object": rec["object"],
            "target": q[h]["target"],
            "q_actual": q[h]["actual"],
            "m_actual": m[h]["actual"],
            "s_actual": r["verdict"],
            "s_confidence": r["confidence"],
            "s_reasons": r["reasons"],
            "elapsed_s": elapsed,
            "n_probe_llm_calls": n_calls,
            "tokens": r.get("tokens", 0),
        }
        results.append(row)
        out_f.write(json.dumps(row) + "\n")
        out_f.flush()

        m_right = m[h]["actual"] == q[h]["target"]
        q_right = q[h]["actual"] == q[h]["target"]
        s_right = r["verdict"] == q[h]["target"]
        marker = (
            "✓→✗" if (m_right and q_right and not s_right) else
            "✗→✓" if (not m_right and not q_right and s_right) else
            "✓→✓" if (m_right and s_right) else
            "✗→✗" if (not m_right and not s_right) else
            "✗→✓" if (not m_right and s_right) else "✓→✗"
        )
        print(f"  [{i + 1}/{len(selected)}] {cls:<4} "
              f"{rec['subject']:>14}-[{rec['stmt_type']:>14}]->{rec['object']:<14}  "
              f"M={m[h]['actual']} Q={q[h]['actual']} S={r['verdict']}  "
              f"{marker} {elapsed:.1f}s ({n_calls} LLM)")

    out_f.close()
    total_elapsed = time.time() - t_start
    print(f"\n=== S8 PROBE COMPLETE — {len(results)}/{len(selected)} "
          f"scored in {total_elapsed / 60:.1f}min ===\n")
    _summarize(results)


def _summarize(results: list[dict]) -> None:
    by_class: dict[str, list[dict]] = {
        "REG": [], "HDG": [], "SGN": [], "GAIN": [],
    }
    for r in results:
        by_class[r["stratum"]].append(r)

    print("=== Per-stratum recovery vs gold ===\n")
    for cls, rows in by_class.items():
        if not rows:
            continue
        total = len(rows)
        s_correct = sum(1 for r in rows if r["s_actual"] == r["target"])
        m_correct = sum(1 for r in rows if r["m_actual"] == r["target"])
        q_correct = sum(1 for r in rows if r["q_actual"] == r["target"])
        print(f"{cls}: S-phase={s_correct}/{total}  "
              f"M12={m_correct}/{total}  Q-phase={q_correct}/{total}")

    print("\n=== Substrate fast-path coverage ===")
    n_zero_llm = sum(1 for r in results if r["n_probe_llm_calls"] == 0)
    print(f"  Records with zero LLM probe calls: "
          f"{n_zero_llm}/{len(results)} "
          f"({100 * n_zero_llm / max(len(results), 1):.0f}%)")
    avg_calls = sum(r["n_probe_llm_calls"] for r in results) / max(
        len(results), 1)
    print(f"  Mean LLM probe calls per record: {avg_calls:.2f}")

    print("\n=== Latency (per-record) ===")
    walls = sorted(r["elapsed_s"] for r in results)
    if walls:
        med = walls[len(walls) // 2]
        p99 = walls[int(len(walls) * 0.99)] if len(walls) > 5 else walls[-1]
        mx = walls[-1]
        print(f"  median {med:.1f}s p99 {p99:.1f}s max {mx:.1f}s")

    print("\n=== Reason breakdown (S-phase) ===")
    rc: Counter[str] = Counter()
    for r in results:
        for reason in r["s_reasons"]:
            rc[reason] += 1
    for k, v in rc.most_common():
        print(f"  {k:<32} {v}")

    print("\n=== Verdict matrix (M12 / S-phase) ===")
    matrix: Counter[tuple[str, str]] = Counter()
    for r in results:
        matrix[(r["m_actual"], r["s_actual"])] += 1
    for (mv, sv), n in sorted(matrix.items()):
        print(f"  M={mv:<10} S={sv:<10} {n}")


if __name__ == "__main__":
    main()

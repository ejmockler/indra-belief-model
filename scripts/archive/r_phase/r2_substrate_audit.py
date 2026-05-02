"""R2 substrate audit — replay expanded CATALOG against Q-phase
regressions + gains and report what would now be substrate-bindable.

Gates this script verifies (consumed by R3 brutalist gate):
  - Recovery rate on 27 absent_relationship regressions (target ≥ 18)
  - No collapse on 25 Q-phase gains (target ≥ 23 still bindable)
  - Total false-positive rate on a clean sample (target < 5%)

Output: stratified breakdown so R3 can audit which records flipped
and which patterns fired.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from indra_belief.data.entity import GroundedEntity
from indra_belief.scorers.context_builder import (
    _bind_to_claim_canonical,
    _expand_synonyms,
)
from indra_belief.scorers.relation_patterns import CATALOG


ROOT = Path(__file__).resolve().parent.parent
Q = ROOT / "data" / "results" / "dec_q_phase.jsonl"
M = ROOT / "data" / "results" / "dec_e3_v18_mphase.jsonl"
SRC = ROOT / "data" / "benchmark" / "holdout_v15_sample.jsonl"


def _load(path):
    return {r["source_hash"]: r for r in (json.loads(l) for l in open(path) if l.strip())}


def _aliases_for(subject: str, obj: str) -> dict[str, frozenset[str]]:
    aliases: dict[str, frozenset[str]] = {}
    for nm in (subject, obj):
        if not nm:
            continue
        try:
            ge = GroundedEntity.resolve(nm, None)
        except Exception:
            continue
        syns = _expand_synonyms(ge)
        if syns:
            aliases[nm] = syns
    return aliases


def _claim_bound(text: str, subject: str, obj: str,
                 aliases: dict[str, frozenset[str]]) -> list[dict]:
    """Return list of CATALOG matches whose (X, Y) alias-bind to (subject,
    object) — same gate as M3 substrate-fallback uses."""
    out = []
    for pat in CATALOG:
        for mm in pat.regex.finditer(text):
            x_text = mm.group("X")
            y_text = mm.group("Y")
            ax = _bind_to_claim_canonical(x_text, aliases)
            ay = _bind_to_claim_canonical(y_text, aliases)
            if not ax or not ay or ax == ay:
                continue
            if (ax == subject and ay == obj) or (ax == obj and ay == subject):
                out.append({
                    "pattern_id": pat.pattern_id,
                    "x_text": x_text,
                    "y_text": y_text,
                    "axis": pat.axis,
                    "sign": pat.sign,
                })
    return out


def _classify_records(q, m, src):
    """Return (regressions, gains, clean_correct).
    regressions: M12 right + Q wrong (with reasons including absent_relationship)
    gains: Q right + M12 wrong
    clean_correct: both Q+M12 right (sample for FP rate measurement)
    """
    common = sorted(set(q) & set(m) & set(src))
    regs, gains, clean = [], [], []
    for h in common:
        qa, ma = q[h]["actual"], m[h]["actual"]
        tgt = q[h]["target"]
        if qa not in ("correct", "incorrect"):
            continue
        if ma not in ("correct", "incorrect"):
            continue
        m_right = ma == tgt
        q_right = qa == tgt
        if m_right and not q_right and "absent_relationship" in (q[h].get("reasons") or []):
            regs.append(h)
        if q_right and not m_right:
            gains.append(h)
        if m_right and q_right and tgt == "correct":
            clean.append(h)
    return regs, gains, clean


def main() -> None:
    q = _load(Q)
    m = _load(M)
    src = _load(SRC)

    regs, gains, clean = _classify_records(q, m, src)

    # === REGRESSION RECOVERY ===
    print(f"=== R2 substrate audit ===\n")
    print(f"CATALOG patterns: {len(CATALOG)}\n")

    print(f"--- 27 absent_relationship regressions (M12 right, Q wrong) ---")
    bound_regs = []
    unbound_regs = []
    pattern_hits = Counter()
    for h in regs:
        qr = q[h]; sr = src[h]
        text = sr.get("evidence_text", "")
        aliases = _aliases_for(qr["subject"], qr["object"])
        matches = _claim_bound(text, qr["subject"], qr["object"], aliases)
        if matches:
            bound_regs.append((h, matches))
            for mm in matches:
                pattern_hits[mm["pattern_id"]] += 1
        else:
            unbound_regs.append(h)

    print(f"\nBOUND (substrate-fallback would lift): {len(bound_regs)}/{len(regs)}")
    print(f"UNBOUND: {len(unbound_regs)}/{len(regs)}")
    print(f"\nPattern hits on regressions:")
    for pid, n in pattern_hits.most_common():
        print(f"  {pid:<40} {n}")

    print(f"\nUNBOUND records (CATALOG still missing):")
    for h in unbound_regs:
        qr = q[h]; sr = src[h]
        print(f"  {qr['subject']:>14}-[{qr['stmt_type']:>14}]->{qr['object']:<14}")
        print(f"    TEXT: {sr.get('evidence_text','')[:200]}")

    # === GAIN PRESERVATION ===
    print(f"\n\n--- 25 Q-phase gains (Q right, M12 wrong) ---")
    bound_gains = 0
    unbound_gains = []
    for h in gains:
        qr = q[h]; sr = src[h]
        text = sr.get("evidence_text", "")
        aliases = _aliases_for(qr["subject"], qr["object"])
        matches = _claim_bound(text, qr["subject"], qr["object"], aliases)
        if matches:
            bound_gains += 1
        else:
            unbound_gains.append(h)
    print(f"BOUND (substrate would also catch): {bound_gains}/{len(gains)}")
    print(f"UNBOUND (gain depended on parser/Q substrate not relations CATALOG): "
          f"{len(unbound_gains)}/{len(gains)}")

    # === FP RATE on a clean sample of correct-correct ===
    print(f"\n\n--- FP rate on 50 'incorrect' records (both Q+M12 said incorrect) ---")
    common = sorted(set(q) & set(m) & set(src))
    incorrect_records = [h for h in common
                         if q[h]["target"] == "incorrect"
                         and q[h]["actual"] == "incorrect"
                         and m[h]["actual"] == "incorrect"][:50]
    fp = 0
    fp_examples = []
    for h in incorrect_records:
        qr = q[h]; sr = src[h]
        text = sr.get("evidence_text", "")
        aliases = _aliases_for(qr["subject"], qr["object"])
        matches = _claim_bound(text, qr["subject"], qr["object"], aliases)
        if matches:
            fp += 1
            if len(fp_examples) < 5:
                fp_examples.append((qr, matches[0], text[:200]))
    print(f"FALSE POSITIVES (substrate would lift incorrect→correct): {fp}/{len(incorrect_records)}")
    if fp_examples:
        print(f"Sample FP regressions:")
        for qr, mm, text in fp_examples:
            print(f"  {qr['subject']}-[{qr['stmt_type']}]->{qr['object']}  "
                  f"pattern={mm['pattern_id']}")
            print(f"    TEXT: {text}")

    # === SUMMARY ===
    print(f"\n\n=== R2 GATE PRECHECK ===")
    print(f"  Regression recovery: {len(bound_regs)}/27  (R3 gate threshold ≥18)")
    print(f"  Gain preservation:   {bound_gains}/{len(gains)}  (informational)")
    print(f"  Clean-incorrect FP:  {fp}/{len(incorrect_records)}  (target ≤5%)")


if __name__ == "__main__":
    main()

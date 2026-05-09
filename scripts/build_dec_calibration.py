"""Build FN and FP calibration sets for dec-pipeline repair tasks.

FN-cal: 54 records where mono-right-dec-wrong AND gold=correct (parse-failures
        + real disagreements that lost dec-vs-mono ground), tagged by pattern.
FP-cal: 16 records where mono-right-dec-wrong AND gold!=correct (dec ratified
        what curators flagged), tagged by failure pattern.

Each set adds ~25-30 negative-test records (where mono and dec both agree
with gold) to guard against regression: a "fix" that drops calibration FN
acc by routing through cascade should not also drop accuracy on these.

Pattern vocabularies (controlled, fixed):
  FN: nominalized | presupposed | indirect_chain | hedging_overfire |
      negation_context | real_disagreement
  FP: co_occurrence | promoter_target | instrumental_role | self_loop |
      fusion_tag | indirect_as_direct | real_match
"""
from __future__ import annotations

import json
import re
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOSSES = Path("/tmp/dec_losses.jsonl")
DEC_FULL = ROOT / "data" / "results" / "dec_full_v6.jsonl"
HOLDOUT = ROOT / "data" / "benchmark" / "holdout_v15_sample.jsonl"
MONO = ROOT / "data" / "results" / "v16_sample.jsonl"
OUT_FN = ROOT / "data" / "benchmark" / "calibration_dec_fn.jsonl"
OUT_FP = ROOT / "data" / "benchmark" / "calibration_dec_fp.jsonl"


def classify_fn(rec: dict) -> str:
    """Tag an FN record with its failure pattern.

    Rule order matters — first match wins.
    """
    reasons = rec.get("dec_reasons") or []
    text = (rec.get("evidence_text") or "").lower()

    # Real-disagreement reasons fire on parse_evidence having extracted
    # something — the mismatch is a substantive semantic field, not a
    # parse miss.
    real_disagreement_reasons = {
        "axis_mismatch", "sign_mismatch", "contradicted",
        "site_mismatch", "grounding_gap",
    }
    if any(r in real_disagreement_reasons for r in reasons):
        return "real_disagreement"

    # Hedging is its own bucket (over-fired detector).
    if "hedging_hypothesis" in reasons:
        return "hedging_overfire"

    # Below here: absent_relationship reason. Sub-classify by surface form.
    # Indirect chain markers — "X via Y", "X thereby Y", "led to", "via X"
    if any(p in text for p in (
        "thereby", " via ", " led to ", "leading to", "results in",
        "resulted in", "indirectly", "downstream of", " in turn ",
        "mediated", "is mediated by",
    )):
        return "indirect_chain"

    # Negation context — "loss of X binding", "decreased binding"
    if re.search(r"\b(loss|decrease|reduce|reduction|diminish)\w*\s+of\s+", text):
        return "negation_context"

    # Presupposition — adjectival modification ("phosphorylated X",
    # "autophosphorylated X", "ubiquitinated X")
    if re.search(r"\b(autophosphorylat|phosphorylat|ubiquitinat|acetylat|methylat)ed\s+\w", text):
        return "presupposed"

    # Nominalization — relation expressed as a noun phrase
    # ("phosphorylation of X by Y", "binding of X to Y", "interaction
    # between X and Y")
    nominalization_patterns = (
        r"\b(phosphorylat|dephosphorylat|ubiquitinat|acetylat|methylat|"
        r"activat|inhibit|induc|suppress|express|bind|interact|"
        r"associat|complex|translocat|conver)\w*ion\b",
        r"\bbinding\s+of\b",
        r"\binteraction\s+between\b",
        r"\bassociation\s+of\b",
    )
    if any(re.search(p, text) for p in nominalization_patterns):
        return "nominalized"

    # Default for absent_relationship without a clear surface marker —
    # treat as nominalized (most common parse-failure shape) but flag
    # for manual audit.
    return "nominalized"


def classify_fp(rec: dict) -> str:
    """Tag an FP record with its failure pattern.

    All FPs have reasons=("match",) — discrimination is by surface form
    of the evidence + claim shape.
    """
    text = (rec.get("evidence_text") or "").lower()
    subj = (rec.get("subject") or "").upper()
    obj = (rec.get("object") or "").upper()
    stmt_type = rec.get("stmt_type", "")

    # Self-loop: subject == object
    if subj == obj:
        return "self_loop"

    # Promoter-as-target: "X promoter", "X gene promoter"
    if re.search(rf"\b{re.escape(obj.lower())}\s+promoter\b", text) or \
       re.search(rf"\b{re.escape(subj.lower())}\s+promoter\b", text):
        return "promoter_target"

    # Fusion-tag construct (e.g., "Myc-EGFR" where one is a tag)
    if re.search(rf"{re.escape(subj.lower())}-{re.escape(obj.lower())}", text) or \
       re.search(rf"{re.escape(obj.lower())}-{re.escape(subj.lower())}", text):
        return "fusion_tag"

    # Co-occurrence: "X and Y bind to Z", "X or Y", "X, Y, and Z bind"
    if re.search(rf"\b{re.escape(subj.lower())}\s+and\s+{re.escape(obj.lower())}\b", text) or \
       re.search(rf"\b{re.escape(subj.lower())}\s+or\s+{re.escape(obj.lower())}\b", text) or \
       re.search(rf"\b{re.escape(obj.lower())}\s+(and|or)\s+{re.escape(subj.lower())}\b", text):
        return "co_occurrence"

    # Instrumental role: "via", "treatment with X", "X treatment"
    if re.search(rf"\b(via|by|with|using)\s+{re.escape(subj.lower())}\b", text) or \
       re.search(rf"\b{re.escape(subj.lower())}\s+treatment\b", text) or \
       re.search(rf"\btreated\s+with\s+{re.escape(subj.lower())}\b", text) or \
       re.search(rf"\bmediated?\b", text):
        return "instrumental_role"

    # Indirect-as-direct: explicit downstream/intermediate marker
    if re.search(r"\b(downstream|intermediate|via|through)\b", text):
        return "indirect_as_direct"

    return "co_occurrence"  # default; most common pattern


def collect_negatives_fn(n: int = 25) -> list[dict]:
    """Pull n records where dec=correct, mono=correct, gold=correct.

    These are "no regression" guard records: a fix that improves FN-cal acc
    must NOT drop these. Sampled deterministically.
    """
    gold = {}
    with open(HOLDOUT) as f:
        for line in f:
            r = json.loads(line)
            gold[r["source_hash"]] = r
    mono = {}
    with open(MONO) as f:
        for line in f:
            r = json.loads(line)
            mono[r["source_hash"]] = r

    candidates = []
    with open(DEC_FULL) as f:
        for line in f:
            d = json.loads(line)
            h = d["source_hash"]
            g = gold.get(h, {}).get("tag", "?").lower()
            mv = mono.get(h, {}).get("verdict")
            if g == "correct" and mv == "correct" and d.get("verdict") == "correct":
                # Capture full record shape with claim + evidence
                candidates.append({
                    "source_hash": h,
                    "stmt_type": d.get("stmt_type"),
                    "subject": d.get("subject"),
                    "object": d.get("object"),
                    "evidence_text": gold[h].get("evidence_text", ""),
                    "gold_tag": g,
                    "mono_verdict": mv,
                    "expected_dec_verdict": "correct",
                    "pattern": "negative_test_should_pass",
                })

    rng = random.Random(42)
    rng.shuffle(candidates)
    return candidates[:n]


def collect_negatives_fp(n: int = 30) -> list[dict]:
    """Pull n records where dec=correct, mono=correct, gold=correct.

    For FP-cal, these guard against the validity backstop incorrectly
    rejecting true positives.
    """
    return collect_negatives_fn(n)  # same shape


def main() -> None:
    losses = [json.loads(l) for l in open(LOSSES)]
    fns = [L for L in losses
           if L.get("gold_tag", "").lower() == "correct"
           and L.get("dec_verdict") == "incorrect"]
    fps = [L for L in losses
           if L.get("gold_tag", "").lower() != "correct"
           and L.get("dec_verdict") == "correct"]

    # Build FN-cal
    fn_records = []
    for rec in fns:
        pattern = classify_fn(rec)
        fn_records.append({
            "source_hash": rec["source_hash"],
            "stmt_type": rec["stmt_type"],
            "subject": rec["subject"],
            "object": rec["object"],
            "evidence_text": rec["evidence_text"],
            "gold_tag": rec["gold_tag"],
            "mono_verdict": rec["mono_verdict"],
            "dec_verdict_baseline": rec["dec_verdict"],
            "dec_reasons_baseline": rec["dec_reasons"],
            "dec_rationale_baseline": rec.get("dec_rationale", ""),
            "expected_post_fix_verdict": "correct",  # via cascade
            "pattern": pattern,
        })
    fn_records += collect_negatives_fn()

    # Build FP-cal
    fp_records = []
    for rec in fps:
        pattern = classify_fp(rec)
        fp_records.append({
            "source_hash": rec["source_hash"],
            "stmt_type": rec["stmt_type"],
            "subject": rec["subject"],
            "object": rec["object"],
            "evidence_text": rec["evidence_text"],
            "gold_tag": rec["gold_tag"],
            "mono_verdict": rec["mono_verdict"],
            "dec_verdict_baseline": rec["dec_verdict"],
            "dec_reasons_baseline": rec["dec_reasons"],
            "dec_rationale_baseline": rec.get("dec_rationale", ""),
            "expected_post_fix_verdict": rec["mono_verdict"],
            "pattern": pattern,
        })
    fp_records += collect_negatives_fp()

    OUT_FN.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FN, "w") as f:
        for r in fn_records:
            f.write(json.dumps(r) + "\n")
    with open(OUT_FP, "w") as f:
        for r in fp_records:
            f.write(json.dumps(r) + "\n")

    # Print pattern distributions
    from collections import Counter
    print(f"FN calibration set: {len(fn_records)} records → {OUT_FN}")
    for p, n in Counter(r["pattern"] for r in fn_records).most_common():
        print(f"  {p:35s} {n}")
    print(f"\nFP calibration set: {len(fp_records)} records → {OUT_FP}")
    for p, n in Counter(r["pattern"] for r in fp_records).most_common():
        print(f"  {p:35s} {n}")


if __name__ == "__main__":
    main()

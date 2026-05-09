"""Classify every FN by failure class to size M-phase recovery.

Heuristic regex/structure tests on (evidence_text, claim, ctx). Class
labels match the M-phase taxonomy:

  A. Nominalization
  B. FPLX expansion gap
  C. Reverse / nested binding
  D. Cascade terminal / chain
  E. Alias normalization
  F. Perturbation inversion
  G. Axis bridge
  H. Ambiguous-alias grounding
  I. Likely gold noise / unrecoverable

A given FN may match multiple; we tag all and report dominant.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Nominalization patterns (M1)
NOM_PATTERNS = [
    re.compile(r"\bphosphorylation of \w+ (?:on|at|by)\b", re.I),
    re.compile(r"\b\w+-(?:induced|mediated|driven|dependent|stimulated|triggered)\b", re.I),
    re.compile(r"\b\w+\s+(?:phosphorylation|activation|expression|degradation|stabilization)\s+of\s+\w+", re.I),
    re.compile(r"\b(?:activation|phosphorylation|expression)\s+of\s+\w+\s+by\s+\w+", re.I),
    re.compile(r"\b\w+\s+(?:triggered|induced|stimulated)\s+the\s+(?:activation|expression|phosphorylation)\s+of\s+\w+", re.I),
]

# FPLX claim entities that typically have sparse Gilda expansion
SPARSE_FPLX = {"GPCR", "p14_3_3", "TCF_LEF", "IKB", "PKC", "PKA", "PI3K",
               "MEK", "ERK", "AKT", "JNK", "p38", "NFkappaB", "FGF", "TCR",
               "Caspase", "Histone_H1", "Histone_H2A", "Hsp70", "Hsp90"}

# Reverse / nested binding patterns
REVERSE_PATTERNS = [
    re.compile(r"\binterferes? with binding of \w+ to \w+\b", re.I),
    re.compile(r"\bcomplex with \w+,\s*\w+,?\s*and\s+\w+\b", re.I),
    re.compile(r"\b\w+\s+binds? to \w+\b", re.I),
    re.compile(r"\bbinding\s+of\s+\w+\s+to\s+\w+\b", re.I),
]

# Cascade-listing patterns
CASCADE_PATTERNS = [
    re.compile(r"\b(?:[A-Z]\w*[/-]){2,}[A-Z]\w*\s+(?:pathway|signaling|cascade)\b"),
    re.compile(r"\b\w+\s+then\s+\w+\b", re.I),
    re.compile(r"\bthereby\s+\w+", re.I),
    re.compile(r"\bvia\s+\w+\s+(?:and|to)\b", re.I),
]

# Alias hyphen / Greek variants
GREEK_HYPHEN_PATTERNS = [
    re.compile(r"[αβγδ]"),
    re.compile(r"\b\w+-\d+\b"),  # HNP-1 style
    re.compile(r"\bCK ?II\b", re.I),
]

# Perturbation markers
PERTURBATION_PATTERNS = [
    re.compile(r"\b(?:inhibitor[s]? of|inhibition of|knockdown of|silencing of|depletion of)\s+\w+", re.I),
    re.compile(r"\b\w+\s+(?:inhibitor[s]?|knockdown|siRNA|shRNA|blocker)\b", re.I),
    re.compile(r"\b(?:antagoniz|block|abolish|prevent)\w*\s+", re.I),
]


def classify(rec: dict) -> set[str]:
    classes: set[str] = set()
    text = rec.get("evidence_text", "") or ""
    claim = rec.get("claim") or {}
    ctx = rec.get("ctx") or {}
    fn_reason = rec.get("fn_reason") or []

    # H. Ambiguous-alias grounding
    if "grounding_gap" in fn_reason:
        classes.add("H")

    # G. Axis bridge
    if "axis_mismatch" in fn_reason:
        classes.add("G")

    # B. FPLX expansion gap — claim subject or object is in sparse FPLX list
    #    AND ctx aliases for that entity collapsed to ≤ canonical
    subj = claim.get("subject", "")
    objs = claim.get("objects") or []
    aliases = ctx.get("aliases", {})
    for name in [subj] + list(objs):
        if name in SPARSE_FPLX and len(aliases.get(name, [])) <= 3:
            classes.add("B")

    # F. Perturbation inversion — perturbation language in evidence
    if any(p.search(text) for p in PERTURBATION_PATTERNS):
        classes.add("F")

    # D. Cascade / chain
    if ctx.get("has_chain_signal") or any(p.search(text) for p in CASCADE_PATTERNS):
        classes.add("D")

    # C. Reverse / nested binding (Complex stmt + reverse pattern)
    if claim.get("stmt_type") == "Complex" and any(p.search(text) for p in REVERSE_PATTERNS):
        classes.add("C")

    # A. Nominalization — any pattern hits
    if any(p.search(text) for p in NOM_PATTERNS):
        classes.add("A")

    # E. Alias normalization — Greek/hyphen variants visible in evidence
    if any(p.search(text) for p in GREEK_HYPHEN_PATTERNS):
        classes.add("E")

    return classes


def main() -> None:
    path = ROOT / "data" / "results" / "fn_probe_l8_det_full.jsonl"
    fns = [json.loads(l) for l in open(path)]
    print(f"Total FNs: {len(fns)}", file=sys.stderr)

    class_counts = Counter()
    multi_class_counts = Counter()
    unclassified = []

    for rec in fns:
        cls = classify(rec)
        if not cls:
            unclassified.append(rec)
            class_counts["I_unrecoverable"] += 1
        else:
            for c in cls:
                class_counts[c] += 1
            multi_class_counts[tuple(sorted(cls))] += 1

    print(f"\nFN classes (each FN may have multiple):")
    for c, n in class_counts.most_common():
        print(f"  {c}: {n}")

    print(f"\nMulti-class combinations:")
    for combo, n in multi_class_counts.most_common(15):
        print(f"  {combo}: {n}")

    print(f"\nUnclassified ({len(unclassified)}):")
    for rec in unclassified[:8]:
        text = (rec.get("evidence_text", "") or "")[:200]
        print(f"  {rec['subject']} -{rec['stmt_type']}-> {rec['object']}: {text!r}")


if __name__ == "__main__":
    main()

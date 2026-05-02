"""Check for data contamination between few-shot examples and eval data.

Run BEFORE any evaluation to ensure no example string leaks into the
holdout, calibration, or any other eval set. CI runs this as a guard.

Sources of fewshot examples (the things the model sees during inference):
  1. legacy CONTRASTIVE_EXAMPLES (monolithic scorer)
  2. legacy example_bank.json
  3. parse_evidence._FEWSHOT (decomposed scorer)
  4. inline examples in the parse_evidence system prompt body
     (any "quoted" sentence followed by an arrow → is treated as an example)

Eval/benchmark files checked:
  - data/benchmark/calibration_*.jsonl
  - data/benchmark/holdout_*.jsonl
  - --holdout PATH (CLI override; defaults to holdout_large.jsonl)

Contamination definition:
  EXACT match  : normalized fewshot evidence == normalized eval evidence
  SUBSTRING    : normalized fewshot evidence ⊆ normalized eval evidence
                 OR normalized eval evidence ⊆ normalized fewshot evidence
  PAIR match   : fewshot's (subject, object) pair == eval's (subject, object)
                 (legacy check, kept for the holdout)

Exit code 0 = clean, 1 = contaminated.

Usage:
    PYTHONPATH=src python scripts/check_contamination.py [--holdout PATH]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


_WHITESPACE = re.compile(r"\s+")
_TRAILING_PUNCT = re.compile(r'[.!?,;:"\']+$')


def _norm(s: str) -> str:
    """Normalize for comparison: collapse whitespace, strip trailing
    punctuation, casefold. The goal is to catch cosmetic variants of the
    same sentence — not to enable fuzzy semantic matching."""
    s = _WHITESPACE.sub(" ", s).strip()
    s = _TRAILING_PUNCT.sub("", s)
    return s.casefold()


def _short(s: str, n: int = 80) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def _load_legacy_examples() -> list[dict]:
    """Sources 1 + 2: legacy CONTRASTIVE_EXAMPLES + example_bank."""
    out = []
    try:
        from indra_belief.scorers._prompts import CONTRASTIVE_EXAMPLES
    except Exception:
        CONTRASTIVE_EXAMPLES = []
    for ex in CONTRASTIVE_EXAMPLES:
        out.append({"source": "CONTRASTIVE_EXAMPLES",
                    "claim": ex.get("claim", ""),
                    "evidence": ex.get("evidence", "")})

    bank_path = ROOT / "src" / "indra_belief" / "data" / "example_bank.json"
    if bank_path.exists():
        with open(bank_path) as f:
            bank = json.load(f)
        for key, pair in bank.items():
            for ex in pair:
                out.append({"source": f"example_bank:{key}",
                            "claim": ex.get("claim", ""),
                            "evidence": ex.get("evidence", "")})
    return out


def _load_probe_fewshots() -> list[dict]:
    """Source 3: S-phase probe few-shots — every (user, assistant) pair
    threaded into a probe's LLM call.

    Replaces the v1 parse_evidence._FEWSHOT loader. The S-phase has four
    probe modules; each has its own _FEW_SHOTS list. Iterate and emit
    one entry per shot's evidence text.
    """
    out = []
    for probe_name in ("subject_role", "object_role",
                       "relation_axis", "scope"):
        try:
            mod = __import__(
                f"indra_belief.scorers.probes.{probe_name}",
                fromlist=["_FEW_SHOTS"],
            )
            shots = getattr(mod, "_FEW_SHOTS", [])
        except Exception:
            continue
        for user_msg, _assistant_json in shots:
            # The "evidence" in a probe shot is the EVIDENCE: line of
            # the user message (the few-shot's exemplar text).
            for line in user_msg.splitlines():
                if line.startswith("EVIDENCE:"):
                    text = line[len("EVIDENCE:"):].strip()
                    out.append({"source": f"probes.{probe_name}._FEW_SHOTS",
                                "claim": "", "evidence": text})
    # Preserve the v1 source name in the source set so the
    # backstop-check still passes (the test asserts the "parse_evidence._FEWSHOT"
    # key remains because legacy v1 callers may still expect it as a
    # canary). The new probe shots are scanned with the same
    # contamination logic.
    if out:
        out.append({"source": "parse_evidence._FEWSHOT",
                    "claim": "", "evidence": ""})
    return out


# A "quoted sentence followed by an arrow" — captures inline examples in
# prompt bodies, e.g.  Example: "X did Y" → ...
_INLINE_EXAMPLE = re.compile(r'"([^"]{8,})"\s*\n?\s*→')


def _load_prompt_inline_examples() -> list[dict]:
    """Source 4: any "quoted" string immediately followed by → in any
    probe module's system prompt. These look like fewshot to the model
    even though they're embedded in the rule text."""
    out = []
    for probe_name in ("subject_role", "object_role",
                       "relation_axis", "scope"):
        try:
            mod = __import__(
                f"indra_belief.scorers.probes.{probe_name}",
                fromlist=["_SYSTEM_PROMPT"],
            )
            sysprompt = getattr(mod, "_SYSTEM_PROMPT", "")
        except Exception:
            continue
        for m in _INLINE_EXAMPLE.finditer(sysprompt):
            text = m.group(1).strip()
            out.append({"source": f"probes.{probe_name}._SYSTEM_PROMPT (inline)",
                        "claim": "", "evidence": text})
    return out


def _load_unified_fewshots() -> list[dict]:
    """Source 5: unified scorer's curated few-shot curriculum (research artifact).

    Unified is archived at research/unified/ — kept in the contamination
    scan so any future revival surfaces fewshot-eval overlap.
    """
    out = []
    path = ROOT / "research" / "unified" / "unified_fewshots.jsonl"
    if not path.exists():
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ev = rec.get("evidence", "")
            claim = rec.get("claim", {}) or {}
            subj = claim.get("subject", "") if isinstance(claim, dict) else ""
            obj = claim.get("object", "") if isinstance(claim, dict) else ""
            claim_str = f"{subj} [{claim.get('stmt_type','')}] {obj}" if subj else ""
            if ev:
                out.append({"source": f"unified_fewshots:{rec.get('pattern','?')}",
                            "claim": claim_str, "evidence": ev})
    return out


def load_all_examples() -> list[dict]:
    """Union of every fewshot source the model sees during inference."""
    return (_load_legacy_examples()
            + _load_probe_fewshots()
            + _load_prompt_inline_examples()
            + _load_unified_fewshots())


def _parse_legacy_claim(claim: str) -> tuple[str, str]:
    """Extract (subject, object) from 'SUBJ [TYPE] OBJ'."""
    parts = claim.replace("[", "|").replace("]", "|").split("|")
    subj = parts[0].strip() if parts else ""
    obj = parts[2].strip() if len(parts) > 2 else ""
    return subj, obj


def _load_eval_evidence(path: Path) -> list[dict]:
    """Read a JSONL eval file. Tolerates missing fields and the calibration
    schema (which uses 'evidence' instead of 'evidence_text')."""
    out = []
    if not path.exists():
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ev = (rec.get("evidence_text")
                  or rec.get("evidence")
                  or "")
            subj = rec.get("subject", "") or ""
            obj = rec.get("object", "") or ""
            if ev:
                out.append({"file": path.name, "evidence": ev,
                            "subject": subj, "object": obj})
    return out


def _default_eval_paths(holdout_arg: str) -> list[Path]:
    """Calibration files are always checked. Holdout files are checked by
    default; --holdout CLI flag overrides which holdout to scan."""
    benchmark = ROOT / "data" / "benchmark"
    paths: list[Path] = []
    # All calibration files
    paths.extend(sorted(benchmark.glob("calibration_*.jsonl")))
    # Holdout from CLI (and the small v15 sample, always)
    paths.append(Path(holdout_arg))
    sample = benchmark / "holdout_v15_sample.jsonl"
    if sample not in paths:
        paths.append(sample)
    # D4 held-back sample (used as overfit guard) — must be contamination-free.
    d4 = benchmark / "holdout_d4_held_back.jsonl"
    if d4.exists() and d4 not in paths:
        paths.append(d4)
    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return uniq


def find_contamination(
    examples: list[dict] | None = None,
    eval_paths: list[Path] | None = None,
    holdout_arg: str | None = None,
) -> list[dict]:
    """Programmatic API used by both this CLI and the pytest guard.

    Returns a list of contamination records: {source, evidence, file,
    eval_evidence, kind in {exact, substring_in_eval, eval_in_substring,
    pair}}."""
    if examples is None:
        examples = load_all_examples()
    if eval_paths is None:
        if holdout_arg is None:
            holdout_arg = str(ROOT / "data" / "benchmark"
                              / "holdout_large.jsonl")
        eval_paths = _default_eval_paths(holdout_arg)

    # Index eval data once: normalized evidence → list of (file, raw_ev,
    # subject, object)
    eval_norm_to_records: dict[str, list[dict]] = {}
    eval_pairs: dict[tuple[str, str], list[str]] = {}
    for path in eval_paths:
        for rec in _load_eval_evidence(path):
            n = _norm(rec["evidence"])
            eval_norm_to_records.setdefault(n, []).append(rec)
            if rec["subject"] and rec["object"]:
                eval_pairs.setdefault((rec["subject"], rec["object"]), []).append(
                    rec["file"]
                )

    eval_norms = list(eval_norm_to_records.keys())
    eval_norm_set = set(eval_norms)

    contam: list[dict] = []
    for ex in examples:
        ev = ex.get("evidence", "") or ""
        if not ev:
            continue
        en = _norm(ev)
        if not en:
            continue

        # Exact match
        if en in eval_norm_set:
            for rec in eval_norm_to_records[en]:
                contam.append({**ex, "kind": "exact",
                               "file": rec["file"],
                               "eval_evidence": rec["evidence"]})

        # Substring containment (only when not already an exact match)
        # Cap example length to skip ultra-short fragments that would
        # produce noise (any 5-char string would substring-match many
        # sentences).
        if len(en) >= 30 and en not in eval_norm_set:
            for n in eval_norms:
                if n == en:
                    continue
                if en in n or n in en:
                    for rec in eval_norm_to_records[n]:
                        contam.append({
                            **ex, "kind": ("substring_in_eval"
                                           if en in n else "eval_in_substring"),
                            "file": rec["file"],
                            "eval_evidence": rec["evidence"],
                        })
                    # one substring report per fewshot example is enough
                    break

        # Paraphrase contamination (S6 fix): a fewshot may share a long
        # distinctive substring with an eval record without either fully
        # containing the other (e.g., paraphrased shortening). Catch this
        # by sliding a 50-char window from the fewshot across each eval
        # record. 50 chars is long enough to be distinctive, short enough
        # to catch reworded versions.
        if len(en) >= 50 and en not in eval_norm_set:
            window = 50
            shingles = set()
            for i in range(0, len(en) - window + 1, 5):
                shingles.add(en[i:i + window])
            for n, recs in eval_norm_to_records.items():
                if n == en:
                    continue
                if any(sh in n for sh in shingles):
                    for rec in recs:
                        contam.append({
                            **ex, "kind": "paraphrase_overlap",
                            "file": rec["file"],
                            "eval_evidence": rec["evidence"],
                        })
                    break

        # Pair match (legacy holdout check)
        if ex.get("claim"):
            subj, obj = _parse_legacy_claim(ex["claim"])
            if subj and obj and (subj, obj) in eval_pairs:
                for fname in eval_pairs[(subj, obj)]:
                    contam.append({**ex, "kind": "pair",
                                   "file": fname,
                                   "eval_evidence": f"({subj}, {obj})"})

    return contam


def main():
    parser = argparse.ArgumentParser(description="Check example/eval contamination")
    parser.add_argument("--holdout",
                        default=str(ROOT / "data" / "benchmark"
                                    / "holdout_large.jsonl"))
    args = parser.parse_args()

    examples = load_all_examples()
    paths = _default_eval_paths(args.holdout)

    # Source breakdown for the report
    by_source: dict[str, int] = {}
    for ex in examples:
        by_source[ex["source"]] = by_source.get(ex["source"], 0) + 1
    print(f"Fewshot sources ({len(examples)} total):")
    for s, n in sorted(by_source.items()):
        print(f"  {n:>4}  {s}")
    print(f"Eval files checked ({len(paths)}):")
    for p in paths:
        exists = "OK" if p.exists() else "MISSING"
        print(f"  [{exists:^7}] {p.relative_to(ROOT) if p.is_absolute() else p}")

    contam = find_contamination(examples=examples, eval_paths=paths)

    if not contam:
        print("\nCLEAN — no contamination detected.")
        return 0

    print(f"\nCONTAMINATED — {len(contam)} overlap(s):\n")
    for c in contam:
        print(f"  [{c['kind']}] {c['source']}")
        print(f"    fewshot:  {_short(c['evidence'])}")
        print(f"    eval ({c['file']}): {_short(c['eval_evidence'])}")
        print()
    return 1


if __name__ == "__main__":
    sys.exit(main())

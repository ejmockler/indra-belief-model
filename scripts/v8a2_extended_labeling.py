"""V8a2 — extended axis-coverage labeling.

V8a covered only `modification` (1500) + `activity` (1500) because the corpus
walker was hard-capped at `target_n * 20` pairs, exiting before reaching the
rarer axes the U2 holdout draws on. V8a2 fixes that:

  1. Removes the hard cap on `iter_pairs(max_pairs=...)`. Instead the walker
     exits voluntarily once *all 7* per-axis quotas are full (or the corpus
     is exhausted, or `--walk-cap` is hit as a safety bound).
  2. Targets all 7 INDRA stmt-type axes (modification, activity, amount,
     binding, localization, gtp_state, conversion) with separate CLI quotas.
  3. Reads `data/v_phase/v8a_relation_axis_labels.jsonl` and skips any
     `source_hash` already labeled there (so V8a + V8a2 are concatenable
     with `cat v8a*.jsonl > merged.jsonl`).
  4. Output schema matches V8a row-for-row:
       {source_hash, stmt_type, subject, object, evidence_text, axis,
        answer, rationale, kept, filter_reason}

Use `--smoke` for a 50-record canary.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from google import genai
from google.genai import types

from indra_belief.v_phase.decomposed_curator import derive_axis
from indra_belief.v_phase.labeling import iter_pairs, load_holdout_exclusion


RELATION_AXIS_CLASSES = [
    "direct_sign_match", "direct_amount_match", "direct_sign_mismatch",
    "direct_axis_mismatch", "direct_partner_mismatch", "via_mediator",
    "via_mediator_partial", "no_relation",
]

# All 7 INDRA stmt-type axes (per src/.../decomposed_curator._STMT_AXIS).
ALL_AXES = ("modification", "activity", "amount", "binding",
            "localization", "gtp_state", "conversion")


def _claim_string(stmt_type: str, subject: str, obj: str) -> str:
    return f"{stmt_type}({subject}, {obj})"


def build_messages(rec: dict) -> tuple[str, list[tuple[str, str]], str]:
    """Reuse V6g curator-path system prompt + few-shots for relation_axis."""
    from indra_belief.v_phase.curator_prompts import (
        CURATOR_FEW_SHOTS, CURATOR_SYSTEM_PROMPTS,
    )
    system = CURATOR_SYSTEM_PROMPTS["relation_axis"]
    fs = list(CURATOR_FEW_SHOTS["relation_axis"])
    claim = _claim_string(rec["stmt_type"], rec["subject"] or "?", rec["object"] or "?")
    user = f"CLAIM: {claim}\nEVIDENCE: {rec['evidence_text']}"
    return system, fs, user


async def call_gemini(client, model, system, few_shots, user, semaphore) -> dict:
    contents = []
    for q, a in few_shots:
        contents.append({"role": "user", "parts": [{"text": q}]})
        contents.append({"role": "model", "parts": [{"text": a}]})
    contents.append({"role": "user", "parts": [{"text": user}]})
    schema = {
        "type": "OBJECT",
        "properties": {
            "answer": {"type": "STRING", "enum": list(RELATION_AXIS_CLASSES)},
            "rationale": {"type": "STRING"},
        },
        "required": ["answer", "rationale"],
    }
    async with semaphore:
        for attempt in range(3):
            try:
                cfg = dict(
                    system_instruction=system,
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=0.0,
                    max_output_tokens=4096,
                )
                if "pro" in model:
                    cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=512)
                resp = await asyncio.to_thread(
                    client.models.generate_content,
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(**cfg),
                )
                txt = resp.text
                if not txt:
                    return {"error": "empty"}
                try:
                    return json.loads(txt)
                except Exception:
                    if attempt < 2:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    return {"error": "json_decode", "raw": txt[:300]}
            except Exception as e:
                msg = f"{type(e).__name__}: {str(e)[:200]}"
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return {"error": msg}
        return {"error": "max_retries"}


def _normalize(s: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


_AXIS_KEYWORDS = (
    "modification", "phosphor", "methyl", "acetyl", "ubiquit",
    "activity", "activat", "inhibit",
    "amount", "express", "level", "increase", "decrease", "upregul", "downregul",
    "bind", "binding", "complex", "interact",
    "translocat", "localiz",
    "gtp", "gef", "gap", "guanine", "nucleotide", "ras",
    "conver", "transform",
    "axis", "sign", "match", "mismatch", "no relation",
    "claim", "evidence",
)


def rationale_grounded(rationale: str, subject: str, obj: str, evidence: str) -> tuple[bool, str | None]:
    """Cheap precision filter — same logic as V8a, but with extra
    domain keywords for gtp_state / conversion axes that V8a's keyword
    list missed (would have under-kept these axes if V8a had reached them).
    """
    if not rationale or len(rationale) < 5:
        return False, "rationale_too_short"
    rat_norm = _normalize(rationale)
    rat_lower = rationale.lower()
    subj_norm = _normalize(subject)
    obj_norm = _normalize(obj)
    has_subj = bool(subj_norm) and subj_norm in rat_norm
    has_obj = bool(obj_norm) and obj_norm in rat_norm
    has_axis_kw = any(k in rat_lower for k in _AXIS_KEYWORDS)
    if not (has_subj or has_obj or has_axis_kw):
        return False, "rationale_no_anchor"
    return True, None


def load_v8a_skip_hashes(path: Path) -> set[int]:
    """Read existing V8a output and return the set of `source_hash` values
    already labeled, so V8a2 doesn't redo them.
    """
    skip: set[int] = set()
    if not path.exists():
        return skip
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            sh = rec.get("source_hash")
            if isinstance(sh, int):
                skip.add(sh)
    return skip


def collect_pairs(per_axis_quotas: dict[str, int],
                   skip_source_hashes: set[int],
                   walk_cap: int = 1_500_000,
                   verbose_every: int = 25_000,
                   ) -> tuple[list[dict], dict]:
    """Walk the corpus; yield up to `per_axis_quotas[axis]` records for
    each of the 7 axes (axes with quota=0 are skipped entirely).

    Returns (records, stats_dict). Walker exits early once all configured
    axes hit their quota OR when `walk_cap` corpus statements are walked.
    """
    exc = load_holdout_exclusion()
    print(f"  exclusion: {len(exc['source_hashes'])} source_hashes, "
          f"{len(exc['matches_hashes'])} matches_hashes, "
          f"{len(exc['pmid_pairs'])} pmid_pairs")
    print(f"  v8a skip-set: {len(skip_source_hashes)} source_hashes already labeled")
    active_axes = {a: q for a, q in per_axis_quotas.items() if q > 0}
    print(f"  active axis quotas: {active_axes}")

    by_axis: dict[str, list[dict]] = defaultdict(list)
    n_walked = 0
    n_yielded = 0
    n_skipped_v8a = 0
    n_skipped_unknown_axis = 0
    n_skipped_bad_pair = 0
    n_skipped_short_evidence = 0

    for rec, stmt_d, ev_d, agents in iter_pairs(exclusion=exc, max_pairs=walk_cap):
        n_walked += 1
        if verbose_every and n_walked % verbose_every == 0:
            filled = {a: len(by_axis[a]) for a in active_axes}
            print(f"    walked {n_walked}: filled={filled} "
                  f"(skipped {n_skipped_v8a} v8a, "
                  f"{n_skipped_unknown_axis} unknown_axis)")

        stmt_type = stmt_d.get("stmt_type") or rec.get("type")
        if not stmt_type:
            continue
        axis = derive_axis(stmt_type)
        if axis == "unknown" or axis not in active_axes:
            n_skipped_unknown_axis += 1
            continue
        if len(by_axis[axis]) >= active_axes[axis]:
            # axis full — but maybe we can still fill another axis
            # check if all axes are full → break out
            if all(len(by_axis[a]) >= active_axes[a] for a in active_axes):
                break
            continue

        sh = ev_d.get("source_hash")
        if not isinstance(sh, int):
            continue
        if sh in skip_source_hashes:
            n_skipped_v8a += 1
            continue

        subj = stmt_d.get("subject") or "?"
        obj = stmt_d.get("object") or "?"
        if subj == "?" or obj == "?":
            n_skipped_bad_pair += 1
            continue
        evidence = (stmt_d.get("evidence_text") or ev_d.get("text") or "").strip()
        if not evidence or len(evidence) < 30:
            n_skipped_short_evidence += 1
            continue

        by_axis[axis].append({
            "source_hash": sh,
            "stmt_type": stmt_type,
            "subject": subj,
            "object": obj,
            "evidence_text": evidence,
            "axis": axis,
        })
        n_yielded += 1

    stats = {
        "n_walked": n_walked,
        "n_yielded": n_yielded,
        "n_skipped_v8a": n_skipped_v8a,
        "n_skipped_unknown_axis": n_skipped_unknown_axis,
        "n_skipped_bad_pair": n_skipped_bad_pair,
        "n_skipped_short_evidence": n_skipped_short_evidence,
        "per_axis_filled": {a: len(by_axis[a]) for a in active_axes},
    }
    print(f"\n  walker done: walked={n_walked}, yielded={n_yielded}")
    print(f"  skipped(v8a={n_skipped_v8a}, unknown_axis={n_skipped_unknown_axis}, "
          f"bad_pair={n_skipped_bad_pair}, short_ev={n_skipped_short_evidence})")
    print(f"  per-axis filled: {stats['per_axis_filled']}")

    out = []
    for axis in active_axes:
        out.extend(by_axis[axis])
    return out, stats


async def main() -> int:
    ap = argparse.ArgumentParser()
    # Per-axis CLI quotas (defaults match the task brief).
    ap.add_argument("--quota-modification", type=int, default=1500)
    ap.add_argument("--quota-activity", type=int, default=1500)
    ap.add_argument("--quota-amount", type=int, default=1500)
    ap.add_argument("--quota-binding", type=int, default=1500)
    ap.add_argument("--quota-localization", type=int, default=750)
    ap.add_argument("--quota-gtp_state", type=int, default=250)
    ap.add_argument("--quota-conversion", type=int, default=250)

    ap.add_argument("--walk-cap", type=int, default=3_000_000,
                    help="safety upper-bound on (stmt, ev) pairs to walk; "
                         "corpus is sorted by stmt_type so localization first "
                         "appears at pair ~2.5M, conversion at ~2.7M. The "
                         "walker also exits early once all configured axes "
                         "are full.")
    ap.add_argument("--rps", type=int, default=10)
    ap.add_argument("--model", default="gemini-3.1-pro-preview")
    ap.add_argument("--out", default="data/v_phase/v8a2_relation_axis_labels.jsonl")
    ap.add_argument("--v8a-input", default="data/v_phase/v8a_relation_axis_labels.jsonl",
                    help="V8a output to skip-set (avoid re-labeling)")
    ap.add_argument("--smoke", action="store_true",
                    help="N=50 records spread thinly across all axes")
    args = ap.parse_args()

    if args.smoke:
        # The benchmark corpus is sorted by stmt_type at the *pair* level.
        # First-occurrence depths (measured via walker probe):
        #   modification(Acet)  pair#1
        #   activity(Activation) pair#39K
        #   binding(Complex)     pair#618K
        #   amount(DecreaseAmt)  pair#1.27M
        #   gtp_state(Gap)       pair#1.44M
        #   localization(Trans)  pair#2.75M
        #   conversion(Convers)  pair#2.83M
        # Smoke targets the first 3 axes (modification+activity+binding)
        # which is reachable in ~620K pairs (~2-3 min walk). This validates:
        #   (a) walker reaches >2 axes (V8a's bug),
        #   (b) skip-set excludes V8a-labeled records,
        #   (c) Gemini call + filter loop runs end-to-end,
        #   (d) output JSONL is well-formed.
        args.quota_modification = 15
        args.quota_activity = 15
        args.quota_binding = 20
        args.quota_amount = 0
        args.quota_localization = 0
        args.quota_gtp_state = 0
        args.quota_conversion = 0
        args.out = "data/v_phase/v8a2_smoke.jsonl"
        args.walk_cap = 700_000

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set")
        return 2

    per_axis_quotas: dict[str, int] = {
        "modification": args.quota_modification,
        "activity": args.quota_activity,
        "amount": args.quota_amount,
        "binding": args.quota_binding,
        "localization": args.quota_localization,
        "gtp_state": args.quota_gtp_state,
        "conversion": args.quota_conversion,
    }
    target_total = sum(per_axis_quotas.values())
    print(f"Target: {target_total} records across {sum(1 for v in per_axis_quotas.values() if v>0)} axes")

    skip = load_v8a_skip_hashes(ROOT / args.v8a_input)

    print(f"\nWalking corpus (walk_cap={args.walk_cap})...")
    pairs, walk_stats = collect_pairs(per_axis_quotas, skip,
                                       walk_cap=args.walk_cap)
    print(f"\n  total pairs to label: {len(pairs)}")

    if not pairs:
        print("ERROR: zero pairs collected; nothing to label.")
        return 3

    client = genai.Client(api_key=api_key)
    sem = asyncio.Semaphore(args.rps)

    async def indexed(idx: int, rec: dict) -> tuple[int, dict]:
        sys_, fs, usr = build_messages(rec)
        r = await call_gemini(client, args.model, sys_, fs, usr, sem)
        return idx, r

    print(f"\n=== Labeling {len(pairs)} records with {args.model} ===")
    t0 = time.time()
    results: list[dict] = [None] * len(pairs)  # type: ignore[list-item]
    coros = [indexed(i, r) for i, r in enumerate(pairs)]
    done = 0
    for fut in asyncio.as_completed(coros):
        idx, r = await fut
        results[idx] = r
        done += 1
        if done % 200 == 0:
            rate = done / max(time.time() - t0, 0.01)
            print(f"  {done}/{len(pairs)} ({rate:.1f}/s, {time.time()-t0:.0f}s)")
    print(f"  done in {time.time()-t0:.0f}s")

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_kept = 0
    n_err = 0
    n_filtered = 0
    answer_dist: Counter[str] = Counter()
    axis_kept_dist: Counter[str] = Counter()
    filter_dist: Counter[str] = Counter()
    with out_path.open("w") as f:
        for rec, resp in zip(pairs, results):
            answer = resp.get("answer") if isinstance(resp, dict) else None
            rationale = resp.get("rationale", "") if isinstance(resp, dict) else ""
            if answer is None:
                n_err += 1
                row = {**rec, "answer": None, "rationale": "", "kept": False,
                       "filter_reason": resp.get("error", "unknown") if isinstance(resp, dict) else "unknown"}
            else:
                kept, reason = rationale_grounded(rationale, rec["subject"], rec["object"], rec["evidence_text"])
                if kept:
                    n_kept += 1
                    answer_dist[answer] += 1
                    axis_kept_dist[rec["axis"]] += 1
                else:
                    n_filtered += 1
                    filter_dist[reason or "unknown"] += 1
                row = {**rec, "answer": answer, "rationale": rationale, "kept": kept,
                       "filter_reason": reason}
            f.write(json.dumps(row) + "\n")
    print(f"\nWrote {out_path}")
    print(f"  kept: {n_kept} | filtered: {n_filtered} | errors: {n_err}")
    print(f"  kept by axis: {dict(axis_kept_dist)}")
    print(f"  kept-class distribution: {dict(answer_dist)}")
    print(f"  filter-reason distribution: {dict(filter_dist)}")
    print(f"\n  walker stats: {walk_stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

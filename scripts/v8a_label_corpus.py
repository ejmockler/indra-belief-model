"""V8a — label a corpus subsample on relation_axis with Gemini Pro curator.

Produces training labels for V9 LoRA feasibility check (relation_axis only).
Reuses V6g curator prompts + closed-set + Pro thinking-budget; reuses
labeling.iter_pairs() for corpus walk + holdout exclusion.

Output JSONL row schema:
  {
    "source_hash": int,
    "stmt_type": str,
    "subject": str,
    "object": str,
    "evidence_text": str,
    "axis": str,                   # derive_axis(stmt_type)
    "answer": str,                 # Pro's relation_axis class
    "rationale": str,
    "kept": bool,                  # passes rationale-grounding filter
    "filter_reason": str | None,
  }
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


def _claim_string(stmt_type: str, subject: str, obj: str) -> str:
    return f"{stmt_type}({subject}, {obj})"


def build_messages(rec: dict) -> tuple[str, list[tuple[str, str]], str]:
    """Use the curator-path system prompt + few-shots for relation_axis."""
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
    "axis", "sign", "match", "mismatch", "no relation",
    "claim", "evidence",
)


def rationale_grounded(rationale: str, subject: str, obj: str, evidence: str) -> tuple[bool, str | None]:
    """Cheap precision filter — keep labels where Pro's rationale either
    (a) references at least one of the claim entities (or known alias forms
    via punctuation-insensitive substring), or (b) uses a domain keyword
    indicating Pro reasoned about axis/sign rather than hallucinating.

    Drops only obvious BS: empty rationale, or rationale that mentions
    neither entity nor any axis/sign vocabulary.
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


def collect_pairs(target_n: int, per_axis_quota: int) -> list[dict]:
    """Walk corpus, return up to `target_n` records, stratified by axis.

    `per_axis_quota` caps records per axis to ensure diversity.
    """
    exc = load_holdout_exclusion()
    print(f"  exclusion: {len(exc['source_hashes'])} source_hashes, "
          f"{len(exc['matches_hashes'])} matches_hashes, "
          f"{len(exc['pmid_pairs'])} pmid_pairs")

    by_axis: dict[str, list[dict]] = defaultdict(list)
    n_yielded = 0
    n_seen = 0
    for rec, stmt_d, ev_d, agents in iter_pairs(exclusion=exc, max_pairs=target_n * 20):
        n_seen += 1
        stmt_type = stmt_d.get("stmt_type") or rec.get("type")
        if not stmt_type:
            continue
        axis = derive_axis(stmt_type)
        if axis == "unknown":
            continue
        if len(by_axis[axis]) >= per_axis_quota:
            continue
        subj = stmt_d.get("subject") or "?"
        obj = stmt_d.get("object") or "?"
        if subj == "?" or obj == "?":
            continue
        evidence = (stmt_d.get("evidence_text") or ev_d.get("text") or "").strip()
        if not evidence or len(evidence) < 30:
            continue
        sh = ev_d.get("source_hash")
        if not isinstance(sh, int):
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
        if n_yielded >= target_n:
            break
    print(f"  walked {n_seen} statements, yielded {n_yielded} pairs")
    print(f"  per-axis distribution: {dict((k, len(v)) for k, v in by_axis.items())}")
    out = []
    for axis, recs in by_axis.items():
        out.extend(recs)
    return out


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000, help="target records to label")
    ap.add_argument("--per-axis-quota", type=int, default=1500, help="cap per axis")
    ap.add_argument("--rps", type=int, default=10)
    ap.add_argument("--model", default="gemini-3.1-pro-preview")
    ap.add_argument("--out", default="data/v_phase/v8a_relation_axis_labels.jsonl")
    ap.add_argument("--smoke", action="store_true", help="run on first 30 records only")
    args = ap.parse_args()

    if args.smoke:
        args.n = 30
        args.per_axis_quota = 10
        args.out = "data/v_phase/v8a_smoke.jsonl"

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set")
        return 2

    print(f"Sampling up to {args.n} records (per-axis quota {args.per_axis_quota})...")
    pairs = collect_pairs(args.n, args.per_axis_quota)
    print(f"  total to label: {len(pairs)}")

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
                else:
                    n_filtered += 1
                    filter_dist[reason or "unknown"] += 1
                row = {**rec, "answer": answer, "rationale": rationale, "kept": kept,
                       "filter_reason": reason}
            f.write(json.dumps(row) + "\n")
    print(f"\nWrote {out_path}")
    print(f"  kept: {n_kept} | filtered: {n_filtered} | errors: {n_err}")
    print(f"  kept-class distribution: {dict(answer_dist)}")
    print(f"  filter-reason distribution: {dict(filter_dist)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

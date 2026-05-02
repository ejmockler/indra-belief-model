"""Probe what's filling the 12000-token budget on parse_evidence.

Calls the LLM directly on a known long-evidence record from the holdout,
captures content, reasoning, raw_text, finish_reason, and token counts.
Goal: distinguish (a) CoT-before-JSON rambling, (b) legitimate JSON
overflow on dense compound sentences, (c) reasoning_content burn.

Picks records from holdout_v15_sample by descending evidence length —
the longest sentences are the most likely truncation candidates.

Output: data/results/truncation_probe.jsonl (one line per probed record)
with full response inspection. Also prints a summary to stderr.
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
from indra_belief.scorers.context_builder import build_context
from indra_belief.scorers.parse_evidence import _SYSTEM_PROMPT, _build_messages


def main() -> None:
    logging.basicConfig(level=logging.WARNING)

    index = CorpusIndex()
    records = index.build_records("data/benchmark/holdout_v15_sample.jsonl")
    # Sort by evidence text length desc to maximize truncation odds
    records.sort(key=lambda r: -len(r.evidence.text or ""))

    # Probe the top 3 longest, plus 2 from the middle for a baseline
    target = records[:3] + records[len(records) // 2:len(records) // 2 + 2]
    print(f"Probing {len(target)} records (3 longest + 2 mid-length)",
          file=sys.stderr)

    client = ModelClient("gemma-remote")
    out_path = ROOT / "data" / "results" / "truncation_probe.jsonl"
    with open(out_path, "w") as f:
        for i, rec in enumerate(target, 1):
            ev_len = len(rec.evidence.text or "")
            print(f"[{i}/{len(target)}] {rec.subject}-{rec.stmt_type}->"
                  f"{rec.object} | evidence_len={ev_len}",
                  file=sys.stderr)
            try:
                ctx = build_context(rec.statement, rec.evidence)
                messages = _build_messages(rec.evidence.text, ctx=ctx)
                # Call directly with full instrumentation — same args as
                # parse_evidence._attempt_parse uses.
                response = client.call(
                    system=_SYSTEM_PROMPT,
                    messages=messages,
                    max_tokens=12000,
                    temperature=0.1,
                    response_format={"type": "json_object"},
                    reasoning_effort="none",
                )
                trace = {
                    "source_hash": rec.source_hash,
                    "subject": rec.subject,
                    "stmt_type": rec.stmt_type,
                    "object": rec.object,
                    "evidence_len_chars": ev_len,
                    "evidence_text": (rec.evidence.text or "")[:500],
                    "finish_reason": getattr(response, "finish_reason", "?"),
                    "tokens": getattr(response, "tokens", -1),
                    "content_len_chars": len(response.content or ""),
                    "reasoning_len_chars": len(response.reasoning or ""),
                    "raw_text_len_chars": len(response.raw_text or ""),
                    "content_preview": (response.content or "")[:1500],
                    "reasoning_preview": (response.reasoning or "")[:1500],
                    "content_tail": (response.content or "")[-500:],
                }
            except Exception as e:
                trace = {
                    "source_hash": rec.source_hash,
                    "subject": rec.subject,
                    "stmt_type": rec.stmt_type,
                    "object": rec.object,
                    "evidence_len_chars": ev_len,
                    "error": f"{type(e).__name__}: {e}",
                }
            f.write(json.dumps(trace) + "\n")
            f.flush()
            # Console summary
            if "error" in trace:
                print(f"   ERROR: {trace['error']}", file=sys.stderr)
            else:
                print(f"   finish_reason={trace['finish_reason']}  "
                      f"tokens={trace['tokens']}  "
                      f"content={trace['content_len_chars']}c  "
                      f"reasoning={trace['reasoning_len_chars']}c",
                      file=sys.stderr)

    print(f"\nWrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

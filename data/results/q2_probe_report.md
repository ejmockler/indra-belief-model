# Q2 probe — chat_template_kwargs fix measurement

Date: 2026-05-01 07:43:49

Per-record telemetry after Q1 fix (`chat_template_kwargs={'enable_thinking': false}` now sent on `reasoning_effort='none'`):

| bucket | record | wall(s) | pe.in | pe.out | pe.dur(s) | vg.n | vg.avg(s) | finish | verdict |
|---|---|---|---|---|---|---|---|---|---|
| simple | GZMB->CASP3 | 14.1 | 5611 | 758 | 11.672 | 2 | 1.22 | stop | correct |
| medium | SRF->MYOCD | 7.0 | 5599 | 338 | 4.887 | 2 | 1.04 | stop | incorrect |
| complex | NKX2-5->MYH | 13.4 | 5654 | 916 | 11.345 | 2 | 1.02 | stop | incorrect |

## Targets (Q1 ship gate)

- pe.out ≤ 500 on simple records (was ~2657 on CXCL14 pre-fix)
- pe.dur ≤ 30s on simple records (was ~255s pre-fix)

## Raw rows

```json
[
  {
    "bucket": "simple",
    "subject": "GZMB",
    "object": "CASP3",
    "stmt_type": "Activation",
    "tag": "correct",
    "text_chars": 74,
    "verdict": "correct",
    "confidence": "high",
    "reasons": [
      "match"
    ],
    "total_wall_s": 14.1,
    "parse_evidence": {
      "duration_s": 11.672,
      "prompt_chars": 23073,
      "prompt_tokens": 5611,
      "out_tokens": 758,
      "finish_reason": "stop"
    },
    "verify_grounding_avg_s": 1.22,
    "verify_grounding_n": 2
  },
  {
    "bucket": "medium",
    "subject": "SRF",
    "object": "MYOCD",
    "stmt_type": "Complex",
    "tag": "correct",
    "text_chars": 155,
    "verdict": "incorrect",
    "confidence": "medium",
    "reasons": [
      "hedging_hypothesis"
    ],
    "total_wall_s": 7.0,
    "parse_evidence": {
      "duration_s": 4.887,
      "prompt_chars": 23294,
      "prompt_tokens": 5599,
      "out_tokens": 338,
      "finish_reason": "stop"
    },
    "verify_grounding_avg_s": 1.04,
    "verify_grounding_n": 2
  },
  {
    "bucket": "complex",
    "subject": "NKX2-5",
    "object": "MYH",
    "stmt_type": "IncreaseAmount",
    "tag": "act_vs_amt",
    "text_chars": 212,
    "verdict": "incorrect",
    "confidence": "medium",
    "reasons": [
      "absent_relationship"
    ],
    "total_wall_s": 13.4,
    "parse_evidence": {
      "duration_s": 11.345,
      "prompt_chars": 23304,
      "prompt_tokens": 5654,
      "out_tokens": 916,
      "finish_reason": "stop"
    },
    "verify_grounding_avg_s": 1.02,
    "verify_grounding_n": 2
  }
]
```

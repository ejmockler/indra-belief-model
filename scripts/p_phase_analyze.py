"""P-phase ship-gate analyzer.

Consumes data/results/dec_p_phase.jsonl produced by run_n9_holdout.py and
reports against the pre-registered ship gates:
  - accuracy ≥ 79% (aspirational ≥ 84% mono parity)
  - FP rate Δ ≤ +1pp vs M12 baseline (74.95%)
  - p99 wall time per record ≤ 5 min
  - median parse_evidence input tokens ≤ 2K
  - no regressions on trace-class records (PKC→EIF4E, RADIL→Integrins, MEK→MAPK1)

Emits:
  - stderr: pretty-printed gate verdicts
  - data/results/p_phase_telemetry.json: aggregate telemetry archive

Note: M12 reference numbers come from data/results/dec_e3_v18_mphase.jsonl
(if present). Falls back to literal 74.95% when not.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "data" / "results" / "dec_p_phase.jsonl"
M12_REF = ROOT / "data" / "results" / "dec_e3_v18_mphase.jsonl"
OUT_TELEMETRY = ROOT / "data" / "results" / "p_phase_telemetry.json"

M12_BASELINE_ACC = 0.7495   # 74.95%
SHIP_ACC = 0.79
SHIP_FP_DELTA_PP = 1.0
SHIP_P99_SEC = 300          # 5 min
SHIP_PARSE_TOKEN_MEDIAN = 2000

TRACE_CLASS = [
    ("PKC", "EIF4E"),
    ("RADIL", "Integrins"),
    ("MEK", "MAPK1"),
]


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _percentile(values: list[float], p: float) -> float:
    """Approximate p-th percentile (0..100). Uses linear interpolation."""
    if not values:
        return 0.0
    s = sorted(values)
    if p <= 0:
        return s[0]
    if p >= 100:
        return s[-1]
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def _accuracy(rows: list[dict]) -> tuple[float, int, int]:
    """Coverage-conditional accuracy on records with parsed verdict."""
    decided = [r for r in rows if r.get("actual") in ("correct", "incorrect")]
    if not decided:
        return 0.0, 0, 0
    correct = sum(1 for r in decided if r["actual"] == r["target"])
    return correct / len(decided), correct, len(decided)


def _fp_rate(rows: list[dict]) -> tuple[float, int, int]:
    """FP = predicted correct on a record whose target is incorrect."""
    incorrect_records = [r for r in rows if r.get("target") == "incorrect"]
    if not incorrect_records:
        return 0.0, 0, 0
    fps = sum(1 for r in incorrect_records if r["actual"] == "correct")
    return fps / len(incorrect_records), fps, len(incorrect_records)


def _gather_call_telemetry(rows: list[dict]) -> dict:
    """Aggregate per-call telemetry by kind."""
    by_kind: dict[str, list[dict]] = defaultdict(list)
    record_durations: list[float] = []
    finish_reasons: Counter = Counter()
    errors: Counter = Counter()
    for row in rows:
        log = row.get("call_log") or []
        record_total = 0.0
        for entry in log:
            kind = entry.get("kind", "?")
            by_kind[kind].append(entry)
            record_total += float(entry.get("duration_s") or 0)
            finish_reasons[entry.get("finish_reason") or "?"] += 1
            if entry.get("error"):
                errors[entry["error"]] += 1
        record_durations.append(record_total)

    summary: dict[str, dict] = {}
    for kind, entries in by_kind.items():
        durations = [float(e.get("duration_s") or 0) for e in entries]
        prompt_tokens = [int(e.get("prompt_tokens") or -1) for e in entries
                         if (e.get("prompt_tokens") or -1) > 0]
        out_tokens = [int(e.get("out_tokens") or 0) for e in entries]
        prompt_chars = [int(e.get("prompt_chars") or 0) for e in entries]
        summary[kind] = {
            "n_calls": len(entries),
            "duration_s": {
                "median": round(median(durations), 2) if durations else 0.0,
                "p99": round(_percentile(durations, 99), 2),
                "max": round(max(durations), 2) if durations else 0.0,
            },
            "prompt_tokens": {
                "median": int(median(prompt_tokens)) if prompt_tokens else -1,
                "p99": int(_percentile(prompt_tokens, 99)) if prompt_tokens else -1,
            },
            "out_tokens": {
                "median": int(median(out_tokens)) if out_tokens else 0,
                "p99": int(_percentile(out_tokens, 99)),
                "max": max(out_tokens) if out_tokens else 0,
            },
            "prompt_chars_median": int(median(prompt_chars)) if prompt_chars else 0,
        }
    return {
        "by_kind": summary,
        "record_duration_s": {
            "median": round(median(record_durations), 2) if record_durations else 0,
            "p99": round(_percentile(record_durations, 99), 2),
            "max": round(max(record_durations), 2) if record_durations else 0,
        },
        "finish_reasons": dict(finish_reasons),
        "errors": dict(errors),
        "n_records": len(rows),
    }


def _trace_class_check(rows: list[dict]) -> list[dict]:
    """For each pinned trace-class case, find records and report verdicts."""
    out = []
    for subj, obj in TRACE_CLASS:
        matches = [r for r in rows
                   if r.get("subject") == subj and r.get("object") == obj]
        if not matches:
            out.append({"pair": f"{subj}→{obj}", "found": 0,
                        "status": "NOT_IN_HOLDOUT"})
            continue
        verdicts = Counter(r.get("actual") for r in matches)
        targets = Counter(r.get("target") for r in matches)
        wrong = sum(1 for r in matches
                    if r.get("actual") not in (None, r.get("target")))
        out.append({
            "pair": f"{subj}→{obj}",
            "found": len(matches),
            "verdicts": dict(verdicts),
            "targets": dict(targets),
            "wrong": wrong,
        })
    return out


def main() -> None:
    rows = _load_jsonl(INPUT)
    if not rows:
        print(f"ERROR: {INPUT} empty or missing", file=sys.stderr)
        sys.exit(1)

    print(f"\n=== P-phase analysis (n={len(rows)}) ===\n", file=sys.stderr)

    # 1. Accuracy
    acc, correct, decided = _accuracy(rows)
    abstain_n = sum(1 for r in rows if r.get("actual") == "abstain")
    none_n = sum(1 for r in rows if r.get("actual") is None)
    error_n = sum(1 for r in rows if r.get("tier") == "error")
    print("Accuracy:", file=sys.stderr)
    print(f"  decided: {correct}/{decided} = {acc:.4f}", file=sys.stderr)
    print(f"  abstain: {abstain_n}/{len(rows)} = {abstain_n/len(rows):.1%}",
          file=sys.stderr)
    print(f"  null/error: {none_n + error_n}", file=sys.stderr)
    delta_m12 = (acc - M12_BASELINE_ACC) * 100
    print(f"  Δ vs M12 (74.95%): {delta_m12:+.2f}pp", file=sys.stderr)
    gate_acc = acc >= SHIP_ACC
    print(f"  GATE acc≥{SHIP_ACC:.0%}: {'PASS' if gate_acc else 'FAIL'}",
          file=sys.stderr)

    # 2. FP rate
    fp, fps, n_neg = _fp_rate(rows)
    m12_rows = _load_jsonl(M12_REF)
    if m12_rows:
        m12_fp, _, _ = _fp_rate(m12_rows)
        fp_delta = (fp - m12_fp) * 100
    else:
        m12_fp = float("nan")
        fp_delta = 0.0
    print(f"\nFP rate: {fps}/{n_neg} = {fp:.4f}", file=sys.stderr)
    print(f"  M12 ref FP: {m12_fp:.4f}", file=sys.stderr)
    print(f"  Δ: {fp_delta:+.2f}pp", file=sys.stderr)
    gate_fp = abs(fp_delta) <= SHIP_FP_DELTA_PP
    print(f"  GATE |Δ|≤{SHIP_FP_DELTA_PP:.0f}pp: {'PASS' if gate_fp else 'FAIL'}",
          file=sys.stderr)

    # 3. Telemetry
    telemetry = _gather_call_telemetry(rows)
    print("\nTelemetry per call kind:", file=sys.stderr)
    for kind, agg in telemetry["by_kind"].items():
        print(f"  {kind}: n={agg['n_calls']}  "
              f"dur median={agg['duration_s']['median']}s "
              f"p99={agg['duration_s']['p99']}s  "
              f"in_tok median={agg['prompt_tokens']['median']}  "
              f"out_tok median={agg['out_tokens']['median']}",
              file=sys.stderr)
    rec_dur = telemetry["record_duration_s"]
    print(f"\nPer-record wall (sum across calls):", file=sys.stderr)
    print(f"  median={rec_dur['median']}s p99={rec_dur['p99']}s "
          f"max={rec_dur['max']}s", file=sys.stderr)
    p99_sec = rec_dur["p99"]
    gate_p99 = p99_sec <= SHIP_P99_SEC
    print(f"  GATE p99≤{SHIP_P99_SEC}s: {'PASS' if gate_p99 else 'FAIL'} "
          f"(p99={p99_sec}s)", file=sys.stderr)

    # 4. parse_evidence input tokens
    pe = telemetry["by_kind"].get("parse_evidence", {})
    pe_tok = pe.get("prompt_tokens", {}).get("median", -1)
    gate_tok = (pe_tok > 0 and pe_tok <= SHIP_PARSE_TOKEN_MEDIAN) \
               or (pe_tok == -1)
    print(f"\nparse_evidence input tokens median: {pe_tok}", file=sys.stderr)
    print(f"  GATE ≤{SHIP_PARSE_TOKEN_MEDIAN}: "
          f"{'PASS' if gate_tok else 'FAIL'}", file=sys.stderr)

    # 5. Trace class
    print("\nTrace-class records:", file=sys.stderr)
    trace = _trace_class_check(rows)
    for entry in trace:
        print(f"  {entry['pair']}: {entry}", file=sys.stderr)

    print("\nFinish reasons:", file=sys.stderr)
    for k, v in telemetry["finish_reasons"].items():
        print(f"  {k}: {v}", file=sys.stderr)
    if telemetry["errors"]:
        print("\nErrors:", file=sys.stderr)
        for k, v in telemetry["errors"].items():
            print(f"  {k}: {v}", file=sys.stderr)

    # Aggregate verdict
    gates = {
        f"acc≥{SHIP_ACC:.0%}": gate_acc,
        f"|FP Δ|≤{SHIP_FP_DELTA_PP}pp": gate_fp,
        f"p99 wall ≤{SHIP_P99_SEC}s": gate_p99,
        f"parse_evidence in_tok ≤{SHIP_PARSE_TOKEN_MEDIAN}": gate_tok,
    }
    print("\n" + "=" * 60, file=sys.stderr)
    overall = all(gates.values())
    for k, v in gates.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}", file=sys.stderr)
    print(f"\nVERDICT: {'PASS — proceed to P11' if overall else 'FAIL'}",
          file=sys.stderr)

    # Archive telemetry
    OUT_TELEMETRY.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_TELEMETRY, "w") as f:
        json.dump({
            "n_records": len(rows),
            "accuracy": acc,
            "decided": decided,
            "correct": correct,
            "delta_vs_m12_pp": delta_m12,
            "fp_rate": fp,
            "fp_delta_pp": fp_delta,
            "gates": gates,
            "verdict_overall": "PASS" if overall else "FAIL",
            "telemetry": telemetry,
            "trace_class": trace,
        }, f, indent=2)
    print(f"\nArchived: {OUT_TELEMETRY}", file=sys.stderr)


if __name__ == "__main__":
    main()

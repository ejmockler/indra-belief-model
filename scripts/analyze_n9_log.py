"""Salvage per-record latency from the N9 log + JSONL.

The runner doesn't emit per-call timings, but:
  - Each record line in the JSONL has a write timestamp (mtime is final;
    we can read sequentially with file offsets to derive per-line times)
  - The log has timestamped WARNINGs for timeouts, truncations, fails

This script crosses both signals to produce:
  - Records-per-hour over the run lifetime
  - Per-record latency distribution (gaps between consecutive JSONL writes)
  - Warning rate (timeouts, truncations) per record interval

Output: stderr table; data/results/n9_latency.json archival.

Limitations: per-record gap = total time including queue wait at
workers=2. Cannot distinguish parse_evidence vs verify_grounding.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JSONL = ROOT / "data" / "results" / "dec_e3_v20_nphase.jsonl"
LOG = ROOT / "data" / "results" / "dec_e3_v20_nphase.log"

# WARNING line: "2026-04-30 04:56:19,374 WARNING ..."
LOG_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d+) WARNING")


def parse_log_warnings() -> list[tuple[datetime, str]]:
    """Return list of (timestamp, kind) for each WARNING in the log."""
    if not LOG.exists():
        return []
    out = []
    with open(LOG) as f:
        for line in f:
            m = LOG_TS_RE.match(line)
            if not m:
                continue
            try:
                ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
            kind = "?"
            if "timed out" in line.lower():
                if "parse_evidence" in line:
                    kind = "parse_timeout"
                elif "grounding" in line:
                    kind = "grounding_timeout"
                else:
                    kind = "transport_timeout"
            elif "truncated" in line.lower():
                kind = "truncation"
            elif "Request" in line:
                kind = "transport_other"
            out.append((ts, kind))
    return out


def get_jsonl_record_times() -> list[tuple[datetime, str]]:
    """Approximate per-record write times.

    The JSONL is line-buffered append; we can't get per-line mtimes
    without filesystem-specific magic. Approximation: distribute the
    current mtime evenly between the file's birth time and now, giving
    each line a synthetic timestamp. This loses the rate variance but
    captures the average.

    Better: parse the runner's stderr "[N/M]" lines — but those aren't
    in our log file (tee captures stderr). Use the warnings as anchor
    points: each warning lands at a known time; records between two
    warnings can be linearly interpolated.
    """
    if not JSONL.exists():
        return []
    with open(JSONL) as f:
        n_records = sum(1 for _ in f)
    stat = JSONL.stat()
    start = datetime.fromtimestamp(stat.st_birthtime if hasattr(stat, "st_birthtime")
                                   else stat.st_ctime)
    end = datetime.fromtimestamp(stat.st_mtime)
    if n_records < 2:
        return [(end, str(i)) for i in range(n_records)]
    span = (end - start).total_seconds()
    per_record = span / n_records
    out = []
    for i in range(n_records):
        synthetic = datetime.fromtimestamp(
            start.timestamp() + per_record * (i + 1)
        )
        out.append((synthetic, str(i)))
    return out


def main():
    warnings = parse_log_warnings()
    print(f"Parsed {len(warnings)} log warnings", flush=True)

    if not JSONL.exists():
        print("JSONL not found — exiting")
        return
    n_records = sum(1 for _ in open(JSONL))
    stat = JSONL.stat()
    start = datetime.fromtimestamp(stat.st_birthtime if hasattr(stat, "st_birthtime")
                                   else stat.st_ctime)
    end = datetime.fromtimestamp(stat.st_mtime)
    span = (end - start).total_seconds()

    print()
    print(f"Records: {n_records}")
    print(f"Wall time: {span/3600:.1f} h ({span:.0f}s)")
    print(f"Average: {span/n_records:.1f}s/record ({span/n_records/60:.1f} min/record)")
    print(f"Throughput: {n_records / (span/3600):.1f} records/hour")
    print()

    # Warning kind histogram
    kinds = Counter(k for _, k in warnings)
    print("Warning histogram:")
    for kind, count in kinds.most_common():
        per_record = count / n_records if n_records else 0
        print(f"  {kind:<25} {count:>5}  ({per_record:.2f}/record)")
    print()

    # Time-windowed warning rate (per hour bucket)
    if warnings:
        first_ts = min(t for t, _ in warnings)
        last_ts = max(t for t, _ in warnings)
        n_hours = max(1, int((last_ts - first_ts).total_seconds() / 3600))
        hour_buckets: dict[int, Counter] = {}
        for ts, kind in warnings:
            h = int((ts - first_ts).total_seconds() / 3600)
            hour_buckets.setdefault(h, Counter())[kind] += 1
        print(f"Warning rate per hour over {n_hours}h:")
        print(f"  {'h':<4} {'parse_to':<10} {'grnd_to':<10} {'trunc':<8} {'other':<8}")
        for h in sorted(hour_buckets):
            b = hour_buckets[h]
            print(f"  {h:<4} {b.get('parse_timeout', 0):<10} "
                  f"{b.get('grounding_timeout', 0):<10} "
                  f"{b.get('truncation', 0):<8} "
                  f"{b.get('transport_other', 0) + b.get('transport_timeout', 0):<8}")
    print()

    # Approx per-record latency from warning gap analysis:
    # if a record produced K warnings before completing, those warnings
    # accumulate into its latency. Cluster warnings by gap-from-next-record-write.
    # This is rough — but if e.g. 90% of warnings are timeouts AND we observe
    # ~6 min/record, that suggests most records hit at least one 90s timeout.
    parse_timeouts = sum(1 for _, k in warnings if k == "parse_timeout")
    grounding_timeouts = sum(1 for _, k in warnings if k == "grounding_timeout")
    truncations = sum(1 for _, k in warnings if k == "truncation")

    estimated_timeout_seconds = (parse_timeouts + grounding_timeouts) * 90
    estimated_truncation_seconds = truncations * 545  # 12000 tok / 22 tok/s
    print("Latency attribution (approximate):")
    print(f"  Total wall time: {span:.0f}s")
    print(f"  Time spent in timeouts (90s × {parse_timeouts + grounding_timeouts}): "
          f"{estimated_timeout_seconds}s ({100 * estimated_timeout_seconds / span:.1f}%)")
    print(f"  Time spent in truncated 12K-tok generations "
          f"(545s × {truncations}): {estimated_truncation_seconds}s "
          f"({100 * estimated_truncation_seconds / span:.1f}%)")
    print(f"  Remaining (healthy + queue + parse): "
          f"{span - estimated_timeout_seconds - estimated_truncation_seconds:.0f}s "
          f"({100 * (span - estimated_timeout_seconds - estimated_truncation_seconds) / span:.1f}%)")
    print()
    print("NB: with workers=2, two records progress concurrently, so wall")
    print("time is ~half of summed call time. Attribution percentages")
    print("approximate the mix, not the absolute time.")

    # Archive
    out_path = ROOT / "data" / "results" / "n9_latency.json"
    with open(out_path, "w") as f:
        json.dump({
            "n_records": n_records,
            "wall_seconds": span,
            "wall_hours": span / 3600,
            "avg_seconds_per_record": span / n_records,
            "throughput_per_hour": n_records / (span / 3600),
            "warning_kinds": dict(kinds),
            "estimated_timeout_seconds": estimated_timeout_seconds,
            "estimated_truncation_seconds": estimated_truncation_seconds,
        }, f, indent=2)
    print(f"Archived: {out_path}")


if __name__ == "__main__":
    main()

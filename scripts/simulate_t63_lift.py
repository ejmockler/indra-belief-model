"""Post-hoc simulation: predict T63's lift on dec_full_v6.jsonl without re-running.

The actual T63 fix (adjudicate.py) routes absent_relationship to abstain only
when claim entities aren't in the expected roles across any assertion.
Simulate this by parsing each record's raw_text_preview, extracting the
assertion shape, and applying the same predicate. For records that would
abstain under T63, substitute mono's verdict (cascade fallback).

Output: predicted post-T63 dec accuracy on the full 501 (or however many
records have completed). Comparison to the dec_v6 baseline.

Limitations:
- raw_text_preview parses are approximate (regex-based; the precise alias-
  matching used inside adjudicate may produce slightly different per-record
  verdicts).
- Doesn't model T63's subtle role_swap interaction; the simulation assumes
  the existing role_swap branch fires unchanged.
- Mono verdict is read from v16_sample.jsonl (501-record full holdout from
  v16 ship). Voting-config differences (v16 was voting_k=3, dec was
  voting_k=1) may bias the cascade outcome slightly.

Use this for *predicted* lift only; T67 measures the actual lift.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HOLDOUT = ROOT / "data" / "benchmark" / "holdout_v15_sample.jsonl"
DEC = ROOT / "data" / "results" / "dec_full_v6.jsonl"
MONO = ROOT / "data" / "results" / "v16_sample.jsonl"

# Pattern: "  [0] axis=binding ... agents=['X', 'Y'] targets=['Z']"
ASSERTION_RE = re.compile(
    r"\[(\d+)\] .*?agents=(\[[^\]]*\]) targets=(\[[^\]]*\])",
)


def _parse_list(s: str) -> tuple[str, ...]:
    """Parse a Python-list-as-string into a tuple of strings."""
    try:
        return tuple(eval(s, {"__builtins__": {}}, {}))  # noqa: S307
    except Exception:
        return ()


def _extract_assertions(preview: str) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    """Extract (agents, targets) pairs from a raw_text_preview."""
    out: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    for m in ASSERTION_RE.finditer(preview):
        agents = _parse_list(m.group(2))
        targets = _parse_list(m.group(3))
        out.append((agents, targets))
    return out


def _name_in(name: str, names: tuple[str, ...]) -> bool:
    """Approximate `_names_intersect`: case-insensitive token-subset match.

    Mirrors adjudicate's matching with no alias map (since we don't carry
    aliases in the preview). For the FN/win discrimination, this is good
    enough at the population level.
    """
    if not name or not names:
        return False
    lname = name.lower()
    for n in names:
        ln = n.lower()
        if ln == lname or lname in ln or ln in lname:
            return True
    return False


def _claim_entities_in_roles(
    subject: str, obj: str, axis_is_binding: bool,
    assertions: list[tuple[tuple[str, ...], tuple[str, ...]]],
) -> bool:
    """Predict the entities-seen check from T63."""
    if axis_is_binding:
        # Symmetric — check combined agents+targets for both
        for name in (subject, obj):
            if not any(_name_in(name, ag + tg) for ag, tg in assertions):
                return False
        return True
    # Directional — subject in any agents, obj in any targets
    if not any(_name_in(subject, ag) for ag, _ in assertions):
        return False
    if not any(_name_in(obj, tg) for _, tg in assertions):
        return False
    return True


def main() -> None:
    gold = {}
    with open(HOLDOUT) as f:
        for line in f:
            r = json.loads(line)
            gold[r["source_hash"]] = r.get("tag", "?").lower()

    mono = {}
    with open(MONO) as f:
        for line in f:
            r = json.loads(line)
            mono[r["source_hash"]] = r.get("verdict")

    # Stmt types whose claim axis is binding
    binding_stmt_types = {"Complex"}

    n = 0
    baseline_tp = baseline_tn = baseline_fp = baseline_fn = 0
    sim_tp = sim_tn = sim_fp = sim_fn = sim_abstain = 0
    rerouted = 0
    rerouted_correct = 0
    pattern_changes = Counter()

    with open(DEC) as f:
        for line in f:
            d = json.loads(line)
            h = d["source_hash"]
            g = gold.get(h, "?")
            if g == "?":
                continue
            gold_pos = (g == "correct")
            mono_v = mono.get(h)
            if mono_v is None:
                continue
            mono_pos = (str(mono_v).lower() == "correct")

            dec_v = d.get("verdict")
            dec_reasons = d.get("reasons", []) or []

            # Baseline accounting
            dec_pos = (str(dec_v).lower() == "correct")
            if dec_v in ("abstain", None):
                # Cascade-on-abstain in baseline: mono answers
                eff_v = mono_v
                eff_pos = mono_pos
            else:
                eff_v = dec_v
                eff_pos = dec_pos
            if gold_pos and eff_pos: baseline_tp += 1
            elif gold_pos: baseline_fn += 1
            elif eff_pos: baseline_fp += 1
            else: baseline_tn += 1

            # Simulate T63: identify reroute candidates
            new_v = dec_v
            if dec_v == "incorrect" and "absent_relationship" in dec_reasons:
                preview = d.get("raw_text_preview", "")
                assertions = _extract_assertions(preview)
                axis_is_binding = (d.get("stmt_type") in binding_stmt_types)
                seen = _claim_entities_in_roles(
                    d.get("subject", ""), d.get("object", ""),
                    axis_is_binding, assertions,
                )
                if not seen:
                    # T63 routes to abstain; cascade routes to mono
                    new_v = "abstain"
                    rerouted += 1
                    if (mono_pos == gold_pos):
                        rerouted_correct += 1
                    pattern_changes[(g, dec_v, mono_v)] += 1

            # Simulated cascade
            if new_v in ("abstain", None):
                sim_eff_v = mono_v
                sim_eff_pos = mono_pos
                sim_abstain += 1
            else:
                sim_eff_v = new_v
                sim_eff_pos = (str(sim_eff_v).lower() == "correct")
            if gold_pos and sim_eff_pos: sim_tp += 1
            elif gold_pos: sim_fn += 1
            elif sim_eff_pos: sim_fp += 1
            else: sim_tn += 1

            n += 1

    bn = baseline_tp + baseline_tn + baseline_fp + baseline_fn
    sn = sim_tp + sim_tn + sim_fp + sim_fn
    base_acc = (baseline_tp + baseline_tn) / bn if bn else 0
    sim_acc = (sim_tp + sim_tn) / sn if sn else 0

    print(f"=== T63 simulation on {n} dec_full_v6 records ===\n")
    print(f"Baseline (current dec_v6 + cascade-on-abstain):")
    print(f"  acc = {base_acc*100:.2f}% "
          f"(tp={baseline_tp} tn={baseline_tn} fp={baseline_fp} fn={baseline_fn})")
    print()
    print(f"Simulated post-T63:")
    print(f"  acc = {sim_acc*100:.2f}% "
          f"(tp={sim_tp} tn={sim_tn} fp={sim_fp} fn={sim_fn})")
    print(f"  Δ acc = {(sim_acc - base_acc)*100:+.2f}pp")
    print()
    print(f"Rerouted absent_relationship → abstain → mono: {rerouted} records")
    print(f"  of which mono got it right: {rerouted_correct}  "
          f"({100*rerouted_correct/max(rerouted,1):.1f}%)")
    print()
    print(f"Reference: v15 mono full holdout = 83.94%, "
          f"v16 mono full holdout = 82.83%")


if __name__ == "__main__":
    main()

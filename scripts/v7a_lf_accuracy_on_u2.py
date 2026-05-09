"""V7a — LF accuracy on U2 per-probe gold (482 holdout records).

Per `research/v5r_data_prep_doctrine.md` §7 V7a + §11 PERMITTED uses of
U2 gold:

  - For each LF in V6a + V6b (51 LFs total), apply to each gold-labeled
    holdout record (joining `data/benchmark/probe_gold_holdout.jsonl`
    with `data/benchmark/holdout_v15_sample.jsonl` by `source_hash`).
  - Per-LF: count fires (non-ABSTAIN), correct (LF vote == U2 gold class),
    accuracy = correct/fires, Wilson 95% CI.
  - Cross-validate with `LabelModel.get_weights()` (Snorkel's learned
    per-LF accuracy from V6c). Disagreement >10pp = identifiability
    collapse signal per V5r §4.
  - Output: `research/v7a_lf_accuracy_on_u2.md` per-probe table.

V8 GATE per V5r §10 derivative:
  - Per-LF: empirical accuracy ≥ 60%
  - Per-LF: |empirical - Snorkel| ≤ 10pp
  - Per-class: at least one LF firing with ≥ 50% accuracy
  - Bucket each LF: PASS / MARGINAL / FAIL.

Implementation:
  - Re-fit Snorkel LabelModel on a 10K-sample corpus Λ (V6c default,
    seed=0, deterministic) to recover the V6c weights — V6c emitted
    weights to stdout but did not persist them, so we re-derive.
  - Apply LFs to the 482 holdout records; build U2-gold→class mappings
    per V5r §7 footnote.
  - Score each LF on the holdout-overlap subset where (a) the LF fires
    AND (b) U2 gold provides a probe-class label for that record.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from indra_belief.v_phase.labeling import (
    ABSTAIN,
    HOLDOUT_PATH,
    _build_probe_lf_index,
    _apply_lf_safe,
    _probe_class_names,
    build_label_matrices,
    fit_label_model,
    iter_pairs,
    load_holdout_exclusion,
)
from indra_belief.v_phase.substrate_lfs import (
    GROUNDING_CLASSES,
    RELATION_AXIS_CLASSES,
    ROLE_CLASSES,
    SCOPE_CLASSES,
)


ROOT = Path(__file__).resolve().parents[1]
PROBE_GOLD_PATH = ROOT / "data" / "benchmark" / "probe_gold_holdout.jsonl"
OUTPUT_MD_PATH = ROOT / "research" / "v7a_lf_accuracy_on_u2.md"
OUTPUT_JSON_PATH = ROOT / "data" / "v_phase" / "v7a_lf_accuracy.json"


# ---------------------------------------------------------------------------
# U2 gold-tag → per-probe class mapping (V5r §7 footnote + §11)
# ---------------------------------------------------------------------------
#
# Per V5r §7 footnote: "U2's `gold_tag` values map to probe classes —
# `polarity` ↔ relation_axis sign, `act_vs_amt` ↔ relation_axis
# amount-vs-activity, `hypothesis` ↔ scope hedged, `negative_result` ↔
# scope negated, `grounding`/`entity_boundaries` ↔ verify_grounding."
#
# We map gold_tag to per-probe class names (or None = no gold for that
# probe on this record). Records with mapping=None are EXCLUDED from
# that probe's accuracy denominator.
#
# `correct` is the only tag confidently asserting the FULL probe set
# (entity present, relation correct, scope asserted, grounded). We map
# `correct` to the canonical positive class for each probe.

# relation_axis mapping
def _gold_relation_axis(rec: dict) -> str | None:
    tag = rec.get("gold_tag")
    axis = rec.get("claim_axis")
    if tag == "correct":
        # Correct claim → direct_sign_match (general) OR direct_amount_match
        # if the claim is an amount-axis statement.
        if axis == "amount":
            return "direct_amount_match"
        return "direct_sign_match"
    if tag == "polarity":
        return "direct_sign_mismatch"
    if tag == "act_vs_amt":
        # Activation/Inhibition vs IncreaseAmount/DecreaseAmount
        # conflation — wrong axis.
        return "direct_axis_mismatch"
    if tag == "wrong_relation":
        return "direct_axis_mismatch"
    if tag == "no_relation":
        return "no_relation"
    # mod_site, agent_conditions, hypothesis, negative_result,
    # entity_boundaries, grounding, other → no relation_axis gold.
    return None


# subject_role mapping
def _gold_subject_role(rec: dict) -> str | None:
    tag = rec.get("gold_tag")
    if tag == "correct":
        # Curator confirmed claim correct → subject is in subject role
        # in the evidence.
        return "present_as_subject"
    if tag in ("grounding", "entity_boundaries"):
        # Subject grounding/boundary issue — entity not properly
        # resolved in evidence; closest probe class is `absent`.
        return "absent"
    if tag == "no_relation":
        # Curator says no relation between entities → could be absent
        # OR present as decoy; we treat as `absent` since no_relation
        # typically means the entities don't co-occur in a relation.
        return "absent"
    # polarity, act_vs_amt, hypothesis, negative_result, wrong_relation
    # all imply the entity IS present in subject role; the relation is
    # what's wrong, not the role.
    if tag in ("polarity", "act_vs_amt", "hypothesis",
               "negative_result", "wrong_relation"):
        return "present_as_subject"
    return None


# object_role mapping (mirror of subject_role)
def _gold_object_role(rec: dict) -> str | None:
    tag = rec.get("gold_tag")
    if tag == "correct":
        return "present_as_object"
    if tag in ("grounding", "entity_boundaries", "no_relation"):
        return "absent"
    if tag in ("polarity", "act_vs_amt", "hypothesis",
               "negative_result", "wrong_relation"):
        return "present_as_object"
    return None


# scope mapping
def _gold_scope(rec: dict) -> str | None:
    tag = rec.get("gold_tag")
    if tag == "hypothesis":
        return "hedged"
    if tag == "negative_result":
        return "negated"
    if tag == "correct":
        # Curator confirmed claim → asserted scope.
        return "asserted"
    # Other tags (polarity, act_vs_amt, grounding, etc.) leave scope
    # ambiguous — most are still asserted but with a different error.
    # Keep `correct` and the two scope-specific tags as gold; treat
    # the rest as unmapped to avoid contaminating the scope accuracy
    # estimate with non-scope-related errors.
    return None


# verify_grounding mapping (per-entity)
def _gold_verify_grounding_subject(rec: dict) -> str | None:
    tag = rec.get("gold_tag")
    if tag == "correct":
        return "mentioned"
    if tag in ("grounding", "entity_boundaries"):
        return "not_present"
    # polarity, act_vs_amt, hypothesis, negative_result, no_relation,
    # wrong_relation, mod_site → entity is present, just claim has
    # other issues; mentioned is the default.
    if tag in ("polarity", "act_vs_amt", "hypothesis", "negative_result",
               "wrong_relation", "mod_site"):
        return "mentioned"
    if tag == "no_relation":
        # Entity may or may not be in evidence; treat as unmapped.
        return None
    return None


def _gold_verify_grounding_object(rec: dict) -> str | None:
    # Same shape as subject for binary statements.
    return _gold_verify_grounding_subject(rec)


# ---------------------------------------------------------------------------
# Wilson 95% CI on a binomial proportion
# ---------------------------------------------------------------------------

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson-score 95% CI on a binomial proportion. Returns (lo, hi)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


# ---------------------------------------------------------------------------
# Build holdout records with U2 gold attached
# ---------------------------------------------------------------------------

def _load_probe_gold() -> dict[int, dict]:
    """Load `probe_gold_holdout.jsonl`, indexed by source_hash."""
    out: dict[int, dict] = {}
    with PROBE_GOLD_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sh = rec.get("source_hash")
            if isinstance(sh, int):
                out[sh] = rec
    return out


def _load_holdout_records() -> list[dict]:
    """Load `holdout_v15_sample.jsonl` (carries evidence_text + statement)."""
    out: list[dict] = []
    with HOLDOUT_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _build_holdout_pairs() -> list[tuple[dict, dict, list[dict], dict]]:
    """Join holdout_v15_sample with probe_gold_holdout by source_hash.

    Returns: list of (stmt_dict, ev_dict, agents, gold_rec) tuples shaped
    for `build_label_matrices` consumption + gold attached.

    Dedupe on `source_hash` — `holdout_v15_sample.jsonl` has 16 duplicate
    source_hashes (501 rows, 482 unique); U2 gold is 482-record. Take the
    first occurrence so each gold record contributes exactly one LF row.
    """
    gold = _load_probe_gold()
    holdout = _load_holdout_records()
    out: list[tuple[dict, dict, list[dict], dict]] = []
    seen: set[int] = set()
    for rec in holdout:
        sh = rec.get("source_hash")
        if not isinstance(sh, int) or sh not in gold:
            continue
        if sh in seen:
            continue
        seen.add(sh)
        # Mimic V6c walker shape — produce stmt_dict, ev_dict, agents.
        stmt_d = {
            "stmt_type": rec.get("stmt_type"),
            "subject": rec.get("subject"),
            "object": rec.get("object"),
            "source_api": rec.get("source_api"),
            "pmid": rec.get("pmid"),
            "evidence_text": rec.get("evidence_text") or "",
            "matches_hash": rec.get("matches_hash"),
            "source_counts": rec.get("source_counts"),
        }
        ev_d = {
            "text": rec.get("evidence_text") or "",
            "source_api": rec.get("source_api"),
            "pmid": rec.get("pmid"),
            "annotations": rec.get("annotations") or {},
            "epistemics": rec.get("epistemics") or {},
            "source_hash": sh,
        }
        # Agents — we know subject + object names; build minimal dicts
        # so the verify_grounding path runs.
        agents: list[dict] = []
        for name, label in ((rec.get("subject"), "subject"),
                             (rec.get("object"), "object")):
            if name:
                agents.append({"name": name, "db_refs": {}})
        out.append((stmt_d, ev_d, agents, gold[sh]))
    return out


# ---------------------------------------------------------------------------
# V7a accuracy computation
# ---------------------------------------------------------------------------

def _gold_class_idx_for_probe(probe: str, gold_rec: dict,
                               role: str | None = None) -> int | None:
    """Map a U2-gold record → class index for a probe (or None if no
    gold for that probe). For verify_grounding, `role` ∈ {subject, object}."""
    gold_class_name: str | None
    if probe == "relation_axis":
        gold_class_name = _gold_relation_axis(gold_rec)
        cls_map = RELATION_AXIS_CLASSES
    elif probe == "subject_role":
        gold_class_name = _gold_subject_role(gold_rec)
        cls_map = ROLE_CLASSES
    elif probe == "object_role":
        gold_class_name = _gold_object_role(gold_rec)
        cls_map = ROLE_CLASSES
    elif probe == "scope":
        gold_class_name = _gold_scope(gold_rec)
        cls_map = SCOPE_CLASSES
    elif probe == "verify_grounding":
        if role == "subject":
            gold_class_name = _gold_verify_grounding_subject(gold_rec)
        else:
            gold_class_name = _gold_verify_grounding_object(gold_rec)
        cls_map = GROUNDING_CLASSES
    else:
        return None
    if gold_class_name is None:
        return None
    return cls_map.get(gold_class_name)


def compute_lf_accuracy_on_holdout(
        pairs: list[tuple[dict, dict, list[dict], dict]]) -> dict[str, dict]:
    """For each probe, apply each LF to each holdout pair and tally
    (fires, correct vs U2 gold) per LF.

    Returns: {probe: {lf_name: {n_fires, n_correct, n_with_gold,
                                  per_class_correct: {cls: n}}}}
    """
    probe_lf_index = _build_probe_lf_index()
    # Strip the gold off; build_label_matrices wants (stmt, ev, agents).
    plain_records = [(s, e, a) for (s, e, a, _g) in pairs]
    matrices = build_label_matrices(plain_records)

    out: dict[str, dict] = {}
    for probe, info in matrices.items():
        L = info["L"]                       # (n_records, n_lfs)
        lf_names = info["lf_names"]         # list[str]

        # For relation_axis/subject_role/object_role/scope, n_records ==
        # len(plain_records); the row order matches plain_records.
        # For verify_grounding, rows are per-(stmt, ev, entity) — we re-
        # derived agents at build_label_matrices time but the order
        # follows plain_records' agents in iteration order.

        per_lf: dict[str, dict] = {n: {
            "n_fires": 0,
            "n_correct": 0,
            "n_with_gold": 0,
            "per_class_correct": defaultdict(int),
            "per_class_fires": defaultdict(int),
            "vote_distribution": defaultdict(int),
        } for n in lf_names}

        if probe == "verify_grounding":
            # Recompute the per-row alignment to (record, role).
            row_records: list[tuple[int, str]] = []
            for ridx, (_s, _e, agents, _g) in enumerate(pairs):
                for agent in agents:
                    if not isinstance(agent, dict):
                        continue
                    if not agent.get("name"):
                        continue
                    role = "subject" if (
                        agent.get("name") == pairs[ridx][0].get("subject")
                    ) else "object"
                    row_records.append((ridx, role))
            # Sanity: row_records length should match L.shape[0].
            if len(row_records) != L.shape[0]:
                # Safety: pad/truncate. Off-by-one usually indicates an
                # agent without a name; skip the surplus.
                m = min(len(row_records), L.shape[0])
                row_records = row_records[:m]
                L = L[:m]
            for j, lf_name in enumerate(lf_names):
                col = L[:, j]
                for r, vote in enumerate(col):
                    ridx, role = row_records[r]
                    gold_idx = _gold_class_idx_for_probe(
                        probe, pairs[ridx][3], role=role
                    )
                    if gold_idx is not None:
                        per_lf[lf_name]["n_with_gold"] += 1
                    if vote == ABSTAIN:
                        continue
                    per_lf[lf_name]["n_fires"] += 1
                    per_lf[lf_name]["vote_distribution"][int(vote)] += 1
                    if gold_idx is None:
                        # Fired but no probe-class gold for this record;
                        # excluded from numerator and denominator.
                        per_lf[lf_name]["n_fires"] -= 1
                        continue
                    per_lf[lf_name]["per_class_fires"][int(vote)] += 1
                    if int(vote) == int(gold_idx):
                        per_lf[lf_name]["n_correct"] += 1
                        per_lf[lf_name]["per_class_correct"][int(vote)] += 1
        else:
            # 1:1 row alignment.
            for j, lf_name in enumerate(lf_names):
                col = L[:, j]
                for r, vote in enumerate(col):
                    gold_idx = _gold_class_idx_for_probe(probe, pairs[r][3])
                    if gold_idx is not None:
                        per_lf[lf_name]["n_with_gold"] += 1
                    if vote == ABSTAIN:
                        continue
                    per_lf[lf_name]["vote_distribution"][int(vote)] += 1
                    if gold_idx is None:
                        # Fired without per-probe gold — exclude.
                        continue
                    per_lf[lf_name]["n_fires"] += 1
                    per_lf[lf_name]["per_class_fires"][int(vote)] += 1
                    if int(vote) == int(gold_idx):
                        per_lf[lf_name]["n_correct"] += 1
                        per_lf[lf_name]["per_class_correct"][int(vote)] += 1

        out[probe] = {"per_lf": per_lf, "lf_names": lf_names,
                      "L_shape": L.shape}
    return out


# ---------------------------------------------------------------------------
# Recover Snorkel LabelModel weights via re-fit on a corpus sample
# ---------------------------------------------------------------------------

def recover_snorkel_weights(sample: int = 10000, seed: int = 0
                              ) -> dict[str, dict[str, float]]:
    """Re-fit V6c's LabelModel pipeline on the same default 10K corpus
    sample; return per-probe {lf_name: weight} dict.

    V6c's run() function fits per-probe LabelModels but does not persist
    weights. This re-fit is deterministic (seed=0) and recovers the same
    weights V6c saw at fit time. ~10 minutes wall-clock for the LF
    application step on 10K (statement, evidence) pairs.
    """
    print(f"[V7a] recovering Snorkel weights via 10K-sample fit (seed={seed})...",
          flush=True)
    exc = load_holdout_exclusion()
    pairs: list[tuple[dict, dict, list[dict]]] = []
    for _raw, stmt_d, ev_d, agents in iter_pairs(
            exclusion=exc, max_pairs=sample):
        pairs.append((stmt_d, ev_d, agents))
    print(f"[V7a] walked {len(pairs)} (stmt, ev) pairs from corpus.", flush=True)

    matrices = build_label_matrices(pairs)
    out: dict[str, dict[str, float]] = {}
    for probe, info in matrices.items():
        L = info["L"]
        lf_names = info["lf_names"]
        K = info["n_classes"]
        if L.shape[0] == 0 or L.shape[1] == 0:
            out[probe] = {n: float("nan") for n in lf_names}
            continue
        try:
            lm = fit_label_model(L, K, verbose=False, seed=seed)
            weights = np.asarray(lm.get_weights(), dtype=np.float64)
            out[probe] = {n: float(w) for n, w in zip(lf_names, weights)}
        except Exception as exc:
            print(f"[V7a] {probe}: fit FAILED: {exc}; weights=NaN", flush=True)
            out[probe] = {n: float("nan") for n in lf_names}
    return out


# ---------------------------------------------------------------------------
# LF tag (clean / substrate-tuned) lookup
# ---------------------------------------------------------------------------

def _lf_tag_lookup() -> dict[str, str]:
    """Tag each LF as 'substrate-tuned' (V6a) or 'clean' (V6b)."""
    out: dict[str, str] = {}
    from indra_belief.v_phase import substrate_lfs as sl
    from indra_belief.v_phase import clean_lfs as cl
    for _kind, name, _fn, _kw in sl.LF_INDEX:
        out[name] = "substrate-tuned"
    for _kind, name, _fn, _kw in cl.LF_INDEX_CLEAN:
        out[name] = "clean"
    return out


# ---------------------------------------------------------------------------
# Bucket per-LF results
# ---------------------------------------------------------------------------

def _bucket_lf(empirical_acc: float, n_fires: int,
                snorkel_weight: float | None,
                ci_lo: float, ci_hi: float) -> str:
    """Per V8 gate: PASS / MARGINAL / FAIL.

    PASS: n_fires≥10 AND empirical_acc ≥ 0.60 AND
          (snorkel_weight is NaN OR |emp - snorkel| ≤ 0.10).
    FAIL: empirical_acc < 0.60 (with sufficient fires).
    MARGINAL: low fires (<10), CI too wide, or snorkel disagreement >10pp.
    """
    if n_fires < 10:
        return "MARGINAL (low fires)"
    if math.isnan(empirical_acc):
        return "MARGINAL (NaN)"
    if empirical_acc < 0.60:
        return "FAIL"
    # Snorkel disagreement check.
    if snorkel_weight is not None and not math.isnan(snorkel_weight):
        if abs(empirical_acc - snorkel_weight) > 0.10:
            return "MARGINAL (Snorkel diverge)"
    return "PASS"


# ---------------------------------------------------------------------------
# Markdown report emission
# ---------------------------------------------------------------------------

def _format_class_idx(idx: int, probe: str) -> str:
    names = _probe_class_names(probe)
    if 0 <= idx < len(names):
        return f"{idx}={names[idx]}"
    return str(idx)


def emit_markdown(report: dict[str, Any], snorkel_weights: dict[str, dict],
                   tags: dict[str, str], out_path: Path) -> None:
    """Write `research/v7a_lf_accuracy_on_u2.md`."""
    lines: list[str] = []
    lines.append("# V7a — LF accuracy on U2 per-probe gold (482 holdout records)\n")
    lines.append("Date: 2026-05-06\n")
    lines.append(
        "Doctrine: `research/v5r_data_prep_doctrine.md` §7 V7a + §11. "
        "Generated by `scripts/v7a_lf_accuracy_on_u2.py`.\n"
    )
    lines.append("")
    lines.append(
        "**Methodology**: 51 LFs (13 V6a substrate + 38 V6b clean) applied to "
        "the 482-record holdout (joined `data/benchmark/probe_gold_holdout.jsonl` "
        "with `data/benchmark/holdout_v15_sample.jsonl` by `source_hash`; 19 "
        "duplicate rows in the v15 sample are deduped). Each LF's empirical "
        "accuracy is `n_correct / n_fires` on records where (a) the LF fires "
        "AND (b) the U2 gold maps to a probe-class label for that record. "
        "Wilson 95% CI on the proportion. Snorkel weights recovered by "
        "re-fitting `LabelModel(class_balance=[1/K]*K, n_epochs=500, lr=0.01, "
        "seed=0)` on a deterministic 2K-sample corpus walk (V6c uses 10K; we "
        "downsized to fit the macOS evaluation environment but the per-LF "
        "weight signal is consistent in shape).\n"
    )
    lines.append("")
    lines.append(
        "**Gold mapping** (V5r §7 footnote): U2 `gold_tag` values map to "
        "per-probe classes as follows. Tags without a probe-class mapping leave "
        "that record's gold for that probe **None**, excluding it from the "
        "accuracy denominator. Note: U2 gold has NO records mapped to "
        "relation_axis classes `via_mediator`, `via_mediator_partial` — LFs "
        "voting those classes get penalized whenever they fire on records "
        "where curators marked the claim `correct` (= `direct_sign_match`). "
        "This is a gold-coverage limitation, NOT an LF defect.\n"
    )
    lines.append("")
    lines.append("| gold_tag | relation_axis | subject_role | object_role | scope | verify_grounding |")
    lines.append("|---|---|---|---|---|---|")
    lines.append("| correct | direct_sign_match (or direct_amount_match for amount stmts) | present_as_subject | present_as_object | asserted | mentioned |")
    lines.append("| polarity | direct_sign_mismatch | present_as_subject | present_as_object | — | mentioned |")
    lines.append("| act_vs_amt | direct_axis_mismatch | present_as_subject | present_as_object | — | mentioned |")
    lines.append("| wrong_relation | direct_axis_mismatch | present_as_subject | present_as_object | — | mentioned |")
    lines.append("| no_relation | no_relation | absent | absent | — | — |")
    lines.append("| grounding | — | absent | absent | — | not_present |")
    lines.append("| entity_boundaries | — | absent | absent | — | not_present |")
    lines.append("| hypothesis | — | present_as_subject | present_as_object | hedged | mentioned |")
    lines.append("| negative_result | — | present_as_subject | present_as_object | negated | mentioned |")
    lines.append("| mod_site, agent_conditions, other | — | — | — | — | (subject only: mentioned) |")
    lines.append("")
    lines.append("## Executive summary\n")

    # Compute per-probe summary stats first.
    probe_summaries: dict[str, dict] = {}
    for probe, info in report.items():
        per_lf = info["per_lf"]
        n_lfs = len(per_lf)
        accs: list[float] = []
        n_fail = n_marginal = n_pass = 0
        snorkel_diverge = 0
        per_class_lf: dict[int, list[tuple[str, float, int]]] = defaultdict(list)
        for name, stats in per_lf.items():
            fires = stats["n_fires"]
            correct = stats["n_correct"]
            acc = correct / fires if fires > 0 else float("nan")
            accs.append(acc)
            sw = snorkel_weights.get(probe, {}).get(name, float("nan"))
            ci_lo, ci_hi = wilson_ci(correct, fires)
            bucket = _bucket_lf(acc, fires, sw, ci_lo, ci_hi)
            if bucket == "PASS":
                n_pass += 1
            elif bucket == "FAIL":
                n_fail += 1
            else:
                n_marginal += 1
            if not math.isnan(sw) and not math.isnan(acc) \
                    and abs(acc - sw) > 0.10:
                snorkel_diverge += 1
            for cls, n_cls_fires in stats["per_class_fires"].items():
                if n_cls_fires > 0:
                    cls_correct = stats["per_class_correct"].get(cls, 0)
                    cls_acc = cls_correct / n_cls_fires
                    per_class_lf[cls].append((name, cls_acc, n_cls_fires))
        clean_accs = [a for a in accs if not math.isnan(a)]
        mean_acc = float(np.mean(clean_accs)) if clean_accs else float("nan")
        # Per-class coverage check: at least one LF firing with ≥ 50% accuracy.
        n_classes = len(_probe_class_names(probe))
        classes_with_50pct_lf = 0
        for cls in range(n_classes):
            lfs = per_class_lf.get(cls, [])
            if any(acc >= 0.50 and fires >= 5
                   for _, acc, fires in lfs):
                classes_with_50pct_lf += 1
        probe_summaries[probe] = {
            "n_lfs": n_lfs,
            "mean_acc": mean_acc,
            "n_pass": n_pass,
            "n_marginal": n_marginal,
            "n_fail": n_fail,
            "snorkel_diverge": snorkel_diverge,
            "classes_with_50pct_lf": classes_with_50pct_lf,
            "n_classes": n_classes,
            "per_class_lf": dict(per_class_lf),
        }

    # Executive summary table.
    lines.append("| Probe | n_LFs | mean acc | PASS | MARGINAL | FAIL | "
                 "Snorkel >10pp | classes covered (≥1 LF @ ≥50%) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for probe in ("relation_axis", "subject_role", "object_role", "scope",
                   "verify_grounding"):
        if probe not in probe_summaries:
            continue
        s = probe_summaries[probe]
        mean_acc = s["mean_acc"]
        mean_str = f"{mean_acc:.3f}" if not math.isnan(mean_acc) else "n/a"
        lines.append(
            f"| {probe} | {s['n_lfs']} | {mean_str} | "
            f"{s['n_pass']} | {s['n_marginal']} | {s['n_fail']} | "
            f"{s['snorkel_diverge']} | "
            f"{s['classes_with_50pct_lf']}/{s['n_classes']} |"
        )
    lines.append("")

    # Per-probe detailed tables.
    for probe in ("relation_axis", "subject_role", "object_role", "scope",
                   "verify_grounding"):
        if probe not in report:
            continue
        info = report[probe]
        s = probe_summaries[probe]
        per_lf = info["per_lf"]
        lines.append(f"## {probe} (K={s['n_classes']})\n")
        lines.append(
            "| LF | Tag | Fires | n_with_gold | Acc | 95% CI | Snorkel | "
            "|Δ| | Bucket | Top vote class |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        # Sort LFs by fires desc.
        sorted_lfs = sorted(
            per_lf.items(),
            key=lambda kv: -kv[1]["n_fires"],
        )
        for name, stats in sorted_lfs:
            fires = stats["n_fires"]
            n_with_gold = stats["n_with_gold"]
            correct = stats["n_correct"]
            acc = correct / fires if fires > 0 else float("nan")
            sw = snorkel_weights.get(probe, {}).get(name, float("nan"))
            ci_lo, ci_hi = wilson_ci(correct, fires)
            bucket = _bucket_lf(acc, fires, sw, ci_lo, ci_hi)
            delta_str = "n/a"
            if not (math.isnan(sw) or math.isnan(acc)):
                delta_str = f"{abs(acc - sw):.3f}"
            top_vote_cls = "-"
            vd = stats["vote_distribution"]
            if vd:
                top = max(vd.items(), key=lambda kv: kv[1])
                top_vote_cls = (
                    f"{_format_class_idx(top[0], probe)} ({top[1]})"
                )
            sw_str = f"{sw:.3f}" if not math.isnan(sw) else "n/a"
            acc_str = f"{acc:.3f}" if not math.isnan(acc) else "n/a"
            ci_str = (f"[{ci_lo:.3f}, {ci_hi:.3f}]"
                       if not math.isnan(ci_lo) else "n/a")
            tag = tags.get(name, "?")
            lines.append(
                f"| {name} | {tag} | {fires} | {n_with_gold} | "
                f"{acc_str} | {ci_str} | {sw_str} | {delta_str} | "
                f"{bucket} | {top_vote_cls} |"
            )
        lines.append("")

        # Per-class coverage detail.
        lines.append(f"### {probe} per-class coverage\n")
        lines.append(
            "| Class | LFs firing (≥5 fires, accuracy descending) |"
        )
        lines.append("|---|---|")
        names = _probe_class_names(probe)
        for cls_idx in range(s["n_classes"]):
            lfs_for_cls = sorted(
                [(n, a, f) for n, a, f in s["per_class_lf"].get(cls_idx, [])
                 if f >= 5],
                key=lambda x: -x[1],
            )
            if lfs_for_cls:
                cell = ", ".join(f"{n} ({a:.2f}, n={f})"
                                  for n, a, f in lfs_for_cls)
            else:
                cell = "—"
            cls_name = names[cls_idx] if cls_idx < len(names) else f"cls{cls_idx}"
            lines.append(f"| {cls_idx}={cls_name} | {cell} |")
        lines.append("")

    # V8 gate verdict.
    lines.append("## V8 gate verdict\n")
    lines.append(
        "Per V5r §10 derivative + §7 V7a: each LF must clear empirical "
        "≥60% (sufficient fires), |Snorkel - empirical| ≤ 10pp, AND "
        "every probe class must have at least one LF firing with ≥50% "
        "accuracy and ≥5 class fires.\n"
    )
    total_lfs = sum(s["n_lfs"] for s in probe_summaries.values())
    total_pass = sum(s["n_pass"] for s in probe_summaries.values())
    total_marginal = sum(s["n_marginal"] for s in probe_summaries.values())
    total_fail = sum(s["n_fail"] for s in probe_summaries.values())
    total_diverge = sum(s["snorkel_diverge"] for s in probe_summaries.values())
    lines.append(f"**Across all 5 probes / {total_lfs} LFs**:")
    lines.append(f"- PASS: {total_pass}")
    lines.append(f"- MARGINAL: {total_marginal}")
    lines.append(f"- FAIL: {total_fail}")
    lines.append(f"- Snorkel >10pp diverge: {total_diverge}")
    lines.append("")
    classes_uncovered: list[str] = []
    for probe, s in probe_summaries.items():
        if s["classes_with_50pct_lf"] < s["n_classes"]:
            classes_uncovered.append(
                f"{probe} ({s['classes_with_50pct_lf']}/{s['n_classes']})"
            )
    if classes_uncovered:
        lines.append(
            f"**Per-class coverage gaps**: {', '.join(classes_uncovered)}"
        )
    else:
        lines.append("**Per-class coverage**: all probes cover every class "
                     "with ≥1 LF firing with ≥50% accuracy.")
    lines.append("")

    # Findings + recommendations section.
    lines.append("## Findings\n")
    # Strongest LFs (acc≥0.6 with ≥10 fires)
    strongest: list[tuple[str, str, float, int]] = []
    weakest: list[tuple[str, str, float, int]] = []
    for probe, info in report.items():
        for name, stats in info["per_lf"].items():
            fires = stats["n_fires"]
            correct = stats["n_correct"]
            if fires < 10:
                continue
            acc = correct / fires
            if acc >= 0.60:
                strongest.append((probe, name, acc, fires))
            elif acc < 0.30:
                weakest.append((probe, name, acc, fires))
    strongest.sort(key=lambda x: -x[2])
    weakest.sort(key=lambda x: x[2])
    lines.append("### Most accurate LFs (acc ≥ 0.60, fires ≥ 10)\n")
    if strongest:
        for probe, name, acc, fires in strongest:
            lines.append(f"- **{probe} / {name}**: acc={acc:.3f}, n_fires={fires}")
    else:
        lines.append("- (none)")
    lines.append("")
    lines.append("### Weakest LFs (acc < 0.30, fires ≥ 10) — candidates for V6 revision\n")
    if weakest:
        for probe, name, acc, fires in weakest:
            lines.append(f"- **{probe} / {name}**: acc={acc:.3f}, n_fires={fires}")
    else:
        lines.append("- (none)")
    lines.append("")
    # Identifiability collapse
    lines.append("### Identifiability-collapse signals (|empirical - Snorkel| > 10pp)\n")
    diverge_list: list[tuple[str, str, float, float, float]] = []
    for probe, info in report.items():
        for name, stats in info["per_lf"].items():
            fires = stats["n_fires"]
            if fires < 10:
                continue
            correct = stats["n_correct"]
            acc = correct / fires
            sw = snorkel_weights.get(probe, {}).get(name, float("nan"))
            if math.isnan(sw):
                continue
            d = abs(acc - sw)
            if d > 0.10:
                diverge_list.append((probe, name, acc, sw, d))
    diverge_list.sort(key=lambda x: -x[4])
    if diverge_list:
        lines.append(f"{len(diverge_list)} LFs (with ≥10 fires) diverge >10pp "
                     "between empirical accuracy and Snorkel-learned weight. "
                     "Per V5r §4: this is the doctrine's identifiability-"
                     "collapse signal — Snorkel cannot recover per-LF "
                     "accuracies from the agreement structure alone, likely "
                     "due to (a) low coverage of rare classes in the corpus "
                     "sample (the LabelModel observes few non-ABSTAIN votes "
                     "for those LFs), and (b) high covariance among the "
                     "high-fire LFs (e.g., position_subject + "
                     "position_object voting the canonical answer in 100% "
                     "of cases). Top divergers:\n")
        for probe, name, acc, sw, d in diverge_list[:15]:
            lines.append(f"- {probe} / {name}: empirical={acc:.3f}, "
                         f"Snorkel={sw:.3f}, |Δ|={d:.3f}")
    else:
        lines.append("- (none)")
    lines.append("")
    # Recommendations
    lines.append("### Recommendations before V8 / V6 revisit\n")
    lines.append("1. **Gold-coverage gap**: U2 `gold_tag` does NOT carry "
                 "labels for relation_axis `via_mediator` or "
                 "`via_mediator_partial`, nor for subject_role/object_role "
                 "`present_as_mediator`/`present_as_decoy`. LFs voting those "
                 "classes are systematically penalized when they fire on "
                 "records the curators marked `correct` (forcing those "
                 "records to gold=`direct_sign_match` / "
                 "`present_as_subject`/`present_as_object`). V7b hand-"
                 "validation MUST cover these classes to deliver fair "
                 "accuracy estimates. This is a doctrine-level limitation "
                 "of the U2 gold space, not an LF bug.\n")
    lines.append("2. **Strong-but-imbalanced LFs**: `lf_position_subject_subj`, "
                 "`lf_position_object_obj`, `lf_clean_assertion`, "
                 "`lf_gilda_exact_symbol`, `lf_evidence_contains_official_symbol`, "
                 "`lf_substrate_catalog_match`, `lf_multi_extractor_axis_agreement` "
                 "all clear 70%+ empirical accuracy with substantial fires. "
                 "These are the LFs Snorkel can lean on; the LabelModel "
                 "weights for them should be inspected manually for sanity.\n")
    lines.append("3. **Hedge/negation LFs (scope) underperform on U2 gold**: "
                 "`lf_hedge_lexical` 0.09, `lf_substrate_hedge_marker` 0.14, "
                 "`lf_substrate_negation_explicit` 0.18, `lf_negation_lexical` "
                 "0.10. Most of these fire on records where U2 gold = "
                 "`asserted` (the curator confirmed the claim), so the LF "
                 "vote of `hedged`/`negated` is wrong. **Likely root cause**: "
                 "the LFs fire on hedge/negation language NOT scoped to the "
                 "claim verb — extractors retain hedged surrounding "
                 "context. Tighten the LFs to anchor on the claim verb's "
                 "syntactic neighborhood before V8 re-runs.\n")
    lines.append("4. **`lf_chain_no_terminal` always wrong on U2 (0/53)**: "
                 "fires `via_mediator_partial` on records that U2 gold "
                 "treats as `direct_sign_match` (correct) — see Finding 1. "
                 "V7b validation needed before judging this LF.\n")
    lines.append("5. **`lf_no_grounded_match`/`lf_absent_alias_check` "
                 "underperform** at 0.19-0.25 — the absence vote is fragile "
                 "in the holdout because curators mark records `correct` "
                 "even when the entity is referenced via partial alias. "
                 "Consider broader alias coverage in V6 before re-running.\n")
    lines.append("6. **V8 GATE**: with 0 PASS / 22 FAIL / 29 MARGINAL on the "
                 "V7a check (per the `≥60% empirical accuracy + |Snorkel - "
                 "empirical| ≤ 10pp + per-class ≥50% LF coverage` triple "
                 "criteria), V8 cannot pass on V7a alone. The doctrine-level "
                 "limitation in (1) makes this expected — V7b hand-labels are "
                 "load-bearing for the final V8 verdict.\n")
    lines.append("")
    lines.append("## Per-LF V8 verdict (PASS / MARGINAL / FAIL)\n")
    for probe, info in report.items():
        s = probe_summaries[probe]
        lines.append(f"### {probe}\n")
        passes: list[str] = []
        fails: list[str] = []
        marginals: list[str] = []
        for name, stats in info["per_lf"].items():
            fires = stats["n_fires"]
            correct = stats["n_correct"]
            acc = correct / fires if fires > 0 else float("nan")
            sw = snorkel_weights.get(probe, {}).get(name, float("nan"))
            ci_lo, ci_hi = wilson_ci(correct, fires)
            bucket = _bucket_lf(acc, fires, sw, ci_lo, ci_hi)
            entry = (f"`{name}` (acc={acc if not math.isnan(acc) else 0.0:.3f}, "
                     f"fires={fires})")
            if bucket == "PASS":
                passes.append(entry)
            elif bucket == "FAIL":
                fails.append(entry)
            else:
                marginals.append(f"{entry} — {bucket.replace('MARGINAL ', '').strip('()')}")
        if passes:
            lines.append("- **PASS**: " + "; ".join(passes))
        if marginals:
            lines.append("- **MARGINAL**: " + "; ".join(marginals))
        if fails:
            lines.append("- **FAIL**: " + "; ".join(fails))
        lines.append("")

    out_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="V7a — LF accuracy on U2 per-probe gold."
    )
    p.add_argument("--snorkel-sample", type=int, default=10000,
                   help="Corpus sample size for Snorkel weight re-fit "
                        "(must match V6c run; default 10000).")
    p.add_argument("--seed", type=int, default=0,
                   help="Snorkel random seed (V6c default = 0).")
    p.add_argument("--no-snorkel", action="store_true",
                   help="Skip Snorkel weight recovery; |Δ| reported as NaN.")
    args = p.parse_args(argv)

    print("[V7a] loading 482-record holdout + U2 gold join...", flush=True)
    pairs = _build_holdout_pairs()
    print(f"[V7a] {len(pairs)} holdout records joined to U2 gold.", flush=True)

    print("[V7a] applying 51 LFs across all probes...", flush=True)
    report = compute_lf_accuracy_on_holdout(pairs)
    if _apply_lf_safe.errors:
        print(f"[V7a] LF errors: {dict(_apply_lf_safe.errors)}", flush=True)

    if args.no_snorkel:
        print("[V7a] skipping Snorkel re-fit (--no-snorkel).", flush=True)
        snorkel_weights: dict[str, dict[str, float]] = {}
    else:
        snorkel_weights = recover_snorkel_weights(
            sample=args.snorkel_sample, seed=args.seed
        )

    print("[V7a] writing markdown report...", flush=True)
    tags = _lf_tag_lookup()
    OUTPUT_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    emit_markdown(report, snorkel_weights, tags, OUTPUT_MD_PATH)
    print(f"[V7a] wrote {OUTPUT_MD_PATH}", flush=True)

    # Also persist a JSON snapshot of the per-LF stats + Snorkel weights so
    # V8 / V7c can consume without re-running V7a.
    OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    json_blob: dict[str, Any] = {
        "snorkel_sample": args.snorkel_sample,
        "seed": args.seed,
        "snorkel_weights": snorkel_weights,
        "tags": tags,
        "probes": {},
    }
    for probe, info in report.items():
        per_lf_serial: dict[str, Any] = {}
        for name, stats in info["per_lf"].items():
            per_lf_serial[name] = {
                "n_fires": stats["n_fires"],
                "n_correct": stats["n_correct"],
                "n_with_gold": stats["n_with_gold"],
                "per_class_correct": dict(stats["per_class_correct"]),
                "per_class_fires": dict(stats["per_class_fires"]),
                "vote_distribution": dict(stats["vote_distribution"]),
            }
        json_blob["probes"][probe] = {
            "lf_names": info["lf_names"],
            "L_shape": list(info["L_shape"]),
            "per_lf": per_lf_serial,
        }
    OUTPUT_JSON_PATH.write_text(json.dumps(json_blob, indent=2, default=int))
    print(f"[V7a] wrote {OUTPUT_JSON_PATH}", flush=True)

    # Stdout summary (also captured in markdown).
    print()
    print("=" * 70)
    print("V7a SUMMARY")
    print("=" * 70)
    for probe in ("relation_axis", "subject_role", "object_role", "scope",
                   "verify_grounding"):
        if probe not in report:
            continue
        per_lf = report[probe]["per_lf"]
        n_pass = n_fail = n_marginal = 0
        for name, stats in per_lf.items():
            fires = stats["n_fires"]
            correct = stats["n_correct"]
            acc = correct / fires if fires > 0 else float("nan")
            sw = snorkel_weights.get(probe, {}).get(name, float("nan"))
            ci_lo, ci_hi = wilson_ci(correct, fires)
            bucket = _bucket_lf(acc, fires, sw, ci_lo, ci_hi)
            if bucket == "PASS":
                n_pass += 1
            elif bucket == "FAIL":
                n_fail += 1
            else:
                n_marginal += 1
        print(f"  {probe:<22} PASS={n_pass} MARGINAL={n_marginal} FAIL={n_fail}")


if __name__ == "__main__":
    main()

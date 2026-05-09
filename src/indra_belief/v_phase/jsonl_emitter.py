"""V6d — stratified subsample + JSONL training/val emission.

Per `research/v5r_data_prep_doctrine.md`:
  - §6: per-probe sample-size targets + LLM-templating synthetic oracle.
  - §8: JSONL output schema (messages, completion, soft_labels, synthetic).
  - §10: synthetic_flag tagging.
  - §12: two-filter contamination guard (existing script + trigram-Jaccard).

Pipeline:
  1. Read V6c parquets at `data/v_phase/labels/{probe}{suffix}.parquet`.
  2. Re-walk corpus to recover (claim, evidence) text per record_id.
  3. Stratified subsample per V5r §6 targets.
  4. For classes < target, augment via synthetic-oracle templates from
     `data/v_phase/synthetic_oracles/{probe}_{class}.yaml`.
  5. Render production prompt for each record by importing the probe's
     `_SYSTEM_PROMPT` + `_FEW_SHOTS` verbatim (byte-equality with
     inference-time prompts).
  6. Apply two-filter contamination guard (Filter 1 = `check_contamination`;
     Filter 2 = trigram-Jaccard at 95th-pct natural-overlap threshold).
     Synthetic records SKIP both filters.
  7. Write 90/10 split JSONL files at
     `data/v_phase/{train,val}/{probe}{suffix}.jsonl`.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

# Late imports of optional deps at usage sites; keep top-level light so
# unit tests that exercise utility functions don't pay full data-stack cost.

# Reuse production prompt builders verbatim — V5r §8 byte-equality contract.
from indra_belief.scorers.probes import (
    object_role as _po,
    relation_axis as _pr,
    scope as _ps,
    subject_role as _psub,
)
from indra_belief.scorers import grounding as _grounding


# ---------------------------------------------------------------------------
# V5r §6 targets per probe (full-corpus). At sample-mode the limits are
# clamped down by what the V6c parquet actually contains.
# ---------------------------------------------------------------------------

PROBE_TARGETS_FULL: dict[str, dict[str, int]] = {
    "relation_axis":     {"total": 24000, "min_per_class": 3000, "K": 8},
    "subject_role":      {"total": 15000, "min_per_class": 3000, "K": 5},
    "object_role":       {"total": 15000, "min_per_class": 3000, "K": 5},
    "scope":             {"total": 15000, "min_per_class": 3000, "K": 5},
    "verify_grounding":  {"total": 28000, "min_per_class": 3000, "K": 4},
}

# Synthetic-oracle target per rare class. V5r §6: 100-300 records;
# V6d brief: 5-10 templates × ~30 placeholders → 150-300.
SYNTHETIC_TARGET_PER_CLASS: int = 200

# Default holdout file for Filter 1 (V5r §12 + check_contamination CLI default).
DEFAULT_HOLDOUT_PATH = ROOT / "data" / "benchmark" / "holdout_v15_sample.jsonl"

# Output paths.
DEFAULT_LABELS_DIR = ROOT / "data" / "v_phase" / "labels"
DEFAULT_TRAIN_DIR = ROOT / "data" / "v_phase" / "train"
DEFAULT_VAL_DIR = ROOT / "data" / "v_phase" / "val"
DEFAULT_THRESHOLD_PATH = ROOT / "data" / "v_phase" / "contamination_threshold.json"
DEFAULT_SYNTHETIC_DIR = ROOT / "data" / "v_phase" / "synthetic_oracles"


# ---------------------------------------------------------------------------
# Probe class ordering (must match V6c parquet column order).
# ---------------------------------------------------------------------------

def _probe_class_names(probe: str) -> list[str]:
    """Return class name list in the order V6c parquet uses for soft-label
    columns (`class_proba_<name>` + argmax_class). Mirrors
    `labeling._probe_class_names` so we have a single source of truth.
    """
    from indra_belief.v_phase.substrate_lfs import (  # noqa: WPS433
        GROUNDING_CLASSES, RELATION_AXIS_CLASSES,
        ROLE_CLASSES, SCOPE_CLASSES,
    )
    src = {
        "relation_axis": RELATION_AXIS_CLASSES,
        "subject_role": ROLE_CLASSES,
        "object_role": ROLE_CLASSES,
        "scope": SCOPE_CLASSES,
        "verify_grounding": GROUNDING_CLASSES,
    }[probe]
    return [name for name, _ in sorted(src.items(), key=lambda kv: kv[1])]


# ---------------------------------------------------------------------------
# Trigram-Jaccard utilities (V5r §12 Filter 2)
# ---------------------------------------------------------------------------

# Biomedical-stopword list (PubMed-derived top-50 by frequency, per V5r §12).
_BIOMED_STOPWORDS: frozenset[str] = frozenset({
    "cell", "cells", "protein", "proteins", "expression", "level", "levels",
    "study", "studies", "result", "results", "data", "figure", "analysis",
    "using", "shown", "observed", "effect", "effects", "patient", "patients",
    "tumor", "tumors", "tissue", "tissues", "human", "mouse", "mice", "rat",
    "treatment", "control", "samples", "vivo", "vitro", "induced", "increased",
    "decreased", "found", "showed", "demonstrated", "indicated", "suggested",
    "table", "method", "methods", "western", "blot", "experiment",
    # plus standard English stopwords (compact list, sufficient for trigrams).
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at",
    "by", "for", "with", "without", "as", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "this", "that", "these", "those", "we", "our", "their", "its", "it",
    "from", "into", "than", "then", "such", "which", "while", "where",
    "also", "may", "can", "not", "no", "only", "other", "more", "most",
    "however", "thus", "therefore",
})

_PUNCT_RE = re.compile(r"[^a-z0-9 ]+")
_WS_RE = re.compile(r"\s+")


def _trigram_normalize(text: str) -> list[str]:
    """Normalize a string for trigram-Jaccard: lowercase, strip punctuation,
    drop stopwords, collapse whitespace. Returns the resulting token list.
    """
    if not text:
        return []
    s = text.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    tokens = [tok for tok in s.split() if tok and tok not in _BIOMED_STOPWORDS]
    return tokens


def _trigrams(text: str) -> set[tuple[str, str, str]]:
    """Generate the set of word-level trigrams (n=3) for a normalized text."""
    tokens = _trigram_normalize(text)
    if len(tokens) < 3:
        return set()
    return {tuple(tokens[i:i + 3]) for i in range(len(tokens) - 2)}


def trigram_jaccard(a: str, b: str) -> float:
    """Compute trigram-Jaccard similarity. Empty trigram sets → 0.0."""
    sa = _trigrams(a)
    sb = _trigrams(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def calibrate_trigram_threshold(
    texts: list[str], n_pairs: int = 10000, percentile: float = 95.0,
    seed: int = 0,
) -> float:
    """Sample `n_pairs` random pairs from `texts` (without self-pairing),
    compute trigram-Jaccard distribution, return the `percentile`th value.

    V5r §12 commits to 95th-percentile-of-natural threshold so the cut is
    calibrated to the corpus's natural overlap distribution.
    """
    rng = random.Random(seed)
    if len(texts) < 2:
        return 0.0
    samples: list[float] = []
    for _ in range(n_pairs):
        i, j = rng.sample(range(len(texts)), 2)
        samples.append(trigram_jaccard(texts[i], texts[j]))
    samples.sort()
    if not samples:
        return 0.0
    # Percentile via linear interpolation. Equivalent to numpy.percentile
    # default mode but no numpy dep at this call site.
    k = (len(samples) - 1) * (percentile / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(samples) - 1)
    frac = k - lo
    return samples[lo] * (1 - frac) + samples[hi] * frac


# ---------------------------------------------------------------------------
# Filter 1 wrapper — invoke check_contamination as a library
# ---------------------------------------------------------------------------

def _load_holdout_evidences(holdout_path: Path) -> list[dict]:
    """Load holdout JSONL records {evidence, subject, object, ...}."""
    if not holdout_path.exists():
        return []
    out = []
    with holdout_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ev = rec.get("evidence_text") or rec.get("evidence") or ""
            subj = rec.get("subject", "") or ""
            obj = rec.get("object", "") or ""
            if ev:
                out.append({"evidence": ev, "subject": subj, "object": obj})
    return out


def _build_filter1_index(holdout: list[dict]) -> dict[str, Any]:
    """Pre-build the data structures `check_contamination.find_contamination`
    expects. We reuse the algorithm by routing each candidate through the
    library's `find_contamination` with a single-example list — but to
    avoid 100K library calls, we inline an equivalent index here.
    """
    # check_contamination uses _norm to casefold + collapse-whitespace +
    # strip trailing punctuation. We replicate it inline for index speed.
    from scripts.check_contamination import _norm  # noqa: WPS433

    norm_to_recs: dict[str, list[dict]] = {}
    pair_to_files: dict[tuple[str, str], list[str]] = {}
    for rec in holdout:
        n = _norm(rec["evidence"])
        if not n:
            continue
        norm_to_recs.setdefault(n, []).append(rec)
        if rec["subject"] and rec["object"]:
            pair_to_files.setdefault((rec["subject"], rec["object"]),
                                     []).append("holdout")
    return {
        "norm_to_recs": norm_to_recs,
        "norm_set": set(norm_to_recs.keys()),
        "norms": list(norm_to_recs.keys()),
        "pair_to_files": pair_to_files,
        "_norm": _norm,
    }


def filter1_contaminated(
    candidate: dict, idx: dict, *, min_substr_len: int = 30,
    paraphrase_window: int = 50, paraphrase_stride: int = 5,
) -> str | None:
    """Return contamination kind if the candidate's evidence overlaps with
    a holdout record per the `check_contamination` algorithm, else None.

    Mirrors find_contamination's per-record logic (exact, substring,
    paraphrase_overlap, pair) without the fewshot-source loop overhead.
    """
    ev = candidate.get("evidence", "") or ""
    if not ev:
        return None
    en = idx["_norm"](ev)
    if not en:
        return None
    if en in idx["norm_set"]:
        return "exact"
    if len(en) >= min_substr_len:
        for n in idx["norms"]:
            if n == en:
                continue
            if en in n or n in en:
                return "substring_in_eval" if en in n else "eval_in_substring"
    if len(en) >= paraphrase_window:
        shingles = set()
        for i in range(0, len(en) - paraphrase_window + 1, paraphrase_stride):
            shingles.add(en[i:i + paraphrase_window])
        for n in idx["norms"]:
            if n == en:
                continue
            if any(sh in n for sh in shingles):
                return "paraphrase_overlap"
    subj = candidate.get("subject", "") or ""
    obj = candidate.get("object", "") or ""
    if subj and obj and (subj, obj) in idx["pair_to_files"]:
        return "pair"
    return None


# ---------------------------------------------------------------------------
# Production-prompt rendering (V5r §8 byte-equality)
# ---------------------------------------------------------------------------

def _render_messages_axis(claim_component: str, evidence_text: str,
                           probe_module) -> list[dict]:
    """Render the production few-shot conversation for a non-grounding probe.

    Reuses `probe_module._SYSTEM_PROMPT` + `_FEW_SHOTS` verbatim. Builds
    a `messages` list that — concatenated with the actual question — is
    byte-identical to what `_llm.llm_classify` constructs at inference.
    """
    messages: list[dict] = [
        {"role": "system", "content": probe_module._SYSTEM_PROMPT},
    ]
    for shot_q, shot_a in probe_module._FEW_SHOTS:
        messages.append({"role": "user", "content": shot_q})
        messages.append({"role": "assistant", "content": shot_a})
    # The actual question — same string assembly as in each probe's
    # answer() function (no substrate_hint at training time).
    user_msg_parts = [_q_label(probe_module),
                      f"EVIDENCE: {evidence_text.strip()}"]
    user_msg_parts[0] = f"{_q_label(probe_module)}: {claim_component}"
    user_msg = "\n".join([
        user_msg_parts[0],
        f"EVIDENCE: {evidence_text.strip()}",
    ])
    messages.append({"role": "user", "content": user_msg})
    return messages


def _q_label(module) -> str:
    """Return the user-question label used by each probe at inference.
    Matches the literal strings in each probe's answer() body."""
    name = module.__name__.rsplit(".", 1)[-1]
    return {
        "subject_role": "CLAIM SUBJECT",
        "object_role": "CLAIM OBJECT",
        "relation_axis": "CLAIM",
        "scope": "CLAIM",
    }[name]


def _render_messages_grounding(entity_block: str,
                                evidence_text: str) -> list[dict]:
    """Render the verify_grounding production prompt verbatim.

    `entity_block` is the entity-context block exactly as
    `grounding._build_user_message` would emit (Claim entity, Grounding,
    aliases…). Caller is responsible for assembling it.

    Returns the system+user messages. verify_grounding has NO few-shots
    (matches inference behavior).
    """
    user_msg = (
        f"{entity_block}\n"
        f"\n"
        f'Evidence: "{evidence_text}"\n'
        f"\n"
        f"Does the evidence reference this entity?"
    )
    return [
        {"role": "system", "content": _grounding._SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]


_PROBE_MODULES = {
    "relation_axis": _pr,
    "subject_role":  _psub,
    "object_role":   _po,
    "scope":         _ps,
}


def render_messages(probe: str, claim: str, evidence: str) -> list[dict]:
    """Public dispatcher: produce the inference-equivalent message list."""
    if probe == "verify_grounding":
        return _render_messages_grounding(claim, evidence)
    return _render_messages_axis(claim, evidence, _PROBE_MODULES[probe])


# ---------------------------------------------------------------------------
# Production-prompt byte-equality check (validation utility)
# ---------------------------------------------------------------------------

def verify_byte_equality_relation_axis() -> dict:
    """Compute SHA-256 of the rendered prompt for a synthetic claim and
    check it matches the expected production-prompt assembly. Used by V6d
    test + the `--verify-bytes` CLI flag.

    Output: {"system_sha256", "n_few_shots", "first_user_msg_len"}.
    """
    msgs = render_messages(
        "relation_axis",
        "({ENT_A}, {ENT_B}) — claim axis=activity, sign=positive",
        "Test evidence text.",
    )
    sys_sha = hashlib.sha256(msgs[0]["content"].encode()).hexdigest()
    n_shots = sum(1 for m in msgs[1:-1] if m["role"] == "user")
    return {
        "system_sha256": sys_sha,
        "n_few_shots": n_shots,
        "first_user_msg_len": len(msgs[1]["content"]),
    }


# ---------------------------------------------------------------------------
# Synthetic-oracle template expansion
# ---------------------------------------------------------------------------

def _safe_yaml_load(path: Path) -> dict | None:
    """Minimal YAML reader for our flat templates. We avoid pulling pyyaml
    to keep deps light; the schema is simple enough to parse with regex.

    Falls back to importing yaml if available (more robust).
    """
    try:
        import yaml  # noqa: WPS433
        with path.open() as f:
            return yaml.safe_load(f)
    except Exception:
        # Hand-roll a parser tuned to our 2-space-indented, list-of-dict
        # template files.
        return _hand_yaml_parse(path)


def _hand_yaml_parse(path: Path) -> dict:
    """Conservative YAML parser for our template files. Supports:
      key: value
      key: |
        multi-line
      - key: value
      key:
        - item
    Sufficient for the placeholder + template files we ship.
    """
    out: dict[str, Any] = {}
    cur_list: list | None = None
    cur_obj: dict | None = None
    cur_key: str | None = None
    with path.open() as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            if indent == 0 and ":" in stripped and not stripped.startswith("-"):
                # top-level key
                k, _, v = stripped.partition(":")
                v = v.strip()
                if v:
                    # scalar
                    out[k.strip()] = _strip_quotes(v)
                    cur_list = None
                    cur_obj = None
                else:
                    # nested
                    cur_list = []
                    out[k.strip()] = cur_list
                    cur_obj = None
                    cur_key = k.strip()
            elif stripped.startswith("- ") and cur_list is not None:
                rest = stripped[2:].strip()
                if ":" in rest:
                    cur_obj = {}
                    cur_list.append(cur_obj)
                    k, _, v = rest.partition(":")
                    cur_obj[k.strip()] = _strip_quotes(v.strip())
                else:
                    cur_list.append(_strip_quotes(rest))
                    cur_obj = None
            elif cur_obj is not None and ":" in stripped:
                k, _, v = stripped.partition(":")
                cur_obj[k.strip()] = _strip_quotes(v.strip())
    return out


def _strip_quotes(v: str) -> str:
    if len(v) >= 2 and v[0] == v[-1] and v[0] in '"\'':
        return v[1:-1]
    return v


def load_placeholders(synthetic_dir: Path = DEFAULT_SYNTHETIC_DIR) -> dict[str, list[str]]:
    """Read `_placeholders.yaml`."""
    p = synthetic_dir / "_placeholders.yaml"
    if not p.exists():
        return {}
    data = _safe_yaml_load(p) or {}
    out: dict[str, list[str]] = {}
    for k, v in data.items():
        if isinstance(v, list):
            out[k] = [str(x) for x in v]
    return out


def expand_synthetic_templates(
    probe: str, class_name: str, target_n: int,
    synthetic_dir: Path = DEFAULT_SYNTHETIC_DIR,
    seed: int = 0,
) -> list[dict]:
    """Expand templates × placeholders deterministically. Returns up to
    `target_n` synthetic records {claim, evidence, class}.

    YAML format: list of {claim, evidence} templates with placeholders
    `{ENT_A}`, `{ENT_B}`, `{ENT_X}`, `{VERB}`, `{DNA_ELEMENT}`. The
    placeholder pool is in `_placeholders.yaml`.

    If `target_n` exceeds (templates × placeholder rows), fewer records
    are returned — the caller logs the shortfall.
    """
    yaml_path = synthetic_dir / f"{probe}_{class_name}.yaml"
    if not yaml_path.exists():
        return []
    data = _safe_yaml_load(yaml_path) or {}
    templates = data.get("templates") or []
    if not templates:
        return []
    placeholders = load_placeholders(synthetic_dir)
    rng = random.Random(seed)
    out: list[dict] = []
    # Build a stable expansion grid: each template iterates over each
    # placeholder slot in lockstep — one synthetic record per (template,
    # placeholder index).
    a_pool = placeholders.get("ent_a", ["ENT_A"])
    b_pool = placeholders.get("ent_b", ["ENT_B"])
    x_pool = placeholders.get("ent_x", ["ENT_X"])
    v_pool = placeholders.get("verb", ["regulates"])
    d_pool = placeholders.get("dna_element", ["the consensus DNA element"])

    def _shuf(seq):
        s = list(seq)
        rng.shuffle(s)
        return s

    a_pool = _shuf(a_pool)
    b_pool = _shuf(b_pool)
    x_pool = _shuf(x_pool)
    v_pool = _shuf(v_pool)
    d_pool = _shuf(d_pool)

    # template × placeholder cartesian product, capped at target_n.
    plen = max(len(a_pool), len(b_pool))
    used = set()  # dedupe by (claim, evidence)
    for tpl_idx, tpl in enumerate(templates):
        if len(out) >= target_n:
            break
        claim_t = tpl.get("claim", "")
        ev_t = tpl.get("evidence", "")
        for i in range(plen):
            if len(out) >= target_n:
                break
            subs = {
                "ENT_A": a_pool[i % len(a_pool)],
                "ENT_B": b_pool[i % len(b_pool)],
                "ENT_X": x_pool[i % len(x_pool)],
                "VERB": v_pool[i % len(v_pool)],
                "DNA_ELEMENT": d_pool[i % len(d_pool)],
            }
            claim = claim_t
            ev = ev_t
            for k, v in subs.items():
                claim = claim.replace("{" + k + "}", v)
                ev = ev.replace("{" + k + "}", v)
            key = (claim, ev)
            if key in used:
                continue
            used.add(key)
            out.append({
                "claim": claim,
                "evidence": ev,
                "class": class_name,
                "synthetic": True,
                "subject": subs["ENT_A"],
                "object": subs["ENT_B"],
            })
    return out[:target_n]


# ---------------------------------------------------------------------------
# Corpus re-walk: rebuild (claim, evidence) text from V6c record_ids
# ---------------------------------------------------------------------------

def _index_corpus_by_record_id(corpus_path: Path,
                                 needed: set[str]) -> dict[str, dict]:
    """Walk corpus, return {record_id → record_payload} for record_ids
    matching the V6c id format `<matches_hash>:<source_hash>` or
    `<matches_hash>:<source_hash>:<ent_idx>`.

    `needed`: set of record_ids we want. Walk stops once we've found
    them all (or exhausts corpus).
    """
    from indra_belief.v_phase.labeling import (  # noqa: WPS433
        _make_ev_dict, _make_stmt_dict, _statement_agents,
        _statement_subj_obj, _stream_corpus_records,
    )
    # Decompose needed ids into target set keyed by (matches_hash, source_hash).
    base_keys: set[tuple[str, str]] = set()
    for rid in needed:
        parts = rid.split(":")
        if len(parts) >= 2:
            base_keys.add((parts[0], parts[1]))
    out: dict[str, dict] = {}
    for raw in _stream_corpus_records(corpus_path):
        evidences = raw.get("evidence") or []
        subj, obj = _statement_subj_obj(raw)
        agents = _statement_agents(raw)
        m = str(raw.get("matches_hash"))
        for ev in evidences:
            if not isinstance(ev, dict):
                continue
            stmt_d = _make_stmt_dict(raw, subj, obj, ev)
            ev_d = _make_ev_dict(ev)
            sh = str(ev_d.get("source_hash"))
            key = (m, sh)
            if key not in base_keys:
                continue
            base_id = f"{m}:{sh}"
            payload = {
                "stmt": stmt_d,
                "ev": ev_d,
                "agents": agents,
                "raw_type": raw.get("type"),
            }
            out[base_id] = payload
            for ent_idx in range(len(agents)):
                out[f"{base_id}:{ent_idx}"] = payload
        if len(out) >= len(needed):
            break
    return out


# ---------------------------------------------------------------------------
# Per-probe claim_component rendering (matches router.py)
# ---------------------------------------------------------------------------

def _claim_axis_for_stmt_type(t: str) -> str:
    return {
        "Phosphorylation": "modification", "Dephosphorylation": "modification",
        "Methylation": "modification", "Demethylation": "modification",
        "Acetylation": "modification", "Deacetylation": "modification",
        "Ubiquitination": "modification", "Deubiquitination": "modification",
        "Sumoylation": "modification", "Desumoylation": "modification",
        "Activation": "activity", "Inhibition": "activity",
        "IncreaseAmount": "amount", "DecreaseAmount": "amount",
        "Complex": "binding",
        "Translocation": "localization",
    }.get(t, "activity")


def _claim_sign_for_stmt_type(t: str) -> str:
    if t in ("Phosphorylation", "Methylation", "Acetylation",
             "Ubiquitination", "Sumoylation", "Activation", "IncreaseAmount"):
        return "positive"
    if t in ("Dephosphorylation", "Demethylation", "Deacetylation",
             "Deubiquitination", "Desumoylation", "Inhibition",
             "DecreaseAmount"):
        return "negative"
    return "neutral"


def render_claim_component(probe: str, stmt: dict) -> str:
    """Render the production claim_component string for a probe.

    Mirrors `router.py` line patterns (see V6d brief). The router itself
    builds these from EvidenceContext; for training we don't have the
    perturbation marker (substrate-only), so the literal claim sign is
    used as `effective_sign`.
    """
    subj = stmt.get("subject") or ""
    obj = stmt.get("object") or ""
    t = stmt.get("stmt_type") or ""
    axis = _claim_axis_for_stmt_type(t)
    sign = _claim_sign_for_stmt_type(t)
    if probe == "subject_role":
        return f"{subj} (axis={axis}, sign={sign}, objects={[obj] if obj else []})"
    if probe == "object_role":
        return f"{obj} (axis={axis}, sign={sign}, subject={subj})"
    if probe == "relation_axis":
        return f"({subj}, {obj}) — claim axis={axis}, sign={sign}"
    if probe == "scope":
        return f"relation between {subj} and {obj}"
    raise ValueError(f"unknown probe {probe!r}")


def render_grounding_block(stmt: dict, ent_idx: int,
                            agents: list[dict]) -> tuple[str, str]:
    """Render the `_build_user_message`-equivalent entity block for an
    agent in `agents[ent_idx]`. Returns (entity_block, entity_name).

    Without re-running Gilda at training time (heavy), we inline the
    minimal fields the production prompt exposes: name + grounding-or-
    `<none>`. V5r §6 notes verify_grounding's V6 dry-run validates
    byte-equality on a holdout-excluded sample; that's V7-stage work.
    For V6d we ensure the FORMAT matches inference.
    """
    if ent_idx >= len(agents):
        return "Claim entity: ?\nGrounding: <none — possibly a generic class noun>", "?"
    ag = agents[ent_idx] or {}
    name = ag.get("name") or "?"
    db_refs = ag.get("db_refs") or {}
    db = None
    db_id = None
    for cand_db in ("HGNC", "FPLX", "UP", "MESH", "CHEBI", "NCBI Gene"):
        if cand_db in db_refs:
            db = cand_db
            db_id = db_refs[cand_db]
            break
    parts = [f"Claim entity: {name}"]
    if db and db_id:
        parts.append(f"Grounding: {db}:{db_id}")
    else:
        parts.append("Grounding: <none — possibly a generic class noun>")
    return "\n".join(parts), name


# ---------------------------------------------------------------------------
# Stratified subsample
# ---------------------------------------------------------------------------

def stratified_subsample(
    df, probe: str, *, target_total: int, min_per_class: int,
    seed: int = 0,
) -> list[int]:
    """Return list of df row indices chosen by stratified subsample.

    Strategy: aim for `min_per_class` per class. If a class has more
    than its share of the budget, downsample. Total clipped at
    `target_total`.
    """
    rng = random.Random(seed)
    classes = sorted(set(df["argmax_class"].tolist()))
    by_class: dict[str, list[int]] = defaultdict(list)
    for i, cls in enumerate(df["argmax_class"]):
        if df["kept_for_training"].iloc[i]:
            by_class[cls].append(i)
    chosen: list[int] = []
    per_class_quota = min_per_class
    for cls in classes:
        pool = by_class.get(cls, [])
        if len(pool) <= per_class_quota:
            chosen.extend(pool)
        else:
            chosen.extend(rng.sample(pool, per_class_quota))
    # If we have remaining budget, fill with extras above quota.
    remaining = target_total - len(chosen)
    if remaining > 0:
        extras: list[int] = []
        for cls in classes:
            pool = by_class.get(cls, [])
            taken = set(idx for idx in chosen if df["argmax_class"].iloc[idx] == cls)
            extras.extend(idx for idx in pool if idx not in taken)
        if extras:
            rng.shuffle(extras)
            chosen.extend(extras[:remaining])
    chosen = chosen[:target_total]
    rng.shuffle(chosen)
    return chosen


# ---------------------------------------------------------------------------
# JSONL emission
# ---------------------------------------------------------------------------

def _make_completion(class_name: str) -> str:
    """Production-shape completion JSON: {"answer": "<class>", "rationale": ""}."""
    return json.dumps({"answer": class_name, "rationale": ""},
                      separators=(", ", ": "))


def _make_grounding_completion(status: str) -> str:
    return json.dumps({"status": status, "rationale": ""},
                      separators=(", ", ": "))


def write_jsonl(records: list[dict], path: Path) -> None:
    """Atomic write: open tmp, write records, rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def emit_for_probe(
    probe: str, *,
    labels_dir: Path, suffix: str,
    holdout_path: Path,
    train_dir: Path, val_dir: Path,
    val_fraction: float = 0.10,
    threshold: float | None = None,
    threshold_path: Path | None = None,
    full_targets: bool = False,
    seed: int = 0,
    log: list[str] | None = None,
) -> dict:
    """Emit train+val JSONL for one probe. Returns summary stats.

    `threshold`: if None, we calibrate the trigram-Jaccard threshold from
    the V6c parquet's evidence text pool. If provided (e.g., from a prior
    run's `contamination_threshold.json`), we skip calibration.
    """
    import pandas as pd  # local import (heavy)

    log = log if log is not None else []
    summary: dict[str, Any] = {"probe": probe}
    parquet_path = labels_dir / f"{probe}{suffix}.parquet"
    if not parquet_path.exists():
        summary["error"] = f"missing parquet: {parquet_path}"
        return summary
    df = pd.read_parquet(parquet_path)
    summary["n_parquet_rows"] = int(df.shape[0])
    summary["n_kept_for_training"] = int(df["kept_for_training"].sum())

    # Re-walk corpus to recover (stmt, ev) by record_id for kept rows.
    needed = set(df.loc[df["kept_for_training"], "record_id"].tolist())
    corpus_path = ROOT / "data" / "benchmark" / "indra_benchmark_corpus.json.gz"
    log.append(f"[{probe}] re-walking corpus for {len(needed)} record_ids...")
    t0 = time.time()
    rid_to_payload = _index_corpus_by_record_id(corpus_path, needed)
    log.append(f"[{probe}]   recovered {len(rid_to_payload)}/{len(needed)} "
               f"(elapsed {time.time()-t0:.1f}s)")

    # Stratified subsample.
    targets = PROBE_TARGETS_FULL[probe]
    if not full_targets:
        # Cap each class at what's available × at most min_per_class
        # (sample-size mode, ceiling). We still record the full-corpus
        # targets in summary for V8.
        kept_per_class: dict[str, int] = (
            df.loc[df["kept_for_training"], "argmax_class"]
              .value_counts().to_dict()
        )
        target_total = sum(min(c, targets["min_per_class"])
                           for c in kept_per_class.values())
    else:
        target_total = targets["total"]

    chosen_idx = stratified_subsample(
        df, probe, target_total=target_total,
        min_per_class=targets["min_per_class"], seed=seed,
    )
    summary["n_chosen_natural"] = len(chosen_idx)
    log.append(f"[{probe}] stratified subsample: {len(chosen_idx)} natural records")

    class_names = _probe_class_names(probe)
    proba_cols = [f"class_proba_{c}" for c in class_names]

    # Build natural records.
    natural: list[dict] = []
    for idx in chosen_idx:
        rid = df["record_id"].iloc[idx]
        cls = df["argmax_class"].iloc[idx]
        soft_labels = {
            class_names[k]: float(df[col].iloc[idx])
            for k, col in enumerate(proba_cols)
        }
        payload = rid_to_payload.get(rid)
        if payload is None:
            continue
        stmt = payload["stmt"]
        ev_text = payload["ev"].get("text") or ""
        if not ev_text:
            continue
        if probe == "verify_grounding":
            ent_idx = int(rid.rsplit(":", 1)[-1])
            entity_block, ent_name = render_grounding_block(
                stmt, ent_idx, payload["agents"],
            )
            messages = render_messages(probe, entity_block, ev_text)
            completion = _make_grounding_completion(cls)
            subject_for_filter = stmt.get("subject") or ""
            object_for_filter = stmt.get("object") or ""
        else:
            claim_str = render_claim_component(probe, stmt)
            messages = render_messages(probe, claim_str, ev_text)
            completion = _make_completion(cls)
            subject_for_filter = stmt.get("subject") or ""
            object_for_filter = stmt.get("object") or ""
        natural.append({
            "messages": messages,
            "completion": completion,
            "soft_labels": soft_labels,
            "synthetic": False,
            # auxiliary fields used by contamination filter only;
            # stripped before JSONL write.
            "_evidence": ev_text,
            "_subject": subject_for_filter,
            "_object": object_for_filter,
            "_class": cls,
            "_record_id": rid,
        })
    summary["n_natural_built"] = len(natural)

    # Per-class natural counts (post-subsample).
    natural_per_class: dict[str, int] = defaultdict(int)
    for r in natural:
        natural_per_class[r["_class"]] += 1

    # Synthetic augmentation for rare classes.
    synthetic: list[dict] = []
    rare_threshold = targets["min_per_class"] if full_targets else 50
    for cls in class_names:
        nat_n = natural_per_class.get(cls, 0)
        if nat_n >= rare_threshold:
            continue
        # Read templates for this rare class — emit up to SYNTHETIC_TARGET_PER_CLASS.
        syn_recs = expand_synthetic_templates(
            probe, cls, target_n=SYNTHETIC_TARGET_PER_CLASS, seed=seed,
        )
        for s in syn_recs:
            soft = {c: 0.0 for c in class_names}
            soft[cls] = 1.0
            if probe == "verify_grounding":
                # Build a plausible entity block from the synthetic claim line.
                entity_block = s["claim"]
                messages = render_messages(probe, entity_block, s["evidence"])
                completion = _make_grounding_completion(cls)
            else:
                messages = render_messages(probe, s["claim"], s["evidence"])
                completion = _make_completion(cls)
            synthetic.append({
                "messages": messages,
                "completion": completion,
                "soft_labels": soft,
                "synthetic": True,
                "_evidence": s["evidence"],
                "_subject": s.get("subject", ""),
                "_object": s.get("object", ""),
                "_class": cls,
                "_record_id": f"synthetic:{probe}:{cls}:{len(synthetic)}",
            })
    summary["n_synthetic"] = len(synthetic)
    log.append(f"[{probe}] synthetic augmentation: {len(synthetic)} records")

    # Filter 1: contamination via check_contamination algorithm.
    holdout_recs = _load_holdout_evidences(holdout_path)
    filter1_idx = _build_filter1_index(holdout_recs)

    # Filter 2 threshold: calibrate from the natural evidence pool if not
    # supplied. Use natural records only (synthetic shouldn't bias
    # threshold).
    nat_evs = [r["_evidence"] for r in natural]
    if threshold is None:
        threshold = calibrate_trigram_threshold(nat_evs, n_pairs=10000, seed=seed)
        log.append(f"[{probe}] trigram threshold (95th pct): {threshold:.4f}")
    summary["trigram_threshold_calibrated"] = float(threshold)

    # Pre-compute holdout trigram sets once.
    holdout_trigram_sets = [_trigrams(r["evidence"]) for r in holdout_recs]

    # V5r §12: "Any training record exceeding this against any holdout
    # evidence is dropped" — strict greater-than. Floor at 0.05 so a
    # degenerate empty-overlap calibration (sample-size mode) doesn't
    # trip on every trigram match.
    effective_threshold = max(threshold, 0.05)
    summary["trigram_threshold"] = float(effective_threshold)
    if threshold_path is not None:
        threshold_path.parent.mkdir(parents=True, exist_ok=True)
        with threshold_path.open("w") as f:
            json.dump({
                "threshold": float(effective_threshold),
                "calibrated_threshold": float(threshold),
                "min_floor": 0.05,
                "probe": probe,
                "calibration_pairs": 10000,
                "percentile": 95.0,
            }, f, indent=2)

    def filter2_contaminated(ev: str) -> float | None:
        """Return max trigram-Jaccard with any holdout, if STRICTLY above
        the calibrated threshold."""
        cand_set = _trigrams(ev)
        if not cand_set:
            return None
        max_overlap = 0.0
        for hs in holdout_trigram_sets:
            if not hs:
                continue
            inter = len(cand_set & hs)
            if inter == 0:
                continue
            union = len(cand_set | hs)
            j = inter / union if union else 0.0
            if j > max_overlap:
                max_overlap = j
        return max_overlap if max_overlap > effective_threshold else None

    drops_f1 = drops_f2 = drops_both = 0
    surviving: list[dict] = []
    for r in natural + synthetic:
        if r["synthetic"]:
            surviving.append(r)
            continue
        f1 = filter1_contaminated(
            {"evidence": r["_evidence"], "subject": r["_subject"],
             "object": r["_object"]},
            filter1_idx,
        )
        f2 = filter2_contaminated(r["_evidence"])
        if f1 and f2 is not None:
            drops_both += 1
            continue
        if f1:
            drops_f1 += 1
            continue
        if f2 is not None:
            drops_f2 += 1
            continue
        surviving.append(r)
    summary["filter_drops"] = {
        "f1_only": drops_f1, "f2_only": drops_f2, "both": drops_both,
        "total_dropped": drops_f1 + drops_f2 + drops_both,
        "total_input": len(natural) + len(synthetic),
        "total_surviving": len(surviving),
    }
    log.append(f"[{probe}] filter drops: f1_only={drops_f1} f2_only={drops_f2} "
               f"both={drops_both}")

    # 90/10 split.
    rng = random.Random(seed + 1)
    rng.shuffle(surviving)
    n_val = max(1, int(len(surviving) * val_fraction)) if surviving else 0
    val = surviving[:n_val]
    train = surviving[n_val:]

    # Strip auxiliary "_…" fields for JSONL.
    def _strip(records):
        return [{k: v for k, v in r.items() if not k.startswith("_")}
                for r in records]

    train_path = train_dir / f"{probe}{suffix}.jsonl"
    val_path = val_dir / f"{probe}{suffix}.jsonl"
    write_jsonl(_strip(train), train_path)
    write_jsonl(_strip(val), val_path)
    summary["paths"] = {"train": str(train_path), "val": str(val_path)}
    summary["n_train"] = len(train)
    summary["n_val"] = len(val)

    # Per-class final counts.
    final_per_class = defaultdict(int)
    for r in surviving:
        final_per_class[r["_class"]] += 1
    summary["per_class_final"] = dict(final_per_class)
    summary["per_class_natural"] = dict(natural_per_class)
    log.append(f"[{probe}] wrote {train_path.name} (n={len(train)}) + "
               f"{val_path.name} (n={len(val)})")
    return summary


def run(
    *, from_sample: bool = False, full_targets: bool = False,
    holdout_path: Path = DEFAULT_HOLDOUT_PATH,
    labels_dir: Path = DEFAULT_LABELS_DIR,
    train_dir: Path = DEFAULT_TRAIN_DIR,
    val_dir: Path = DEFAULT_VAL_DIR,
    threshold_path: Path = DEFAULT_THRESHOLD_PATH,
    seed: int = 0,
) -> dict:
    """End-to-end V6d run across all 5 probes."""
    suffix = "_sample" if from_sample else ""
    log: list[str] = []
    log.append(f"[V6d] starting suffix={suffix!r} full_targets={full_targets}")

    # Byte-equality check (V5r §8) — log once.
    be = verify_byte_equality_relation_axis()
    log.append(f"[V6d] relation_axis byte-equality: system_sha={be['system_sha256'][:16]} "
               f"n_few_shots={be['n_few_shots']} first_user_len={be['first_user_msg_len']}")

    summaries: dict[str, dict] = {}
    threshold = None
    for probe in ["relation_axis", "subject_role", "object_role",
                  "scope", "verify_grounding"]:
        s = emit_for_probe(
            probe,
            labels_dir=labels_dir, suffix=suffix,
            holdout_path=holdout_path,
            train_dir=train_dir, val_dir=val_dir,
            threshold=threshold,
            threshold_path=threshold_path if threshold is None else None,
            full_targets=full_targets, seed=seed, log=log,
        )
        summaries[probe] = s
        # Reuse the first probe's calibrated threshold across remaining probes
        # so all five share one cut (V5r §12 — single threshold, not per-probe).
        if threshold is None and "trigram_threshold" in s:
            threshold = s["trigram_threshold"]

    overall = {
        "probes": summaries,
        "byte_equality_check": be,
        "trigram_threshold": threshold,
        "log": log,
    }
    for line in log:
        print(line)
    return overall


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="V6d — stratified subsample + JSONL emission",
    )
    p.add_argument("--from-sample", action="store_true",
                   help="Read V6c sample parquets (`{probe}_sample.parquet`)")
    p.add_argument("--full-targets", action="store_true",
                   help="Apply full-corpus per-probe targets (V5r §6).")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    run(from_sample=args.from_sample, full_targets=args.full_targets,
        seed=args.seed)


if __name__ == "__main__":
    main()

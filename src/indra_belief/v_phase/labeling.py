"""V6c — per-probe label-matrix builder + Snorkel LabelModel fit.

Per `research/v5r_data_prep_doctrine.md`:
  - §4: aggregation strategy with corrected Snorkel API (no `Y_dev`,
    explicit `class_balance=[1/K]*K`, manual numpy.corrcoef diagnosis).
  - §5: holdout exclusion by source_hash, matches_hash, or
    (pmid, entity_pair).
  - §6: per-probe sample-size targets (verify_grounding multiplier).
  - §11: NO `Y_dev` path — `Y_dev` would route U2 gold into
    `class_balance` and create a hidden filtering path.

V6a + V6b deliver the 13 substrate + 38 clean LFs (51 total). This
module:

  1. Walks `data/benchmark/indra_benchmark_corpus.json.gz`, applies V5r
     §5 holdout exclusion, yields filtered (statement_dict, evidence_dict)
     records.
  2. For each probe, builds an integer Λ matrix (n_records × n_LFs_for_probe)
     of class-index votes ∈ {-1=ABSTAIN, 0..K-1}.
  3. Diagnoses pairwise LF correlations via `numpy.corrcoef` on records
     where both LFs fire (overlap ≥50). Reports any |r| > 0.5 to stdout
     for V5r §4 step 2 follow-up (LFs are NOT auto-merged in V6c).
  4. Fits `LabelModel(cardinality=K, verbose=True).fit(L_train=Λ,
     class_balance=[1/K]*K, n_epochs=500, lr=0.01)` per probe.
  5. Emits `data/v_phase/labels/{probe}{suffix}.parquet` with columns
     (record_id, class_proba_<class>, argmax_class, max_proba,
     kept_for_training).

Use `--sample N` to limit to the first N (stmt, evidence) pairs (default
10000); `--sample 0` runs the full corpus. Probe-level `verify_grounding`
runs per-entity (multiplier ~2.3×).
"""
from __future__ import annotations

import argparse
import gzip
import itertools
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np

# Snorkel + parquet are heavyweight; import inside main paths so unit
# tests that exercise the corpus walker / LF application don't pay the cost.

# ---------------------------------------------------------------------------
# Holdout exclusion (V5r §5)
# ---------------------------------------------------------------------------

ABSTAIN: int = -1

ROOT = Path(__file__).resolve().parents[3]  # repo root
HOLDOUT_PATH = ROOT / "data" / "benchmark" / "holdout_v15_sample.jsonl"
CORPUS_PATH = ROOT / "data" / "benchmark" / "indra_benchmark_corpus.json.gz"
OUTPUT_DIR = ROOT / "data" / "v_phase" / "labels"


def _entity_pair_key(subj: str | None, obj: str | None) -> frozenset:
    """Order-independent key for a (subject, object) pair."""
    return frozenset((s for s in (subj, obj) if s))


def load_holdout_exclusion(holdout_path: Path = HOLDOUT_PATH) -> dict[str, set]:
    """Build the V5r §5 holdout exclusion sets:
      - source_hashes: int set of holdout evidence-level hashes
      - matches_hashes: str set of holdout statement-level hashes
      - pmid_pairs: set of (pmid, frozenset({subj, obj})) tuples

    Records are dropped if ANY of the three keys overlap. `pmid`-only
    overlap (no entity match) is NOT in the exclusion set per V5r §5.
    """
    source_hashes: set[int] = set()
    matches_hashes: set[str] = set()
    pmid_pairs: set[tuple[str, frozenset]] = set()
    if not holdout_path.exists():
        return {"source_hashes": source_hashes,
                "matches_hashes": matches_hashes,
                "pmid_pairs": pmid_pairs}
    with holdout_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sh = rec.get("source_hash")
            if isinstance(sh, int):
                source_hashes.add(sh)
            mh = rec.get("matches_hash")
            if mh is not None:
                matches_hashes.add(str(mh))
            pmid = rec.get("pmid")
            subj = rec.get("subject")
            obj = rec.get("object")
            ep = _entity_pair_key(subj, obj)
            if pmid is not None and ep:
                pmid_pairs.add((str(pmid), ep))
    return {
        "source_hashes": source_hashes,
        "matches_hashes": matches_hashes,
        "pmid_pairs": pmid_pairs,
    }


def is_holdout_excluded(stmt_dict: dict, ev_dict: dict, exc: dict) -> bool:
    """Return True if the (stmt, ev) pair must be dropped per V5r §5."""
    sh = ev_dict.get("source_hash")
    if isinstance(sh, int) and sh in exc["source_hashes"]:
        return True
    mh = stmt_dict.get("matches_hash")
    if mh is not None and str(mh) in exc["matches_hashes"]:
        return True
    pmid = ev_dict.get("pmid") or stmt_dict.get("pmid")
    subj = stmt_dict.get("subject")
    obj = stmt_dict.get("object")
    if pmid and (subj or obj):
        ep = _entity_pair_key(subj, obj)
        if ep and (str(pmid), ep) in exc["pmid_pairs"]:
            return True
    return False


# ---------------------------------------------------------------------------
# Corpus walker
# ---------------------------------------------------------------------------

# Statement-type → fields that name the agent dicts (subj-position, obj-position).
# For Complex statements, agents live in `members` list. For unary types
# (Autophosphorylation), only `enz`/`agent` exists.
_AGENT_FIELDS_SUBJ = ("enz", "subj", "subject", "agent", "factor")
_AGENT_FIELDS_OBJ = ("sub", "obj", "object", "target")


def _agent_name(agent: Any) -> str | None:
    if not isinstance(agent, dict):
        return None
    n = agent.get("name")
    return n if isinstance(n, str) and n else None


def _statement_subj_obj(rec: dict) -> tuple[str | None, str | None]:
    """Best-effort subject/object name extraction from an INDRA dict."""
    t = rec.get("type")
    if t == "Complex" and isinstance(rec.get("members"), list):
        m = [_agent_name(x) for x in rec["members"]]
        m = [x for x in m if x]
        return (m[0] if m else None), (m[1] if len(m) >= 2 else None)
    subj = obj = None
    for key in _AGENT_FIELDS_SUBJ:
        if key in rec and isinstance(rec[key], dict):
            subj = _agent_name(rec[key])
            if subj:
                break
    for key in _AGENT_FIELDS_OBJ:
        if key in rec and isinstance(rec[key], dict):
            obj = _agent_name(rec[key])
            if obj:
                break
    # Translocation has from_location/to_location strings — leave them out
    # of the entity_pair key (V5r §5 needs name-level membership only).
    return subj, obj


def _statement_agents(rec: dict) -> list[dict]:
    """All claim-agent dicts (Complex members, plus subj/obj for binaries)."""
    out: list[dict] = []
    t = rec.get("type")
    if t == "Complex" and isinstance(rec.get("members"), list):
        for m in rec["members"]:
            if isinstance(m, dict) and m.get("name"):
                out.append(m)
        return out
    for key in _AGENT_FIELDS_SUBJ + _AGENT_FIELDS_OBJ:
        a = rec.get(key)
        if isinstance(a, dict) and a.get("name"):
            out.append(a)
    return out


def _make_stmt_dict(raw: dict, subj: str | None, obj: str | None,
                     ev: dict) -> dict:
    """Build the dict-shape that V6a/V6b LFs accept."""
    return {
        "stmt_type": raw.get("type"),
        "subject": subj,
        "object": obj,
        "source_api": ev.get("source_api"),
        "pmid": ev.get("pmid") or (ev.get("text_refs") or {}).get("PMID"),
        "evidence_text": ev.get("text") or "",
        "matches_hash": raw.get("matches_hash"),
        "source_counts": raw.get("source_counts"),
    }


def _make_ev_dict(ev: dict) -> dict:
    pmid = ev.get("pmid")
    if not pmid:
        pmid = (ev.get("text_refs") or {}).get("PMID")
    return {
        "text": ev.get("text") or "",
        "source_api": ev.get("source_api"),
        "pmid": pmid,
        "annotations": ev.get("annotations") or {},
        "epistemics": ev.get("epistemics") or {},
        "source_hash": ev.get("source_hash"),
    }


def _stream_corpus_records(corpus_path: Path) -> Iterator[dict]:
    """Stream each top-level INDRA-statement object out of the gzipped JSON
    array. The corpus is heavily indented (2-space JSON). We track brace
    depth to detect statement boundaries.
    """
    with gzip.open(corpus_path, "rt", encoding="utf-8") as f:
        # Skip leading whitespace/`[`.
        ch = f.read(1)
        while ch and ch in " \n\r\t":
            ch = f.read(1)
        if ch != "[":
            raise ValueError(f"Expected '[' at start of corpus, got {ch!r}")
        buf: list[str] = []
        depth = 0
        in_string = False
        escape = False
        while True:
            ch = f.read(1)
            if not ch:
                break
            if escape:
                buf.append(ch)
                escape = False
                continue
            if ch == "\\" and in_string:
                buf.append(ch)
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                buf.append(ch)
                continue
            if not in_string:
                if ch == "{":
                    if depth == 0:
                        buf = []  # start fresh at statement boundary
                    depth += 1
                    buf.append(ch)
                elif ch == "}":
                    depth -= 1
                    buf.append(ch)
                    if depth == 0:
                        try:
                            obj = json.loads("".join(buf))
                        except json.JSONDecodeError:
                            buf = []
                            continue
                        yield obj
                        buf = []
                elif ch == "]":
                    if depth == 0:
                        return
                    buf.append(ch)
                else:
                    if depth > 0:
                        buf.append(ch)
            else:
                buf.append(ch)


def iter_pairs(corpus_path: Path = CORPUS_PATH,
               exclusion: dict | None = None,
               max_pairs: int = 0) -> Iterator[tuple[dict, dict, dict, list[dict]]]:
    """Walk the corpus, yielding `(raw_record, stmt_dict, ev_dict, agents)`
    tuples after V5r §5 holdout exclusion.

    `agents`: list of agent-dicts for the statement (for verify_grounding
    multiplicity).
    """
    if exclusion is None:
        exclusion = load_holdout_exclusion()
    yielded = 0
    n_seen = 0
    n_dropped_holdout = 0

    def _stash():
        iter_pairs.last_walk_stats = {  # type: ignore[attr-defined]
            "n_statements_seen": n_seen,
            "n_pairs_yielded": yielded,
            "n_pairs_dropped_holdout": n_dropped_holdout,
        }

    try:
        for rec in _stream_corpus_records(corpus_path):
            n_seen += 1
            evidences = rec.get("evidence") or []
            subj, obj = _statement_subj_obj(rec)
            agents = _statement_agents(rec)
            for ev in evidences:
                if not isinstance(ev, dict):
                    continue
                stmt_d = _make_stmt_dict(rec, subj, obj, ev)
                ev_d = _make_ev_dict(ev)
                if is_holdout_excluded(stmt_d, ev_d, exclusion):
                    n_dropped_holdout += 1
                    continue
                yield rec, stmt_d, ev_d, agents
                yielded += 1
                if max_pairs and yielded >= max_pairs:
                    _stash()
                    return
    finally:
        _stash()


# ---------------------------------------------------------------------------
# LF application — assemble per-probe label matrices
# ---------------------------------------------------------------------------

# Probe → [(lf_name, lf_callable, kwargs, signature_kind)]
# signature_kind ∈ {"stmt_ev", "stmt_ev_kw", "entity_ev", "entity_ev_role"}.
# - "stmt_ev": fn(statement, evidence) → int
# - "stmt_ev_kw": fn(statement, evidence, **kwargs) → int (V6a `entity=`,
#   V6b role-keyed LFs that take role= kwarg, or V6a substrate fragment LFs)
# - "entity_ev": fn(entity, evidence) → int (V6b grounding LFs)


def _build_probe_lf_index() -> dict[str, list[tuple]]:
    """Group all 51 LFs by probe, with their callable+signature spec.

    The output is a dict probe_kind → list of (lf_name, fn, kwargs, sig_kind)
    where sig_kind ∈ {"stmt_ev", "stmt_ev_kw", "entity_ev"}.
    """
    from indra_belief.v_phase import substrate_lfs as sl  # noqa: WPS433
    from indra_belief.v_phase import clean_lfs as cl  # noqa: WPS433

    out: dict[str, list[tuple]] = defaultdict(list)
    # V6a substrate LFs all take (statement, evidence, mode=...). We inject
    # mode="tuned" by default.
    for kind, name, fn, kwargs in sl.LF_INDEX:
        out[kind].append((name, fn, dict(kwargs), "stmt_ev_kw"))
    # V6b clean LFs split between (statement, evidence, **role_kw) and
    # (entity, evidence) signatures. Detect by probe kind.
    for kind, name, fn, kwargs in cl.LF_INDEX_CLEAN:
        if kind == "verify_grounding":
            out[kind].append((name, fn, dict(kwargs), "entity_ev"))
        elif kwargs:
            out[kind].append((name, fn, dict(kwargs), "stmt_ev_kw"))
        else:
            out[kind].append((name, fn, dict(kwargs), "stmt_ev"))
    return dict(out)


def _apply_lf_safe(fn, args, kwargs, sig_kind: str) -> int:
    """Apply an LF; convert any exception to ABSTAIN (V6c task brief: any
    LF that throws is a V6c bug, but we surface that via the post-walk
    error-count summary instead of crashing the whole run)."""
    try:
        if sig_kind == "stmt_ev":
            return int(fn(*args))
        if sig_kind == "stmt_ev_kw":
            return int(fn(*args, **kwargs))
        if sig_kind == "entity_ev":
            return int(fn(*args, **kwargs))
        return ABSTAIN
    except Exception as exc:  # pragma: no cover — bubbled to summary
        _apply_lf_safe.errors[fn.__name__] = (
            _apply_lf_safe.errors.get(fn.__name__, 0) + 1  # type: ignore[attr-defined]
        )
        return ABSTAIN


_apply_lf_safe.errors = {}  # type: ignore[attr-defined]


def build_label_matrices(records: list[tuple[dict, dict, list[dict]]]
                          ) -> dict[str, dict]:
    """For each probe, build an integer Λ matrix.

    `records`: list of (stmt_dict, ev_dict, agents_list) tuples.

    Returns: probe_kind → {
      "L": np.ndarray (n_records × n_lfs),
      "lf_names": list[str],
      "record_ids": list[str],
      "n_classes": int,
    }

    For verify_grounding, n_records = sum(len(agents) for each (stmt, ev)).
    """
    from indra_belief.v_phase import substrate_lfs as sl

    probe_lf_index = _build_probe_lf_index()
    probe_n_classes = {
        "subject_role": 5, "object_role": 5,
        "relation_axis": 8, "scope": 5,
        "verify_grounding": 4,
    }

    out: dict[str, dict] = {}

    for probe, lf_specs in probe_lf_index.items():
        lf_names = [s[0] for s in lf_specs]
        if probe == "verify_grounding":
            # Build per-(stmt, ev, entity) record list.
            rows: list[list[int]] = []
            record_ids: list[str] = []
            for stmt_d, ev_d, agents in records:
                for ent_idx, agent in enumerate(agents):
                    if not isinstance(agent, dict):
                        continue
                    if not agent.get("name"):
                        continue
                    # V6b LFs take a normalized entity dict; V6a substrate
                    # `lf_fragment_processed_form_{subject,object}` takes
                    # (statement, evidence) and resolves the role internally.
                    # For substrate fragment LFs, we need the entity-role-anchor
                    # to match this agent; we compute by checking whether
                    # the agent is the named subject vs object (or "subject"
                    # if neither).
                    role = "subject"
                    if (stmt_d.get("subject") and
                            agent.get("name") == stmt_d.get("subject")):
                        role = "subject"
                    elif (stmt_d.get("object") and
                            agent.get("name") == stmt_d.get("object")):
                        role = "object"
                    entity_dict = {
                        "name": agent.get("name"),
                        "canonical": (agent.get("db_refs") or {}).get("HGNC")
                                     or (agent.get("db_refs") or {}).get("FPLX")
                                     or agent.get("name"),
                    }
                    row: list[int] = []
                    for lf_name, fn, kw, sig in lf_specs:
                        if sig == "entity_ev":
                            v = _apply_lf_safe(fn, (entity_dict, ev_d), kw, sig)
                        elif sig == "stmt_ev_kw":
                            # V6a substrate fragment LF: route by the entity
                            # role we computed above. Keep substrate LF kwargs
                            # but override `entity=` if the LF accepts it.
                            kw2 = dict(kw)
                            if "entity" in kw2:
                                kw2["entity"] = role
                            v = _apply_lf_safe(fn, (stmt_d, ev_d), kw2, sig)
                        else:
                            v = _apply_lf_safe(fn, (stmt_d, ev_d), kw, sig)
                        row.append(v)
                    rows.append(row)
                    rec_id = (
                        f"{stmt_d.get('matches_hash')}:"
                        f"{ev_d.get('source_hash')}:"
                        f"{ent_idx}"
                    )
                    record_ids.append(rec_id)
            L = np.asarray(rows, dtype=np.int8) if rows else \
                np.zeros((0, len(lf_names)), dtype=np.int8)
        else:
            # Per-(stmt, ev) records.
            rows = []
            record_ids = []
            for stmt_d, ev_d, _agents in records:
                row = []
                for lf_name, fn, kw, sig in lf_specs:
                    v = _apply_lf_safe(fn, (stmt_d, ev_d), kw, sig)
                    row.append(v)
                rows.append(row)
                rec_id = (
                    f"{stmt_d.get('matches_hash')}:"
                    f"{ev_d.get('source_hash')}"
                )
                record_ids.append(rec_id)
            L = np.asarray(rows, dtype=np.int8) if rows else \
                np.zeros((0, len(lf_names)), dtype=np.int8)
        out[probe] = {
            "L": L,
            "lf_names": lf_names,
            "record_ids": record_ids,
            "n_classes": probe_n_classes[probe],
        }
    return out


# ---------------------------------------------------------------------------
# Correlation diagnosis (V5r §4 step 2)
# ---------------------------------------------------------------------------

def diagnose_lf_correlations(L: np.ndarray, lf_names: list[str],
                              min_overlap: int = 50,
                              threshold: float = 0.5) -> list[tuple]:
    """Compute pairwise Pearson correlation on records where both LFs fire.
    Returns a list of (lf_i, lf_j, n_overlap, correlation) for any pair
    exceeding |r| > threshold (after the min_overlap gate).
    """
    flagged: list[tuple] = []
    n_lfs = L.shape[1]
    if n_lfs < 2:
        return flagged
    for i, j in itertools.combinations(range(n_lfs), 2):
        col_i = L[:, i]
        col_j = L[:, j]
        both = (col_i != ABSTAIN) & (col_j != ABSTAIN)
        n = int(both.sum())
        if n < min_overlap:
            continue
        a = col_i[both].astype(np.float64)
        b = col_j[both].astype(np.float64)
        # If either is constant, corrcoef returns NaN; treat as 0 (no info).
        if a.std() == 0 or b.std() == 0:
            continue
        r = float(np.corrcoef(a, b)[0, 1])
        if not np.isfinite(r):
            continue
        if abs(r) > threshold:
            flagged.append((lf_names[i], lf_names[j], n, r))
    return flagged


# ---------------------------------------------------------------------------
# Snorkel LabelModel fit
# ---------------------------------------------------------------------------

def fit_label_model(L: np.ndarray, n_classes: int, *,
                     n_epochs: int = 500, lr: float = 0.01,
                     verbose: bool = False, seed: int = 0):
    """Fit a Snorkel LabelModel on Λ. Per V5r §4: explicit equal
    `class_balance`, NO `Y_dev`. Returns the fit model.
    """
    from snorkel.labeling.model import LabelModel
    lm = LabelModel(cardinality=n_classes, verbose=verbose)
    lm.fit(L_train=L, class_balance=[1.0 / n_classes] * n_classes,
           n_epochs=n_epochs, lr=lr, seed=seed)
    return lm


def predict_proba_safe(lm, L: np.ndarray) -> np.ndarray:
    """Run `lm.predict_proba(L)` and verify no NaN/inf."""
    p = lm.predict_proba(L)
    if not np.isfinite(p).all():
        bad = np.argwhere(~np.isfinite(p))
        raise RuntimeError(f"predict_proba returned NaN/inf at {len(bad)} entries")
    return p


# ---------------------------------------------------------------------------
# Parquet emit
# ---------------------------------------------------------------------------

def write_parquet(probe: str, suffix: str, record_ids: list[str],
                   class_proba: np.ndarray, class_names: list[str],
                   out_dir: Path = OUTPUT_DIR) -> Path:
    """Emit `data/v_phase/labels/{probe}{suffix}.parquet` with columns:
      record_id, class_proba_<class>... (one per class), argmax_class,
      max_proba, kept_for_training (bool, max_proba ≥ 0.5).
    """
    import pandas as pd
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{probe}{suffix}.parquet"
    if class_proba.shape[0] == 0:
        df = pd.DataFrame({
            "record_id": pd.Series([], dtype="object"),
            **{f"class_proba_{c}": pd.Series([], dtype="float32")
               for c in class_names},
            "argmax_class": pd.Series([], dtype="object"),
            "max_proba": pd.Series([], dtype="float32"),
            "kept_for_training": pd.Series([], dtype="bool"),
        })
        df.to_parquet(out_path, index=False)
        return out_path
    argmax_idx = np.argmax(class_proba, axis=1)
    max_p = class_proba[np.arange(len(argmax_idx)), argmax_idx]
    df_data: dict[str, Any] = {"record_id": record_ids}
    for k, name in enumerate(class_names):
        df_data[f"class_proba_{name}"] = class_proba[:, k].astype(np.float32)
    df_data["argmax_class"] = [class_names[i] for i in argmax_idx]
    df_data["max_proba"] = max_p.astype(np.float32)
    df_data["kept_for_training"] = (max_p >= 0.5)
    df = pd.DataFrame(df_data)
    df.to_parquet(out_path, index=False)
    return out_path


# ---------------------------------------------------------------------------
# Probe-class-name maps for parquet column naming
# ---------------------------------------------------------------------------

def _probe_class_names(probe: str) -> list[str]:
    from indra_belief.v_phase.substrate_lfs import (
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
# Per-LF / per-probe summary stats
# ---------------------------------------------------------------------------

def per_lf_firing_rates(L: np.ndarray, lf_names: list[str]
                         ) -> list[tuple[str, int, float]]:
    out = []
    n = max(1, L.shape[0])
    for j, name in enumerate(lf_names):
        fires = int((L[:, j] != ABSTAIN).sum())
        out.append((name, fires, fires / n))
    return out


def per_class_vote_distribution(L: np.ndarray, n_classes: int
                                 ) -> list[int]:
    """Total non-ABSTAIN votes for each class index 0..K-1."""
    counts = [0] * n_classes
    for c in range(n_classes):
        counts[c] = int((L == c).sum())
    return counts


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def run(sample: int, *, log_path: Path | None = None,
         seed: int = 0, verbose_lm: bool = True) -> dict:
    """End-to-end pipeline: walk corpus, build matrices, diagnose, fit,
    emit parquets. Returns a dict of summary statistics.

    `sample`: 0 = full corpus, >0 = first N (statement, evidence) pairs
    after holdout exclusion.
    """
    print(f"[V6c] starting; sample={sample or 'full'}; "
          f"corpus={CORPUS_PATH.name}", flush=True)
    t_start = time.time()

    exc = load_holdout_exclusion()
    print(f"[V6c] holdout exclusion: "
          f"{len(exc['source_hashes'])} source_hashes, "
          f"{len(exc['matches_hashes'])} matches_hashes, "
          f"{len(exc['pmid_pairs'])} pmid+pair tuples")

    # Walk corpus.
    pairs: list[tuple[dict, dict, list[dict]]] = []
    walk_t0 = time.time()
    for raw, stmt_d, ev_d, agents in iter_pairs(
            corpus_path=CORPUS_PATH, exclusion=exc, max_pairs=sample):
        pairs.append((stmt_d, ev_d, agents))
    walk_dt = time.time() - walk_t0
    walk_stats = getattr(iter_pairs, "last_walk_stats",
                          {"n_statements_seen": -1,
                           "n_pairs_yielded": len(pairs),
                           "n_pairs_dropped_holdout": -1})
    print(f"[V6c] corpus walk: yielded {len(pairs)} (stmt, ev) pairs "
          f"from {walk_stats.get('n_statements_seen', '?')} statements; "
          f"dropped {walk_stats.get('n_pairs_dropped_holdout', '?')} "
          f"by holdout; {walk_dt:.1f}s")

    # Build label matrices.
    print(f"[V6c] applying 51 LFs across all probes...", flush=True)
    lf_t0 = time.time()
    matrices = build_label_matrices(pairs)
    lf_dt = time.time() - lf_t0
    if _apply_lf_safe.errors:  # type: ignore[attr-defined]
        print(f"[V6c] LF errors (treated as ABSTAIN): "
              f"{dict(_apply_lf_safe.errors)}")  # type: ignore[attr-defined]
    print(f"[V6c] LF application: {lf_dt:.1f}s")

    # Per-probe Snorkel fit + emit.
    suffix = "_sample" if sample else ""
    summary: dict = {
        "n_pairs": len(pairs),
        "walk_stats": walk_stats,
        "elapsed": {"walk": walk_dt, "lfs": lf_dt},
        "probes": {},
    }
    for probe, info in matrices.items():
        L = info["L"]
        lf_names = info["lf_names"]
        record_ids = info["record_ids"]
        K = info["n_classes"]
        n = L.shape[0]
        print(f"\n[V6c] === probe: {probe} ===")
        print(f"[V6c] {probe}: Λ shape = {L.shape} (records × LFs); K={K}")

        # Per-LF firing rates.
        firing = per_lf_firing_rates(L, lf_names)
        firing_sorted = sorted(firing, key=lambda r: -r[1])
        print(f"[V6c] {probe}: per-LF fires (top 10):")
        for name, fires, rate in firing_sorted[:10]:
            print(f"    {name:<48} {fires:>8} ({rate:.1%})")
        print(f"[V6c] {probe}: per-LF fires (bottom 10):")
        for name, fires, rate in firing_sorted[-10:]:
            print(f"    {name:<48} {fires:>8} ({rate:.1%})")

        # Per-class vote distribution.
        votes_per_class = per_class_vote_distribution(L, K)
        print(f"[V6c] {probe}: total per-class non-ABSTAIN votes:")
        for c, count in enumerate(votes_per_class):
            print(f"    class {c}: {count}")

        # Correlation diagnosis.
        flagged = diagnose_lf_correlations(L, lf_names, min_overlap=50,
                                            threshold=0.5)
        print(f"[V6c] {probe}: correlation flags (|r|>0.5, overlap≥50): "
              f"{len(flagged)}")
        for a, b, n_ov, r in flagged:
            print(f"    {a:<40} <> {b:<40} n={n_ov:>5} r={r:+.3f}")

        # Snorkel fit.
        if n == 0 or L.shape[1] == 0:
            print(f"[V6c] {probe}: no records OR no LFs; skipping LM fit.")
            continue

        # Snorkel needs at least one non-ABSTAIN vote per class for
        # identifiability of class_balance; guard against degenerate slices.
        observed_classes = set()
        for c in range(K):
            if (L == c).any():
                observed_classes.add(c)
        print(f"[V6c] {probe}: classes with ≥1 non-ABSTAIN vote: "
              f"{sorted(observed_classes)} of {K}")

        try:
            lm = fit_label_model(L, K, verbose=verbose_lm, seed=seed)
        except Exception as exc:
            print(f"[V6c] {probe}: LabelModel.fit FAILED with: {exc}")
            continue

        try:
            P = predict_proba_safe(lm, L)
        except Exception as exc:
            print(f"[V6c] {probe}: predict_proba FAILED with: {exc}")
            continue

        # LM weights summary.
        try:
            weights = lm.get_weights()
            w_arr = np.asarray(weights, dtype=np.float64)
            print(f"[V6c] {probe}: LabelModel weights: "
                  f"min={w_arr.min():.3f} mean={w_arr.mean():.3f} "
                  f"max={w_arr.max():.3f}")
            for name, w in zip(lf_names, w_arr):
                print(f"    weight {name:<48} {w:+.3f}")
        except Exception as exc:
            print(f"[V6c] {probe}: get_weights FAILED with: {exc}")

        # Emit parquet.
        class_names = _probe_class_names(probe)
        # Padding: snorkel may return fewer columns if cardinality<K - but
        # we passed cardinality=K, so P.shape == (n, K).
        if P.shape[1] != K:
            print(f"[V6c] {probe}: WARN predict_proba shape {P.shape} "
                  f"!= (n, {K})")
        out_path = write_parquet(probe, suffix, record_ids, P, class_names)
        argmax_idx = np.argmax(P, axis=1)
        max_p = P[np.arange(len(argmax_idx)), argmax_idx]
        kept = int((max_p >= 0.5).sum())
        print(f"[V6c] {probe}: wrote {out_path}; "
              f"kept_for_training: {kept}/{n} ({kept/n:.1%})")

        summary["probes"][probe] = {
            "n_records": n,
            "n_lfs": len(lf_names),
            "lf_names": lf_names,
            "firing_rates": firing,
            "per_class_votes": votes_per_class,
            "correlation_flags": flagged,
            "lm_weights": w_arr.tolist() if 'w_arr' in locals() else None,
            "kept_for_training": kept,
            "kept_fraction": kept / n,
            "parquet_path": str(out_path),
        }

    summary["elapsed"]["total"] = time.time() - t_start
    print(f"\n[V6c] total: {summary['elapsed']['total']:.1f}s")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="V6c — build per-probe Λ + fit Snorkel LabelModel."
    )
    p.add_argument("--sample", type=int, default=10000,
                   help="N (statement, evidence) pairs to process; "
                        "0 = full corpus")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-verbose-lm", action="store_true",
                   help="Suppress Snorkel LabelModel.fit verbose output.")
    args = p.parse_args(argv)
    run(sample=args.sample, seed=args.seed, verbose_lm=not args.no_verbose_lm)


if __name__ == "__main__":
    main()

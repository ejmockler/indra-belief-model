"""Lossless ingest from INDRA JSON dumps into the corpus DuckDB schema.

Reads INDRA Statement objects via `stmts_from_json_file`, derives the
canonical hashes (Phase 1.5 lock: `Statement.get_hash(shallow=True)` for
`stmt_hash`; `Evidence.get_source_hash()` for `evidence_hash`), and writes
them into the schema's `statement` / `evidence` / `agent` / `supports_edge`
tables. Auto-registers two truth_sets per ingest (Phase 1.3): the per-
statement `indra_published_belief` and the per-evidence `indra_epistemics`
flags (`is_direct` / `is_negated` / `is_curated`).

The loader is intentionally minimal — append-only conflict resolution,
chunking, parallelism all belong to Phase 2.5 / 3.x. This module is the
"every INDRA field reaches DuckDB" guarantee from D9.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Iterable

if TYPE_CHECKING:
    import duckdb
    from indra.statements import Statement, Evidence, Agent

log = logging.getLogger(__name__)


def from_indra_json(path: str | Path) -> Iterator["Statement"]:
    """Stream INDRA Statement objects from a JSON dump."""
    from indra.statements import stmts_from_json_file
    yield from stmts_from_json_file(str(path))


def _hex(n: int, width: int = 16) -> str:
    """INDRA hashes can be negative ints (signed 64-bit); make them stable hex."""
    return f"{n & ((1 << 64) - 1):0{width}x}"


def _agent_hash(agent: "Agent") -> str:
    """Stable 16-hex hash of `Agent.matches_key()` — the canonical identity."""
    h = hashlib.sha256(agent.matches_key().encode("utf-8")).hexdigest()
    return h[:16]


def _upsert_truth_label(
    con: "duckdb.DuckDBPyConnection",
    *,
    truth_set_id: str,
    target_kind: str,
    target_id: str,
    field: str,
    value_text: str | None = None,
    value_json: str | None = None,
    provenance: str | None = None,
) -> None:
    """Idempotent INSERT on the natural key (truth_set_id, target_kind,
    target_id, field). Used by both auto-registered INDRA truth_sets and
    `load_truth_labels`.
    """
    con.execute(
        """DELETE FROM truth_label
           WHERE truth_set_id = ? AND target_kind = ?
             AND target_id = ? AND field = ?""",
        [truth_set_id, target_kind, target_id, field],
    )
    con.execute(
        """INSERT INTO truth_label
           (truth_set_id, target_kind, target_id, field,
            value_text, value_json, provenance)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        [truth_set_id, target_kind, target_id, field,
         value_text, value_json, provenance],
    )


def _agent_role_iter(stmt: "Statement") -> Iterator[tuple[str, int, "Agent"]]:
    """Yield (role, role_index, agent) for all agents on a Statement.

    Roles vary per Statement subclass — Modifications use enz/sub, RegulateActivity
    uses subj/obj, Complex uses members[]. We cover the common ones; unknown
    subclasses fall back to positional `agent_list()` indexing.
    """
    cls = type(stmt).__name__

    def _emit(role: str, idx: int, agent):
        if agent is not None:
            yield role, idx, agent

    if hasattr(stmt, "members") and getattr(stmt, "members", None):
        for i, a in enumerate(stmt.members):
            yield from _emit("member", i, a)
        return

    if hasattr(stmt, "enz") or hasattr(stmt, "sub"):
        yield from _emit("enz", 0, getattr(stmt, "enz", None))
        yield from _emit("sub", 0, getattr(stmt, "sub", None))
        return

    if hasattr(stmt, "subj") or hasattr(stmt, "obj"):
        yield from _emit("subj", 0, getattr(stmt, "subj", None))
        yield from _emit("obj", 0, getattr(stmt, "obj", None))
        return

    # Fallback: positional
    for i, a in enumerate(stmt.agent_list() or []):
        yield from _emit("agent", i, a)


def ingest_statements(
    con: "duckdb.DuckDBPyConnection",
    stmts: Iterable["Statement"],
    *,
    source_dump_id: str | None = None,
    register_indra_truth: bool = True,
) -> dict[str, int]:
    """Write a stream of INDRA Statements to the corpus DuckDB.

    Returns a counters dict: {n_statements, n_evidences, n_agents, n_edges, n_truth_labels}.

    All inserts use INSERT OR REPLACE on the natural key, making the loader
    idempotent for re-ingests of the same dump. Append-only semantics across
    `scorer_version` (Phase 2.5) is layered separately.

    `register_indra_truth=True` (default) auto-emits `truth_label` rows under
    truth_sets `indra_published_belief` and `indra_epistemics`.
    """
    counters = {"n_statements": 0, "n_evidences": 0, "n_agents": 0,
                "n_edges": 0, "n_truth_labels": 0}

    if register_indra_truth:
        _register_indra_truth_sets(con, source_dump_id)

    for stmt in stmts:
        stmt_hash = _hex(stmt.get_hash(shallow=True))
        extraction_hash = _hex(stmt.get_hash(shallow=False))
        indra_uuid = getattr(stmt, "uuid", None)
        indra_type = type(stmt).__name__
        belief = float(getattr(stmt, "belief", 1.0) or 1.0)
        supports = list(getattr(stmt, "supports", []) or [])
        supported_by = list(getattr(stmt, "supported_by", []) or [])

        # INDRA quirk: when supports/supported_by are UUID strings (the JSON-loaded
        # form), `Statement.to_json()` tries to assign a `.uuid` attribute to them
        # and crashes. Clear → serialize → restore, then patch UUIDs back into the
        # raw JSON manually so raw_json is truly lossless.
        _saved = (stmt.supports, stmt.supported_by)
        try:
            stmt.supports = []
            stmt.supported_by = []
            raw = stmt.to_json()
        finally:
            stmt.supports, stmt.supported_by = _saved
        if supports:
            raw["supports"] = [str(u) for u in supports]
        if supported_by:
            raw["supported_by"] = [str(u) for u in supported_by]
        raw_json = json.dumps(raw, default=str)

        con.execute(
            """INSERT OR REPLACE INTO statement
               (stmt_hash, extraction_hash, indra_uuid, indra_type, indra_belief,
                supports_count, supported_by_count, raw_json, source_dump_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [stmt_hash, extraction_hash, indra_uuid, indra_type, belief,
             len(supports), len(supported_by), raw_json, source_dump_id],
        )
        counters["n_statements"] += 1

        if register_indra_truth:
            _upsert_truth_label(
                con,
                truth_set_id="indra_published_belief",
                target_kind="stmt",
                target_id=stmt_hash,
                field="belief",
                value_text=str(belief),
                provenance=source_dump_id,
            )
            counters["n_truth_labels"] += 1

        # Agents
        for role, role_index, agent in _agent_role_iter(stmt):
            ahash = _agent_hash(agent)
            db_refs = json.dumps(getattr(agent, "db_refs", {}) or {}, default=str)
            mods = json.dumps(
                [m.to_json() for m in (getattr(agent, "mods", None) or [])],
                default=str,
            ) if getattr(agent, "mods", None) else None
            mutations = json.dumps(
                [m.to_json() for m in (getattr(agent, "mutations", None) or [])],
                default=str,
            ) if getattr(agent, "mutations", None) else None
            bound = json.dumps(
                [b.to_json() for b in (getattr(agent, "bound_conditions", None) or [])],
                default=str,
            ) if getattr(agent, "bound_conditions", None) else None
            activity = json.dumps(
                agent.activity.to_json(), default=str,
            ) if getattr(agent, "activity", None) else None
            location = getattr(agent, "location", None)

            con.execute(
                """INSERT OR REPLACE INTO agent
                   (stmt_hash, agent_hash, role, role_index, name, db_refs_json,
                    mods_json, mutations_json, bound_conditions_json, activity_json, location)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [stmt_hash, ahash, role, role_index, agent.name, db_refs,
                 mods, mutations, bound, activity, location],
            )
            counters["n_agents"] += 1

            # Per-agent grounding truth (indra_grounding truth_set)
            if register_indra_truth and agent.db_refs:
                _upsert_truth_label(
                    con,
                    truth_set_id="indra_grounding",
                    target_kind="agent",
                    target_id=ahash,
                    field="db_refs",
                    value_json=db_refs,
                    provenance=source_dump_id,
                )
                counters["n_truth_labels"] += 1

        # Evidences
        for ev in (getattr(stmt, "evidence", None) or []):
            try:
                evhash = _hex(ev.get_source_hash())
            except Exception:
                # Some Evidence objects may have None source_hash; fall back
                # to a content-derived stable hash.
                evhash = hashlib.sha256(
                    f"{ev.source_api}|{ev.source_id}|{ev.pmid}|{ev.text}".encode("utf-8")
                ).hexdigest()[:16]
            epistemics = dict(getattr(ev, "epistemics", {}) or {})
            annotations = dict(getattr(ev, "annotations", {}) or {})
            is_direct = epistemics.get("direct")
            is_negated = epistemics.get("negated")
            is_curated = (
                bool(epistemics.get("curated"))
                if "curated" in epistemics
                else None
            )
            ev_raw = json.dumps(ev.to_json(), default=str)
            con.execute(
                """INSERT OR REPLACE INTO evidence
                   (evidence_hash, stmt_hash, source_api, source_id, pmid, text,
                    is_direct, is_negated, is_curated,
                    epistemics_json, annotations_json, raw_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [evhash, stmt_hash, ev.source_api, ev.source_id, ev.pmid, ev.text,
                 is_direct, is_negated, is_curated,
                 json.dumps(epistemics, default=str),
                 json.dumps(annotations, default=str),
                 ev_raw],
            )
            counters["n_evidences"] += 1

            if register_indra_truth and epistemics:
                for fld_key, fld_val in (
                    ("is_direct", is_direct),
                    ("is_negated", is_negated),
                    ("is_curated", is_curated),
                ):
                    if fld_val is None:
                        continue
                    _upsert_truth_label(
                        con,
                        truth_set_id="indra_epistemics",
                        target_kind="evidence",
                        target_id=evhash,
                        field=fld_key,
                        value_text=str(bool(fld_val)).lower(),
                        provenance=ev.source_api,
                    )
                    counters["n_truth_labels"] += 1

        # Supports/supported_by edges (UUIDs in the dump; resolved post-ingest
        # in a separate pass since the referenced statement may not have been
        # ingested yet)
        for to_uuid in supports:
            con.execute(
                """INSERT OR REPLACE INTO supports_edge
                   (from_stmt_hash, to_stmt_hash, kind, source_dump_id)
                   VALUES (?, ?, 'supports', ?)""",
                [stmt_hash, to_uuid, source_dump_id],
            )
            counters["n_edges"] += 1
        for from_uuid in supported_by:
            con.execute(
                """INSERT OR REPLACE INTO supports_edge
                   (from_stmt_hash, to_stmt_hash, kind, source_dump_id)
                   VALUES (?, ?, 'supported_by', ?)""",
                [stmt_hash, from_uuid, source_dump_id],
            )
            counters["n_edges"] += 1

    # Resolve supports_edge UUIDs → stmt_hashes. INDRA's Statement.to_json()
    # writes UUIDs (Statement.uuid) into supports/supported_by, NOT
    # stmt_hashes — the column-name `to_stmt_hash` was misleading until this
    # resolution pass runs. Edges that reference a stmt outside this dump
    # stay as UUIDs (an honest "out-of-corpus" marker).
    n_resolved = con.execute(
        """UPDATE supports_edge se
           SET to_stmt_hash = (
               SELECT s.stmt_hash FROM statement s
               WHERE s.indra_uuid = se.to_stmt_hash
           )
           WHERE EXISTS (
               SELECT 1 FROM statement s WHERE s.indra_uuid = se.to_stmt_hash
           )"""
    ).fetchall()
    if counters["n_edges"]:
        log.info("supports_edge: resolved UUIDs → stmt_hashes for in-corpus refs")

    log.info("ingest counters: %s", counters)
    return counters


def _register_indra_truth_sets(
    con: "duckdb.DuckDBPyConnection",
    source_dump_id: str | None,
) -> None:
    """Register the three INDRA-derived truth_sets (idempotent)."""
    for tset_id, name, source, desc in [
        (
            "indra_published_belief",
            "INDRA published Statement.belief",
            "indra_aggregator",
            "Per-statement belief score from upstream INDRA aggregator",
        ),
        (
            "indra_epistemics",
            "INDRA Evidence.epistemics flags",
            "indra_evidence_epistemics",
            "Per-evidence is_direct / is_negated / is_curated from source_api curators",
        ),
        (
            "indra_grounding",
            "INDRA Agent.db_refs",
            "indra_curators",
            "Per-agent database namespace IDs (HGNC, UP, MESH, FamPlex, …)",
        ),
    ]:
        con.execute(
            """INSERT OR REPLACE INTO truth_set
               (id, name, source, version, description)
               VALUES (?, ?, ?, ?, ?)""",
            [tset_id, name, source, source_dump_id or "default", desc],
        )


def register_truth_set(
    con: "duckdb.DuckDBPyConnection",
    *,
    id: str,
    name: str,
    source: str | None = None,
    version: str | None = None,
    description: str | None = None,
) -> None:
    """Register a truth_set row (idempotent on `id`)."""
    con.execute(
        """INSERT OR REPLACE INTO truth_set
           (id, name, source, version, description) VALUES (?, ?, ?, ?, ?)""",
        [id, name, source, version, description],
    )


def load_truth_labels(
    con: "duckdb.DuckDBPyConnection",
    truth_set_id: str,
    labels: Iterable[dict],
) -> int:
    """Bulk-write truth_label rows for a registered truth_set.

    Each label dict needs at least: target_kind, target_id, field, and one of
    value_text or value_json. Optional: relation_target_id, confidence, provenance.

    Idempotent on the natural key (truth_set_id, target_kind, target_id, field):
    re-loading the same labels does not duplicate rows. This matches the
    schema-doc contract ("INSERT OR REPLACE on the upsert path") that DuckDB's
    native ON CONFLICT can't express on a non-PK index, so we DELETE+INSERT
    per label.

    Raises `ValueError` if `truth_set_id` is not a registered truth_set.
    The schema documents this column as FK truth_set.id but DuckDB doesn't
    enforce it; the app-level check prevents orphaned labels (a typo'd
    truth_set_id silently inserts rows the dashboard can't display).
    """
    if not con.execute(
        "SELECT 1 FROM truth_set WHERE id = ? LIMIT 1", [truth_set_id]
    ).fetchone():
        raise ValueError(
            f"truth_set {truth_set_id!r} not registered — "
            f"call register_truth_set(con, id={truth_set_id!r}, ...) first"
        )
    n = 0
    for lbl in labels:
        # Per-label upsert on the natural key. Uses a fuller column set than
        # the shared helper because user-provided labels may carry
        # relation_target_id and confidence (auto-registered INDRA labels
        # don't).
        con.execute(
            """DELETE FROM truth_label
               WHERE truth_set_id = ? AND target_kind = ?
                 AND target_id = ? AND field = ?""",
            [truth_set_id, lbl["target_kind"], lbl["target_id"], lbl["field"]],
        )
        con.execute(
            """INSERT INTO truth_label
               (truth_set_id, target_kind, target_id, relation_target_id,
                field, value_text, value_json, confidence, provenance)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [truth_set_id,
             lbl["target_kind"], lbl["target_id"], lbl.get("relation_target_id"),
             lbl["field"],
             lbl.get("value_text"),
             json.dumps(lbl["value_json"], default=str) if lbl.get("value_json") is not None else None,
             lbl.get("confidence"),
             lbl.get("provenance")],
        )
        n += 1
    return n

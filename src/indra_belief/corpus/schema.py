"""DuckDB schema for the INDRA belief-rescoring corpus.

Specified by Phase 2.1 of `research/rasmachine_task_graph.md`. The schema
must survive scorer-architecture iterations: every scorer step is its own
row keyed by `(stmt_hash, scorer_version, step_kind)`, append-only with
versioned migrations layered alongside (never overwriting).

Truth-set support is foundational (D7). `truth_set` registers labelled
sources (INDRA published belief, INDRA epistemics flags, gold pool, etc.)
and `truth_label` is a polymorphic attachment that can target any of:
statement, evidence, agent, scorer_step, supports_edge.

Hashes (per Phase 1.5 lock):
  - `stmt_hash` = `Statement.get_hash(shallow=True)` 14-nibble hex, stored as VARCHAR
  - `extraction_hash` = `Statement.get_hash(shallow=False)` 16-nibble, identifies
    a specific dump's statement-with-evidence
  - `evidence_hash` = `Evidence.source_hash` (INDRA content-addressed; NOT
    `matches_key()` — that has dict-ordering instability)
  - `agent_hash`, `step_hash` = stable hashes computed at write time
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import duckdb

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1


# Step kinds emitted by the composed scorer (per Phase 1.2 catalog of
# the 9-step pipeline).
SCORER_STEP_KINDS = (
    "parse_claim",
    "build_context",
    "substrate_route",
    "subject_role_probe",
    "object_role_probe",
    "relation_axis_probe",
    "scope_probe",
    "grounding",
    "adjudicate",
)

# Target kinds for polymorphic truth labels (per Phase 1.3 catalog).
TRUTH_TARGET_KINDS = (
    "stmt",
    "evidence",
    "agent",
    "scorer_step",
    "supports_edge",
)


_DDL = f"""
-- ─── INDRA-native object tables ────────────────────────────────────────
CREATE TABLE IF NOT EXISTS statement (
    stmt_hash         VARCHAR PRIMARY KEY,        -- Statement.get_hash(shallow=True)
    extraction_hash   VARCHAR,                    -- Statement.get_hash(shallow=False)
    indra_uuid        VARCHAR,                    -- per-extraction UUID
    indra_type        VARCHAR NOT NULL,           -- discriminator: Phosphorylation, Activation, …
    indra_belief      DOUBLE,                     -- INDRA's published belief ∈ [0,1]
    supports_count    INTEGER NOT NULL DEFAULT 0,
    supported_by_count INTEGER NOT NULL DEFAULT 0,
    raw_json          JSON NOT NULL,              -- canonical full Statement.to_json()
    loaded_at         TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    source_dump_id    VARCHAR                     -- which corpus ingest brought this in
);
CREATE INDEX IF NOT EXISTS idx_statement_indra_type ON statement(indra_type);
CREATE INDEX IF NOT EXISTS idx_statement_source_dump ON statement(source_dump_id);

CREATE TABLE IF NOT EXISTS evidence (
    evidence_hash     VARCHAR PRIMARY KEY,        -- Evidence.source_hash
    stmt_hash         VARCHAR NOT NULL,           -- FK statement.stmt_hash
    source_api        VARCHAR,                    -- reach, biopax, biogrid, signor, …
    source_id         VARCHAR,
    pmid              VARCHAR,
    text              TEXT,
    -- Top-level epistemics columns (G1: query-friendly mirror of epistemics_json):
    is_direct         BOOLEAN,
    is_negated        BOOLEAN,
    is_curated        BOOLEAN,
    epistemics_json   JSON,
    annotations_json  JSON,
    raw_json          JSON NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_evidence_stmt ON evidence(stmt_hash);
CREATE INDEX IF NOT EXISTS idx_evidence_source_api ON evidence(source_api);
CREATE INDEX IF NOT EXISTS idx_evidence_pmid ON evidence(pmid);

-- Agent participation in a statement. Same agent (by agent_hash) can appear
-- in many statements; PK is the participation, not the identity.
CREATE TABLE IF NOT EXISTS agent (
    stmt_hash         VARCHAR NOT NULL,           -- FK statement.stmt_hash
    agent_hash        VARCHAR NOT NULL,           -- canonical agent identity (matches_key hash)
    role              VARCHAR NOT NULL,           -- subj, obj, enz, sub, member, …
    role_index        INTEGER NOT NULL DEFAULT 0, -- ordinal within role (e.g. members[0..N])
    name              VARCHAR NOT NULL,
    db_refs_json      JSON NOT NULL,
    mods_json         JSON,
    mutations_json    JSON,
    bound_conditions_json JSON,
    activity_json     JSON,
    location          VARCHAR,
    PRIMARY KEY (stmt_hash, agent_hash, role, role_index)
);
CREATE INDEX IF NOT EXISTS idx_agent_stmt ON agent(stmt_hash);
CREATE INDEX IF NOT EXISTS idx_agent_hash ON agent(agent_hash);
CREATE INDEX IF NOT EXISTS idx_agent_name ON agent(name);

-- INDRA's supports/supported_by are UUIDs in the JSON dump, NOT live refs.
-- Reconstruction is a separate post-ingest pass; this table holds the edges.
CREATE TABLE IF NOT EXISTS supports_edge (
    from_stmt_hash    VARCHAR NOT NULL,
    to_stmt_hash      VARCHAR NOT NULL,
    kind              VARCHAR NOT NULL,           -- 'supports' or 'supported_by'
    source_dump_id    VARCHAR,
    PRIMARY KEY (from_stmt_hash, to_stmt_hash, kind)
);
CREATE INDEX IF NOT EXISTS idx_supports_from ON supports_edge(from_stmt_hash);
CREATE INDEX IF NOT EXISTS idx_supports_to ON supports_edge(to_stmt_hash);

-- ─── Truth-set registry + polymorphic labels ───────────────────────────
CREATE TABLE IF NOT EXISTS truth_set (
    id            VARCHAR PRIMARY KEY,            -- e.g., 'indra_published_belief', 'gold_pool_v15'
    name          VARCHAR NOT NULL,
    source        VARCHAR,                        -- 'indra_evidence_epistemics', 'project_annotators', …
    version       VARCHAR,
    loaded_at     TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    description   TEXT
);

CREATE SEQUENCE IF NOT EXISTS truth_label_id_seq;
CREATE TABLE IF NOT EXISTS truth_label (
    label_id            BIGINT PRIMARY KEY DEFAULT nextval('truth_label_id_seq'),
    truth_set_id        VARCHAR NOT NULL,         -- FK truth_set.id
    target_kind         VARCHAR NOT NULL,         -- stmt | evidence | agent | scorer_step | supports_edge
    target_id           VARCHAR NOT NULL,         -- the relevant *_hash
    relation_target_id  VARCHAR,                  -- second hash for edge labels (supports_edge.to_stmt_hash)
    field               VARCHAR NOT NULL,         -- e.g., 'belief', 'is_direct', 'verdict', 'subj_grounding'
    value_text          VARCHAR,                  -- scalar value as text
    value_json          JSON,                     -- non-scalar value (e.g., grounding correction)
    confidence          DOUBLE,                   -- annotator confidence if known
    provenance          VARCHAR                   -- 'biogrid_v4.4_curated', 'annotator_eric', …
);
-- App-level uniqueness on the natural key is enforced in `loader.py` via
-- `_upsert_truth_label`, which DELETEs on (truth_set_id, target_kind,
-- target_id, field) then INSERTs. DuckDB's INSERT OR REPLACE / ON
-- CONFLICT requires the conflict target to be a UNIQUE/PK constraint —
-- the natural key is an index, not a constraint, because this table
-- uses a surrogate label_id PK. The DELETE-then-INSERT pattern is the
-- equivalent. Locked by `test_load_truth_labels_is_idempotent` and
-- `test_re_ingest_idempotent` in tests/test_corpus_loader.py.
CREATE INDEX IF NOT EXISTS idx_truth_label_natural ON truth_label(truth_set_id, target_kind, target_id, field);
CREATE INDEX IF NOT EXISTS idx_truth_label_target ON truth_label(target_kind, target_id);
CREATE INDEX IF NOT EXISTS idx_truth_label_set ON truth_label(truth_set_id);
CREATE INDEX IF NOT EXISTS idx_truth_label_field ON truth_label(field);

-- ─── Scorer traces ─────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS score_run (
    run_id             VARCHAR PRIMARY KEY,
    scorer_version     VARCHAR NOT NULL,          -- git commit hash of indra-belief
    indra_version      VARCHAR NOT NULL,          -- indra.__version__ at run time
    model_id_default   VARCHAR,                   -- LLM identifier (per-step may override)
    started_at         TIMESTAMP NOT NULL,
    finished_at        TIMESTAMP,
    n_stmts            INTEGER,
    status             VARCHAR NOT NULL,          -- running | succeeded | failed | aborted
    cost_estimate_usd  DOUBLE,
    cost_actual_usd    DOUBLE,
    notes              TEXT
);

CREATE TABLE IF NOT EXISTS scorer_step (
    step_hash             VARCHAR PRIMARY KEY,    -- composite hash over (stmt_hash, evidence_hash, scorer_version, model_id, step_kind, input_payload)
    stmt_hash             VARCHAR NOT NULL,
    evidence_hash         VARCHAR,                -- nullable: parse_claim is statement-level, no evidence
    run_id                VARCHAR,                -- FK score_run.run_id
    scorer_version        VARCHAR NOT NULL,
    model_id              VARCHAR,                -- nullable for substrate-only steps
    step_kind             VARCHAR NOT NULL,       -- enumerated in SCORER_STEP_KINDS
    is_substrate_answered BOOLEAN,                -- true for substrate-answered probe rows; null for non-probe steps
    input_payload_json    JSON,
    output_json           JSON,                   -- shape varies per step_kind; see scorer-output catalog
    latency_ms            INTEGER,
    prompt_tokens         INTEGER,
    out_tokens            INTEGER,
    finish_reason         VARCHAR,
    error                 VARCHAR,
    started_at            TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_scorer_step_stmt ON scorer_step(stmt_hash);
CREATE INDEX IF NOT EXISTS idx_scorer_step_evidence ON scorer_step(evidence_hash);
CREATE INDEX IF NOT EXISTS idx_scorer_step_run ON scorer_step(run_id);
CREATE INDEX IF NOT EXISTS idx_scorer_step_kind ON scorer_step(step_kind);
CREATE INDEX IF NOT EXISTS idx_scorer_step_version ON scorer_step(scorer_version);

-- ─── Metrics ───────────────────────────────────────────────────────────
CREATE SEQUENCE IF NOT EXISTS metric_id_seq;
CREATE TABLE IF NOT EXISTS metric (
    metric_id     BIGINT PRIMARY KEY DEFAULT nextval('metric_id_seq'),
    run_id        VARCHAR NOT NULL,               -- FK score_run.run_id
    truth_set_id  VARCHAR,                        -- nullable: no-truth-mode metrics omit it
    metric_name   VARCHAR NOT NULL,               -- e.g., 'indra_belief_calibration', 'parse_claim_precision'
    value         DOUBLE NOT NULL,
    slice_json    JSON,                           -- e.g., {{"step": "grounding", "source_api": "reach"}}
    computed_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_metric_run ON metric(run_id);
CREATE INDEX IF NOT EXISTS idx_metric_truthset ON metric(truth_set_id);
CREATE INDEX IF NOT EXISTS idx_metric_natural ON metric(run_id, metric_name);

-- ─── Schema version tracking ───────────────────────────────────────────
CREATE TABLE IF NOT EXISTS schema_meta (
    key    VARCHAR PRIMARY KEY,
    value  VARCHAR NOT NULL
);
"""


def apply_schema(con: "duckdb.DuckDBPyConnection") -> None:
    """Apply the corpus schema to a DuckDB connection (idempotent).

    Records `SCHEMA_VERSION` in the `schema_meta` table. Existing tables
    are preserved (CREATE IF NOT EXISTS). Future migrations append new
    tables / columns rather than rewriting; that's the append-only
    contract Phase 2.6 will enforce.
    """
    con.execute(_DDL)
    con.execute(
        "INSERT OR REPLACE INTO schema_meta (key, value) VALUES ('schema_version', ?)",
        [str(SCHEMA_VERSION)],
    )
    log.info("corpus schema applied (version=%s)", SCHEMA_VERSION)


def schema_ddl() -> str:
    """Return the canonical DDL string. Useful for inspection / dumping."""
    return _DDL

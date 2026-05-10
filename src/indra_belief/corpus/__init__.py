"""Corpus persistence + scoring orchestration for INDRA Statement dumps.

Public surface (per `research/rasmachine_task_graph.md` D6 — viewer at
`viewer/` IS the serve layer; no Python serve() exists):
  - `from_indra_json(path) -> Iter[Statement]` — lossless loader (Phase 2)
  - `apply_schema(con)` / `ingest_statements(con, stmts)` (Phase 2)
  - `register_truth_set(...)` / `load_truth_labels(...)` — truth-set ingestion (Phase 2)
  - `estimate_cost(stmts, model_id) -> dict` — pre-run audit (Phase 0.1)
  - `score_corpus(con, stmts, *, decompose, with_validity, cost_threshold_usd) -> str` (Phase 3)
  - `compute_validity(con, run_id) -> dict` — idempotent metric refresh (Phase 4)
  - `aggregate_beliefs(...)` / `export_beliefs(...)` / `model_card(...)` (Phase 6)

Persistence is content-addressed and append-only; surviving scorer-architecture
iterations is a hard requirement. Schema lives in `corpus.schema`.
"""

from indra_belief.corpus.schema import apply_schema, SCHEMA_VERSION
from indra_belief.corpus.loader import (
    from_indra_json,
    ingest_statements,
    register_truth_set,
    load_truth_labels,
)
from indra_belief.corpus.scoring import score_corpus
from indra_belief.corpus.validity import compute_validity
from indra_belief.corpus.export import (
    aggregate_beliefs,
    export_beliefs,
    model_card,
)
from indra_belief.corpus.cost import estimate_cost, MODEL_PRICES_PER_M_TOKENS

__all__ = [
    "apply_schema",
    "SCHEMA_VERSION",
    "from_indra_json",
    "ingest_statements",
    "register_truth_set",
    "load_truth_labels",
    "score_corpus",
    "compute_validity",
    "aggregate_beliefs",
    "export_beliefs",
    "model_card",
    "estimate_cost",
    "MODEL_PRICES_PER_M_TOKENS",
]

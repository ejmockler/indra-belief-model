"""Belief export — Phase 6 of the rasmachine task graph.

Closes the loop: takes a `score_run`, aggregates per-evidence scores into
a per-statement belief, and emits an INDRA-native JSON dump where each
Statement's `belief` field is replaced with our score. The collaborator
can rebuild the EMMAA / rasmachine model from this and measure
downstream lift.

Aggregation rule (default, **mean of per-evidence scores**) is the no-gold
default and matches the validity layer's calibration computation. Phase
6.1's open-question framing kept the rule swappable: pass `aggregator=`
to override (noisy-OR, epistemics-weighted, supports-graph-aware are all
research questions to be settled by Phase 4 + 6 calibration on rasmachine
once we run the real corpus).
"""

from __future__ import annotations

import json
import logging
import math
import statistics
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    import duckdb

log = logging.getLogger(__name__)


def _assert_run_exists(con: "duckdb.DuckDBPyConnection", run_id: str) -> None:
    """Raise `ValueError` if `run_id` is not in `score_run`.

    Used at the top of every public function that takes a `run_id`. Cheap
    (`SELECT 1 ... LIMIT 1` against the small `score_run` table) and
    surfaces typo'd run_ids immediately rather than yielding empty/hollow
    results that look successful.
    """
    if not con.execute(
        "SELECT 1 FROM score_run WHERE run_id = ? LIMIT 1", [run_id]
    ).fetchone():
        raise ValueError(f"score_run {run_id} not found")


def _mean_aggregator(scores: list[float]) -> float:
    """Default aggregator: arithmetic mean of per-evidence scores."""
    return statistics.fmean(scores) if scores else 0.0


def _noisy_or_aggregator(scores: list[float]) -> float:
    """Noisy-OR: 1 - prod(1 - s_i). Matches INDRA's own aggregation when
    each evidence is treated as an independent positive signal."""
    if not scores:
        return 0.0
    p = 1.0
    for s in scores:
        p *= max(0.0, min(1.0, 1.0 - s))
    return 1.0 - p


AGGREGATORS: dict[str, Callable[[list[float]], float]] = {
    "mean": _mean_aggregator,
    "noisy_or": _noisy_or_aggregator,
}


def aggregate_beliefs(
    con: "duckdb.DuckDBPyConnection",
    run_id: str,
    *,
    aggregator: str | Callable[[list[float]], float] = "mean",
) -> dict[str, float]:
    """For each stmt scored in this run, return aggregated belief.

    Returns: `{stmt_hash: aggregated_belief}`.

    Raises `ValueError` if `run_id` doesn't exist in `score_run` — aligned
    with the rest of the run_id consumers (compute_validity, export_beliefs,
    model_card). Empty-dict return on missing run_id was a foot-gun.
    """
    _assert_run_exists(con, run_id)

    fn = aggregator if callable(aggregator) else AGGREGATORS[aggregator]

    rows = con.execute(
        """
        SELECT stmt_hash,
               array_agg(CAST(json_extract(output_json, '$.score') AS DOUBLE)) AS scores
        FROM scorer_step
        WHERE run_id = ?
          AND step_kind = 'aggregate'
          AND json_extract(output_json, '$.score') IS NOT NULL
        GROUP BY stmt_hash
        """,
        [run_id],
    ).fetchall()
    out: dict[str, float] = {}
    for stmt_hash, scores in rows:
        cleaned = [float(s) for s in (scores or []) if s is not None]
        if not cleaned:
            continue
        out[stmt_hash] = float(fn(cleaned))
    return out


def export_beliefs(
    con: "duckdb.DuckDBPyConnection",
    run_id: str,
    out_path: str | Path,
    *,
    aggregator: str | Callable[[list[float]], float] = "mean",
    only_scored: bool = True,
) -> Path:
    """Emit an INDRA-native JSON dump with `belief` replaced by our scores.

    Schema: standard INDRA `Statement.to_json()` list, lossless on every
    field that survived `from_indra_json` ingest. Only the `belief` value
    on statements present in this `run_id` is replaced; statements without
    a scoring row keep their original belief (or are omitted if
    `only_scored=True`).

    Returns the absolute Path of the written JSON file.

    Raises `ValueError` if `run_id` doesn't exist in `score_run` — matches
    `model_card`'s strictness. Silently emitting an empty-list export was
    a foot-gun (typo'd run_id → empty file → user thinks "no scored stmts"
    rather than "no such run").
    """
    out_path = Path(out_path).resolve()
    _assert_run_exists(con, run_id)
    new_beliefs = aggregate_beliefs(con, run_id, aggregator=aggregator)

    where_clause = (
        f"WHERE stmt_hash IN ({','.join(['?']*len(new_beliefs))})"
        if only_scored and new_beliefs else ""
    )
    params: list = list(new_beliefs.keys()) if (only_scored and new_beliefs) else []

    rows = con.execute(
        f"""SELECT stmt_hash, raw_json::VARCHAR AS raw_json
            FROM statement
            {where_clause}
            ORDER BY stmt_hash""",
        params,
    ).fetchall()

    out: list[dict] = []
    for stmt_hash, raw_json in rows:
        try:
            stmt_dict = json.loads(raw_json)
        except Exception as e:
            log.warning("export: failed to parse raw_json for %s: %s", stmt_hash, e)
            continue
        if stmt_hash in new_beliefs:
            stmt_dict["belief"] = new_beliefs[stmt_hash]
        out.append(stmt_dict)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    log.info("export_beliefs: wrote %d statements to %s", len(out), out_path)
    return out_path


def model_card(
    con: "duckdb.DuckDBPyConnection",
    run_id: str,
    *,
    out_path: str | Path | None = None,
) -> dict:
    """Generate a model card describing what was scored, how, and limitations.

    Phase 6.4. Returns the card as a dict; if `out_path` is provided,
    writes it as JSON for hand-off alongside the belief export.
    """
    _assert_run_exists(con, run_id)
    run = con.execute(
        """SELECT scorer_version, indra_version, model_id_default,
                  started_at::VARCHAR, finished_at::VARCHAR, status, n_stmts,
                  cost_estimate_usd, cost_actual_usd
           FROM score_run WHERE run_id = ?""",
        [run_id],
    ).fetchone()
    (scorer_version, indra_version, model_id, started_at, finished_at,
     status, n_stmts, cost_est, cost_act) = run

    metrics = con.execute(
        """SELECT metric_name, value, slice_json::VARCHAR, truth_set_id
           FROM metric WHERE run_id = ?
           ORDER BY metric_name, COALESCE(truth_set_id, '')""",
        [run_id],
    ).fetchall()

    # Truth-set coverage: count labels whose target is *any* stmt/evidence/agent
    # scored in this run. Per-target-kind dispatch — `target_id IN (stmt_hashes)`
    # only catches stmt-level labels and silently drops evidence/agent labels
    # (which carry evidence_hash / agent_hash, not stmt_hash).
    truth_label_counts = dict(
        con.execute(
            """SELECT truth_set_id, COUNT(*)
               FROM truth_label tl
               WHERE
                   (tl.target_kind = 'stmt' AND tl.target_id IN (
                       SELECT DISTINCT stmt_hash FROM scorer_step WHERE run_id = ?
                   ))
                OR (tl.target_kind = 'evidence' AND tl.target_id IN (
                       SELECT DISTINCT evidence_hash FROM evidence
                       WHERE stmt_hash IN (SELECT DISTINCT stmt_hash FROM scorer_step WHERE run_id = ?)
                   ))
                OR (tl.target_kind = 'agent' AND tl.target_id IN (
                       SELECT DISTINCT agent_hash FROM agent
                       WHERE stmt_hash IN (SELECT DISTINCT stmt_hash FROM scorer_step WHERE run_id = ?)
                   ))
               GROUP BY truth_set_id""",
            [run_id, run_id, run_id],
        ).fetchall()
    )

    card = {
        # Card-format version: bump on any breaking change to this dict shape.
        # v1: dict-keyed metrics (collapsed multi-truth_set; bug)
        # v2: list of metric records {metric_name, truth_set_id, value, slice}
        "card_format_version": 2,
        "run_id": run_id,
        "scorer_version": scorer_version,
        "indra_version": indra_version,
        "model_id": model_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "status": status,
        "n_stmts_scored": n_stmts,
        "cost_estimate_usd": cost_est,
        "cost_actual_usd": cost_act,
        # Metrics are run-keyed AND optionally truth_set-keyed. The same
        # metric_name (e.g. `truth_present.aggregate.f1`) may have multiple
        # rows when scored against different verdict-grade gold pools. List
        # form preserves all of them; dict-keying by metric_name alone
        # silently collapsed multi-truth_set entries (last-write-wins).
        "metrics": [
            {
                "metric_name": name,
                "truth_set_id": tset,
                "value": (None if value is None or (isinstance(value, float) and math.isnan(value)) else value),
                "slice": json.loads(slice_json or "{}"),
            }
            for name, value, slice_json, tset in metrics
        ],
        "truth_set_coverage": {tid: int(n) for tid, n in truth_label_counts.items()},
        "limitations": [
            "Belief is the mean of per-evidence scores. The aggregation rule "
            "is research-grade; alternatives (noisy-OR, epistemics-weighted) "
            "are available via the `aggregator=` parameter.",
            "When `truth_set_coverage` is sparse, calibration metrics report "
            "their `unavailable_reason` rather than imputing zero (G4 "
            "honest-failure clause).",
            "Per-step traces (parse_claim / probes / grounding / adjudicate) "
            "are stored as a single 'aggregate' row in this run; Phase 3.4 "
            "decomposition lands per-step rows.",
        ],
    }

    if out_path is not None:
        out_path = Path(out_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(card, fh, indent=2, default=str)
        log.info("model_card: wrote %s", out_path)

    return card

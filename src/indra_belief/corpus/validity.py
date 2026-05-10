"""Validity-layer metric computation — Phase 4 of the rasmachine task graph.

Branches on truth presence at runtime (D7):

  - **4a (truth-present)**: per-step precision/recall/F1, confusion matrices,
    score-level calibration vs gold. Requires `truth_label` rows for the
    relevant scorer_step / stmt / evidence target_kinds.

  - **4b (no-truth)**: INDRA-belief calibration, inter-evidence consistency,
    gold-overlap subset, supports-graph plausibility. Always available so
    long as the corpus carries `indra_published_belief` truth labels (the
    loader auto-registers these on every ingest).

Metrics are written to the `metric` table keyed by (run_id, optional
truth_set_id, metric_name, slice_json). G4 honest-failure: any metric
that *cannot* be computed from the loaded slice writes a row with
`value=NaN` and a `slice_json.unavailable_reason` field rather than
silently substituting zero.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import duckdb

log = logging.getLogger(__name__)


def _q(stmt_hash: str) -> str:
    return stmt_hash.replace("'", "''")


def _stmts_with_aggregate_score(con: "duckdb.DuckDBPyConnection", run_id: str):
    """Return [(stmt_hash, indra_belief, our_aggregate, n_evidences)] for the run.

    Aggregate rule: mean of per-evidence scores in this run. Phase 6.1 will
    install a real aggregator; this mean is the no-gold default.
    """
    return con.execute(
        """
        SELECT s.stmt_hash,
               s.indra_belief,
               AVG(CAST(json_extract(ss.output_json, '$.score') AS DOUBLE)) AS our_score,
               COUNT(*) AS n_ev
        FROM statement s
        JOIN scorer_step ss
          ON ss.stmt_hash = s.stmt_hash
         AND ss.run_id = ?
         AND ss.step_kind = 'aggregate'
         AND json_extract(ss.output_json, '$.score') IS NOT NULL
        GROUP BY s.stmt_hash, s.indra_belief
        """,
        [run_id],
    ).fetchall()


def _verdict_counts(con: "duckdb.DuckDBPyConnection", run_id: str) -> dict[str, int]:
    rows = con.execute(
        """
        SELECT json_extract(output_json, '$.verdict')::VARCHAR AS verdict, COUNT(*)
        FROM scorer_step
        WHERE run_id = ? AND step_kind = 'aggregate'
        GROUP BY 1
        """,
        [run_id],
    ).fetchall()
    out: dict[str, int] = {}
    for v, n in rows:
        # JSON extract returns quoted string '"correct"'; strip
        key = (v or "").strip('"') or "unknown"
        out[key] = int(n)
    return out


def _inter_evidence_consistency(con: "duckdb.DuckDBPyConnection", run_id: str):
    """For statements with N>1 evidences: stdev of per-evidence scores.

    Returns list of (stmt_hash, n_ev, mean, stdev). Lower stdev = higher
    inter-evidence agreement. NaN stdev for n=1 stmts is filtered out.
    """
    rows = con.execute(
        """
        SELECT stmt_hash,
               array_agg(CAST(json_extract(output_json, '$.score') AS DOUBLE)) AS scores
        FROM scorer_step
        WHERE run_id = ? AND step_kind = 'aggregate'
          AND json_extract(output_json, '$.score') IS NOT NULL
        GROUP BY stmt_hash
        HAVING COUNT(*) > 1
        """,
        [run_id],
    ).fetchall()
    out = []
    for h, scores in rows:
        if not scores:
            continue
        n = len(scores)
        mean = statistics.fmean(scores)
        stdev = statistics.pstdev(scores) if n > 1 else 0.0
        out.append((h, n, mean, stdev))
    return out


def _calibration_residuals(stmts) -> list[tuple[str, float, float, float]]:
    """For each stmt with INDRA belief: (stmt_hash, indra_belief, our_score, residual).

    Skips stmts where indra_belief is None (no anchor).
    """
    out = []
    for stmt_hash, indra_belief, our_score, _n_ev in stmts:
        if indra_belief is None or our_score is None:
            continue
        out.append((stmt_hash, float(indra_belief), float(our_score),
                    float(our_score) - float(indra_belief)))
    return out


def _write_metric(
    con: "duckdb.DuckDBPyConnection",
    run_id: str,
    name: str,
    value: float,
    *,
    truth_set_id: str | None = None,
    slice_: dict | None = None,
):
    con.execute(
        """INSERT INTO metric (run_id, truth_set_id, metric_name, value, slice_json)
           VALUES (?, ?, ?, ?, ?)""",
        [run_id, truth_set_id, name,
         value if not math.isnan(value) else float("nan"),
         json.dumps(slice_ or {}, default=str)],
    )


def _stratified_calibration(
    con: "duckdb.DuckDBPyConnection",
    run_id: str,
    *,
    by: str,
) -> list[tuple[str, int, float, float]]:
    """Per-stratum (MAE, bias, n_stmts) for a given grouping column.

    Returns [(stratum_value, n_stmts, mae, bias), …] for strata with ≥2 stmts.
    Strata with only 1 stmt are filtered out — single-point MAE is the
    abs(residual) which is dominated by noise.
    """
    if by == "indra_type":
        col = "s.indra_type"
        join = ""
    elif by == "source_api":
        # Pick the lexicographically-first source_api per stmt (deterministic
        # tie-break; an auditor can drill down by hand for mixed-source stmts)
        col = "(SELECT MIN(e.source_api) FROM evidence e WHERE e.stmt_hash = s.stmt_hash)"
        join = ""
    else:
        return []

    rows = con.execute(
        f"""
        WITH stmt_scores AS (
            SELECT s.stmt_hash, s.indra_belief,
                   {col} AS stratum,
                   AVG(CAST(json_extract(ss.output_json, '$.score') AS DOUBLE)) AS our_score
            FROM statement s
            JOIN scorer_step ss
              ON ss.stmt_hash = s.stmt_hash
             AND ss.run_id = ?
             AND ss.step_kind = 'aggregate'
             AND json_extract(ss.output_json, '$.score') IS NOT NULL
            {join}
            GROUP BY s.stmt_hash, s.indra_belief, {col}
        )
        SELECT stratum,
               COUNT(*) AS n_stmts,
               AVG(ABS(our_score - indra_belief)) AS mae,
               AVG(our_score - indra_belief) AS bias
        FROM stmt_scores
        WHERE indra_belief IS NOT NULL AND our_score IS NOT NULL
              AND stratum IS NOT NULL
        GROUP BY stratum
        HAVING COUNT(*) >= 2
        ORDER BY mae DESC
        """,
        [run_id],
    ).fetchall()
    return [(str(r[0]), int(r[1]), float(r[2]), float(r[3])) for r in rows]


def _truth_present_metrics(
    con: "duckdb.DuckDBPyConnection",
    run_id: str,
    truth_set_id: str,
) -> dict:
    """Phase 4a — per-step P/R/F1 when gold labels exist for this run's
    scorer_step targets under the given truth_set.

    Compares `scorer_step.output_json -> $.verdict` (the load-bearing
    decision per S-phase doctrine) against `truth_label.value_text` where
    `target_kind='evidence'` and `field='verdict'`. Per-step_kind precision,
    recall, F1 are computed against `verdict='correct'` as the positive class.

    Returns `{step_kind: {n_compared, precision, recall, f1, tp, fp, fn}}`.
    Empty dict if no overlap exists between gold labels and scorer outputs
    (legitimate signal: caller can render "no gold overlap on this run").
    """
    rows = con.execute(
        """
        WITH scorer_verdicts AS (
            SELECT step_kind,
                   evidence_hash,
                   replace(json_extract(output_json, '$.verdict')::VARCHAR, '"', '') AS our_verdict
            FROM scorer_step
            WHERE run_id = ?
              AND step_kind = 'aggregate'
              AND json_extract(output_json, '$.verdict') IS NOT NULL
              AND evidence_hash IS NOT NULL
        ),
        gold AS (
            SELECT target_id AS evidence_hash,
                   value_text AS gold_verdict
            FROM truth_label
            WHERE truth_set_id = ?
              AND target_kind = 'evidence'
              AND field = 'verdict'
        )
        SELECT sv.step_kind, sv.our_verdict, g.gold_verdict
        FROM scorer_verdicts sv
        JOIN gold g USING (evidence_hash)
        """,
        [run_id, truth_set_id],
    ).fetchall()

    if not rows:
        return {}

    by_step: dict[str, dict] = {}
    for step_kind, our, gold in rows:
        d = by_step.setdefault(
            step_kind, {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "n": 0}
        )
        d["n"] += 1
        # 'correct' is the positive class
        our_pos = our == "correct"
        gold_pos = gold == "correct"
        if our_pos and gold_pos:
            d["tp"] += 1
        elif our_pos and not gold_pos:
            d["fp"] += 1
        elif not our_pos and gold_pos:
            d["fn"] += 1
        else:
            d["tn"] += 1

    out = {}
    for step_kind, d in by_step.items():
        tp, fp, fn = d["tp"], d["fp"], d["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        out[step_kind] = {
            "n_compared": d["n"],
            "tp": tp, "fp": fp, "fn": fn, "tn": d["tn"],
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return out


def compute_validity(
    con: "duckdb.DuckDBPyConnection",
    run_id: str,
) -> dict:
    """Compute and persist validity metrics for a score run.

    Returns a summary dict for caller convenience; the canonical form is
    in the `metric` table.

    Idempotent: existing metric rows for this run_id are deleted first
    so re-running accumulates no duplicates. Caller can re-compute freely
    after registering new truth labels or schema-shape changes.

    Raises `ValueError` if `run_id` doesn't exist in `score_run` — aligned
    with `export_beliefs` / `model_card` strictness so a typo'd run_id
    surfaces immediately rather than producing a hollow summary dict that
    looks "successful" with empty metrics.
    """
    if not con.execute(
        "SELECT 1 FROM score_run WHERE run_id = ? LIMIT 1", [run_id]
    ).fetchone():
        raise ValueError(f"score_run {run_id} not found")

    summary: dict = {"run_id": run_id}

    con.execute("DELETE FROM metric WHERE run_id = ?", [run_id])

    # Verdict distribution
    verdicts = _verdict_counts(con, run_id)
    summary["verdicts"] = verdicts
    total = sum(verdicts.values())
    if total:
        for v, n in verdicts.items():
            _write_metric(
                con, run_id, f"verdict_share.{v}", n / total,
                slice_={"verdict": v, "n": n, "total": total},
            )

    # 4b.1 calibration vs indra_published_belief (no-gold default)
    stmts = _stmts_with_aggregate_score(con, run_id)
    residuals = _calibration_residuals(stmts)
    if residuals:
        per_diffs = [r[3] for r in residuals]
        mae = statistics.fmean([abs(d) for d in per_diffs])
        rmse = math.sqrt(statistics.fmean([d * d for d in per_diffs]))
        bias = statistics.fmean(per_diffs)
        _write_metric(
            con, run_id, "indra_belief_calibration.mae", mae,
            truth_set_id="indra_published_belief",
            slice_={"n_stmts": len(residuals)},
        )
        _write_metric(
            con, run_id, "indra_belief_calibration.rmse", rmse,
            truth_set_id="indra_published_belief",
            slice_={"n_stmts": len(residuals)},
        )
        _write_metric(
            con, run_id, "indra_belief_calibration.bias", bias,
            truth_set_id="indra_published_belief",
            slice_={"n_stmts": len(residuals),
                    "interp": "positive bias = our scores higher than INDRA's"},
        )
        summary["calibration"] = {
            "n_stmts": len(residuals),
            "mae": mae,
            "rmse": rmse,
            "bias": bias,
        }
    else:
        # G4 honest-failure: write the unavailability rather than skipping
        _write_metric(
            con, run_id, "indra_belief_calibration.mae", float("nan"),
            truth_set_id="indra_published_belief",
            slice_={"unavailable_reason": "no statements with both indra_belief and our_score"},
        )
        summary["calibration"] = {"n_stmts": 0,
                                  "unavailable_reason": "no overlap"}

    # 4b.2 inter-evidence consistency
    consistency = _inter_evidence_consistency(con, run_id)
    if consistency:
        stdevs = [s for _h, _n, _m, s in consistency]
        mean_stdev = statistics.fmean(stdevs)
        _write_metric(
            con, run_id, "inter_evidence_consistency.mean_stdev", mean_stdev,
            slice_={"n_multi_evidence_stmts": len(consistency),
                    "interp": "lower stdev = higher agreement"},
        )
        summary["inter_evidence_consistency"] = {
            "n_multi_evidence_stmts": len(consistency),
            "mean_stdev": mean_stdev,
            "max_stdev": max(stdevs),
        }
    else:
        _write_metric(
            con, run_id, "inter_evidence_consistency.mean_stdev", float("nan"),
            slice_={"unavailable_reason": "no statements with N>1 evidences"},
        )
        summary["inter_evidence_consistency"] = {
            "n_multi_evidence_stmts": 0,
            "unavailable_reason": "all singletons",
        }

    # 4a — truth-present P/R/F1 against any registered truth_set carrying
    # `target_kind='evidence', field='verdict'` rows (e.g. project gold pool).
    # When no gold overlap exists, returns {} and we skip writing rows.
    candidate_truth_sets = [
        r[0] for r in con.execute(
            """SELECT DISTINCT truth_set_id FROM truth_label
               WHERE target_kind = 'evidence' AND field = 'verdict'"""
        ).fetchall()
    ]
    summary["truth_present_metrics"] = {}
    for tset_id in candidate_truth_sets:
        per_step = _truth_present_metrics(con, run_id, tset_id)
        if not per_step:
            continue
        summary["truth_present_metrics"][tset_id] = per_step
        for step_kind, m in per_step.items():
            for metric_name in ("precision", "recall", "f1"):
                _write_metric(
                    con, run_id, f"truth_present.{step_kind}.{metric_name}",
                    m[metric_name], truth_set_id=tset_id,
                    slice_={"step_kind": step_kind, "n_compared": m["n_compared"],
                            "tp": m["tp"], "fp": m["fp"], "fn": m["fn"]},
                )

    # 4a.4 / 4b stratified calibration (post-iter-31): break MAE/bias by
    # indra_type and source_api so an auditor can spot type-specific or
    # source-specific systematic miscalibration. Strata with <2 stmts are
    # suppressed (single-point MAE is abs-residual noise).
    summary["calibration_by_indra_type"] = []
    for stratum, n, mae, bias in _stratified_calibration(con, run_id, by="indra_type"):
        _write_metric(
            con, run_id, "indra_belief_calibration_by_type.mae", mae,
            truth_set_id="indra_published_belief",
            slice_={"stratum": "indra_type", "value": stratum, "n": n},
        )
        _write_metric(
            con, run_id, "indra_belief_calibration_by_type.bias", bias,
            truth_set_id="indra_published_belief",
            slice_={"stratum": "indra_type", "value": stratum, "n": n},
        )
        summary["calibration_by_indra_type"].append(
            {"stratum": stratum, "n": n, "mae": mae, "bias": bias}
        )

    summary["calibration_by_source_api"] = []
    for stratum, n, mae, bias in _stratified_calibration(con, run_id, by="source_api"):
        _write_metric(
            con, run_id, "indra_belief_calibration_by_source.mae", mae,
            truth_set_id="indra_published_belief",
            slice_={"stratum": "source_api", "value": stratum, "n": n},
        )
        _write_metric(
            con, run_id, "indra_belief_calibration_by_source.bias", bias,
            truth_set_id="indra_published_belief",
            slice_={"stratum": "source_api", "value": stratum, "n": n},
        )
        summary["calibration_by_source_api"].append(
            {"stratum": stratum, "n": n, "mae": mae, "bias": bias}
        )

    # 4b.4 supports-graph plausibility (sketch): mean score for stmts with
    # supports vs without. Real signal once rasmachine's supports edges populate.
    rows = con.execute(
        """
        SELECT (s.supports_count > 0 OR s.supported_by_count > 0) AS has_supports,
               AVG(CAST(json_extract(ss.output_json, '$.score') AS DOUBLE)) AS mean_score,
               COUNT(*) AS n_ev
        FROM statement s
        JOIN scorer_step ss
          ON ss.stmt_hash = s.stmt_hash
         AND ss.run_id = ?
         AND ss.step_kind = 'aggregate'
         AND json_extract(ss.output_json, '$.score') IS NOT NULL
        GROUP BY 1
        """,
        [run_id],
    ).fetchall()
    if len(rows) == 2:
        # both buckets populated — real comparison
        with_s = next((r for r in rows if r[0]), None)
        without_s = next((r for r in rows if not r[0]), None)
        if with_s and without_s and with_s[1] is not None and without_s[1] is not None:
            delta = float(with_s[1]) - float(without_s[1])
            _write_metric(
                con, run_id, "supports_graph_plausibility.delta", delta,
                slice_={"n_with_supports": int(with_s[2]),
                        "n_without_supports": int(without_s[2]),
                        "interp": "positive = supports-rich score higher"},
            )
            summary["supports_graph_plausibility"] = {
                "delta": delta,
                "n_with_supports": int(with_s[2]),
                "n_without_supports": int(without_s[2]),
            }
        else:
            _write_metric(
                con, run_id, "supports_graph_plausibility.delta", float("nan"),
                slice_={"unavailable_reason": "missing aggregate score in one bucket"},
            )
            summary["supports_graph_plausibility"] = {
                "unavailable_reason": "missing data in one bucket"
            }
    else:
        _write_metric(
            con, run_id, "supports_graph_plausibility.delta", float("nan"),
            slice_={"unavailable_reason": "all evidences are in same supports-bucket"},
        )
        summary["supports_graph_plausibility"] = {
            "unavailable_reason": "all-same-bucket — needs supports-edge variety"
        }

    log.info("compute_validity %s: %s", run_id, summary)
    return summary

"""Viewer-spawned worker (U3 of the pipeline-in-viewer hypergraph).

The SvelteKit viewer invokes this module via
    python -m indra_belief.worker <verb> [--args...]

Subcommands:
    ingest                 — load an INDRA-Statement JSON into corpus.duckdb
    register-truth-set     — load a JSONL of tagged records as a truth_set
    score                  — (deferred to U5) run score_corpus end-to-end

Output convention: each non-fatal status event is a single JSON object
written to stdout followed by a newline + flush, so the viewer endpoint
can stream events line-by-line for SSE-style progress. Final events
include `event: "done"` with summary fields. Failures raise; stderr
carries the traceback for the endpoint to surface.
"""
from __future__ import annotations

import argparse
import json
import sys
import time


def emit(event: dict) -> None:
    """Write a single newline-terminated JSON event to stdout."""
    sys.stdout.write(json.dumps(event))
    sys.stdout.write("\n")
    sys.stdout.flush()


def do_ingest(args: argparse.Namespace) -> int:
    import duckdb
    from indra.statements import stmts_from_json_file

    from indra_belief.corpus import apply_schema, ingest_statements

    emit({"event": "started", "verb": "ingest", "path": args.path})
    t0 = time.time()
    con = duckdb.connect(args.db)
    try:
        apply_schema(con)
        stmts = stmts_from_json_file(args.path)
        emit({"event": "loaded", "n_statements": len(stmts)})
        ingest_statements(con, stmts, source_dump_id=args.source_dump_id)
        emit({
            "event": "done",
            "n_statements": len(stmts),
            "duration_s": round(time.time() - t0, 2),
        })
        return 0
    finally:
        con.close()


def do_estimate_cost(args: argparse.Namespace) -> int:
    """Read a JSON of INDRA Statements, estimate cost per supported model."""
    from indra.statements import stmts_from_json_file

    from indra_belief.corpus.cost import MODEL_PRICES_PER_M_TOKENS, estimate_cost

    emit({"event": "started", "verb": "estimate-cost", "path": args.path})
    t0 = time.time()
    stmts = stmts_from_json_file(args.path)
    estimates: list[dict] = []
    for model_id in MODEL_PRICES_PER_M_TOKENS.keys():
        e = estimate_cost(stmts, model_id=model_id)
        estimates.append({
            "model_id": model_id,
            "cost_usd": e["cost_usd"],
            "n_stmts": e["n_stmts"],
            "n_evidences_est": e["n_evidences_est"],
            "n_llm_calls_est": e["n_llm_calls_est"],
        })
    emit({
        "event": "done",
        "n_statements": len(stmts),
        "estimates": estimates,
        "duration_s": round(time.time() - t0, 2),
    })
    return 0


def do_score(args: argparse.Namespace) -> int:
    """Ingest (idempotent) + score a corpus end-to-end. Emits per-evidence progress."""
    import duckdb
    from indra.statements import stmts_from_json_file

    from indra_belief.corpus import (
        apply_schema,
        ingest_statements,
        score_corpus,
    )
    from indra_belief.model_client import ModelClient

    emit({
        "event": "started",
        "verb": "score",
        "path": args.path,
        "model": args.model,
        "scorer_version": args.scorer_version,
    })
    t0 = time.time()
    con = duckdb.connect(args.db)
    try:
        apply_schema(con)
        stmts = stmts_from_json_file(args.path)
        # Count evidences so the viewer's progress bar has a real denominator
        # rather than a fabricated multiplier.
        n_evidences = sum(
            len(getattr(s, "evidence", None) or []) for s in stmts
        )
        emit({
            "event": "loaded",
            "n_statements": len(stmts),
            "n_evidences": n_evidences,
        })

        # Idempotent ingest — safe to call even if the user already clicked
        # [ingest] before. INSERT OR REPLACE on the natural key (stmt_hash).
        ingest_statements(con, stmts, source_dump_id=args.source_dump_id)
        emit({"event": "ingested"})

        client = ModelClient(args.model)
        evidences_done = [0]

        def on_ev(stmt_hash: str, ev_hash: str, _result: dict) -> None:
            evidences_done[0] += 1
            n = evidences_done[0]
            # Emit progress: every evidence for the first 5, every 5 thereafter,
            # then every 25. Keeps the SSE stream legible.
            if n <= 5 or (n <= 50 and n % 5 == 0) or n % 25 == 0:
                emit({
                    "event": "progress",
                    "n_evidences_done": n,
                    "latest_stmt_hash": stmt_hash,
                })

        run_id = score_corpus(
            con,
            stmts,
            client=client,
            scorer_version=args.scorer_version,
            model_id_default=args.model,
            decompose=True,
            cost_threshold_usd=args.cost_threshold_usd,
            on_evidence=on_ev,
        )
        emit({
            "event": "done",
            "run_id": run_id,
            "n_statements": len(stmts),
            "n_evidences_done": evidences_done[0],
            "duration_s": round(time.time() - t0, 2),
        })
        return 0
    finally:
        con.close()


def do_register_truth_set(args: argparse.Namespace) -> int:
    import duckdb

    from indra_belief.corpus import apply_schema
    from indra_belief.corpus.loader import _hex

    emit({
        "event": "started",
        "verb": "register-truth-set",
        "path": args.path,
        "truth_set_id": args.truth_set_id,
    })
    t0 = time.time()
    con = duckdb.connect(args.db)
    try:
        apply_schema(con)
        # Register truth_set row (idempotent — DELETE-then-INSERT in case a
        # prior load used a different name/description).
        con.execute(
            "DELETE FROM truth_set WHERE id=?",
            [args.truth_set_id],
        )
        con.execute(
            "INSERT INTO truth_set (id, name, description) VALUES (?, ?, ?)",
            [args.truth_set_id, args.truth_set_name,
             f"loaded from {args.path} via worker"],
        )

        n_loaded = 0
        n_skipped = 0
        n_missing_target = 0
        n_missing_field = 0
        skipped_examples: list[str] = []

        with open(args.path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    n_skipped += 1
                    continue

                value = rec.get(args.field)
                if value is None:
                    n_missing_field += 1
                    if len(skipped_examples) < 3:
                        skipped_examples.append(
                            f"no `{args.field}` on record"
                        )
                    continue

                # Resolve target_id from INDRA hashes
                if args.target_kind == "evidence":
                    raw = rec.get(args.target_hash_field or "source_hash")
                elif args.target_kind == "stmt":
                    raw = rec.get(args.target_hash_field or "matches_hash")
                else:
                    raise ValueError(
                        f"unsupported target-kind {args.target_kind}"
                    )
                if raw is None:
                    n_missing_target += 1
                    continue

                try:
                    target_id = _hex(int(raw))
                except (TypeError, ValueError):
                    n_missing_target += 1
                    continue

                # Idempotent write on the natural key
                con.execute(
                    "DELETE FROM truth_label "
                    "WHERE truth_set_id=? AND target_kind=? AND target_id=? AND field=?",
                    [args.truth_set_id, args.target_kind, target_id, args.field],
                )
                con.execute(
                    "INSERT INTO truth_label "
                    "(truth_set_id, target_kind, target_id, field, value_text, provenance) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    [args.truth_set_id, args.target_kind, target_id,
                     args.field, str(value), args.path],
                )
                n_loaded += 1
                if n_loaded % 200 == 0:
                    emit({
                        "event": "progress",
                        "n_loaded": n_loaded,
                        "n_missing_target": n_missing_target,
                        "n_missing_field": n_missing_field,
                    })

        # Optionally re-compute validity for the latest succeeded run so the
        # viewer's validity panel grows a P/R/F1 row automatically. Without
        # this, the new truth_set is registered but invisible until the next
        # score_corpus.
        recomputed_run_id: str | None = None
        if args.recompute_latest_validity:
            from indra_belief.corpus import compute_validity

            r = con.execute(
                "SELECT run_id FROM score_run WHERE status='succeeded' "
                "ORDER BY started_at DESC LIMIT 1"
            ).fetchone()
            if r:
                recomputed_run_id = r[0]
                try:
                    compute_validity(con, recomputed_run_id)
                    emit({
                        "event": "validity_recomputed",
                        "run_id": recomputed_run_id,
                    })
                except Exception as e:
                    emit({
                        "event": "validity_recompute_failed",
                        "run_id": recomputed_run_id,
                        "error": str(e),
                    })

        emit({
            "event": "done",
            "n_loaded": n_loaded,
            "n_missing_target": n_missing_target,
            "n_missing_field": n_missing_field,
            "n_skipped": n_skipped,
            "skipped_examples": skipped_examples,
            "recomputed_run_id": recomputed_run_id,
            "duration_s": round(time.time() - t0, 2),
        })
        return 0
    finally:
        con.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="indra_belief.worker",
        description="viewer-spawned worker for ingest / truth-set / score verbs",
    )
    sub = parser.add_subparsers(dest="cmd")

    p_ingest = sub.add_parser("ingest")
    p_ingest.add_argument("--db", required=True)
    p_ingest.add_argument("--path", required=True)
    p_ingest.add_argument("--source-dump-id", required=True)

    p_est = sub.add_parser("estimate-cost")
    p_est.add_argument("--path", required=True)

    p_score = sub.add_parser("score")
    p_score.add_argument("--db", required=True)
    p_score.add_argument("--path", required=True)
    p_score.add_argument("--source-dump-id", required=True)
    p_score.add_argument("--model", required=True)
    p_score.add_argument("--scorer-version", required=True)
    p_score.add_argument("--cost-threshold-usd", type=float, default=None,
                         help="abort if estimated cost exceeds this dollar cap")

    p_truth = sub.add_parser("register-truth-set")
    p_truth.add_argument("--db", required=True)
    p_truth.add_argument("--path", required=True)
    p_truth.add_argument("--truth-set-id", required=True)
    p_truth.add_argument("--truth-set-name", required=True)
    p_truth.add_argument("--target-kind", required=True,
                         choices=["stmt", "evidence"])
    p_truth.add_argument("--field", required=True,
                         help="record field whose value becomes the label")
    p_truth.add_argument("--target-hash-field", default=None,
                         help="record field carrying the INDRA hash; "
                              "defaults to source_hash for evidence, matches_hash for stmt")
    p_truth.add_argument("--recompute-latest-validity", action="store_true",
                         help="after registering, re-run compute_validity for "
                              "the latest succeeded run so the new truth_set's "
                              "P/R/F1 row appears in the viewer")

    args = parser.parse_args(argv)

    if args.cmd == "ingest":
        return do_ingest(args)
    if args.cmd == "estimate-cost":
        return do_estimate_cost(args)
    if args.cmd == "score":
        return do_score(args)
    if args.cmd == "register-truth-set":
        return do_register_truth_set(args)

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())

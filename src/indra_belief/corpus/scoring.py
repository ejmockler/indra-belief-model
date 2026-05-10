"""Score-corpus orchestration — Phase 3.1 of the rasmachine task graph.

`score_corpus(con, stmts, client, ...)` iterates statements through
`indra_belief.score_evidence` and writes per-evidence aggregate rows to
`scorer_step` plus a single `score_run` summary row.

This is the minimum viable orchestration: one `scorer_step(step_kind='aggregate')`
per evidence, capturing the full per-evidence dict (score / verdict /
confidence / reasons / call_log) as `output_json`. Phase 3.4 decomposes
this into per-step rows (parse_claim / build_context / substrate_route /
4 probes / grounding / adjudicate) by emitting structured events from
within `score_via_probes` rather than parsing the aggregate dict.

Append-only contract (Phase 2.5): a re-run with a different `scorer_version`
lands new rows alongside the old; same `(stmt_hash, evidence_hash,
scorer_version, step_kind)` is upserted by `step_hash`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable, Iterable, Protocol

if TYPE_CHECKING:
    import duckdb
    from indra.statements import Statement

log = logging.getLogger(__name__)


class ScoreEvidenceFn(Protocol):
    """Callable that scores one (Statement, Evidence) pair and returns a dict."""

    def __call__(self, statement, evidence, client) -> dict: ...


def _hex(n: int, width: int = 16) -> str:
    return f"{n & ((1 << 64) - 1):0{width}x}"


def _step_hash(stmt_hash: str, evidence_hash: str, scorer_version: str,
               model_id: str, step_kind: str) -> str:
    raw = f"{stmt_hash}|{evidence_hash}|{scorer_version}|{model_id}|{step_kind}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _detect_indra_version() -> str:
    try:
        import indra
        return getattr(indra, "__version__", "unknown")
    except Exception:
        return "unknown"


def _decompose_steps(stmt, ev) -> list[tuple[str, dict, bool | None]]:
    """Phase 3.4 partial: capture deterministic substeps as separate rows.

    Returns a list of `(step_kind, output_json_dict, is_substrate_answered)`
    tuples for steps 1-3 (parse_claim, build_context, substrate_route) plus
    one row per substrate-answered probe (steps 4-7 when substrate hits).

    Steps 4-7 (LLM-escalated probes), 8 (grounding), 9 (adjudicate) are
    entangled with the LLM call and remain captured in the aggregate row
    until the orchestrator emits them as structured events.
    """
    out: list[tuple[str, dict, bool | None]] = []

    try:
        from dataclasses import asdict, is_dataclass
        from indra_belief.scorers.parse_claim import parse_claim
        from indra_belief.scorers.context_builder import build_context
        from indra_belief.scorers.probes.router import substrate_route
    except Exception as e:
        log.warning("decompose imports failed: %s", e)
        return out

    def _to_dict(obj):
        if obj is None:
            return None
        if is_dataclass(obj):
            return asdict(obj)
        if hasattr(obj, "_asdict"):
            return obj._asdict()
        if isinstance(obj, (str, int, float, bool, list, dict, tuple)):
            return obj
        # Fallback: stringify
        return str(obj)

    try:
        claim = parse_claim(stmt)
        out.append(("parse_claim", _to_dict(claim) or {}, None))
    except Exception as e:
        out.append(("parse_claim", {"error": str(e)}, None))
        return out

    try:
        ctx = build_context(stmt, ev)
        ctx_summary = {
            "stmt_type": getattr(ctx, "stmt_type", None),
            "n_aliases": len(getattr(ctx, "aliases", {}) or {}),
            "n_detected_relations": len(getattr(ctx, "detected_relations", []) or []),
            "n_modifications_sites": len(getattr(ctx, "detected_sites", set()) or set()),
            "has_chain_signal": getattr(ctx, "has_chain_signal", False),
            "is_complex": getattr(ctx, "is_complex", False),
            "is_modification": getattr(ctx, "is_modification", False),
            "subject_class": getattr(ctx, "subject_class", "unknown"),
            "object_class": getattr(ctx, "object_class", "unknown"),
        }
        out.append(("build_context", ctx_summary, None))
    except Exception as e:
        out.append(("build_context", {"error": str(e)}, None))
        return out

    try:
        evidence_text = (getattr(ev, "text", "") or "").strip()
        routes = substrate_route(claim, ctx, evidence_text)
        # Top-level row summarizing how each probe was routed
        route_summary = {}
        substrate_answered = {}
        for kind, route in routes.items():
            source = getattr(route, "source", None)
            answer = getattr(route, "answer", None)
            confidence = getattr(route, "confidence", None)
            route_summary[kind] = {
                "source": source,
                "answer": answer,
                "confidence": confidence,
            }
            substrate_answered[kind] = (source == "substrate")
        out.append(("substrate_route", route_summary, None))

        # Per-probe rows for substrate-answered probes (steps 4-7 lit when hit)
        kind_to_step = {
            "subject_role": "subject_role_probe",
            "object_role": "object_role_probe",
            "relation_axis": "relation_axis_probe",
            "scope": "scope_probe",
        }
        for kind, route in routes.items():
            if substrate_answered.get(kind):
                step_kind = kind_to_step[kind]
                out.append((
                    step_kind,
                    {
                        "answer": getattr(route, "answer", None),
                        "confidence": getattr(route, "confidence", None),
                        "source": "substrate",
                        "span": getattr(route, "span", None),
                        "rationale": getattr(route, "rationale", None),
                    },
                    True,
                ))
    except Exception as e:
        out.append(("substrate_route", {"error": str(e)}, None))

    return out


def score_corpus(
    con: "duckdb.DuckDBPyConnection",
    stmts: Iterable["Statement"],
    *,
    client=None,
    scorer_version: str = "dev",
    model_id_default: str = "unknown",
    score_evidence: ScoreEvidenceFn | None = None,
    on_evidence: Callable[[str, str, dict], None] | None = None,
    decompose: bool = False,
    with_validity: bool = True,
    cost_threshold_usd: float | None = None,
) -> str:
    """Score a stream of INDRA Statements and write rows to the corpus DB.

    Args:
        con: a DuckDB connection with the corpus schema applied.
        stmts: iterable of INDRA Statement objects (already ingested via
            `ingest_statements`; this function does NOT re-ingest).
        client: a `ModelClient` (or compatible) — passed straight to
            `score_evidence`. Required for real scoring.
        scorer_version: identifier for this run's scorer code (typically
            a git commit hash). Multiple runs at different versions land
            alongside each other in `scorer_step`; never overwrite.
        model_id_default: LLM identifier recorded on `score_run` and
            propagated to per-step rows when not overridden.
        score_evidence: override the default `indra_belief.score_evidence`.
            Useful for tests with a mock that returns deterministic dicts.
        on_evidence: optional callback `(stmt_hash, evidence_hash, dict)`
            fired after each evidence is scored. Phase 3.7 SSE live-tail
            hooks here.

    Returns:
        The `run_id` (UUID hex) so the caller can `JOIN metric ON metric.run_id`.
    """
    if score_evidence is None:
        # Default: use the project scorer. Requires a real ModelClient — we
        # fail fast here rather than letting every evidence collapse into
        # an 'abstain' row from the per-evidence exception handler when
        # `client.call(...)` hits None. Tests pass mock score_evidence and
        # never hit this branch.
        if client is None:
            raise ValueError(
                "score_corpus requires either client= (a ModelClient) or "
                "score_evidence= (a callable). Got both None."
            )
        from indra_belief import score_evidence as default_score_evidence
        score_evidence = default_score_evidence  # type: ignore[assignment]

    # Always estimate cost upfront — needed for the threshold gate, AND
    # persisted to score_run.cost_estimate_usd as part of the audit trail
    # (the model_card surfaces it; the column was schema-defined but
    # forever-NULL until this iter).
    stmts = list(stmts)  # type: ignore[assignment]
    from indra_belief.corpus.cost import estimate_cost
    estimate = estimate_cost(stmts, model_id=model_id_default)  # type: ignore[arg-type]
    cost_estimate = estimate["cost_usd"]

    if cost_threshold_usd is not None and cost_estimate > cost_threshold_usd:
        raise ValueError(
            f"score_corpus aborted: estimated cost ${cost_estimate:.2f} "
            f"exceeds threshold ${cost_threshold_usd:.2f} "
            f"({estimate['n_stmts']} stmts × {estimate['n_evidences_est']} ev "
            f"→ ~{estimate['n_llm_calls_est']:,} LLM calls on model {model_id_default}). "
            f"Raise cost_threshold_usd= or sample smaller stmts."
        )

    run_id = uuid.uuid4().hex
    started_at = datetime.now(timezone.utc)
    indra_version = _detect_indra_version()

    con.execute(
        """INSERT INTO score_run
           (run_id, scorer_version, indra_version, model_id_default,
            started_at, status, n_stmts, cost_estimate_usd)
           VALUES (?, ?, ?, ?, ?, 'running', 0, ?)""",
        [run_id, scorer_version, indra_version, model_id_default, started_at,
         cost_estimate],
    )

    n_stmts = 0
    n_evidences = 0
    status = "running"

    try:
        for stmt in stmts:
            stmt_hash = _hex(stmt.get_hash(shallow=True))
            evidences = list(getattr(stmt, "evidence", None) or [])
            if not evidences:
                n_stmts += 1
                continue

            for ev in evidences:
                try:
                    evhash = _hex(ev.get_source_hash())
                except Exception:
                    evhash = hashlib.sha256(
                        f"{ev.source_api}|{ev.source_id}|{ev.pmid}|{ev.text}".encode("utf-8")
                    ).hexdigest()[:16]

                t0 = time.perf_counter()

                # Phase 3.4 partial: capture deterministic substeps into
                # their own scorer_step rows BEFORE running the aggregate.
                # Lights rail positions 1-3 (and 4-7 for substrate-answered
                # probes) deterministically; LLM-escalated steps + adjudicate
                # remain in the aggregate row until orchestrator emits them.
                if decompose:
                    for det_kind, det_payload, det_substrate in _decompose_steps(stmt, ev):
                        det_step_hash = _step_hash(stmt_hash, evhash, scorer_version,
                                                    model_id_default, det_kind)
                        try:
                            con.execute(
                                """INSERT OR REPLACE INTO scorer_step
                                   (step_hash, stmt_hash, evidence_hash, run_id,
                                    scorer_version, model_id, step_kind, is_substrate_answered,
                                    input_payload_json, output_json, latency_ms,
                                    prompt_tokens, out_tokens, finish_reason, error)
                                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                                [det_step_hash, stmt_hash, evhash, run_id, scorer_version,
                                 model_id_default, det_kind, det_substrate,
                                 None, json.dumps(det_payload, default=str),
                                 None, None, None, None, None],
                            )
                        except Exception as e:
                            log.warning("decompose write failed for %s: %s", det_kind, e)

                try:
                    result = score_evidence(stmt, ev, client)
                except Exception as e:
                    log.warning("score_evidence failed for %s/%s: %s", stmt_hash, evhash, e)
                    result = {
                        "score": None, "verdict": "abstain", "confidence": "low",
                        "error": str(e), "call_log": [],
                    }
                latency_ms = int((time.perf_counter() - t0) * 1000)

                model_id = result.get("model_id") or model_id_default
                step_kind = "aggregate"
                step_hash = _step_hash(stmt_hash, evhash, scorer_version, model_id, step_kind)

                # Sum probe call latency / tokens from call_log if present
                call_log = result.get("call_log") or []
                prompt_tokens = sum(
                    (c.get("prompt_tokens") or 0) for c in call_log
                ) or None
                out_tokens = sum(
                    (c.get("out_tokens") or 0) for c in call_log
                ) or result.get("tokens") or None

                con.execute(
                    """INSERT OR REPLACE INTO scorer_step
                       (step_hash, stmt_hash, evidence_hash, run_id,
                        scorer_version, model_id, step_kind, is_substrate_answered,
                        input_payload_json, output_json, latency_ms,
                        prompt_tokens, out_tokens, finish_reason, error)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    [step_hash, stmt_hash, evhash, run_id, scorer_version,
                     model_id, step_kind,
                     None,  # is_substrate_answered N/A on aggregate
                     None,  # input_payload_json N/A on aggregate
                     json.dumps(result, default=str),
                     latency_ms,
                     prompt_tokens,
                     out_tokens,
                     None,
                     result.get("error")],
                )
                n_evidences += 1

                if on_evidence is not None:
                    try:
                        on_evidence(stmt_hash, evhash, result)
                    except Exception:
                        log.exception("on_evidence callback raised")
            n_stmts += 1
        status = "succeeded"
    except Exception:
        status = "failed"
        raise
    finally:
        finished_at = datetime.now(timezone.utc)
        # Compute actual cost from observed tokens × model rate. Sums across
        # all scorer_step rows for this run; unknown models contribute 0
        # (mock runs land here). This was the second forever-NULL column on
        # score_run; iter-90 fixed cost_estimate_usd, iter-92 the actual.
        from indra_belief.corpus.cost import MODEL_PRICES_PER_M_TOKENS
        prices = MODEL_PRICES_PER_M_TOKENS.get(model_id_default, (0.0, 0.0))
        in_p, out_p = prices
        actual_row = con.execute(
            """SELECT
                 COALESCE(SUM(prompt_tokens), 0) AS total_in,
                 COALESCE(SUM(out_tokens), 0) AS total_out
               FROM scorer_step WHERE run_id = ? AND step_kind = 'aggregate'""",
            [run_id],
        ).fetchone()
        total_in_tokens, total_out_tokens = actual_row or (0, 0)
        cost_actual = (
            (total_in_tokens or 0) * in_p / 1_000_000
            + (total_out_tokens or 0) * out_p / 1_000_000
        )
        con.execute(
            """UPDATE score_run
               SET finished_at = ?, status = ?, n_stmts = ?,
                   cost_actual_usd = ?
               WHERE run_id = ?""",
            [finished_at, status, n_stmts, cost_actual, run_id],
        )
        log.info(
            "score_corpus run_id=%s status=%s n_stmts=%d n_evidences=%d",
            run_id, status, n_stmts, n_evidences,
        )

    # Auto-compute validity at end of successful run unless caller opts out.
    # Tightens the workflow: ingest → score (validity computed) → export.
    if with_validity and status == "succeeded":
        try:
            from indra_belief.corpus.validity import compute_validity
            compute_validity(con, run_id)
        except Exception as e:
            log.warning("auto compute_validity failed for %s: %s", run_id, e)

    return run_id

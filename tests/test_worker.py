"""Tests for the viewer-spawned worker (src/indra_belief/worker.py).

The U-phase pipeline-in-viewer architecture (see
research/pipeline_in_viewer_task_graph.md) spawns this worker via
node:child_process. The SvelteKit endpoints rely on its output
contract: newline-JSON events on stdout, structured `done` /
`error` final events, exit code 0 on success.

These tests exercise three of the four verbs (ingest,
estimate-cost, register-truth-set) without LLM credentials. The
score verb is exercised only structurally — it requires a live
ModelClient + API key; the existing end-to-end smoke path
(curl /api/runs/score) is the integration test.
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import duckdb
import pytest

from indra.statements import (
    Activation,
    Agent,
    Evidence,
    Phosphorylation,
)

from indra_belief import worker as W
from indra_belief.corpus import apply_schema, ingest_statements


# ---------- shared fixtures ---------------------------------------------------


@pytest.fixture()
def tmp_db(tmp_path: Path) -> str:
    db_path = tmp_path / "corpus.duckdb"
    con = duckdb.connect(str(db_path))
    apply_schema(con)
    con.close()
    return str(db_path)


@pytest.fixture()
def tiny_indra_json(tmp_path: Path) -> str:
    """Three INDRA Statements serialized to JSON — the same shape
    `stmts_from_json_file` consumes."""
    stmts = [
        Phosphorylation(
            Agent("MAP2K1", db_refs={"HGNC": "6840"}),
            Agent("MAPK1", db_refs={"HGNC": "6871"}),
            residue="T",
            position="185",
            evidence=[Evidence(source_api="reach",
                               text="MAP2K1 phosphorylates MAPK1.",
                               pmid="11111")],
        ),
        Activation(
            Agent("RAF1", db_refs={"HGNC": "9829"}),
            Agent("MAP2K1", db_refs={"HGNC": "6840"}),
            evidence=[Evidence(source_api="reach",
                               text="RAF1 activates MAP2K1.",
                               pmid="22222")],
        ),
        Activation(
            Agent("KRAS", db_refs={"HGNC": "6407"}),
            Agent("RAF1", db_refs={"HGNC": "9829"}),
            evidence=[Evidence(source_api="reach",
                               text="KRAS activates RAF1.",
                               pmid="33333")],
        ),
    ]
    for s in stmts:
        s.belief = 0.85
    path = tmp_path / "tiny_indra.json"
    path.write_text(json.dumps([s.to_json() for s in stmts]))
    return str(path)


@pytest.fixture()
def tiny_jsonl(tmp_path: Path) -> str:
    """Three benchmark records with `tag` field and `source_hash` for
    evidence-kind truth-set registration."""
    records = [
        {
            "matches_hash": "111111111111111111",
            "source_hash": "-1000000000000000001",
            "stmt_type": "Phosphorylation",
            "subject": "MAP2K1",
            "object": "MAPK1",
            "evidence_text": "MAP2K1 phosphorylates MAPK1.",
            "tag": "correct",
        },
        {
            "matches_hash": "222222222222222222",
            "source_hash": "-1000000000000000002",
            "stmt_type": "Activation",
            "subject": "RAF1",
            "object": "MAP2K1",
            "evidence_text": "RAF1 activates MAP2K1.",
            "tag": "correct",
        },
        {
            "matches_hash": "333333333333333333",
            "source_hash": "-1000000000000000003",
            "stmt_type": "Activation",
            "subject": "KRAS",
            "object": "RAF1",
            "evidence_text": "KRAS activates RAF1.",
            "tag": "negative_result",
        },
    ]
    path = tmp_path / "tiny_bench.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return str(path)


def _parse_events(captured_stdout: str) -> list[dict]:
    """Parse newline-JSON events from the worker's stdout capture."""
    events: list[dict] = []
    for line in captured_stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            # Non-JSON content (shouldn't happen in passing tests, but be safe)
            pass
    return events


def _find_event(events: list[dict], kind: str) -> dict:
    matches = [e for e in events if e.get("event") == kind]
    assert matches, f"no event of kind {kind!r}; saw: {[e.get('event') for e in events]}"
    return matches[-1]


# ---------- estimate-cost -----------------------------------------------------


def test_estimate_cost_returns_per_model_estimates(
    tiny_indra_json: str, capsys: pytest.CaptureFixture[str]
):
    """estimate-cost emits one `done` event with a per-model estimates list.

    Required: every supported model in MODEL_PRICES_PER_M_TOKENS appears
    in the result with a non-negative cost_usd. Without this, the cost
    preflight panel on the dashboard renders empty / no rows.
    """
    args = argparse.Namespace(path=tiny_indra_json)
    rc = W.do_estimate_cost(args)
    assert rc == 0

    events = _parse_events(capsys.readouterr().out)
    done = _find_event(events, "done")
    assert done["n_statements"] == 3

    estimates = done["estimates"]
    assert isinstance(estimates, list)
    model_ids = {e["model_id"] for e in estimates}
    # The dashboard expects these to render in the preflight table.
    assert "claude-sonnet-4-6" in model_ids
    assert "claude-opus-4-7" in model_ids
    assert "gemini-2.5-flash" in model_ids
    for e in estimates:
        assert e["cost_usd"] >= 0
        assert e["n_stmts"] == 3
        assert e["n_evidences_est"] >= 3  # one evidence per stmt minimum
        assert e["n_llm_calls_est"] > 0


def test_estimate_cost_per_model_ordering_makes_sense(
    tiny_indra_json: str, capsys: pytest.CaptureFixture[str]
):
    """Opus should never be cheaper than Sonnet on the same workload —
    if this inverts, MODEL_PRICES_PER_M_TOKENS has drifted from
    actual provider pricing."""
    rc = W.do_estimate_cost(argparse.Namespace(path=tiny_indra_json))
    assert rc == 0
    events = _parse_events(capsys.readouterr().out)
    by_model = {e["model_id"]: e for e in _find_event(events, "done")["estimates"]}
    assert by_model["claude-opus-4-7"]["cost_usd"] > by_model["claude-sonnet-4-6"]["cost_usd"]
    assert by_model["claude-sonnet-4-6"]["cost_usd"] > by_model["claude-haiku-4-5"]["cost_usd"]
    assert by_model["gemini-2.5-pro"]["cost_usd"] > by_model["gemini-2.5-flash"]["cost_usd"]


# ---------- ingest ------------------------------------------------------------


def test_ingest_writes_statements_to_corpus(
    tmp_db: str, tiny_indra_json: str, capsys: pytest.CaptureFixture[str]
):
    """ingest verb emits `loaded` then `done` events and the DB ends up
    with the expected statement count."""
    args = argparse.Namespace(
        db=tmp_db, path=tiny_indra_json, source_dump_id="test_ingest"
    )
    rc = W.do_ingest(args)
    assert rc == 0

    events = _parse_events(capsys.readouterr().out)
    _find_event(events, "started")
    loaded = _find_event(events, "loaded")
    done = _find_event(events, "done")
    assert loaded["n_statements"] == 3
    assert done["n_statements"] == 3
    assert done["duration_s"] >= 0

    # Verify the DB rows
    con = duckdb.connect(tmp_db, read_only=True)
    try:
        n = con.execute(
            "SELECT COUNT(*) FROM statement WHERE source_dump_id='test_ingest'"
        ).fetchone()[0]
        assert n == 3
    finally:
        con.close()


def test_ingest_is_idempotent(
    tmp_db: str, tiny_indra_json: str, capsys: pytest.CaptureFixture[str]
):
    """Re-ingesting the same JSON should NOT duplicate rows. The U-phase
    score verb relies on this — it calls ingest+score on the same input
    even when [ingest] was already clicked."""
    args = argparse.Namespace(
        db=tmp_db, path=tiny_indra_json, source_dump_id="test_idem"
    )
    assert W.do_ingest(args) == 0
    capsys.readouterr()  # drain
    assert W.do_ingest(args) == 0  # second run

    con = duckdb.connect(tmp_db, read_only=True)
    try:
        n = con.execute("SELECT COUNT(*) FROM statement").fetchone()[0]
        # 3 statements, not 6.
        assert n == 3
    finally:
        con.close()


# ---------- register-truth-set -----------------------------------------------


def test_register_truth_set_writes_truth_labels(
    tmp_db: str, tiny_jsonl: str, capsys: pytest.CaptureFixture[str]
):
    args = argparse.Namespace(
        db=tmp_db,
        path=tiny_jsonl,
        truth_set_id="test_bench",
        truth_set_name="test benchmark",
        target_kind="evidence",
        field="tag",
        target_hash_field=None,
        recompute_latest_validity=False,
    )
    rc = W.do_register_truth_set(args)
    assert rc == 0

    events = _parse_events(capsys.readouterr().out)
    done = _find_event(events, "done")
    assert done["n_loaded"] == 3
    assert done["n_missing_target"] == 0
    assert done["n_missing_field"] == 0

    con = duckdb.connect(tmp_db, read_only=True)
    try:
        rows = con.execute(
            "SELECT target_kind, target_id, field, value_text "
            "FROM truth_label WHERE truth_set_id='test_bench' "
            "ORDER BY target_id"
        ).fetchall()
        assert len(rows) == 3
        for target_kind, _target_id, field, value_text in rows:
            assert target_kind == "evidence"
            assert field == "tag"
            assert value_text in {"correct", "negative_result"}
    finally:
        con.close()


def test_register_truth_set_is_idempotent(
    tmp_db: str, tiny_jsonl: str, capsys: pytest.CaptureFixture[str]
):
    """The viewer's [register tag as truth_set] button can be clicked
    twice; the second click must NOT duplicate rows or fail."""
    base = dict(
        db=tmp_db,
        path=tiny_jsonl,
        truth_set_id="test_idem",
        truth_set_name="test idem",
        target_kind="evidence",
        field="tag",
        target_hash_field=None,
        recompute_latest_validity=False,
    )
    assert W.do_register_truth_set(argparse.Namespace(**base)) == 0
    capsys.readouterr()
    assert W.do_register_truth_set(argparse.Namespace(**base)) == 0

    con = duckdb.connect(tmp_db, read_only=True)
    try:
        n = con.execute(
            "SELECT COUNT(*) FROM truth_label WHERE truth_set_id='test_idem'"
        ).fetchone()[0]
        assert n == 3
    finally:
        con.close()


def test_register_truth_set_handles_missing_field(
    tmp_db: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """Records without the configured field are counted as
    n_missing_field, not silently dropped. The dashboard surfaces this
    in the response's `done` event."""
    path = tmp_path / "missing_field.jsonl"
    path.write_text(json.dumps({
        "matches_hash": "1",
        "source_hash": "1",
        # no `tag` field
        "evidence_text": "no tag here",
    }) + "\n" + json.dumps({
        "matches_hash": "2",
        "source_hash": "2",
        "tag": "correct",
    }) + "\n")

    args = argparse.Namespace(
        db=tmp_db,
        path=str(path),
        truth_set_id="test_partial",
        truth_set_name="partial",
        target_kind="evidence",
        field="tag",
        target_hash_field=None,
        recompute_latest_validity=False,
    )
    assert W.do_register_truth_set(args) == 0
    events = _parse_events(capsys.readouterr().out)
    done = _find_event(events, "done")
    assert done["n_loaded"] == 1
    assert done["n_missing_field"] == 1


def test_register_truth_set_handles_missing_target_hash(
    tmp_db: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """Records without the configured target-hash field are counted as
    n_missing_target. Curator can re-run with --target-hash-field to
    redirect."""
    path = tmp_path / "missing_target.jsonl"
    path.write_text(json.dumps({
        "matches_hash": "1",
        # no `source_hash` field — required for evidence-kind
        "tag": "correct",
    }) + "\n")

    args = argparse.Namespace(
        db=tmp_db,
        path=str(path),
        truth_set_id="test_missing_target",
        truth_set_name="missing target",
        target_kind="evidence",
        field="tag",
        target_hash_field=None,
        recompute_latest_validity=False,
    )
    assert W.do_register_truth_set(args) == 0
    events = _parse_events(capsys.readouterr().out)
    done = _find_event(events, "done")
    assert done["n_loaded"] == 0
    assert done["n_missing_target"] == 1


# ---------- CLI dispatcher ----------------------------------------------------


def test_main_unknown_verb_returns_nonzero(capsys: pytest.CaptureFixture[str]):
    """Unknown / missing subcommands print help + return non-zero exit
    so the SvelteKit endpoint's spawn-error path triggers."""
    assert W.main([]) == 2
    out = capsys.readouterr()
    # argparse prints to stdout when --help is invoked, stderr when args invalid.
    # We just check we didn't silently succeed.


def test_main_dispatches_estimate_cost(
    tiny_indra_json: str, capsys: pytest.CaptureFixture[str]
):
    """End-to-end through the argv dispatcher — what the SvelteKit
    endpoint actually invokes when it spawns the worker."""
    rc = W.main(["estimate-cost", "--path", tiny_indra_json])
    assert rc == 0
    events = _parse_events(capsys.readouterr().out)
    _find_event(events, "done")

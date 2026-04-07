"""Corpus loader with native INDRA deserialization.

Builds a source_hash index over the benchmark corpus. Statements are
deserialized from JSON on demand via stmts_from_json(), avoiding the
cost of loading 894K Statement objects upfront.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from indra.statements import Statement, Evidence

from indra_belief.data.scoring_record import ScoringRecord

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CORPUS = ROOT / "data" / "benchmark" / "indra_benchmark_corpus.json.gz"


class CorpusIndex:
    """Source-hash index over the INDRA benchmark corpus.

    Stores lightweight JSON dicts. Deserializes to native INDRA objects
    on demand when a specific (source_hash, subject, object) is requested.
    """

    def __init__(self, corpus_path: str | Path = DEFAULT_CORPUS):
        self._corpus_path = Path(corpus_path)
        # source_hash → list of (stmt_json, evidence_index)
        self._index: dict[int, list[tuple[dict, int]]] = {}
        self._loaded = False

    def load(self) -> None:
        """Build the source_hash index. ~30s for 894K statements."""
        if self._loaded:
            return

        print(f"Loading corpus from {self._corpus_path} ...")
        with gzip.open(self._corpus_path, "rt", encoding="utf-8") as fh:
            corpus: list[dict] = json.load(fh)

        total = len(corpus)
        print(f"  {total:,} statements loaded. Building index ...")

        for i, stmt_json in enumerate(corpus):
            if (i + 1) % 100_000 == 0:
                print(f"  ... processed {i + 1:,} / {total:,}")

            for ei, ev in enumerate(stmt_json.get("evidence", [])):
                sh = ev.get("source_hash")
                if sh is None:
                    continue
                sh = int(sh)
                if sh not in self._index:
                    self._index[sh] = []
                self._index[sh].append((stmt_json, ei))

        self._loaded = True
        print(f"  Done. {len(self._index):,} source_hash entries.")

    def get(
        self,
        source_hash: int,
        subject: str,
        obj: str,
    ) -> tuple[Statement, Evidence] | None:
        """Find and deserialize the Statement + Evidence for a source_hash.

        Matches by subject/object names against the statement's agent list.
        Returns None if no match found.
        """
        from indra.statements import stmts_from_json

        self.load()
        entries = self._index.get(int(source_hash), [])
        if not entries:
            return None

        claim_set = {subject.lower(), obj.lower()} - {"?", ""}

        for stmt_json, ei in entries:
            stmts = stmts_from_json([stmt_json])
            if not stmts:
                continue
            stmt = stmts[0]
            agent_names = {a.name.lower() for a in stmt.agent_list() if a}
            if claim_set <= agent_names:
                # Exact match — both entities found
                if ei < len(stmt.evidence):
                    return stmt, stmt.evidence[ei]

        # Fallback: return first entry (may be wrong statement but
        # evidence text is shared across statements with same source_hash)
        stmt_json, ei = entries[0]
        stmts = stmts_from_json([stmt_json])
        if stmts and ei < len(stmts[0].evidence):
            return stmts[0], stmts[0].evidence[ei]

        return None

    def build_records(
        self,
        holdout_path: str | Path,
    ) -> list[ScoringRecord]:
        """Build ScoringRecords for an entire holdout file."""
        self.load()

        with open(holdout_path) as f:
            holdout = [json.loads(line) for line in f]

        records = []
        skipped = 0
        for h in holdout:
            result = self.get(h["source_hash"], h["subject"], h["object"])
            if result is None:
                skipped += 1
                continue
            stmt, ev = result
            records.append(ScoringRecord.from_holdout(h, stmt, ev))

        if skipped:
            print(f"  Warning: {skipped}/{len(holdout)} records not found in corpus")

        return records

    def __len__(self) -> int:
        self.load()
        return len(self._index)

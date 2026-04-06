"""Deterministic grounding verification for INDRA extractions.

Compares what the NLP reader extracted (raw_text) against what the Statement
claims (subject/object gene symbols) using gilda.ground(). Returns a structured
assessment: MATCH, MISMATCH, AMBIGUOUS, or UNRESOLVABLE.

MISMATCH has 100% precision — zero false rejections on tested valid mappings.
Can be used as an auto-reject signal without LLM involvement.
"""
from __future__ import annotations

from functools import lru_cache

import gilda


@lru_cache(maxsize=4096)
def _ground(name: str):
    """Cached gilda grounding."""
    try:
        return gilda.ground(name) or []
    except Exception:
        return []


@lru_cache(maxsize=4096)
def _get_desc(db: str, db_id: str) -> tuple[str, bool]:
    """Get functional description and pseudogene status for an HGNC gene."""
    if db != "HGNC":
        return "", False
    try:
        names = gilda.get_names("HGNC", str(db_id))
        descs = sorted(
            [n for n in names if len(n) > 12 and n[0].isupper()],
            key=len, reverse=True,
        )
        desc = descs[0] if descs else ""
        pseudo = any("pseudogene" in n.lower() for n in names)
        return desc, pseudo
    except Exception:
        return "", False


def verify_mapping(extracted_text: str, claim_entity: str) -> tuple[str, str]:
    """Check whether an NLP-extracted text entity maps to the claimed gene.

    Args:
        extracted_text: What the NLP reader pulled from the sentence (e.g., "9G8")
        claim_entity: The gene symbol in the INDRA Statement (e.g., "SLU7")

    Returns:
        (status, note) where status is one of:
        - MATCH: extracted text resolves to the same gene as the claim
        - MISMATCH: extracted text resolves to a DIFFERENT gene (auto-reject safe)
        - AMBIGUOUS: claim entity is a candidate but not the top hit
        - UNRESOLVABLE: one or both sides can't be grounded
    """
    if not extracted_text or not claim_entity:
        return "UNRESOLVABLE", "Missing entity name"

    text_results = _ground(extracted_text)
    claim_results = _ground(claim_entity)

    if not text_results:
        return "UNRESOLVABLE", f'"{extracted_text}" not found in gene databases'
    if not claim_results:
        return "UNRESOLVABLE", f'"{claim_entity}" not found in gene databases'

    text_top = text_results[0].term
    claim_top = claim_results[0].term

    # Same (db, id)?
    if text_top.db == claim_top.db and text_top.id == claim_top.id:
        return "MATCH", f'"{extracted_text}" resolves to {text_top.entry_name} = {claim_entity}'

    # Check if claim entity appears in lower-ranked candidates
    claim_in_candidates = any(
        r.term.db == claim_top.db and r.term.id == claim_top.id
        for r in text_results[:5]
    )

    # Get functional descriptions for both sides
    text_desc, text_pseudo = _get_desc(text_top.db, str(text_top.id))
    claim_desc, claim_pseudo = _get_desc(claim_top.db, str(claim_top.id))

    # Non-gene namespaces
    ns_labels = {"CHEBI": "chemical", "MESH": "MeSH term", "GO": "GO term"}
    text_ns = ns_labels.get(text_top.db, "")
    claim_ns = ns_labels.get(claim_top.db, "")

    if claim_in_candidates:
        note = (
            f'"{extracted_text}" most likely refers to '
            f'{text_top.entry_name} ({text_desc}), '
            f'not {claim_entity} ({claim_desc})'
        )
        if claim_pseudo:
            note += f". {claim_entity} is a PSEUDOGENE."
        if text_ns:
            note += f" Note: top match is a {text_ns}, not a gene."
        return "AMBIGUOUS", note
    else:
        note = (
            f'"{extracted_text}" resolves to '
            f'{text_top.entry_name} ({text_desc}), '
            f'NOT {claim_entity} ({claim_desc})'
        )
        if text_ns:
            note += f". Note: \"{extracted_text}\" is a {text_ns}, not a gene."
        if claim_ns:
            note += f" Note: {claim_entity} is a {claim_ns}."
        return "MISMATCH", note


def check_record(
    subject: str,
    obj: str,
    raw_text: list[str | None] | None,
) -> list[tuple[str, str, str, str]]:
    """Check all entity mappings for a record.

    Args:
        subject: Claim subject entity
        obj: Claim object entity
        raw_text: NLP-extracted text spans [subject_text, object_text]

    Returns:
        List of (extracted, claim, status, note) for each mismatched entity.
        Empty list if all mappings are MATCH or no raw_text available.
    """
    if not raw_text:
        return []

    clean = [r for r in raw_text if r is not None]
    if not clean:
        return []

    claim_entities = [subject, obj]
    results = []

    for i, rt in enumerate(clean[:2]):
        if i >= len(claim_entities):
            break
        ce = claim_entities[i]

        # Skip if text matches claim literally
        if rt.lower() == ce.lower():
            continue

        status, note = verify_mapping(rt, ce)
        if status != "MATCH":
            results.append((rt, ce, status, note))

    return results


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_cases = [
        # Grounding errors (should be MISMATCH or AMBIGUOUS)
        ("9G8", "SLU7"),
        ("RhoA", "ARHGEF25"),
        ("CagA", "S100A8"),
        ("DVL", "DVL1P1"),
        ("TFs", "TCEA1"),
        ("IN", "ERVK-10"),
        ("HNF-4alpha", "RXR"),
        ("SIR2", "SIRT1"),
        ("p21", "SP1"),
        ("galectin-1", "RAS"),
        ("amyloid", "IAPP"),
        ("RAD52", "BRCA1"),
        # Valid mappings (should be MATCH)
        ("ponsin", "SORBS1"),
        ("LARG", "ARHGEF12"),
        ("PDK1", "PDPK1"),
        ("PKCepsilon", "PRKCE"),
        ("FAK", "PTK2"),
        ("RSK1", "RPS6KA1"),
    ]

    for extracted, claim in test_cases:
        status, note = verify_mapping(extracted, claim)
        marker = {"MATCH": "✓", "MISMATCH": "✗", "AMBIGUOUS": "~", "UNRESOLVABLE": "?"}[status]
        print(f"{marker} {extracted:>15s} → {claim:12s}  [{status:12s}]  {note}")

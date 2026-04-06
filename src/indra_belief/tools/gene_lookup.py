"""Gene name grounding service for LLM belief scorer.

Uses gilda (Gyori lab's biomedical entity grounder) for robust resolution.
This is the same grounding system INDRA uses internally.

Provides:
- lookup_gene(name) → structured grounding info (HGNC/FPLX/other)
- find_entities_in_text(text) → all entities with groundings
- entities_match(claim_entity, text) → does claim entity appear in text?
"""
from __future__ import annotations

from functools import lru_cache

import gilda


@lru_cache(maxsize=4096)
def lookup_gene(name: str) -> dict:
    """Look up a gene/protein name via gilda.

    Returns:
      {
        "query": original name,
        "resolved": bool (True if high-confidence grounding found),
        "canonical_name": name from the grounding,
        "namespace": "HGNC" | "FPLX" | "UP" | "MESH" | etc.
        "identifier": db-specific id,
        "score": gilda confidence score,
        "match_type": "gene" (HGNC), "family" (FPLX), "other" (MESH etc.),
        "alternatives": list of alternative groundings,
        "note": explanation,
      }
    """
    if not name or name.strip() in ("", "?"):
        return {
            "query": name,
            "resolved": False,
            "canonical_name": None,
            "namespace": None,
            "identifier": None,
            "score": 0.0,
            "match_type": "empty",
            "alternatives": [],
            "note": "Empty or placeholder entity name",
        }

    try:
        results = gilda.ground(name)
    except Exception as e:
        return {
            "query": name,
            "resolved": False,
            "canonical_name": None,
            "namespace": None,
            "identifier": None,
            "score": 0.0,
            "match_type": "error",
            "alternatives": [],
            "note": f"Grounding error: {e}",
        }

    if not results:
        return {
            "query": name,
            "resolved": False,
            "canonical_name": None,
            "namespace": None,
            "identifier": None,
            "score": 0.0,
            "match_type": "unresolved",
            "alternatives": [],
            "note": f"No grounding found for '{name}'",
        }

    top = results[0]
    namespace = top.term.db
    identifier = top.term.id
    canonical = top.term.entry_name
    score = top.score

    # Determine match_type from namespace
    if namespace == "HGNC":
        match_type = "gene"
    elif namespace == "FPLX":
        match_type = "family"
    elif namespace == "UP":
        match_type = "protein"
    elif namespace == "MESH":
        match_type = "mesh"
    elif namespace == "CHEBI":
        match_type = "chemical"
    else:
        match_type = namespace.lower()

    # Low-score results may be unreliable
    resolved = score >= 0.5

    # Alternatives: other high-scoring groundings
    alternatives = []
    for r in results[1:4]:  # top 3 alts
        alternatives.append({
            "namespace": r.term.db,
            "identifier": r.term.id,
            "canonical_name": r.term.entry_name,
            "score": round(r.score, 3),
        })

    # Note: is this ambiguous?
    is_ambiguous = len(results) > 1 and results[1].score >= 0.5

    note = f"Grounded to {namespace}:{identifier} ({canonical}), score={score:.2f}"
    if match_type == "family":
        note += " [protein family]"
    if is_ambiguous:
        note += f" [ambiguous — alternatives: {', '.join(a['canonical_name'] for a in alternatives[:2])}]"

    return {
        "query": name,
        "resolved": resolved,
        "canonical_name": canonical,
        "namespace": namespace,
        "identifier": identifier,
        "score": round(score, 3),
        "match_type": match_type,
        "alternatives": alternatives,
        "note": note,
        "is_ambiguous": is_ambiguous,
    }


@lru_cache(maxsize=2048)
def find_entities_in_text(text: str) -> tuple[tuple, ...]:
    """Extract all gene/protein/entity mentions from text with groundings.

    Returns tuple of (span_text, namespace, identifier, canonical_name, score).
    Using tuple to allow lru_cache.
    """
    if not text:
        return ()
    try:
        annotations = gilda.annotate(text)
    except Exception:
        return ()

    results = []
    for a in annotations:
        if not a.matches:
            continue
        top = a.matches[0]
        results.append((
            a.text,
            top.term.db,
            top.term.id,
            top.term.entry_name,
            round(top.score, 3),
        ))
    return tuple(results)


def entities_match(claim_entity: dict, text: str) -> dict:
    """Check if claim's grounded entity appears in text.

    Uses gilda.annotate to find all entities in text, then checks if any
    match the claim entity's grounding (by identifier or canonical name).

    Returns:
      {"match": bool | None, "via": str, "note": str}
      match=None if claim entity can't be verified (placeholder, unresolved)
    """
    # Handle placeholder entities
    if claim_entity.get("match_type") == "empty":
        return {
            "match": None,
            "via": "unspecified",
            "note": "Claim entity is unspecified (?) — cannot verify",
        }

    if not claim_entity.get("resolved"):
        # Unresolved claim entity — can't check precisely
        query = claim_entity.get("query", "")
        # Simple substring check as fallback
        if query and query.lower() in text.lower():
            return {
                "match": True,
                "via": "literal_substring",
                "note": f"Unresolved entity '{query}' appears literally in text",
            }
        return {
            "match": False,
            "via": None,
            "note": f"Unresolved entity '{query}' not in text",
        }

    # Get all entities in text
    text_entities = find_entities_in_text(text)

    claim_canonical = claim_entity["canonical_name"]
    claim_id = claim_entity["identifier"]
    claim_namespace = claim_entity["namespace"]

    # Check for ID match (most reliable)
    for span_text, ns, ident, canonical, score in text_entities:
        if ns == claim_namespace and ident == claim_id:
            return {
                "match": True,
                "via": "grounding_id",
                "note": f"Text mentions '{span_text}' → {ns}:{ident} ({canonical}) matching claim",
            }

    # Fallback: check string forms (query, canonical, alternatives) against text
    # gilda.annotate may miss entities that gilda.ground resolves fine
    text_lower = text.lower()
    text_collapsed = text_lower.replace("-", "").replace(" ", "").replace("_", "")

    def text_has(needle: str) -> bool:
        if not needle or len(needle) < 3:
            return False
        nl = needle.lower()
        if nl in text_lower:
            return True
        nc = nl.replace("-", "").replace(" ", "").replace("_", "")
        return nc and nc in text_collapsed

    # Try canonical name
    if text_has(claim_canonical):
        return {
            "match": True,
            "via": "canonical_literal",
            "note": f"Canonical '{claim_canonical}' found via string match",
        }

    # Try the original query (user's name) if different from canonical
    query = claim_entity.get("query", "")
    if query != claim_canonical and text_has(query):
        return {
            "match": True,
            "via": "query_literal",
            "note": f"Query form '{query}' found via string match",
        }

    # For genes/proteins, try looking up each text token via gilda
    # to catch cases where annotate() missed them but ground() would resolve
    import gilda
    if claim_namespace in ("HGNC", "UP"):
        import re
        # Generate candidates: individual words AND hyphen-joined compounds AND sub-parts
        candidates = set()
        # Individual alphanumeric tokens
        for m in re.finditer(r'[A-Za-z][A-Za-z0-9]{2,20}', text):
            candidates.add(m.group())
        # Hyphen-separated: "TRAIL-induced" → "TRAIL" + "induced" + "TRAIL-induced"
        for m in re.finditer(r'[A-Za-z][A-Za-z0-9\-]{2,20}', text):
            compound = m.group()
            candidates.add(compound)
            for part in compound.split("-"):
                if len(part) >= 3:
                    candidates.add(part)
        for cand in candidates:
            try:
                results = gilda.ground(cand)
                if results and results[0].score >= 0.7:
                    top = results[0]
                    if top.term.db == claim_namespace and top.term.id == claim_id:
                        return {
                            "match": True,
                            "via": "token_grounding",
                            "note": f"Token '{cand}' grounds to {claim_namespace}:{claim_id}",
                        }
            except Exception:
                pass

    # For FamPlex families: literal family name may appear
    if claim_namespace == "FPLX":
        if text_has(claim_canonical):
            return {
                "match": True,
                "via": "family_literal",
                "note": f"Family name '{claim_canonical}' in text",
            }

    return {
        "match": False,
        "via": None,
        "note": (
            f"Claim entity {claim_namespace}:{claim_id} ({claim_canonical}) "
            f"not found via annotate or string/token matching"
        ),
    }

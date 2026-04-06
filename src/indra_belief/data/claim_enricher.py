"""Claim enrichment utilities for the LLM-based INDRA evidence scorer.

Two capabilities:

1. **Claim enrichment via Statement metadata** -- surfaces hidden Statement
   fields (residue, position, mutations, bound conditions) that the LLM
   needs to verify a text-mining extraction but currently cannot see.

2. **Alias context via gilda** -- resolves entity names to canonical forms
   and provides synonyms / family members so the LLM can recognize that
   e.g. LARG = ARHGEF12.

Usage as a module::

    from experiments.belief_benchmark.claim_enricher import (
        build_corpus_index,
        enrich_claim,
        format_entity_context,
    )

    index = build_corpus_index("data/benchmark/indra_benchmark_corpus.json.gz")
    claim = enrich_claim("AURKB", "Phosphorylation", "ATXN10", 2197027780787608736, index)
    ctx = format_entity_context("LARG", "NXF1")
"""
from __future__ import annotations

import gzip
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = ROOT / "data" / "benchmark" / "indra_benchmark_corpus.json.gz"

# Modification statement types that can carry residue/position
_MODIFICATION_TYPES = frozenset({
    "Phosphorylation", "Dephosphorylation",
    "Ubiquitination", "Deubiquitination",
    "Sumoylation", "Desumoylation",
    "Acetylation", "Deacetylation",
    "Hydroxylation", "Dehydroxylation",
    "Glycosylation", "Deglycosylation",
    "Ribosylation", "Deribosylation",
    "Methylation", "Demethylation",
    "Palmitoylation", "Depalmitoylation",
    "Myristoylation", "Demyristoylation",
    "Farnesylation", "Defarnesylation",
    "Geranylgeranylation", "Degeranylgeranylation",
    "Autophosphorylation",
    "Transphosphorylation",
})


# ---------------------------------------------------------------------------
# 1. Corpus index
# ---------------------------------------------------------------------------

def build_corpus_index(
    corpus_path: str | Path = DEFAULT_CORPUS,
) -> dict[int, dict]:
    """Build a source_hash -> Statement mapping from the benchmark corpus.

    For each Statement, for each evidence entry, the evidence's
    ``source_hash`` is mapped to a copy of the Statement dict with the
    ``evidence`` list stripped out (to save memory).

    Parameters
    ----------
    corpus_path
        Path to the gzipped JSON corpus file.

    Returns
    -------
    dict
        Mapping from ``int(source_hash)`` to statement dict (sans evidence).
    """
    corpus_path = Path(corpus_path)
    print(f"Loading corpus from {corpus_path} ...")

    with gzip.open(corpus_path, "rt", encoding="utf-8") as fh:
        statements: list[dict] = json.load(fh)

    total = len(statements)
    print(f"  {total:,} statements loaded. Building source_hash index ...")

    index: dict[int, dict] = {}
    report_every = 100_000

    for i, stmt in enumerate(statements):
        if (i + 1) % report_every == 0:
            print(f"  ... processed {i + 1:,} / {total:,} statements")

        evidence_list = stmt.get("evidence", [])
        if not evidence_list:
            continue

        # Build a lightweight copy without the evidence list
        stmt_light: dict[str, Any] = {
            k: v for k, v in stmt.items() if k != "evidence"
        }

        for ev in evidence_list:
            sh = ev.get("source_hash")
            if sh is not None:
                index[int(sh)] = stmt_light

    print(f"  Done. Index contains {len(index):,} source_hash entries.")
    return index


def build_corpus_index_v8(
    corpus_path: str | Path = DEFAULT_CORPUS,
) -> dict[str, dict]:
    """Build Statement + evidence-level indexes in a single corpus pass.

    Returns dict with two keys:
      - "statements": source_hash → stmt_light (same as build_corpus_index)
      - "evidence_meta": source_hash → {"raw_text": [...], "direct": bool|None}
    """
    corpus_path = Path(corpus_path)
    print(f"Loading corpus from {corpus_path} ...")

    with gzip.open(corpus_path, "rt", encoding="utf-8") as fh:
        statements: list[dict] = json.load(fh)

    total = len(statements)
    print(f"  {total:,} statements loaded. Building v8 indexes ...")

    stmt_index: dict[int, dict] = {}
    ev_meta: dict[int, dict] = {}
    report_every = 100_000

    for i, stmt in enumerate(statements):
        if (i + 1) % report_every == 0:
            print(f"  ... processed {i + 1:,} / {total:,} statements")

        evidence_list = stmt.get("evidence", [])
        if not evidence_list:
            continue

        stmt_light: dict[str, Any] = {
            k: v for k, v in stmt.items() if k != "evidence"
        }

        for ev in evidence_list:
            sh = ev.get("source_hash")
            if sh is None:
                continue
            sh = int(sh)
            stmt_index[sh] = stmt_light

            # Extract evidence-level metadata
            agents = ev.get("annotations", {}).get("agents", {})
            raw_text = agents.get("raw_text")  # list of extracted text spans
            epistemics = ev.get("epistemics", {})
            direct = epistemics.get("direct")  # True, False, or None

            ev_meta[sh] = {
                "raw_text": raw_text,
                "direct": direct,
            }

    print(f"  Done. {len(stmt_index):,} statement entries, {len(ev_meta):,} evidence entries.")
    return {"statements": stmt_index, "evidence_meta": ev_meta}


def _gilda_resolves_to(text_form: str, claim_entity: str) -> bool:
    """Check if gilda resolves a text form to the same entity as the claim."""
    try:
        import gilda
        # Ground the text form
        text_results = gilda.ground(text_form)
        if not text_results:
            return False
        # Ground the claim entity
        claim_results = gilda.ground(claim_entity)
        if not claim_results:
            return False
        # Match if same (db, id)
        t = text_results[0].term
        c = claim_results[0].term
        return t.db == c.db and t.id == c.id
    except Exception:
        return False


@lru_cache(maxsize=4096)
def _validated_alias(text_form: str, claim_entity: str) -> bool:
    """Cached check: does text_form resolve to the same gene as claim_entity?"""
    if not text_form or not claim_entity:
        return False
    # Exact or substring match = obviously fine
    if text_form.lower() == claim_entity.lower():
        return True
    if text_form.lower() in claim_entity.lower() or claim_entity.lower() in text_form.lower():
        return True
    # Ask gilda
    return _gilda_resolves_to(text_form, claim_entity)


def get_extraction_context(
    subject: str,
    obj: str,
    source_hash: int,
    evidence_meta: dict[int, dict],
) -> str:
    """Format extraction context showing what INDRA actually extracted.

    Only fires when gilda CANNOT resolve the extracted text span to the
    claim entity — i.e., a genuine grounding mismatch, not just a standard
    alias like RSK1→RPS6KA1.
    """
    meta = evidence_meta.get(int(source_hash))
    if meta is None:
        return ""

    raw_text = meta.get("raw_text")
    if not raw_text or not isinstance(raw_text, list):
        return ""

    raw_text = [r for r in raw_text if r is not None]
    if not raw_text:
        return ""

    # Check each extracted span against its corresponding claim entity
    claim_entities = [subject, obj]
    mismatches = []

    for i, rt in enumerate(raw_text[:2]):
        if i >= len(claim_entities):
            break
        ce = claim_entities[i]
        if _validated_alias(rt, ce):
            continue  # gilda confirms this is a valid alias — skip
        mismatches.append((rt, ce))

    if not mismatches:
        return ""  # all spans resolve to claim entities — no context needed

    # Only report mismatches where the extracted text resolves to a DIFFERENT
    # gene/protein than the claim entity. Skip cases where gilda can't resolve
    # the text (ambiguous short names, abbreviations) — these are noise.
    import gilda
    severe = []
    for rt, ce in mismatches:
        try:
            rt_results = gilda.ground(rt)
            ce_results = gilda.ground(ce)
        except Exception:
            continue
        if not rt_results or not ce_results:
            continue  # can't resolve one side — skip
        rt_top = rt_results[0].term
        ce_top = ce_results[0].term
        # Same namespace, different ID = definitive mismatch
        if rt_top.db == ce_top.db and rt_top.id != ce_top.id:
            severe.append((rt, ce, rt_top.entry_name))
        # Different namespace entirely (e.g., MESH vs HGNC) = likely mismatch
        elif rt_top.db != ce_top.db:
            severe.append((rt, ce, f"{rt_top.db}:{rt_top.entry_name}"))

    if not severe:
        return ""

    parts = []
    for rt, ce, resolved_to in severe:
        parts.append(f'"{rt}" (resolves to {resolved_to}, mapped to {ce})')

    return "Extraction mismatch: " + ", ".join(parts)


def get_evidence_directness(
    source_hash: int,
    evidence_meta: dict[int, dict],
) -> bool | None:
    """Return epistemics.direct for this evidence.

    True = direct experimental evidence
    False = indirect/review/computational
    None = unknown
    """
    meta = evidence_meta.get(int(source_hash))
    if meta is None:
        return None
    return meta.get("direct")


# ---------------------------------------------------------------------------
# 2. Claim enrichment
# ---------------------------------------------------------------------------

def _format_mutation(mut: dict) -> str:
    """Format a single INDRA mutation dict as a human-readable string.

    Returns empty string if all fields are None/empty (INDRA sometimes
    stores placeholder mutation dicts with all-None values).
    """
    res_from = mut.get("residue_from") or ""
    pos = mut.get("position") or ""
    res_to = mut.get("residue_to") or ""
    if res_from and res_to:
        return f"{res_from}{pos}{res_to}"
    if res_from and pos:
        return f"{res_from}{pos}"
    if pos:
        return str(pos)
    return ""


def _agent_annotations(agent: dict | None) -> str:
    """Return parenthetical annotations for an agent's metadata."""
    if not agent:
        return ""

    parts: list[str] = []

    # Mutations
    mutations = agent.get("mutations")
    if mutations:
        mut_strs = [s for s in (_format_mutation(m) for m in mutations) if s]
        if mut_strs:
            parts.append(f"mutation: {', '.join(mut_strs)}")

    # Bound conditions
    bound = agent.get("bound_conditions")
    if bound:
        partners = []
        for bc in bound:
            partner_agent = bc.get("agent", {})
            name = partner_agent.get("name", "?")
            is_bound = bc.get("is_bound", True)
            if is_bound:
                partners.append(name)
            else:
                partners.append(f"not {name}")
        parts.append(f"bound to: {', '.join(partners)}")

    # Activity
    activity = agent.get("activity")
    if activity:
        act_type = activity.get("activity_type", "activity")
        is_active = activity.get("is_active", True)
        qualifier = "" if is_active else "inactive "
        parts.append(f"{qualifier}{act_type}")

    # Location
    location = agent.get("location")
    if location:
        parts.append(f"location: {location}")

    if not parts:
        return ""
    return f" ({'; '.join(parts)})"


def enrich_claim(
    subject: str,
    stmt_type: str,
    obj: str,
    source_hash: int | str,
    corpus_index: dict[int, dict],
) -> str:
    """Build an enriched claim string by surfacing Statement metadata.

    Parameters
    ----------
    subject, stmt_type, obj
        The bare claim components (e.g. "AURKB", "Phosphorylation", "ATXN10").
    source_hash
        Evidence source_hash used to look up the Statement.
    corpus_index
        Mapping from source_hash to Statement dict, as built by
        :func:`build_corpus_index`.

    Returns
    -------
    str
        Enriched claim, e.g. ``"AURKB [Phosphorylation] ATXN10 @S77"``.
    """
    sh = int(source_hash)
    stmt = corpus_index.get(sh)

    if stmt is None:
        return f"{subject} [{stmt_type}] {obj}"

    # --- Agent annotations ---
    # Modification types use enz/sub; regulation types use subj/obj
    if stmt_type in _MODIFICATION_TYPES:
        enz_agent = stmt.get("enz")
        sub_agent = stmt.get("sub")
        subj_ann = _agent_annotations(enz_agent)
        obj_ann = _agent_annotations(sub_agent)
    else:
        subj_agent = stmt.get("subj")
        obj_agent = stmt.get("obj")
        subj_ann = _agent_annotations(subj_agent)
        obj_ann = _agent_annotations(obj_agent)

    claim = f"{subject}{subj_ann} [{stmt_type}] {obj}{obj_ann}"

    # --- Residue / position for modification types ---
    if stmt_type in _MODIFICATION_TYPES:
        residue = stmt.get("residue")
        position = stmt.get("position")
        if residue and position:
            claim += f" @{residue}{position}"
        elif residue:
            claim += f" @{residue}"
        elif position:
            claim += f" @{position}"

    return claim


# ---------------------------------------------------------------------------
# 3. Alias context via gilda
# ---------------------------------------------------------------------------

# Aliases that are common abbreviations, domain names, or highly ambiguous.
# These cause false matches when they appear as substrings of unrelated terms.
_AMBIGUOUS_ALIASES = frozenset({
    "AF-1", "AF1", "AF-2", "AF2",  # activation function domains
    "CD", "Antigen",  # too generic
    "PI",  # phosphatidylinositol, not insulin
    "HR",  # hormone receptor domain
    "NR",  # nuclear receptor
    "AD",  # activation domain
    "BD",  # binding domain
    "KD",  # kinase domain / knockdown
    "TF",  # transcription factor (generic)
    "Receptor", "Receptors",  # too generic for aliases
    "Protein", "Ligand",  # too generic
})


def _filter_aliases(aliases: list[str], entity_name: str, canonical: str) -> list[str]:
    """Filter aliases to keep only informative, unambiguous ones.

    Priority: short symbol-like aliases first, then longer descriptive ones.
    Allows lowercase-starting names if they look like protein identifiers
    (e.g., p70S6K, p105, ponsin).
    """
    candidates = []
    for a in aliases:
        if a == canonical or a == entity_name:
            continue
        if a in _AMBIGUOUS_ALIASES:
            continue
        if len(a) <= 1:
            continue
        # Skip generic English words
        a_lower = a.lower()
        if a_lower in ("antigen", "protein", "receptor", "ligand", "factor",
                        "kinase", "enzyme", "inhibitor", "substrate"):
            continue
        # Skip multi-word descriptive names (>= 2 spaces)
        if a.count(" ") >= 2:
            continue
        # Skip very long names
        if len(a) > 20:
            continue
        # Score: shorter symbol-like names rank higher
        # Uppercase or starts with lowercase p/c (protein convention) = good
        is_symbol = len(a) <= 10 and a.count(" ") == 0
        score = 0
        if is_symbol:
            score = 100 - len(a)  # shorter symbols ranked first
        else:
            score = 50 - len(a)
        candidates.append((score, a))

    # Sort by score (descending), take top 6
    candidates.sort(key=lambda x: -x[0])
    return [a for _, a in candidates[:6]]


@lru_cache(maxsize=2048)
def get_alias_context(entity_name: str) -> str:
    """Resolve an entity name via gilda and return a context string.

    Returns
    -------
    str
        One of:
        - ``"entity (HGNC: canonical, aliases: alias1, alias2)"``
        - ``"entity (family: canonical — if text names a specific member like X, that is more precise than the family name)"``
        - ``""`` if unresolved or trivially obvious.
    """
    import gilda  # deferred to avoid import-time cost

    if not entity_name or entity_name == "?":
        return ""

    matches = gilda.ground(entity_name)
    if not matches:
        return ""

    top = matches[0]
    db = top.term.db
    db_id = str(top.term.id)
    canonical = top.term.entry_name

    if db == "HGNC":
        # Get all synonyms for this gene
        all_names = gilda.get_names("HGNC", db_id)
        aliases = _filter_aliases(all_names, entity_name, canonical)
        if not aliases and canonical == entity_name:
            return ""
        parts = f"{entity_name} (HGNC: {canonical}"
        if aliases:
            parts += f", aliases: {', '.join(aliases)}"
        parts += ")"
        return parts

    if db == "FPLX":
        # Protein family — warn about specificity
        members = _get_fplx_members(db_id)
        parts = f"{entity_name} (family: {canonical}"
        if members:
            parts += f", members: {', '.join(members[:6])}"
        # Add specificity warning
        parts += " — if text names a specific member, the claim should use that member, not the family"
        parts += ")"
        return parts

    # Other namespace (CHEBI, MESH, etc.) -- not useful for gene scoring
    return ""


def _get_fplx_members(fplx_id: str) -> list[str]:
    """Return gene-symbol names of direct children of a FamPlex family."""
    try:
        from indra.ontology.bio import bio_ontology
        children = bio_ontology.get_children("FPLX", fplx_id)
        names = []
        for child_db, child_id in children:
            name = bio_ontology.get_name(child_db, child_id)
            if name:
                names.append(name)
        return sorted(names)
    except Exception:
        return []


def format_entity_context(subject: str, obj: str) -> str:
    """Build a one-line entity context string for both claim entities.

    Parameters
    ----------
    subject, obj
        Entity names from the claim.

    Returns
    -------
    str
        Context line like ``"Entities: LARG (HGNC: ARHGEF12, aliases: LARG)
        | NXF1 (HGNC: NXF1, aliases: TAP, Mex67)"``, or empty string if
        neither entity yields useful context.
    """
    subj_ctx = get_alias_context(subject)
    obj_ctx = get_alias_context(obj)

    # Deduplicate when both entities are the same
    if subject == obj:
        parts = [subj_ctx] if subj_ctx else []
    else:
        parts = [p for p in (subj_ctx, obj_ctx) if p]

    if not parts:
        return ""
    return "Entities: " + " | ".join(parts)


# ---------------------------------------------------------------------------
# __main__ -- build index and run smoke tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Build corpus index
    # ------------------------------------------------------------------
    index = build_corpus_index()

    # ------------------------------------------------------------------
    # Test 1: AURKB / ATXN10 (Phosphorylation @S77)
    # ------------------------------------------------------------------
    # source_hash 2197027780787608736 maps to the S77 statement via the
    # top-level (no-residue) statement's evidence list, which shares
    # hashes with the site-specific statement.  But in the corpus the
    # evidence list belongs to whichever Statement it's attached to.
    # We look for a source_hash that resolves to the @S77 variant.

    print("\n=== Test: AURKB / ATXN10 Phosphorylation ===")
    # Try multiple known hashes from the AURKB/ATXN10 evidence entries
    test_hashes = [
        2197027780787608736,
        -8228455341112843992,
        6806114382927229841,
        -4457023483705503340,
        -880497081637330206,
        -5770620258845694030,
    ]
    for sh in test_hashes:
        stmt = index.get(sh)
        if stmt:
            residue = stmt.get("residue")
            position = stmt.get("position")
            claim = enrich_claim("AURKB", "Phosphorylation", "ATXN10", sh, index)
            print(f"  hash={sh}: residue={residue}, position={position}")
            print(f"    -> {claim}")

    # ------------------------------------------------------------------
    # Test 2: Activation (should show no extra metadata)
    # ------------------------------------------------------------------
    print("\n=== Test: Activation (no extra metadata expected) ===")
    test_activation_hash = 5768355959642762169  # TNFSF10 -> CASP3
    claim = enrich_claim("TNFSF10", "Activation", "CASP3", test_activation_hash, index)
    print(f"  hash={test_activation_hash}")
    print(f"    -> {claim}")

    # ------------------------------------------------------------------
    # Test 3: Entity alias context
    # ------------------------------------------------------------------
    print("\n=== Test: Alias context ===")
    ctx = format_entity_context("LARG", "NXF1")
    print(f"  LARG / NXF1: {ctx}")

    ctx2 = format_entity_context("ERK", "MAPK1")
    print(f"  ERK / MAPK1: {ctx2}")

    ctx3 = format_entity_context("TP53", "TP53")
    print(f"  TP53 / TP53: {ctx3}")

    # ------------------------------------------------------------------
    # Test 4: Enrichment with agent-level metadata
    # ------------------------------------------------------------------
    print("\n=== Test: Agent-level metadata scan ===")
    found_examples = {"mutation": False, "bound": False}
    for sh, stmt in index.items():
        if found_examples["mutation"] and found_examples["bound"]:
            break
        for agent_key in ("enz", "sub", "subj", "obj"):
            agent = stmt.get(agent_key)
            if not agent:
                continue
            if agent.get("mutations") and not found_examples["mutation"]:
                found_examples["mutation"] = True
                stype = stmt.get("type", "?")
                # Figure out subject/object names
                subj_name = (stmt.get("enz") or stmt.get("subj") or {}).get("name", "?")
                obj_name = (stmt.get("sub") or stmt.get("obj") or {}).get("name", "?")
                claim = enrich_claim(subj_name, stype, obj_name, sh, index)
                print(f"  Mutation example: {claim}")
            if agent.get("bound_conditions") and not found_examples["bound"]:
                found_examples["bound"] = True
                stype = stmt.get("type", "?")
                subj_name = (stmt.get("enz") or stmt.get("subj") or {}).get("name", "?")
                obj_name = (stmt.get("sub") or stmt.get("obj") or {}).get("name", "?")
                claim = enrich_claim(subj_name, stype, obj_name, sh, index)
                print(f"  Bound example: {claim}")

    print("\nDone.")

"""Gilda grounding tools for agentic LLM evidence scoring.

Provides tool declarations (Gemma 4 native format) and executors
for gilda.ground() and gilda.get_names(), registered as callable
tools for the ModelClient.call_with_tools() interface.
"""
from __future__ import annotations

import gilda


# ---------------------------------------------------------------------------
# Gemma 4 native tool declarations
# ---------------------------------------------------------------------------

TOOL_DECLARATIONS = """<|tool>declaration:lookup_gene{description:<|"|>Look up a gene or protein name to find candidate identities. Returns top matches with database IDs and scores. Use this when the evidence text mentions an entity name that differs from the claim entity and you need to verify whether they refer to the same gene.<|"|>,parameters:{properties:{entity_name:{description:<|"|>The gene or protein name to look up, exactly as it appears in the text<|"|>,type:<|"|>STRING<|"|>}},required:[<|"|>entity_name<|"|>],type:<|"|>OBJECT<|"|>}}<tool|>"""


# ---------------------------------------------------------------------------
# Tool executors
# ---------------------------------------------------------------------------

def execute_lookup_gene(args: dict) -> dict:
    """Execute gilda.ground() and return enriched results with functional descriptions."""
    name = args.get("entity_name") or args.get("name") or ""
    name = name.strip().strip("'\"")
    if not name:
        return {"error": "No entity name provided"}

    results = gilda.ground(name)
    if not results:
        return {"entity": name, "candidates": [], "note": "No matches found"}

    candidates = []
    for r in results[:4]:
        entry = {
            "db": r.term.db,
            "id": r.term.id,
            "name": r.term.entry_name,
            "score": round(r.score, 3),
        }

        # Enrich with functional description + alias provenance
        if r.term.db == "HGNC":
            try:
                all_names = gilda.get_names("HGNC", str(r.term.id))
                # Functional description (longest descriptive name)
                descs = sorted(
                    [n for n in all_names if len(n) > 12 and n != r.term.entry_name
                     and not n[0].islower()],
                    key=len, reverse=True,
                )
                if descs:
                    entry["description"] = descs[0]
                # Pseudogene detection
                if any("pseudogene" in n.lower() for n in all_names):
                    entry["pseudogene"] = True
                # Is the query a known alias for this candidate?
                entry["query_is_alias"] = name in all_names
                entry["alias_count"] = len([n for n in all_names if len(n) < 12])
            except Exception:
                pass

        candidates.append(entry)

    return {"entity": name, "candidates": candidates}


def format_tool_result(result: dict) -> str:
    """Format tool result with functional descriptions for disambiguation."""
    if "error" in result:
        return f"Error: {result['error']}"

    entity = result.get("entity", "?")
    candidates = result.get("candidates", [])

    if not candidates:
        return f'lookup_gene("{entity}"): No matches found in gene databases.'

    lines = [f'lookup_gene("{entity}") candidates:']
    for i, c in enumerate(candidates):
        desc = c.get("description", "")
        pseudo = c.get("pseudogene", False)
        query_is_alias = c.get("query_is_alias")
        alias_count = c.get("alias_count", 0)
        db = c["db"]

        parts = [f"  [{i}] {c['name']}"]
        if desc:
            parts.append(f" — {desc}")
        if pseudo:
            parts.append(" [PSEUDOGENE]")
        if db == "CHEBI":
            parts.append(" (chemical, not a gene)")
        elif db == "MESH":
            parts.append(" (MeSH term, not a specific gene)")
        elif db == "FPLX":
            parts.append(" (protein family)")
        # Alias provenance — key disambiguation signal
        if query_is_alias is True:
            parts.append(f' ("{entity}" is a known alias)')
        elif query_is_alias is False:
            parts.append(f' ("{entity}" is NOT a known alias for this gene)')
        parts.append(f" score={c['score']}")
        lines.append("".join(parts))
    return "\n".join(lines)


def lookup_gene_executor(args: dict) -> str:
    """Combined executor: run gilda, format result as text."""
    result = execute_lookup_gene(args)
    return format_tool_result(result)


# ---------------------------------------------------------------------------
# Tool registry for ModelClient.call_with_tools()
# ---------------------------------------------------------------------------

TOOLS = {
    "lookup_gene": lambda args: lookup_gene_executor(args),
}


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_names = ["9G8", "CagA", "DVL", "RSK1", "PKB", "TFs", "p63RhoGEF"]
    for name in test_names:
        print(lookup_gene_executor({"entity_name": name}))
        print()

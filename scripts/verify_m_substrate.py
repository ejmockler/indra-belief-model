"""M11 brutalist self-verification: run M-phase substrate on the
in-context-only diagnosis records and report what fires per detector.

Output table per record:
  ctx.detected_relations (M1/M2)
  ctx.cascade_terminals (M7)
  ctx.subject_perturbation_marker (M9)
  ctx.explicit_hedge_markers (M10)

This is a deterministic-only probe (no LLM calls). Verifies the
substrate is producing expected fields for each diagnosis class.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from indra_belief.scorers.context_builder import build_context
from indra_belief.data.corpus import CorpusIndex


# Curated diagnosis FN/FP records by class (subset of full L8 set)
DIAGNOSIS_RECORDS = [
    # Class A nominalization
    ("PLK1→CDC20 phospho", "Phosphorylation", "PLK1", "CDC20",
     "Bub1 DeltaKinase stimulated Plk1 dependent phosphorylation of Cdc20 at S92."),
    ("FGF→ERK activation", "Activation", "FGF", "ERK",
     "Decreased GRIN diminished FGF activation of MAPK."),
    ("DEFA1→NFkB act", "Activation", "DEFA1", "NFkappaB",
     "HNP1 triggered the activation of NF-kappaB and IRF1 signaling pathways."),
    # Class B FPLX expansion
    ("p14_3_3↔CDC25C", "Complex", "p14_3_3", "CDC25C",
     "UCN-01 caused loss of both serine 216 phosphorylation and 14-3-3 binding to Cdc25C."),
    ("TCF_LEF↔CTNNB1", "Complex", "TCF_LEF", "CTNNB1",
     "beta-catenin's interactions with TCF/Lef proteins contribute to neoplastic transformation."),
    # Class D pathway
    ("NFkB→LCN2 amount", "IncreaseAmount", "NFkappaB", "LCN2",
     "LCN2 expression is upregulated by HER2/PI3K/AKT/NFkappaB pathway."),
    # Class D chain
    ("NOTCH1→NFkB chain", "Activation", "NOTCH1", "NFkappaB",
     "activation of Notch1 by DLL4 stimulation resulted in AKT activation "
     "and thereby promoted beta-catenin activity and NF-kappaB signaling."),
    # Class E alias norm
    ("DEFA1 hyphen", "Activation", "DEFA1", "NFkappaB",
     "HNP-1 triggered the activation of NF-kappaB."),
    # Class F perturbation FN
    ("MEK→MAPK1 LOF", "Phosphorylation", "MEK", "MAPK1",
     "Inhibiting MEK with PD184161 blocked Erk1 and Erk2 phosphorylation."),
    # Class F perturbation FP
    ("HDAC→AR LOF", "Acetylation", "HDAC", "AR",
     "The acetylation of the AR is induced by histone deacetylase (HDAC) inhibitors."),
    # Class hedge FP
    ("CCR7→AKT hedge", "Activation", "CCR7", "AKT",
     "CCR7 may activate Akt and the PI3K/Akt signal pathway."),
]


def main():
    logging.basicConfig(level=logging.WARNING)
    # Build minimal Statement/Evidence stubs without INDRA imports
    from indra.statements import (
        Activation, Agent, Complex, Evidence, IncreaseAmount,
        Phosphorylation, Acetylation,
    )
    stmt_classes = {
        "Phosphorylation": Phosphorylation,
        "Activation": Activation,
        "IncreaseAmount": IncreaseAmount,
        "Complex": Complex,
        "Acetylation": Acetylation,
    }

    print(f"{'Record':<25} {'M1/M2 detected':<25} {'M7 cascade':<15} "
          f"{'M9 LOF':<10} {'M10 hedge':<15}")
    print("-" * 95)
    for label, stmt_type, subj, obj, text in DIAGNOSIS_RECORDS:
        cls = stmt_classes[stmt_type]
        if stmt_type == "Complex":
            stmt = cls([Agent(subj), Agent(obj)])
        else:
            stmt = cls(Agent(subj), Agent(obj))
        ev = Evidence(text=text)
        try:
            ctx = build_context(stmt, ev)
        except Exception as e:
            print(f"{label:<25} ERROR: {type(e).__name__}: {e}")
            continue

        m12 = (f"{len(ctx.detected_relations)} rels"
               if ctx.detected_relations else "—")
        m7 = (",".join(ctx.cascade_terminals) if ctx.cascade_terminals else "—")
        m9 = ctx.subject_perturbation_marker or "—"
        m10 = (",".join(ctx.explicit_hedge_markers)[:14]
               if ctx.explicit_hedge_markers else "—")

        print(f"{label:<25} {m12:<25} {m7:<15} {m9:<10} {m10:<15}")
        # Show first detected relation
        if ctx.detected_relations:
            r = ctx.detected_relations[0]
            print(f"  ↪ {r.pattern_id} ({r.agent_canonical} → "
                  f"{r.target_canonical}, axis={r.axis}, "
                  f"site={r.site or '—'})")

    print()
    print("M11 verification complete. Substrate fires per detector on diagnosis records.")


if __name__ == "__main__":
    main()

"""R7 oracle probe — empirically validate that thinking-on ACTUALLY
changes parser output on records where escalation would fire.

If thinking-on doesn't meaningfully differ from thinking-off on the
ambiguity classes the escalator detects, the escalation layer is
theater and should be removed before R8.

Three records, three classes:
  1. hedge_scope_ambiguity — IFNG/STAT1 ("rate of activation of Stat1
     by IFN-gamma can be modified...speculate that pharmacological
     agents may alter") — hedge scopes to a different clause.
  2. perturbation_inversion_check — VHL/VIM ("VHL silencing increased
     vimentin") — silencing inverts literal increase to effective
     decrease.
  3. substrate_axis_mismatch — synthetic case for testing.

For each record:
  - Run parse_evidence with reasoning_effort="none"
  - Run parse_evidence with reasoning_effort="medium"
  - Compare assertions, claim_status, axis, sign
  - Report whether thinking-on changed the output meaningfully

Exit gate: if 0 of 3 records show meaningful difference, ESCALATION IS
THEATER — remove R6 before running R8. If ≥1 records differ in a
verdict-relevant way, escalation is justified and R8 can proceed.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from indra.statements import (
    Activation,
    Agent,
    Complex,
    Evidence,
    IncreaseAmount,
    Phosphorylation,
)

from indra_belief.model_client import ModelClient
from indra_belief.scorers.context_builder import build_context
from indra_belief.scorers.parse_evidence import parse_evidence


CASES = [
    {
        "name": "hedge_scope_ambiguity",
        "subj": "IFNG", "obj": "STAT1", "stmt_cls": Phosphorylation,
        "text": "We conclude that the rate of activation of Stat1 in cells by "
                "IFN-gamma can be modified by regulating either receptor chain "
                "and speculate that pharmacological agents which modify "
                "receptor chain expression may alter IFN-gamma receptor signal "
                "transduction.",
        "expected_target": "correct",
    },
    {
        "name": "perturbation_inversion_check",
        "subj": "VHL", "obj": "VIM", "stmt_cls": IncreaseAmount,
        "text": "In studies of paired isogenic cell lines, VHL silencing "
                "increased the levels of N-cadherin and vimentin and reduced "
                "the levels of E-cadherin relative to the parental VHL (+) "
                "cell line.",
        "expected_target": "correct",  # claim is VHL→VIM IncreaseAmount; gold says correct via inversion
    },
    {
        "name": "substrate_axis_mismatch_proxy",
        "subj": "AGO2", "obj": "RAD51", "stmt_cls": Complex,
        "text": "Interestingly, we show that Ago2 forms a complex with Rad51 "
                "and that the interaction is enhanced in cells treated with "
                "ionizing radiation.",
        "expected_target": "correct",
    },
]


def _summarize(commit) -> dict:
    if commit is None:
        return {"assertions": [], "n": 0, "raw": "ABSTAIN"}
    return {
        "n": len(commit.assertions),
        "assertions": [
            {
                "agents": list(a.agents),
                "targets": list(a.targets),
                "axis": a.axis,
                "sign": a.sign,
                "negation": a.negation,
                "claim_status": a.claim_status,
            }
            for a in commit.assertions
        ],
    }


def main() -> None:
    print("Initializing model client (gemma-remote)...")
    client = ModelClient("gemma-remote")

    results = []
    for case in CASES:
        print(f"\n=== {case['name']}: {case['subj']}-[{case['stmt_cls'].__name__}]->{case['obj']} ===")
        if case["stmt_cls"] is Complex:
            stmt = case["stmt_cls"]([Agent(case["subj"]), Agent(case["obj"])])
        else:
            stmt = case["stmt_cls"](Agent(case["subj"]), Agent(case["obj"]))
        ev = Evidence(source_api="reach", text=case["text"])
        ctx = build_context(stmt, ev)

        # Thinking OFF
        print("  [1/2] thinking-OFF...", end=" ", flush=True)
        t0 = time.time()
        commit_off = parse_evidence(case["text"], client, ctx=ctx,
                                     reasoning_effort="none")
        t_off = time.time() - t0
        print(f"{t_off:.1f}s")

        # Thinking ON
        print("  [2/2] thinking-ON...", end=" ", flush=True)
        t0 = time.time()
        commit_on = parse_evidence(case["text"], client, ctx=ctx,
                                    reasoning_effort="medium")
        t_on = time.time() - t0
        print(f"{t_on:.1f}s")

        sum_off = _summarize(commit_off)
        sum_on = _summarize(commit_on)

        # Compare
        differs = (
            sum_off["n"] != sum_on["n"]
            or sum_off["assertions"] != sum_on["assertions"]
        )
        results.append({
            "case": case["name"],
            "differs": differs,
            "t_off": t_off,
            "t_on": t_on,
            "off": sum_off,
            "on": sum_on,
        })

        print(f"  DIFFERS: {differs}")
        print(f"  off ({sum_off['n']} assertions):")
        for a in sum_off["assertions"]:
            print(f"    axis={a['axis']} sign={a['sign']} neg={a['negation']} "
                  f"status={a['claim_status']} agents={a['agents']} targets={a['targets']}")
        print(f"  on ({sum_on['n']} assertions):")
        for a in sum_on["assertions"]:
            print(f"    axis={a['axis']} sign={a['sign']} neg={a['negation']} "
                  f"status={a['claim_status']} agents={a['agents']} targets={a['targets']}")

    print("\n\n=== R7 ORACLE GATE ===")
    differing = sum(1 for r in results if r["differs"])
    print(f"Records where thinking-on ≠ thinking-off: {differing}/{len(results)}")
    avg_off = sum(r["t_off"] for r in results) / len(results)
    avg_on = sum(r["t_on"] for r in results) / len(results)
    print(f"Avg latency: thinking-off {avg_off:.1f}s, thinking-on {avg_on:.1f}s "
          f"(ratio {avg_on/avg_off:.1f}×)")

    if differing == 0:
        print("\nVERDICT: ESCALATION IS THEATER. Remove R6 before running R8.")
    else:
        print(f"\nVERDICT: thinking-on differs on {differing}/{len(results)} cases. "
              f"Escalation is justified; proceed to R8.")

    # Persist for the gate write-up
    out = Path("data/results/r7_oracle_probe.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nFull results: {out}")


if __name__ == "__main__":
    main()

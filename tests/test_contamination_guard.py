"""Hard guard: no fewshot example may overlap any eval/calibration record.

Contamination has bitten this project twice (gate-#36 caught in review,
gate-#39 caught post-calibration). The runtime cost is real: any
calibration run that contains contaminated evidence MEASURES MEMORIZATION
not GENERALIZATION, and the resulting reliability number is invalid.

This test runs the same `find_contamination` function used by the CI
guard (`scripts/check_contamination.py`) and asserts zero overlaps. If
this fails, the offending fewshot example or eval record must be
changed before any further calibration is trusted.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import check_contamination  # noqa: E402  (script imported as a module)


def test_no_fewshot_eval_overlap():
    contam = check_contamination.find_contamination()
    if contam:
        # Build a readable failure report so the dev sees exactly what's wrong.
        lines = [f"\n{len(contam)} fewshot/eval contamination(s) detected:\n"]
        for c in contam:
            lines.append(
                f"  [{c['kind']}] source={c['source']!s}\n"
                f"    fewshot:  {c['evidence']!s}\n"
                f"    eval ({c['file']}): {c['eval_evidence']!s}\n"
            )
        raise AssertionError("\n".join(lines))


def test_examples_load_from_every_known_source():
    """Defensive: if a fewshot source goes silent (import error, etc.) we
    want to know — silent zero would let new contamination slip in
    unnoticed."""
    examples = check_contamination.load_all_examples()
    sources = {ex["source"] for ex in examples}
    # parse_evidence._FEWSHOT must be present (it's the active sub-call's
    # fewshot). Legacy sources are nice-to-have but not required.
    assert any(s == "parse_evidence._FEWSHOT" for s in sources), (
        f"parse_evidence._FEWSHOT did not load — got sources: {sources}"
    )
    # Sanity: at least one example must have evidence text.
    assert any(ex["evidence"] for ex in examples), (
        "every loaded example had empty evidence text"
    )

"""Fail if Python files contain v{n} version-label references.

Version labels belong in PR titles, changelogs, and benchmark-run
filenames — not in living source. The labels are useful while iterating,
but become archaeology once merged. A reader of the code should not
need to know history to understand the current state.

Scope: src/, tests/, scripts/ (Python files).

Ignored contexts (not flagged):
  - `/v1` URL paths (OpenAI API convention)
  - Filename references like `eval_set_v4.jsonl`, `fewshot_pool_v4.jsonl`,
    `holdout_v15_sample.jsonl` — those name specific data artifacts whose
    identity includes the version.
  - `scripts/build_holdout.py` — entirely exempt. It encodes historical
    example-hash provenance (e.g., V6_V7_EXAMPLE_HASHES) that is tied to
    specific prior prompt bank states. Renaming would destroy traceable
    lineage for data decontamination.

Exit 0: clean. Exit 1: violations found.

Intended as a pre-commit hook or CI step. Run:
    python scripts/check_no_version_labels.py            # quiet
    python scripts/check_no_version_labels.py --verbose  # show allow-list hits
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


# v{n} anywhere a reader would interpret it as a version label: standalone
# in prose ("v5 baseline"), embedded in identifiers ("simulate_v11_verdicts",
# "V11_ACCURACY"), or at end of names ("test_v16"). Case-insensitive so
# `V11` is caught too.
#
# Excluded via negative lookbehind/lookahead:
#   /  — URL paths ("/v1/chat/completions")
#   .  — IP/version-number octets ("192.168.v5.1" — not a label, likely a typo
#        but false-positive not worth the noise)
#   \d — dotted version tuples ("1.2v3" — rare)
#
# Deliberately NOT excluded:
#   _ — identifiers with underscore neighbors are exactly what this check
#       must catch (e.g., `simulate_v11_verdicts`)
_VERSION_LABEL = re.compile(
    r"(?<![/.\d])v\d+(?![.\d])",
    re.IGNORECASE,
)

# URL-style `/v1/...` paths are API conventions, not version labels.
_URL_V = re.compile(r"/v\d+(/|\b)")

# Allowed substrings: data-file basenames whose identity includes the
# version tag. These are stripped BEFORE the version-label regex runs,
# so they never trigger a violation.
_ALLOWED_PATH_PATTERNS = (
    re.compile(r"eval_set_v\d+\."),
    re.compile(r"fewshot_pool_v\d+\."),
    re.compile(r"holdout(_v\d+)?"),
)

# Fully exempt files. Two legitimate classes:
#   - Data-prep scripts whose identity encodes prior bank/holdout
#     versions — renaming breaks decontamination lineage.
#   - This guard itself, whose docstrings and comments cite concrete
#     example strings (e.g., "simulate_v11_verdicts") to describe what
#     it catches; those citations aren't archaeology.
# Paths are repo-relative forward-slash form (Path.as_posix()).
_EXEMPT_FILES = {
    "scripts/build_holdout.py",
    "scripts/check_no_version_labels.py",
}

# Directories searched for violations. Paths are relative to repo root.
_SCAN_DIRS = ("src", "tests", "scripts")


def file_has_violations(path: Path) -> list[tuple[int, str]]:
    """Return (lineno, line) violations in `path`. Empty list = clean."""
    violations = []
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, FileNotFoundError):
        return []

    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line
        stripped = _URL_V.sub("", stripped)
        for allowed in _ALLOWED_PATH_PATTERNS:
            stripped = allowed.sub("", stripped)
        if _VERSION_LABEL.search(stripped):
            violations.append((lineno, line.rstrip()))
    return violations


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    verbose = "--verbose" in argv or "-v" in argv

    root = Path(__file__).resolve().parents[1]

    candidates: list[Path] = []
    for scan_dir in _SCAN_DIRS:
        candidates.extend(sorted((root / scan_dir).rglob("*.py")))

    total_violations = 0
    exempt_skipped = 0
    for py_path in candidates:
        rel = py_path.relative_to(root)
        rel_posix = rel.as_posix()
        if rel_posix in _EXEMPT_FILES:
            exempt_skipped += 1
            if verbose:
                print(f"  [exempt] {rel_posix}")
            continue
        hits = file_has_violations(py_path)
        for lineno, line in hits:
            print(f"{rel_posix}:{lineno}: {line}")
            total_violations += 1

    if total_violations:
        print(f"\n{total_violations} version-label violation(s) found.")
        print("Version labels belong in PR titles and benchmark filenames,")
        print("not in living source comments or identifiers.")
        return 1

    scope = ", ".join(_SCAN_DIRS)
    suffix = f" ({exempt_skipped} exempt)" if exempt_skipped and verbose else ""
    print(f"OK: no v{{n}} version labels in {scope}/{suffix}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Fail if source .py files contain v{n} version-label references.

Version labels belong in PR titles, changelogs, and benchmark-run
filenames — not in living source. The labels are useful while iterating,
but become archaeology once merged. A reader of the code should not
need to know history to understand the current state.

Ignored contexts:
  - `/v1` URL paths (OpenAI API)
  - comment lines in data-prep scripts that reference concrete filenames
    like "eval_set_v4.jsonl" (those are provenance of data, not version
    labels of the code)

Exit 0: clean. Exit 1: violations found.

Intended as a pre-commit hook or CI step. Run:
    python scripts/check_no_version_labels.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


# Standalone v{n} where n is one or more digits. The negative lookbehind
# and lookahead exclude paths like `/v1/chat/completions` and filename
# references like `eval_set_v4.jsonl`.
_VERSION_LABEL = re.compile(
    r"(?<![/a-zA-Z_\-.])v\d+(?![a-zA-Z_0-9])",
)

# URL-style `/v1/...` paths are OpenAI API conventions, not version
# labels. Match them to exclude.
_URL_V = re.compile(r"/v\d+(/|\b)")

# Allowed hits: file basenames that intentionally encode the version
# (data files, one-off analysis scripts moved to scripts/archive).
_ALLOWED_PATH_PATTERNS = (
    re.compile(r"eval_set_v\d+\."),          # benchmark data files
    re.compile(r"fewshot_pool_v\d+\."),
    re.compile(r"holdout(_v\d+)?"),          # holdout file names
)


def file_has_violations(path: Path) -> list[tuple[int, str]]:
    """Return a list of (lineno, line) violations in `path`. Empty → clean."""
    violations = []
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, FileNotFoundError):
        return []

    for lineno, line in enumerate(text.splitlines(), start=1):
        # Strip allowed patterns so they don't trigger.
        stripped = line
        stripped = _URL_V.sub("", stripped)
        for allowed in _ALLOWED_PATH_PATTERNS:
            stripped = allowed.sub("", stripped)
        if _VERSION_LABEL.search(stripped):
            violations.append((lineno, line.rstrip()))
    return violations


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"

    total_violations = 0
    for py_path in sorted(src.rglob("*.py")):
        hits = file_has_violations(py_path)
        if hits:
            rel = py_path.relative_to(root)
            for lineno, line in hits:
                print(f"{rel}:{lineno}: {line}")
                total_violations += 1

    if total_violations:
        print(f"\n{total_violations} version-label violation(s) found.")
        print("Version labels belong in PR titles and benchmark filenames,")
        print("not in living source comments or identifiers.")
        return 1
    print("OK: no v{n} version labels in src/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

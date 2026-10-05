#!/usr/bin/env python3
"""Pre-commit guard for docs/SPEC.md section 12.3 rule 9.

Reads `.private/banned_terms.txt` (gitignored, one term per line) and fails the
commit if any staged file contains any of them, case-insensitively.

Design notes:

* The term list is **not** in the repository, so the terms themselves never leak
  into git history. That is the point of keeping it in `.private/`.
* An absent list is a hard failure, not a silent pass. A guard that quietly does
  nothing when its configuration is missing is worse than no guard, because it
  reads as protection that is not there. Use `--allow-missing` only when you
  have decided you do not want this check.
* An empty list passes with a warning, because an empty list is a deliberate
  "nothing to ban yet" state.
* Only the staged content is scanned, which is what is about to be committed.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

TERMS_FILE = Path(".private/banned_terms.txt")


def staged_files() -> list[str]:
    out = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
        capture_output=True,
        text=True,
        check=False,
    )
    return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]


def load_terms() -> list[str]:
    if not TERMS_FILE.is_file():
        return []
    terms: list[str] = []
    for raw in TERMS_FILE.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            terms.append(line.lower())
    return terms


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="files to scan; default is the staged set")
    ap.add_argument(
        "--allow-missing",
        action="store_true",
        help="pass instead of failing when the term list is absent",
    )
    args = ap.parse_args(argv)

    if not TERMS_FILE.is_file():
        if args.allow_missing:
            print(f"warning: {TERMS_FILE} absent; banned-term check skipped")
            return 0
        print(
            f"ERROR: {TERMS_FILE} is missing.\n"
            "docs/SPEC.md section 12.3 rule 9 requires a banned-term list. Create it "
            "with one term per line (it is gitignored), or pass --allow-missing if "
            "you have decided not to use this check.",
            file=sys.stderr,
        )
        return 1

    terms = load_terms()
    if not terms:
        print(f"warning: {TERMS_FILE} has no terms; nothing to check")
        return 0

    targets = args.files or staged_files()
    hits: list[str] = []
    for name in targets:
        path = Path(name)
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace").lower()
        except OSError:
            continue
        for term in terms:
            if term in text:
                # Do not echo the term: that would put it in the commit output.
                hits.append(f"{name}: contains a banned term (index {terms.index(term)})")

    if hits:
        print("ERROR: banned term(s) found in staged content:", file=sys.stderr)
        for h in hits:
            print(f"  - {h}", file=sys.stderr)
        print(
            "\nThe term itself is not printed on purpose. Check "
            f"{TERMS_FILE} by index and remove the text before committing.",
            file=sys.stderr,
        )
        return 1

    print(f"banned-term check: {len(targets)} file(s) clean against {len(terms)} term(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

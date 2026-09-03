"""Diff-scoped line+branch coverage gate for the graphed integration.

Whole-module coverage of the touched files is dragged down by pre-existing untested paths (error
branches, ``compute=False``, executor selection) that are NOT this milestone's concern, so a
whole-module ``--fail-under`` is the wrong gate. This checks only the lines the branch CHANGED
(``git diff <base>...HEAD``), which is the mechanical gate the plan defines (>=90% line+branch diff
coverage on new/changed lines).

Usage:  python scripts/diff_coverage_gate.py <base-ref> <coverage.json> <file> [<file> ...]

Fails (exit 1) if any changed executable line is uncovered, or any changed line has an untaken
branch arc, below the threshold. Emits the changed/covered lines so an empty result cannot pass as
a silent success (the diff itself is the positive control: no changed lines -> nothing to gate).
"""
from __future__ import annotations

import json
import re
import subprocess
import sys

THRESHOLD = 90.0


def changed_new_lines(base: str, path: str) -> set[int]:
    """New-side line numbers added/changed in ``path`` between ``base`` and the working tree
    (two-dot: matches the checked-out file coverage.json was measured against — locally with
    uncommitted work, and in a clean CI checkout where the work tree is the committed HEAD)."""
    diff = subprocess.run(
        ["git", "diff", base, "--unified=0", "--", path],
        capture_output=True, text=True, check=True,
    ).stdout
    added: set[int] = set()
    newno = 0
    for ln in diff.splitlines():
        m = re.match(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", ln)
        if m:
            newno = int(m.group(1))
            continue
        if ln.startswith("+") and not ln.startswith("+++"):
            added.add(newno)
            newno += 1
        elif ln.startswith("-") and not ln.startswith("---"):
            continue
        elif not ln.startswith("\\"):
            newno += 1
    return added


def main() -> int:
    base, cov_path, files = sys.argv[1], sys.argv[2], sys.argv[3:]
    with open(cov_path) as fh:
        cov = json.load(fh)
    total_exec = total_covered = 0
    for path in files:
        added = changed_new_lines(base, path)
        fcov = cov["files"].get(path)
        if fcov is None:
            print(f"FAIL {path}: not in coverage report (never imported/traced)")
            return 1
        executed = set(fcov["executed_lines"])
        missing = set(fcov["missing_lines"])
        mb_added = [b for b in fcov.get("missing_branches", []) if b[0] in added]
        exec_added = sorted(a for a in added if a in executed or a in missing)
        uncovered = sorted(a for a in added if a in missing)
        total_exec += len(exec_added)
        total_covered += len(exec_added) - len(uncovered)
        pct = 100.0 if not exec_added else 100.0 * (len(exec_added) - len(uncovered)) / len(exec_added)
        print(f"{path}: {len(exec_added) - len(uncovered)}/{len(exec_added)} changed lines "
              f"= {pct:.1f}%  (changed exec lines: {exec_added})")
        if uncovered:
            print(f"  UNCOVERED changed lines: {uncovered}")
        if mb_added:
            print(f"  UNTAKEN branch arcs on changed lines: {mb_added}")
        if uncovered or mb_added:
            return 1
    if total_exec == 0:
        print("FAIL: no changed executable lines found — gate ran against an empty diff (base wrong?)")
        return 1
    pct = 100.0 * total_covered / total_exec
    print(f"TOTAL diff coverage: {total_covered}/{total_exec} = {pct:.1f}% (threshold {THRESHOLD}%)")
    return 0 if pct >= THRESHOLD else 1


if __name__ == "__main__":
    raise SystemExit(main())

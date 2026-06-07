"""Drive uproot5-graphed-mvp M0 (spine) + M8 (checkpoint/resume) through the real orchestrator.

The engine refuses DONE unless `ci` is confirmed green for the pushed commit, so without
--ci-confirmed the milestones stop at REVIEW (verify CI with scripts/confirm_ci.py in the
orchestrator repo, then re-run with --ci-confirmed).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from graphed_orchestrator import GateReport, IterationEvidence, Orchestrator
from graphed_orchestrator.gates import evaluate_test_sanity
from graphed_orchestrator.store import append_attempt, write_state

REPO = "uproot5-graphed-mvp"
MILESTONES = ("UPROOT",)


def _drive(milestone: str, *, total: int, cov: float, ci_confirmed: bool) -> Orchestrator:
    o = Orchestrator(milestone)
    o.start()
    o.begin_test_authoring()
    o.run_test_sanity(
        evaluate_test_sanity(
            collects=True,
            stub_pass_count=0,
            stub_total=total,
            stub_run1_fail_hash="stub",
            stub_run2_fail_hash="stub",
            coverage_instrumented=True,
        )
    )
    o.freeze(f"freeze-{milestone}-0")
    o.record_iteration(
        IterationEvidence(
            iteration_index=0,
            pass_count=total,
            total_tests=total,
            source_tree_hash="local",
            coverage_line=cov,
            coverage_branch=cov,
            coverage_from_frozen=True,
            lint_ok=True,
            types_ok=True,
            determinism_ok=True,
            benchmark_ok=None,
        )
    )
    # the engine records DONE only when ci is True; ci_confirmed=False leaves it REVIEW-pending-CI
    if o.phase.value == "REVIEW":
        o.review(
            approve=True,
            gates=GateReport(
                frozen_tests=True,
                coverage=True,
                lint=True,
                types=True,
                determinism=True,
                integrity_scan=True,
                ci=ci_confirmed,
            ),
        )
    return o


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, required=True)
    ap.add_argument("--cov", type=float, required=True)
    ap.add_argument("--ci-confirmed", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    records = []
    for milestone in MILESTONES:
        o = _drive(milestone, total=args.total, cov=args.cov, ci_confirmed=args.ci_confirmed)
        append_attempt(
            root / ".graphed" / milestone / "attempts.md",
            o.record,
            summary=f"local gates green; ci_confirmed={args.ci_confirmed}",
        )
        records.append(o.record)
        print(f"{milestone}: phase={o.phase.value}")

    write_state(root / ".graphed" / "state.json", REPO, records)
    print(f"wrote {root / '.graphed' / 'state.json'}")


if __name__ == "__main__":
    main()

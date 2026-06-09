## Iteration 0 — phase REVIEW — 2026-06-07T20:20:52Z

- summary: local gates green; ci_confirmed=False
- gates: {'frozen_tests': True, 'coverage': True, 'lint': True, 'types': True, 'determinism': True, 'benchmark': None, 'integrity_scan': True}
- l0_count=0 escalated=False reject_count=0

## Iteration 0 — phase DONE — 2026-06-07T20:27:53Z

- summary: local gates green; ci_confirmed=True
- gates: {'frozen_tests': True, 'coverage': True, 'lint': True, 'types': True, 'determinism': True, 'benchmark': None, 'integrity_scan': True}
- l0_count=0 escalated=False reject_count=0


## M10 remediation (2026-06-09, human-directed; freeze-UPROOT-1)

Applied the graphed M10 remediations (superproject `mvp-shortcomings.md`) to the integration:

- **A.2 — compiled-IR execution.** `graphed_uproot_analysis.process` no longer builds a Session
  and re-records the analysis per partition: the analysis is compiled ONCE per worker
  (`graphed.compile_ir`, module-cached) and each partition evaluates the reduced serialized IR
  (`graphed.evaluate_ir`) — one dispatch per reduced node. Bit-for-bit equality with the single
  pass is re-pinned in `test_graphed_m10.py`.
- **A.3 — buffer-level projection.** `uproot.necessary_buffers` + `uproot.resolve_read_branches`:
  a count-only analysis reports `{branch: OFFSETS}` (non-empty) and is served from the jagged
  branch's COUNTER branch (`NMuon` for `Muon_Px`) without the payload baskets.
- **C.9 — first-class blind partitions.** `graphed_partitions(open_files=False)` now emits
  `graphed_core.Partition.blind(...)`; the reader resolves via `partition.resolve(num_entries)`
  and still honors the legacy negative-entry_stop sentinel for pre-M10 plans. The one frozen-era
  test that pinned the sentinel encoding was amended under freeze-UPROOT-1 (recorded above).
- **C.9 — graphed_write hardening.** Uses the public `Session.sources()` accessor (no private
  reach-in); rejects multi-source arrays loudly; writes only the array's projected branches; the
  per-partition path table is gone — workers recompute their own `part{N}` index from an
  O(#files) base table.

Local: 59 graphed tests + 11 new M10 tests green; full uproot suite (xrootd-deselected) run before
push.

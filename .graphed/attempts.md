
## UPROOT-2 — graphed parity follow-ups (M11-M20 functionality reaches the reader) — 2026-06-10

Test-first throughout (each feature's suite verified failing before implementation; pre-freeze
authoring fixes recorded below). New test files (the existing frozen set is UNTOUCHED):
test_graphed_to_parquet.py (6), test_graphed_behaviors.py (4), test_graphed_head_and_surface.py (9).

- P1.1 graphed_to_parquet: ROOT in, parquet out, partition by partition — blind partitions (no
  file opened at planning), compiled-IR evaluation per partition (R7.8), specialized on the
  graphed.write base (write_plan / file_bases over (uri, tree) keys / blind_part_index /
  part_path), ProcessExecutor default, disabled==enabled consistency (R15.4), multi-source
  rejected. FINDING: the read list must be the graph's SYNTACTIC source-field accesses — the
  buffer projection under-supplies evaluation (a zip's pz/E legs are replayed even when only .pt
  is consumed); derived via a session.walk.
- P1.2 uproot.graphed(behavior=...): forwarded to AwkwardBackend (graphed M18); vector Momentum4D
  records/evaluates/projects (pt -> exactly {px, py} branches). For process workers,
  graphed_to_parquet(behavior=) also accepts an importable "module:attr" reference (behavior
  dicts contain lambdas and do not pickle). Authoring notes: vector.register_awkward() installs
  GLOBAL behaviors (the no-kwarg failure premise was wrong -> re-pinned on unknown attributes);
  references are vector's own computation (np.hypot differs in the last ULP from sqrt(px2+py2)).
- P2.3 uproot.graphed_head(expr, n): first file's leading entries only, projected branches only,
  through the compiled IR — witnessed by corrupting every later file (head unaffected; whole
  materialize fails); clamps to the first file's entry count.
- P2.4 fusion witness: per-event (axis=1) reductions live INSIDE stages (graphed M16); default
  SingleUse keeps the fanned-out field op its own stage (the frozen M4 diamond pin) -> source +
  2 stages; maximal_fusion=True -> source + ONE stage. Pinned so the gain cannot regress.
- P2.5 inherited-surface pins over uproot sources: record-subset getitem narrows the projection,
  axis-0 slices, the M11 ufunc tier, M17 structure ops.
- P3.6 (user-directed): the partitioned-write machinery moved to a format-agnostic graphed.write
  base (graphed M20); graphed_to_parquet specializes it. graphed_write deliberately does NOT —
  two of its behaviors are frozen pins predating the base (tasks return None; empty step ranges
  are skipped); documented in the module, alignment would be a recorded freeze bump.

## UPROOT-3 / freeze-UPROOT-2 — P3.6 revision (user-confirmed plan, 2026-06-10)

USER-AUTHORIZED FROZEN AMENDMENTS to tests/test_graphed_write.py (freeze tag bumped
freeze-UPROOT-1 -> freeze-UPROOT-2; each amendment named in the confirmed plan):
  1. `result.value is None` -> workers REPORT their part paths (the graphed.write base contract);
  2. part-name pins -> the base's part_path naming (part-00000.root / data-00000.root);
  3. the module/suite docstrings describing nothing-returning tasks.
No other frozen test was altered.

- uproot.graphed_to_parquet REMOVED: `graphed_awkward.io.to_parquet(uproot.graphed(...), ...)`
  is the same functionality through ONE generic entry point — _GraphedTTreeSource implements
  graphed.write.PartitionedSource (blind partitions; open-once partition reads). Efficiency
  WITNESSED in the rewritten tests: the source's whole-dataset loader never runs
  (last_columns_read stays None), planning opens no files, the read list is pinned.
- uproot.graphed_write now SPECIALIZES graphed.write: blind partitions (the driver no longer
  opens every file for num_entries), write_plan with path-reporting workers, base part naming,
  blind_part_index from the partition alone. A step resolving empty (fewer entries than steps)
  is skipped — no empty part files; numbering may gap in that corner case (documented).
- Recorded for future work: compile_ir output-accumulation footgun (compiling two different
  expressions from one session yields a multi-output IR).

## 2026-06-10 — freeze-UPROOT-3 (user-authorized respin, graphed-core freeze-M22-1)

- mark_output was removed from graphed-core's public API (outputs are per compile request). The
  frozen report helper (tests/graphed_uproot_report.py) respun to serialize(outputs=[...]) —
  bytes unchanged for its single-output graph. 78/78 green.

## UPROOT-4 — NanoAOD witnesses (ADL-port P0.2) — 2026-06-11

New frozen file tests/test_graphed_nanoaod.py (+tests/vector_backend_ref.py, the importable
worker backend factory) over a synthetic NanoAOD-style TTree (shared-counter jagged collections
written via mktree; note: this uproot defaults recreate-assignment to RNTuple — mktree forces
the TTree convention). Witnesses (the porting idiom for the ADL queries — per-query gak.zip,
no schema layer): counted-jagged zip + with_name + vector behaviors records with correct forms,
evaluates exactly, and PROJECTS TRUTHFULLY (px reads only {Muon_pt, Muon_phi} though the zip
names five branches); record+record `+` four-vector sums (.mass) over the reader;
jagged-integer-array getitem (Q8's leptons[pair.l1]); the capstone TTree -> collection ->
behavior property -> deferred hist.graphed fill through a SPAWNED process pool with the
behavior forwarded by import ref (tests dir on sys.path via monkeypatch — importlib test mode
keeps it off; spawn children inherit sys.path). 4/4 green; pre-freeze fix recorded (worker
import path).

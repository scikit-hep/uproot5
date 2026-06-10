
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

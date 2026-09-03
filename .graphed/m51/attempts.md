# m51 — ROOT derived-column IR evaluation in `graphed_write` (anchor K)

Scope: derived-column IR evaluation ONLY (non-varied). Varied ROOT write-out is Phase-2.
Freeze: `freeze-m51`. Frozen tree: `tests/frozen/m51/test_derived_columns.py` (4 tests).

## R1 — 2026-09-03

### Change (`src/uproot/writing/_graphed_write.py`)

Pre-m51 `_write_partition` copied source branches verbatim (`record = {name: chunk[name] for name
in chunk.fields}`, zero IR eval), so a derived column — a field computed from an expression, absent
from the source branches — was lost on write. R1 compiles the recorded array once in the driver and
evaluates the compiled IR per partition, mirroring the read-side `uproot._graphed.graphed_head`
pattern (`compiled = compile_ir(session, array)`; `(evaluated,) = evaluate_ir(compiled, backend,
{source_name: chunk})`).

Read list switched from `necessary_columns` (buffer projection) to the SYNTACTIC
`_evaluation_columns` (already in `_graphed.py`, used by `graphed_head`): compiled-IR evaluation
REPLAYS every field the graph accesses — including field reads whose buffers the output never
touches (a zip's pz/E legs when only .pt is consumed) — so buffer projection UNDER-supplies
evaluation and starves the read (plan §6.4f). Projection is still exact for the columns that ARE
read (probe: `g.x + g.y` over `{x,y,z}` reads `(x, y)`, `z` excluded).

Non-record fallback: a bare (non-record) expression evaluates to a FIELDLESS array with no branch
name to write it under. `out_rec = evaluated if evaluated.fields else chunk` falls back to the
source columns the expression reads — the pre-m51 projection behavior — so bare-expression writes
are backward-compatible (strict superset: records gain derived columns; bare exprs unchanged).

### Regression found + fixed

`tests/test_graphed_m10.py::test_write_of_a_projected_array_writes_only_its_branches` (non-frozen)
writes a bare `g.x + g.y` and asserts output `{x, y}`. It PASSED at `freeze-m51`; the initial R1
change (record-only) broke it (`TypeError: RecordArray if len(contents)==0, a 'length' must be
specified` — empty `record` dict from a fieldless evaluation). The non-record fallback restores it.
Blast radius is confined to `graphed_write` (sole entry into this module): 24/24 non-frozen
graphed_write tests green (`test_graphed_write.py` + `test_graphed_m10.py` + `test_graphed_to_parquet.py`).

### CI (`.github/workflows/graphed.yml`)

Added `m51-vary` to push triggers + a `Frozen acceptance` step (`pytest -vv tests/frozen
--noconftest`, isolated from the `RangeHTTPServer` root conftest) ahead of the full-suite step; the
full suite still exercises that conftest.

### Gates

- frozen suite: 4/4 GREEN, deterministic across two runs (identical); `git diff freeze-m51 --
  tests/frozen/` EMPTY (frozen unmodified).
- diff coverage (FROZEN suite, subprocess coverage over the spawn workers): 100% — all 8 executable
  added source lines covered, 0 uncovered, 0 missing branch arcs on added lines. The 9 missed
  whole-file statements are all pre-existing error/alt-branch paths. The bare-expr fallback branch
  itself is exercised by the non-frozen m10 test, not by frozen (frozen writes only records).
- ruff: clean on the changed source. black/mypy: N/A for this fork — the `freeze-m51` baseline
  itself fails black (not the enforced style; absent from `graphed.yml` CI), and there is no
  `[tool.mypy]` (the module is untyped by the fork's convention; `mypy --strict` yields 22
  structural errors on uproot's untyped public API). cov/mypy CI config is the plan's "ride this
  PR" fork infra, out of R1 source scope.
- non-varied scope confirmed: no manifest / per-universe / varied-write code; derived-columns-only.

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

## R1 follow-up — §10 fork-gates (wire the fork's first gated-pipeline CI gates)

The §10 fork-gates ride THIS fork PR (the fork's first frozen tree), so they land now rather than
as deferred infra. Separate commit from the R1 source (cleaner for three-lens review).

### (a) Coverage gate — DIFF-scoped, real

Whole-module coverage of `_graphed_write.py` from the frozen suite is 79% — dragged down by
PRE-EXISTING untested paths (the `except ImportError`, `compression is None`, empty-step skip,
`_select_executor` passthrough, the two `raise TypeError`s, the `compute=False` return) that are not
m51's concern — so a whole-module `--cov-fail-under=90` is unachievable honestly and a diff-scoped
gate is the correct one. `scripts/diff_coverage_gate.py` (stdlib only, no `diff-cover` dep) diffs the
source against `origin/graphed-mvp` and requires ≥90% line+branch coverage on the CHANGED lines.
Result: 9/9 changed lines = 100%. Non-vacuity PROVEN: the identical gate over the FROZEN suite ALONE
returns 55.6% and FAILS (exit 1) — the frozen suite runs `_write_partition` in spawn ProcessPool
children, invisible to a non-subprocess coverage run. `tests/extra/m51/` drives the same worker body
(and the bare-expr fallback branch frozen never reaches) via `executor="thread"`, in-process, so the
100% is genuine coverage of exercised code, not a subprocess-coverage trick.

### (b) mypy — scoped, meaningful, non-vacuous

`[tool.mypy]` (pyproject) scopes to `_graphed_write.py` + `tests/frozen/m51` + `tests/extra/m51`:
`check_untyped_defs=true` (bodies ARE checked — non-vacuous) with `ignore_missing_imports=true`
(the untyped uproot/graphed/awkward boundary is Any; not fork-wide `--strict`, which is neither
achievable nor the goal on untyped upstream). `python_version="3.12"` pins analysis identically on
both CI Pythons. Non-vacuity witness: the SAME config reported 3 real errors (`evaluate_ir` is typed
`list[object]`; `object` has no `.fields`) BEFORE the one-line fix (`evaluated: Any` at the backend
boundary). Now: `Success: no issues found in 5 source files`.

### (c) DoD matrix + trigger — ACCEPTED REDUCED SCOPE

- CI matrix: `graphed.yml` runs ubuntu-latest × CPython {3.11, 3.12}. The full §A.5 matrix
  (Linux/macOS/Windows × x86_64/arm64 × 3.11–3.14 + 3.14t) is NOT provisioned here; the reduced
  ubuntu/3.11–3.12 matrix is recorded as accepted scope for the fork (§10(c)-style reduced-scope
  allowance) — the fork is an integration harness over uproot, not a §A.5 wheel-shipping package.
- Trigger: DONE is keyed on a BRANCH PUSH (`graphed.yml` `on: push: [graphed-mvp, m51-vary]`).
  `pull_request` is deliberately NOT added — the workflow runs only on the graphed branches so it
  does not collide with uproot's own build-test matrix (which runs on main/PRs), per the file's
  header comment.

### Gate results (follow-up)

- diff-coverage gate: 9/9 = 100% (≥90); PROVEN to fail (55.6%) without the extra suite.
- mypy: clean (5 files); witnessed non-vacuous by the pre-fix 3 errors.
- ruff: clean on all changed/new (`_graphed_write.py`, `tests/extra/m51/`, `scripts/*` — a
  `scripts/*` T20 per-file-ignore, mirroring `dev/*`, also cleans the pre-existing `graphed_advance.py`).
- frozen: still 4/4; `git diff freeze-m51 -- tests/frozen/` EMPTY (frozen untouched).
- graphed.yml: valid YAML; adds the mypy + diff-coverage steps and `fetch-depth: 0` (the gate needs
  `origin/graphed-mvp` to diff against).

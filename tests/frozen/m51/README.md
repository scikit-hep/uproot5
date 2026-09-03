# Frozen acceptance — m51 (uproot fork)

The fork's first frozen tree. Frozen = read-only after the `freeze-m51` tag; do not edit, skip, or
weaken. See `vary-m51-decomposition.md` (§3, §4) and `systematics-vary-plan.md` r47 (§10/m51 ROOT
anchor, §6.4f ROOT seam).

## Scope

The fork's m51 half is **derived-column IR evaluation only** in `uproot.graphed_write` — NOT
variation-aware. Varied ROOT write-out (manifest channel, per-TBranch delta storage) is Phase-2
(plan §11) and is NOT frozen here.

## Traceability

| Anchor | File | Satisfied by |
|---|---|---|
| K — ROOT derived-column round-trip | `test_derived_columns.py` | commit R1 (`_graphed_write.py` gains `compile_ir`/`evaluate_ir`, mirroring `_graphed.py::graphed_head`) |

`test_derived_columns.py`:
- `test_plain_source_roundtrips_positive_control` — POSITIVE CONTROL; passes today (verbatim copy),
  proving the write/read harness works.
- `test_derived_column_roundtrips` — derived `doubled = x*2 + 1` beside a passthrough, over 4
  partitions (per-partition eval).
- `test_pure_derived_record_is_named_by_its_field` — output named by the derived field `energy`,
  not the raw branches the expression reads.
- `test_jagged_derived_column_roundtrips` — derived per-object `Jet_pt2 = Jet_pt * 2` over a jagged
  branch (structure preserved).

## Non-vacuity (measured at authoring, against verbatim-copy `graphed_write`)

Today `_write_partition` writes `{name: chunk[name] for name in chunk.fields}` from the RAW read, so:
- derived record `zip({"x": g.x, "doubled": g.x*2+1})` → output fields `["x"]` (derived ABSENT);
- pure-derived `zip({"energy": sqrt(x^2+y^2)})` → output fields `["x", "y"]` (never `energy`);
- jagged `zip({"Jet_pt": g.Jet_pt, "Jet_pt2": g.Jet_pt*2})` → output fields `["Jet_pt"]`.

The derived-column assertions therefore FAIL now and PASS once R1 evaluates the compiled IR
per partition; the plain-source test PASSES now (positive control).

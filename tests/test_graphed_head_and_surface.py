# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
"""Partitioned peeking + the inherited graphed surface over real TTrees (items P2.3-P2.5).

- ``uproot.graphed_head`` evaluates the recorded analysis over ONLY the first file's leading
  entries (witnessed: corrupting every later file does not disturb it; a whole-dataset
  materialize does fail).
- The M16 structural rule fuses per-event (axis=1) reductions into stages — witnessed by the
  compiled IR's node count, so the fusion gain cannot silently regress on the path that ships.
- The M11-M19 user surface (record subsets, axis-0 slices, the full ufunc tier, structure ops)
  is pinned over uproot sources, where a regression would actually reach users.
"""

import os

import awkward as ak
import numpy as np
import pytest
import skhep_testdata

import uproot

pytest.importorskip("graphed.awkward")

from graphed.awkward import gak  # noqa: E402


def _zmumu():
    return skhep_testdata.data_path("uproot-Zmumu.root") + ":events"


def _hzz():
    return skhep_testdata.data_path("uproot-HZZ.root") + ":events"


# ---- partitioned head (P2.3) -------------------------------------------------------------------
def test_head_matches_the_whole_computation(tmp_path):
    g = uproot.graphed(_zmumu(), library="ak", filter_name=["px1", "py1"])
    expr = g.px1 + g.py1
    got = ak.Array(uproot.graphed_head(expr, 7))
    raw = uproot.open(_zmumu()).arrays(["px1", "py1"])
    assert len(got) == 7
    assert ak.array_equal(got, (raw.px1 + raw.py1)[:7])


def test_head_reads_only_the_first_file(tmp_path):
    # two files; the SECOND is corrupted after recording — head must not notice
    src = uproot.open(_zmumu()).arrays(["px1"], entry_stop=20)
    paths = []
    for i in range(2):
        p = os.path.join(tmp_path, f"f{i}.root")
        with uproot.recreate(p) as f:
            f["events"] = {"px1": np.asarray(src.px1) + 100.0 * i}
        paths.append(p + ":events")
    g = uproot.graphed(paths, library="ak")
    expr = g.px1 * 2.0
    with open(os.path.join(tmp_path, "f1.root"), "wb") as f:
        f.write(b"ruined")  # the second file is now garbage
    got = ak.Array(uproot.graphed_head(expr, 5))
    assert ak.array_equal(got, (src.px1 * 2.0)[:5])  # only file 0 was read
    with pytest.raises(Exception):  # noqa: B017  (the dataset really is broken now)
        g.session.materialize(expr)


def test_head_clamps_to_the_first_files_entries(tmp_path):
    p = os.path.join(tmp_path, "tiny.root")
    with uproot.recreate(p) as f:
        f["events"] = {"x": np.arange(3.0)}
    g = uproot.graphed(p + ":events", library="ak")
    got = ak.Array(uproot.graphed_head(g.x, 50))
    assert len(got) == 3  # the first file has only 3 entries (documented MVP clamp)


def test_head_reads_only_projected_branches(tmp_path):
    g = uproot.graphed(_hzz(), library="ak", filter_name=["Muon_Px", "Muon_Py", "MET_px"])
    got = ak.Array(uproot.graphed_head(g.MET_px + 0.0, 4))
    raw = uproot.open(_hzz()).arrays(["MET_px"])
    assert ak.array_equal(got, (raw.MET_px + 0.0)[:4])


# ---- the M16 fusion witness (P2.4) ---------------------------------------------------------------
def test_per_event_reductions_fuse_into_one_stage():
    import graphed.core
    from graphed import compile_ir

    g = uproot.graphed(_hzz(), library="ak", filter_name=["Muon_Px"])
    # pre-M16 each axis=1 reduction was a stage BOUNDARY; now they live INSIDE stages
    expr = gak.sum(g.Muon_Px + 1.0, axis=1) * 2.0 + gak.num(g.Muon_Px, axis=1)
    compiled = compile_ir(g.session, expr)
    nodes = graphed.core.GraphStore.deserialize(compiled.ir).nodes()
    kinds = sorted(n["kind"] for n in nodes)
    assert "reduction" not in kinds, f"a per-event reduction leaked out as a boundary: {kinds}"
    # default SingleUse fusion keeps the fanned-out field op as its own stage (the frozen M4
    # diamond pin): source + 2 stages; under maximal fusion the chain is source + ONE stage
    assert kinds == ["source", "stage", "stage"]
    maximal = compile_ir(g.session, expr, maximal_fusion=True)
    nodes_max = graphed.core.GraphStore.deserialize(maximal.ir).nodes()
    assert sorted(n["kind"] for n in nodes_max) == ["source", "stage"]


# ---- the inherited M11-M19 surface over uproot sources (P2.5) ------------------------------------
def test_record_subset_getitem_narrows_the_projection():
    g = uproot.graphed(_zmumu(), library="ak", filter_name=["px1", "py1", "pz1", "E1"])
    sub = g[["px1", "py1"]]
    assert uproot.necessary_columns(sub) == {"events": frozenset({"px1", "py1"})}
    out = ak.Array(g.session.materialize(sub))
    assert set(out.fields) == {"px1", "py1"}
    raw = uproot.open(_zmumu()).arrays(["px1", "py1"])
    assert ak.array_equal(out.px1, raw.px1)


def test_axis0_slices_evaluate_over_uproot_sources():
    g = uproot.graphed(_zmumu(), library="ak", filter_name=["px1"])
    raw = uproot.open(_zmumu()).arrays(["px1"])
    got = ak.Array(g.session.materialize(g.px1[10:20]))
    assert ak.array_equal(got, raw.px1[10:20])
    got = ak.Array(g.session.materialize(g.px1[::500]))
    assert ak.array_equal(got, raw.px1[::500])


def test_new_ufunc_tier_evaluates_over_uproot_sources():
    g = uproot.graphed(_zmumu(), library="ak", filter_name=["px1", "py1"])
    raw = uproot.open(_zmumu()).arrays(["px1", "py1"])
    for fn in (np.exp, np.log1p, np.arctan, np.cbrt):
        got = ak.Array(g.session.materialize(fn(abs(g.px1))))
        assert ak.array_equal(got, fn(abs(raw.px1)), equal_nan=True)
    got = ak.Array(g.session.materialize(np.logaddexp(g.px1, g.py1)))
    assert ak.array_equal(got, np.logaddexp(raw.px1, raw.py1))


def test_structure_ops_evaluate_over_jagged_branches():
    g = uproot.graphed(_hzz(), library="ak", filter_name=["Muon_Px"])
    raw = uproot.open(_hzz()).arrays(["Muon_Px"])
    got = ak.Array(g.session.materialize(gak.sort(g.Muon_Px, axis=1, ascending=False)))
    assert ak.array_equal(got, ak.sort(raw.Muon_Px, axis=1, ascending=False))
    padded = gak.fill_none(gak.pad_none(g.Muon_Px, 3, axis=1, clip=True), 0.0)
    got = ak.Array(g.session.materialize(padded))
    ref = ak.fill_none(ak.pad_none(raw.Muon_Px, 3, axis=1, clip=True), 0.0)
    assert ak.array_equal(got, ref)

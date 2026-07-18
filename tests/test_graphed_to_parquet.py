# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
"""ROOT -> parquet through the GENERIC writer (P3.6 revision; supersedes uproot.graphed_to_parquet).

``graphed.awkward.io.to_parquet(uproot.graphed(...), ...)`` writes a recorded analysis over each
blind ROOT partition through the compiled IR: the uproot source implements
``graphed.write.PartitionedSource``, so the ONE generic entry point covers parquet datasets and
ROOT trees alike. Efficiency is witnessed, not assumed: the source's whole-dataset loader is
NEVER invoked (no big materializes), planning opens no files, and the per-task read list is the
graph's syntactic needs.
"""

import os

import awkward as ak
import numpy as np
import pytest
import skhep_testdata

import uproot

pytest.importorskip("pyarrow")
pytest.importorskip("graphed_exec_local")
pytest.importorskip("graphed.awkward")

import graphed.awkward.io as gio  # noqa: E402
from graphed.awkward import gak  # noqa: E402


def _zmumu():
    return skhep_testdata.data_path("uproot-Zmumu.root") + ":events"


def _hzz():
    return skhep_testdata.data_path("uproot-HZZ.root") + ":events"


def _source_of(array):
    from uproot._graphed import _GraphedTTreeSource

    ((_nid, src),) = array.session.sources().items()
    assert isinstance(src, _GraphedTTreeSource)
    return src


def test_roundtrip_matches_single_pass_without_materializing(tmp_path):
    g = uproot.graphed(_zmumu(), library="ak", filter_name=["px1", "py1"])
    expr = g.px1 + g.py1
    outdir = os.path.join(tmp_path, "out")
    paths = gio.to_parquet(expr, outdir, steps_per_file=3)
    assert len(paths) == 3
    assert paths == sorted(paths)  # deterministic, key-ordered part names
    assert _source_of(g).last_columns_read is None  # the whole-dataset loader NEVER ran
    back = ak.concatenate([ak.from_parquet(p) for p in paths])
    raw = uproot.open(_zmumu()).arrays(["px1", "py1"])
    assert ak.array_equal(back["data"], raw.px1 + raw.py1)


def test_partitions_are_blind_and_disabled_plan_equals_enabled_run(tmp_path):
    from graphed.core.execution import Plan
    from graphed_exec_local import ProcessExecutor

    g = uproot.graphed(_zmumu(), library="ak", filter_name=["px1"])
    expr = g.px1 * 2.0

    enabled = gio.to_parquet(expr, os.path.join(tmp_path, "en"), steps_per_file=2)

    plan = gio.to_parquet(expr, os.path.join(tmp_path, "dis"), steps_per_file=2, compute=False)
    assert isinstance(plan, Plan)
    assert all(t.partition.is_blind for t in plan.tasks)  # planning opened no files (R7.9)
    assert not os.path.exists(os.path.join(tmp_path, "dis"))  # nothing written yet
    later = ProcessExecutor(max_workers=2).run(plan).value  # any R7 executor runs the same plan

    assert [os.path.basename(p) for p in later] == [os.path.basename(p) for p in enabled]
    for a, b in zip(enabled, later):
        assert ak.array_equal(ak.from_parquet(a), ak.from_parquet(b))  # bit-for-bit (R15.4)
    assert _source_of(g).last_columns_read is None  # still no whole-dataset read


def test_structure_only_needs_read_their_carrier_branch(tmp_path):
    g = uproot.graphed(_hzz(), library="ak", filter_name=["Muon_Px", "MET_px"])
    expr = gak.num(g.Muon_Px, axis=1) + 0 * g.MET_px  # offsets of Muon_Px + data of MET_px
    plan = gio.to_parquet(expr, os.path.join(tmp_path, "o"), compute=False)
    # syntactic accesses {Muon_Px, MET_px}; on a flat TTree each branch is its own carrier
    assert set(plan.process.columns) == {"Muon_Px", "MET_px"}

    paths = gio.to_parquet(expr, os.path.join(tmp_path, "out"))
    back = ak.concatenate([ak.from_parquet(p) for p in paths])
    raw = uproot.open(_hzz()).arrays(["Muon_Px", "MET_px"])
    ref = ak.num(raw.Muon_Px, axis=1) + 0 * raw.MET_px
    assert ak.array_equal(back["data"], ref)


def test_custom_column_name_and_explicit_executor(tmp_path):
    from graphed_exec_local import ThreadExecutor

    g = uproot.graphed(_zmumu(), library="ak", filter_name=["px1"])
    paths = gio.to_parquet(
        np.sqrt(abs(g.px1)),
        os.path.join(tmp_path, "t"),
        executor=ThreadExecutor(max_workers=2),
        column="spx",
    )
    back = ak.concatenate([ak.from_parquet(p) for p in paths])
    raw = uproot.open(_zmumu()).arrays(["px1"])
    assert ak.array_equal(back["spx"], np.sqrt(abs(raw.px1)))


def test_multi_source_arrays_are_rejected(tmp_path):
    # two uproot sources in ONE session (the construction test_graphed_m10 pins for graphed_write)
    import graphed
    from graphed.awkward import AwkwardBackend, AwkwardForm
    from uproot._graphed import _GraphedTTreeSource

    path = skhep_testdata.data_path("uproot-Zmumu.root")
    chunk = uproot.open(path)["events"].arrays(["px1"], entry_stop=4)
    s = graphed.Session(AwkwardBackend())
    arrays = []
    for name in ("a", "b"):
        src = _GraphedTTreeSource([(path, "events")], ["px1"], None, False, {"num_workers": 1})
        form = AwkwardForm(ak.Array(chunk.layout.to_typetracer(forget_length=True)))
        arrays.append(s.source(name, form=form, data=src))
    with pytest.raises(TypeError, match="exactly one"):
        gio.to_parquet(arrays[0].px1 + arrays[1].px1, os.path.join(tmp_path, "nope"))


def test_record_results_write_their_own_fields(tmp_path):
    g = uproot.graphed(_zmumu(), library="ak", filter_name=["px1", "py1"])
    rec = gak.zip({"sum": g.px1 + g.py1, "diff": g.px1 - g.py1})
    paths = gio.to_parquet(rec, os.path.join(tmp_path, "rec"))
    back = ak.concatenate([ak.from_parquet(p) for p in paths])
    assert set(back.fields) == {"sum", "diff"}
    raw = uproot.open(_zmumu()).arrays(["px1", "py1"])
    assert ak.array_equal(back["diff"], raw.px1 - raw.py1)
    assert _source_of(g).last_columns_read is None

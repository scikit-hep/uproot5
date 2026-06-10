# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
"""ROOT -> parquet conversion, partition by partition (graphed parity follow-up, item P1.1).

``uproot.graphed_to_parquet`` evaluates a recorded analysis over each (blind) ROOT partition
through the COMPILED IR (R7.8 — compiled once at the driver) and writes one parquet part per
partition via the graphed.parquet write plan: the first cross-format pipeline. The per-task read
list is wired from the BUFFER projection (a structure-only need reads its branch as the carrier —
counter-only reads cannot feed evaluation without form reconstruction, see mvp-shortcomings).
"""

import os

import awkward as ak
import numpy as np
import pytest
import skhep_testdata

import uproot

pytest.importorskip("pyarrow")
pytest.importorskip("graphed_exec_local")
pytest.importorskip("graphed_awkward")

from graphed_awkward import gak  # noqa: E402


def _zmumu():
    return skhep_testdata.data_path("uproot-Zmumu.root") + ":events"


def _hzz():
    return skhep_testdata.data_path("uproot-HZZ.root") + ":events"


def test_roundtrip_matches_single_pass(tmp_path):
    g = uproot.graphed(_zmumu(), library="ak", filter_name=["px1", "py1"])
    expr = g.px1 + g.py1
    outdir = os.path.join(tmp_path, "out")
    paths = uproot.graphed_to_parquet(expr, outdir, steps_per_file=3)
    assert len(paths) == 3
    assert paths == sorted(paths)  # deterministic, key-ordered part names
    back = ak.concatenate([ak.from_parquet(p) for p in paths])
    raw = uproot.open(_zmumu()).arrays(["px1", "py1"])
    assert ak.array_equal(back["data"], raw.px1 + raw.py1)


def test_partitions_are_blind_and_disabled_plan_equals_enabled_run(tmp_path):
    from graphed_core.execution import Plan
    from graphed_exec_local import ProcessExecutor

    g = uproot.graphed(_zmumu(), library="ak", filter_name=["px1"])
    expr = g.px1 * 2.0

    enabled = uproot.graphed_to_parquet(expr, os.path.join(tmp_path, "en"), steps_per_file=2)

    plan = uproot.graphed_to_parquet(
        expr, os.path.join(tmp_path, "dis"), steps_per_file=2, compute=False
    )
    assert isinstance(plan, Plan)
    assert all(t.partition.is_blind for t in plan.tasks)  # planning opened no files (R7.9)
    assert not os.path.exists(os.path.join(tmp_path, "dis"))  # nothing written yet
    later = ProcessExecutor(max_workers=2).run(plan).value

    assert [os.path.basename(p) for p in later] == [os.path.basename(p) for p in enabled]
    for a, b in zip(enabled, later):
        assert ak.array_equal(ak.from_parquet(a), ak.from_parquet(b))  # bit-for-bit (R15.4)


def test_structure_only_needs_read_their_carrier_branch(tmp_path):
    g = uproot.graphed(_hzz(), library="ak", filter_name=["Muon_Px", "MET_px"])
    expr = gak.num(g.Muon_Px, axis=1) + 0 * g.MET_px  # offsets of Muon_Px + data of MET_px
    plan = uproot.graphed_to_parquet(expr, os.path.join(tmp_path, "o"), compute=False)
    # the read list is wired from the BUFFER projection: the jagged branch itself is the
    # offsets carrier for evaluation; MET_px is a data need; nothing else travels
    assert set(plan.process.keywords["columns"]) == {"Muon_Px", "MET_px"}

    paths = uproot.graphed_to_parquet(expr, os.path.join(tmp_path, "out"))
    back = ak.concatenate([ak.from_parquet(p) for p in paths])
    raw = uproot.open(_hzz()).arrays(["Muon_Px", "MET_px"])
    ref = ak.num(raw.Muon_Px, axis=1) + 0 * raw.MET_px
    assert ak.array_equal(back["data"], ref)


def test_custom_column_name_and_thread_executor(tmp_path):
    g = uproot.graphed(_zmumu(), library="ak", filter_name=["px1"])
    paths = uproot.graphed_to_parquet(
        np.sqrt(abs(g.px1)), os.path.join(tmp_path, "t"), executor="thread", column="spx"
    )
    back = ak.concatenate([ak.from_parquet(p) for p in paths])
    raw = uproot.open(_zmumu()).arrays(["px1"])
    assert ak.array_equal(back["spx"], np.sqrt(abs(raw.px1)))


def test_multi_source_arrays_are_rejected(tmp_path):
    # two uproot sources in ONE session (the construction test_graphed_m10 pins for graphed_write)
    import graphed
    from graphed_awkward import AwkwardBackend, AwkwardForm
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
        uproot.graphed_to_parquet(arrays[0].px1 + arrays[1].px1, os.path.join(tmp_path, "nope"))


def test_record_results_write_their_own_fields(tmp_path):
    g = uproot.graphed(_zmumu(), library="ak", filter_name=["px1", "py1"])
    rec = gak.zip({"sum": g.px1 + g.py1, "diff": g.px1 - g.py1})
    paths = uproot.graphed_to_parquet(rec, os.path.join(tmp_path, "rec"))
    back = ak.concatenate([ak.from_parquet(p) for p in paths])
    assert set(back.fields) == {"sum", "diff"}
    raw = uproot.open(_zmumu()).arrays(["px1", "py1"])
    assert ak.array_equal(back["diff"], raw.px1 - raw.py1)

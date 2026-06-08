# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
"""``uproot.graphed_write`` — one output ROOT file per partition, mirroring ``dask_write``.

``compute=False`` returns the write **task graph** (a ``graphed_core.Plan`` of side-effecting,
nothing-returning write tasks) without writing; ``compute=True`` runs it through a ``graphed-exec-local``
executor (``ProcessExecutor`` by default).
"""

import os

import awkward as ak
import numpy as np
import pytest
import skhep_testdata

import uproot

pytest.importorskip("graphed_exec_local")
graphed_awkward = pytest.importorskip("graphed_awkward")

from graphed_core import Plan
from graphed_exec_local import ProcessExecutor


def _make_root(path, n=20):
    with uproot.recreate(path) as f:
        f["events"] = {"x": np.arange(n, dtype="f8"), "y": np.arange(n, dtype="f8") * 2.0}
    return path + ":events"


def test_writes_one_file_per_partition(tmp_path):
    src = _make_root(os.path.join(tmp_path, "in.root"), n=20)
    g = uproot.graphed(src, library="ak")
    outdir = os.path.join(tmp_path, "out")

    paths = uproot.graphed_write(g, outdir, steps_per_file=4, tree_name="events")

    assert len(paths) == 4
    assert sorted(os.listdir(outdir)) == ["part0.root", "part1.root", "part2.root", "part3.root"]
    pieces = [uproot.open(p + ":events").arrays() for p in paths]
    assert [len(p) for p in pieces] == [5, 5, 5, 5]  # four contiguous quarters of 20 entries
    back = ak.concatenate(pieces)
    ref = uproot.open(src).arrays()
    assert ak.array_equal(back.x, ref.x) and ak.array_equal(back.y, ref.y)


def test_prefix_names_the_part_files(tmp_path):
    src = _make_root(os.path.join(tmp_path, "in.root"), n=10)
    g = uproot.graphed(src, library="ak")
    outdir = os.path.join(tmp_path, "out")

    uproot.graphed_write(g, outdir, steps_per_file=2, prefix="data", tree_name="events")

    assert sorted(os.listdir(outdir)) == ["data-part0.root", "data-part1.root"]


def test_compute_false_returns_task_graph_and_writes_nothing(tmp_path):
    src = _make_root(os.path.join(tmp_path, "in.root"), n=12)
    g = uproot.graphed(src, library="ak")
    outdir = os.path.join(tmp_path, "out")

    plan = uproot.graphed_write(g, outdir, steps_per_file=3, tree_name="events", compute=False)

    # NOT an array: a task graph of 3 write tasks, and nothing written yet
    assert isinstance(plan, Plan)
    assert len(plan.tasks) == 3
    assert os.listdir(outdir) == []

    # running the task graph (each task writes a partition, returns nothing) writes the part files
    result = ProcessExecutor(max_workers=2).run(plan)
    assert result.value is None  # write tasks return nothing
    assert sorted(os.listdir(outdir)) == ["part0.root", "part1.root", "part2.root"]


@pytest.mark.parametrize("executor", ["process", "thread"])
def test_compute_true_executes_via_executor(executor, tmp_path):
    src = _make_root(os.path.join(tmp_path, "in.root"), n=16)
    g = uproot.graphed(src, library="ak")
    outdir = os.path.join(tmp_path, f"out-{executor}")

    paths = uproot.graphed_write(g, outdir, steps_per_file=4, tree_name="events", executor=executor)

    assert len(paths) == 4
    assert all(os.path.exists(p) for p in paths)


def test_multifile_topology_writes_a_part_per_file_and_step(tmp_path):
    src1 = _make_root(os.path.join(tmp_path, "a.root"), n=10)
    src2 = _make_root(os.path.join(tmp_path, "b.root"), n=10)
    g = uproot.graphed([src1, src2], library="ak")
    outdir = os.path.join(tmp_path, "out")

    paths = uproot.graphed_write(g, outdir, steps_per_file=2, tree_name="events")

    assert len(paths) == 4  # 2 files x 2 steps
    assert sum(len(uproot.open(p + ":events").arrays()) for p in paths) == 20


def test_zmumu_roundtrip_via_partitioned_write(tmp_path):
    src = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    g = uproot.graphed(src, library="ak")
    outdir = os.path.join(tmp_path, "out")

    paths = uproot.graphed_write(g, outdir, steps_per_file=4, tree_name="events")

    ref = uproot.open(src).arrays()
    back = ak.concatenate([uproot.open(p + ":events").arrays() for p in paths])
    assert len(back) == len(ref)
    assert set(back.fields) == set(ref.fields)
    assert ak.almost_equal(back.px1, ref.px1)
    assert ak.almost_equal(back.py2, ref.py2)

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
"""``uproot.graphed_write`` corner cases: the compression settings of the part files, steps that
resolve empty, a user-supplied executor class, and the rejected arguments."""

import os

import awkward as ak
import numpy as np
import pytest

import uproot

pytest.importorskip("graphed_executors.local")
pytest.importorskip("graphed.awkward")

import graphed  # noqa: E402
from graphed.awkward import AwkwardBackend, AwkwardForm  # noqa: E402
from graphed_executors.local import ThreadExecutor  # noqa: E402


def _src(tmp_path, n=12, name="in.root"):
    p = os.path.join(tmp_path, name)
    with uproot.recreate(p) as f:
        f["events"] = {
            "x": np.arange(n, dtype="f8"),
            "y": np.arange(n, dtype="f8") * 2.0,
        }
    return p + ":events"


def test_compression_none_uses_the_writers_default(tmp_path):
    """``compression=None`` passes no compression setting to ``uproot.recreate``, so the part
    files carry the writer's default (``ZLIB(1)``) — unlike an explicit codec, which is honored.
    """
    src = _src(tmp_path)
    g = uproot.graphed(src, library="ak")

    default = uproot.graphed_write(
        g,
        os.path.join(tmp_path, "default"),
        steps_per_file=2,
        tree_name="events",
        executor="thread",
        compression=None,
    )
    explicit = uproot.graphed_write(
        g,
        os.path.join(tmp_path, "explicit"),
        steps_per_file=2,
        tree_name="events",
        executor="thread",
        compression="zstd",
        compression_level=5,
    )

    assert [uproot.open(p).file.compression for p in default] == [uproot.ZLIB(1)] * 2
    assert [uproot.open(p).file.compression for p in explicit] == [uproot.ZSTD(5)] * 2

    ref = uproot.open(src).arrays()
    back = ak.concatenate([uproot.open(p + ":events").arrays() for p in default])
    assert ak.array_equal(back.x, ref.x) and ak.array_equal(back.y, ref.y)


def test_steps_beyond_the_entry_count_write_no_empty_parts(tmp_path):
    """More steps than entries: the steps that resolve empty write nothing, so the reported paths
    are exactly the non-empty parts (numbering gaps and all) and are the only files in the
    destination."""
    src = _src(tmp_path, n=2, name="tiny.root")
    g = uproot.graphed(src, library="ak")
    outdir = os.path.join(tmp_path, "out")

    paths = uproot.graphed_write(
        g, outdir, steps_per_file=4, tree_name="events", executor="thread"
    )

    assert [os.path.basename(p) for p in paths] == [
        "part-00001.root",
        "part-00003.root",
    ]
    assert sorted(os.listdir(outdir)) == [os.path.basename(p) for p in paths]

    ref = uproot.open(src).arrays()
    back = ak.concatenate([uproot.open(p + ":events").arrays() for p in paths])
    assert ak.array_equal(back.x, ref.x) and ak.array_equal(back.y, ref.y)


def test_executor_class_is_instantiated_and_run(tmp_path):
    """An executor class passed instead of a name is the executor that runs the write plan."""
    runs = []

    class RecordingExecutor(ThreadExecutor):
        def run(self, plan):
            runs.append(len(plan.tasks))
            return super().run(plan)

    src = _src(tmp_path, n=6)
    g = uproot.graphed(src, library="ak")
    outdir = os.path.join(tmp_path, "out")

    paths = uproot.graphed_write(
        g,
        outdir,
        steps_per_file=3,
        tree_name="events",
        executor=RecordingExecutor,
        max_workers=2,
    )

    assert runs == [3]  # the user's executor ran, once, over the three write tasks
    ref = uproot.open(src).arrays()
    back = ak.concatenate([uproot.open(p + ":events").arrays() for p in paths])
    assert ak.array_equal(back.x, ref.x)


def test_rejects_an_array_that_is_not_a_graphed_array(tmp_path):
    outdir = os.path.join(tmp_path, "out")

    with pytest.raises(TypeError, match="expects a uproot.graphed Array"):
        uproot.graphed_write(
            ak.Array({"x": [1.0, 2.0]}), outdir, tree_name="events", executor="thread"
        )

    assert not os.path.exists(outdir)


def test_rejects_a_graphed_array_without_an_uproot_source(tmp_path):
    """A ``graphed.Array`` whose session has no ``uproot.graphed`` source has no partitions to
    write, so it is rejected before the destination is created."""
    data = ak.Array({"x": [1.0, 2.0]})
    session = graphed.Session(AwkwardBackend())
    array = session.source(
        "plain",
        form=AwkwardForm(ak.Array(data.layout.to_typetracer(forget_length=True))),
        data=data,
    )
    outdir = os.path.join(tmp_path, "out")

    with pytest.raises(TypeError, match="not backed by a uproot.graphed source"):
        uproot.graphed_write(array, outdir, tree_name="events", executor="thread")

    assert not os.path.exists(outdir)

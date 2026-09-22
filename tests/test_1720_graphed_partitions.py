# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
"""``uproot._graphed.graphed_partitions`` / ``read_graphed_partition``: ``uproot.dask``'s
chunking knobs as ``graphed.core.Partition`` lists for hand-built plans."""

from __future__ import annotations

import awkward as ak
import pytest
import skhep_testdata

import uproot

pytest.importorskip("graphed.awkward")

import graphed  # noqa: E402
from graphed.core import SequentialRunner  # noqa: E402

from uproot._graphed import graphed_partitions, read_graphed_partition  # noqa: E402

N = 2304  # entries in uproot-Zmumu.root


def _zmumu():
    return skhep_testdata.data_path("uproot-Zmumu.root") + ":events"


def _ranges(parts):
    return [(p.entry_start, p.entry_stop) for p in parts]


def test_steps_per_file_splits_into_contiguous_ranges():
    parts = graphed_partitions(_zmumu(), steps_per_file=5)
    assert _ranges(parts) == [((i * N) // 5, ((i + 1) * N) // 5) for i in range(5)]
    assert {(p.uri, p.tree) for p in parts} == {(_zmumu().split(":")[0], "/events;1")}
    assert _ranges(graphed_partitions(_zmumu())) == [(0, N)]


def test_step_size_caps_each_chunk():
    parts = graphed_partitions(_zmumu(), step_size=1000)
    assert _ranges(parts) == [(0, 1000), (1000, 2000), (2000, N)]
    by_memory = graphed_partitions(_zmumu(), step_size="50 kB")
    assert len(by_memory) > 1 and by_memory[-1].entry_stop == N
    assert all(a.entry_stop == b.entry_start for a, b in zip(by_memory, by_memory[1:]))


def test_blind_partitions_open_no_file_and_resolve_on_read():
    parts = graphed_partitions(
        "nowhere.root:events", steps_per_file=3, open_files=False
    )
    assert all(p.is_blind for p in parts) and [p.blind_step for p in parts] == [0, 1, 2]
    blind = graphed_partitions(_zmumu(), steps_per_file=3, open_files=False)
    eager = graphed_partitions(_zmumu(), steps_per_file=3)
    tree = uproot.open(_zmumu())
    assert [p.resolve(N) for p in blind] == [
        graphed.core.Partition(_zmumu().split(":")[0], "events", s, e)
        for s, e in _ranges(eager)
    ]
    chunk = read_graphed_partition(blind[1], ["px1"], tree=tree)
    assert ak.array_equal(
        chunk.px1, tree["px1"].array(entry_start=N // 3, entry_stop=(2 * N) // 3)
    )


def test_incompatible_knobs_are_refused():
    with pytest.raises(TypeError, match="mutually exclusive"):
        graphed_partitions(_zmumu(), step_size=10, steps_per_file=2)
    with pytest.raises(TypeError, match="cannot be used with open_files=False"):
        graphed_partitions(_zmumu(), step_size=10, open_files=False)


def test_hand_built_partitions_drive_a_plan():
    g = uproot.graphed(_zmumu())
    plan = graphed.aggregate_plan(
        g.px1,
        reduce=lambda vals: float(ak.sum(vals[0])),
        combine=lambda a, b: a + b,
        empty=lambda: 0.0,
        partitions=graphed_partitions(_zmumu(), step_size="50 kB"),
    )
    assert len(plan.tasks) > 1
    assert SequentialRunner().run(plan).value == pytest.approx(
        float(ak.sum(uproot.open(_zmumu())["px1"].array()))
    )

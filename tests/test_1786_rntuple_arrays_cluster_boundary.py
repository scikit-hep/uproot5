# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations

import numpy
import pytest

import uproot
import uproot.models.RNTuple

ak = pytest.importorskip("awkward")


def _data(start, stop):
    return ak.Array(
        {
            "x": numpy.arange(start, stop, dtype=numpy.int64),
            "jagged": [list(range(i % 5)) for i in range(start, stop)],
            "struct": [{"a": i, "b": float(i)} for i in range(start, stop)],
        }
    )


@pytest.fixture
def ntuple(tmp_path):
    path = tmp_path / "test.root"
    with uproot.recreate(path) as file:
        # two clusters, starting at entries 0 and 600
        writable = file.mkrntuple("ntuple", _data(0, 600))
        writable.extend(_data(600, 1000))
    with uproot.open(path) as file:
        yield file["ntuple"]


@pytest.fixture
def clusters_read(monkeypatch):
    read = []
    original = uproot.models.RNTuple.Model_ROOT_3a3a_RNTuple.read_cluster_range

    def spy(self, col_idx, start_cluster_idx, stop_cluster_idx, *args, **kwargs):
        read.append((int(start_cluster_idx), int(stop_cluster_idx)))
        return original(
            self, col_idx, start_cluster_idx, stop_cluster_idx, *args, **kwargs
        )

    monkeypatch.setattr(
        uproot.models.RNTuple.Model_ROOT_3a3a_RNTuple, "read_cluster_range", spy
    )
    return read


@pytest.mark.parametrize(
    ("entry_start", "entry_stop", "expected_clusters"),
    [
        (0, 600, {0}),  # entry_stop is exactly where cluster 1 starts
        (0, 599, {0}),
        (0, 601, {0, 1}),
        (100, 600, {0}),
        (600, 1000, {1}),
        (599, 601, {0, 1}),
        (0, 1000, {0, 1}),
        (0, 0, set()),
        (600, 600, set()),
    ],
)
def test_arrays_reads_only_needed_clusters(
    ntuple, clusters_read, entry_start, entry_stop, expected_clusters
):
    arrays = ntuple.arrays(entry_start=entry_start, entry_stop=entry_stop)
    assert {i for start, stop in clusters_read for i in range(start, stop)} == (
        expected_clusters
    )
    assert arrays.to_list() == _data(entry_start, entry_stop).to_list()


@pytest.mark.parametrize(
    ("entry_start", "entry_stop"),
    [(0, 600), (600, 600), (0, 0), (100, 600), (599, 601)],
)
def test_arrays_virtual_at_cluster_boundary(ntuple, entry_start, entry_stop):
    arrays = ntuple.arrays(entry_start=entry_start, entry_stop=entry_stop, virtual=True)
    assert ak.materialize(arrays).to_list() == _data(entry_start, entry_stop).to_list()


def test_iterate_cluster_aligned_steps(ntuple, clusters_read):
    chunks = list(ntuple.iterate(step_size=600, filter_name="x"))
    assert [len(chunk) for chunk in chunks] == [600, 400]
    # one column, one cluster per chunk
    assert clusters_read == [(0, 1), (1, 2)]
    assert ak.concatenate(chunks).x.to_list() == list(range(1000))

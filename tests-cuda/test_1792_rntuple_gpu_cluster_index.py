# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
from __future__ import annotations

import numpy
import pytest

import uproot

ak = pytest.importorskip("awkward")
cupy = pytest.importorskip("cupy")
pytestmark = pytest.mark.skipif(
    cupy.cuda.runtime.driverGetVersion() == 0, reason="No available CUDA driver."
)


def _data(start, stop):
    return ak.Array(
        {
            "x": numpy.arange(start, stop, dtype=numpy.int64),
            "jagged": [list(range(i % 5)) for i in range(start, stop)],
        }
    )


@pytest.fixture
def path(tmp_path):
    path = tmp_path / "test.root"
    # the GPU interpreter can decompress ZSTD and LZ4, but not ZLIB
    with uproot.recreate(path, compression=uproot.ZSTD(1)) as file:
        # three clusters, starting at entries 0, 300 and 600
        writable = file.mkrntuple("ntuple", _data(0, 300))
        writable.extend(_data(300, 600))
        writable.extend(_data(600, 1000))
    return path


@pytest.mark.parametrize(
    ("entry_start", "entry_stop"),
    [(300, 600), (600, 1000), (300, 1000), (450, 700), (999, 1000)],
)
def test_arrays_gpu_not_from_first_cluster(path, entry_start, entry_stop):
    with uproot.open(path) as file:
        arrays = file["ntuple"].arrays(
            entry_start=entry_start,
            entry_stop=entry_stop,
            backend="cuda",
            interpreter="gpu",
        )
    assert (
        ak.to_backend(arrays, "cpu").to_list()
        == _data(entry_start, entry_stop).to_list()
    )


def test_iterate_gpu(path):
    with uproot.open(path) as file:
        chunks = list(
            file["ntuple"].iterate(step_size=300, backend="cuda", interpreter="gpu")
        )
    assert [len(chunk) for chunk in chunks] == [300, 300, 300, 100]
    assert ak.to_backend(ak.concatenate(chunks), "cpu").to_list() == (
        _data(0, 1000).to_list()
    )

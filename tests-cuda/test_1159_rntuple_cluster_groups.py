# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
from __future__ import annotations

import pytest
import skhep_testdata

import uproot

ak = pytest.importorskip("awkward")
cupy = pytest.importorskip("cupy")
pytestmark = pytest.mark.skipif(
    cupy.cuda.runtime.driverGetVersion() == 0, reason="No available CUDA driver."
)


@pytest.mark.parametrize(
    ("backend", "interpreter", "library"),
    [("cuda", "cpu", cupy), ("cuda", "gpu", cupy)],
)
def test_multiple_cluster_groups(backend, interpreter, library):
    filename = skhep_testdata.data_path(
        "test_multiple_cluster_groups_rntuple_v1-0-0-0.root"
    )
    with uproot.open(filename) as f:
        obj = f["ntuple"]

        group_ids = [cluster.cluster_group_id for cluster in obj.clusters]
        assert group_ids == [0] * 5 + [1] * 4 + [2] * 3

        assert obj.num_entries == 1000

        arrays = obj.arrays(backend=backend, interpreter=interpreter)

        assert ak.all(arrays.one == library.array(list(range(1000))))
        assert ak.all(
            arrays.int_vector == library.array([[i, i + 1] for i in range(1000)])
        )

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
from __future__ import annotations

import pytest
import skhep_testdata

import uproot

try:
    import cupy
except:
    cupy = None
ak = pytest.importorskip("awkward")


@pytest.mark.parametrize(
    "backend,GDS,library",
    [
        ("cuda", False, cupy),
        pytest.param(
            "cuda",
            True,
            cupy,
            marks=pytest.mark.skipif(
                cupy is None, reason="could not import 'cupy': No module named 'cupy'"
            ),
        ),
    ],
)
def test_schema_extension(backend, GDS, library):
    if GDS and cupy.cuda.runtime.driverGetVersion() == 0:
        pytest.skip("No available CUDA driver.")
    filename = skhep_testdata.data_path("test_index_multicluster_rntuple_v1-0-0-0.root")
    with uproot.open(filename) as f:
        obj = f["ntuple"]

        arrays = obj.arrays(backend=backend, use_GDS=GDS)
        int_vec_array = arrays["int_vector"]

        for j in range(2):
            for i in range(100):
                assert int_vec_array[i + j * 100, 0] == i
                assert int_vec_array[i + j * 100, 1] == i + j

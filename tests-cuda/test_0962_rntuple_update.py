# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import pytest
import uproot
import awkward as ak
import skhep_testdata
import numpy

try:
    import cupy
except ImportError:
    cupy = None


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
def test_new_support_RNTuple_split_int32_reading(backend, GDS, library):
    if GDS and cupy.cuda.runtime.driverGetVersion() == 0:
        pytest.skip("No available CUDA driver.")
    with uproot.open(
        skhep_testdata.data_path("test_int_5e4_rntuple_v1-0-0-0.root")
    ) as f:
        obj = f["ntuple"]
        df = obj.arrays(backend=backend, use_GDS=GDS)
        assert len(df) == 5e4
        assert len(df.one_integers) == 5e4
        assert ak.all(df.one_integers == library.arange(5e4 + 1)[::-1][:-1])


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
def test_new_support_RNTuple_bit_bool_reading(backend, GDS, library):
    if GDS and cupy.cuda.runtime.driverGetVersion() == 0:
        pytest.skip("No available CUDA driver.")
    with uproot.open(skhep_testdata.data_path("test_bit_rntuple_v1-0-0-0.root")) as f:
        obj = f["ntuple"]
        df = obj.arrays(backend=backend, use_GDS=GDS)
        assert ak.all(df.one_bit == library.asarray([1, 0, 0, 1, 0, 0, 1, 0, 0, 1]))


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
def test_new_support_RNTuple_split_int16_reading(backend, GDS, library):
    if GDS and cupy.cuda.runtime.driverGetVersion() == 0:
        pytest.skip("No available CUDA driver.")
    with uproot.open(
        skhep_testdata.data_path("test_int_multicluster_rntuple_v1-0-0-0.root")
    ) as f:
        obj = f["ntuple"]
        df = obj.arrays(backend=backend, use_GDS=GDS)
        assert len(df.one_integers) == 1e8
        assert df.one_integers[0] == 2
        assert df.one_integers[-1] == 1
        assert ak.all(
            library.unique(df.one_integers[: len(df.one_integers) // 2])
            == library.array([2])
        )
        assert ak.all(
            library.unique(df.one_integers[len(df.one_integers) / 2 + 1 :])
            == library.array([1])
        )

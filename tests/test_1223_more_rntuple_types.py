# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
import pytest
import skhep_testdata

import uproot

import numpy
try:
    import cupy
except ImportError:
    cupy = None
ak = pytest.importorskip("awkward")


@pytest.mark.parametrize(
    "backend,GDS,library", [("cpu", False, numpy),
                            pytest.param(
                            "cuda", True, cupy, marks = pytest.mark.skipif(cupy is None, reason = "could not import 'cupy': No module named 'cupy'")
                            ),
                           ]
)
def test_atomic(backend, GDS, library):
    if GDS and cupy.cuda.runtime.driverGetVersion() == 0:
        pytest.skip("No available CUDA driver.")
    filename = skhep_testdata.data_path("test_atomic_bitset_rntuple_v1-0-0-0.root")
    with uproot.open(filename) as f:
        obj = f["ntuple"]

        a = obj.arrays("atomic_int", backend=backend, use_GDS=GDS)

        assert ak.all(a.atomic_int == library.array([1, 2, 3]))


@pytest.mark.parametrize(
    "backend,GDS,library", [("cpu", False, numpy),
                            pytest.param(
                            "cuda", True, cupy, marks = pytest.mark.skipif(cupy is None, reason = "could not import 'cupy': No module named 'cupy'")
                            ),
                           ]
)
def test_bitset(backend, GDS, library):
    if GDS and cupy.cuda.runtime.driverGetVersion() == 0:
        pytest.skip("No available CUDA driver.")
    filename = skhep_testdata.data_path("test_atomic_bitset_rntuple_v1-0-0-0.root")
    with uproot.open(filename) as f:
        obj = f["ntuple"]

        a = obj.arrays("bitset", backend=backend, use_GDS=GDS)

        assert len(a.bitset) == 3
        assert len(a.bitset[0]) == 42
        assert ak.all(a.bitset[0][:6] == library.array([0, 1, 0, 1, 0, 1]))
        assert ak.all(a.bitset[0][6:] == 0)
        assert ak.all(
            a.bitset[1][:16]
            == library.array(
                [
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                ]
            )
        )
        assert ak.all(a.bitset[1][16:] == 0)
        assert ak.all(
            a.bitset[2][:16]
            == library.array(
                [
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    1,
                ]
            )
        )
        assert ak.all(a.bitset[2][16:] == 0)


@pytest.mark.parametrize(
    "backend,GDS,library", [("cpu", False, numpy),
                            pytest.param(
                            "cuda", True, cupy, marks = pytest.mark.skipif(cupy is None, reason = "could not import 'cupy': No module named 'cupy'")
                            ),
                           ]
)
def test_empty_struct(backend, GDS, library):
    if GDS and cupy.cuda.runtime.driverGetVersion() == 0:
        pytest.skip("No available CUDA driver.")
    filename = skhep_testdata.data_path(
        "test_emptystruct_invalidvar_rntuple_v1-0-0-0.root"
    )
    with uproot.open(filename) as f:
        obj = f["ntuple"]

        a = obj.arrays("empty_struct", backend=backend, use_GDS=GDS)

        assert a.empty_struct.tolist() == [(), (), ()]


# cupy doesn't support None or object dtype like numpy
def test_invalid_variant():
    filename = skhep_testdata.data_path(
        "test_emptystruct_invalidvar_rntuple_v1-0-0-0.root"
    )
    with uproot.open(filename) as f:
        obj = f["ntuple"]

        a = obj.arrays("variant.*")

        assert a.variant.tolist() == [1, None, {"i": 2}]

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations

import os

import numpy as np
import pytest

import uproot

da = pytest.importorskip("dask.array")


@pytest.fixture
def sample_rntuples(tmp_path):
    f1 = os.path.join(tmp_path, "sample1.root")
    f2 = os.path.join(tmp_path, "sample2.root")
    x1 = np.arange(10.0)
    y1 = np.arange(10, dtype=np.int32) * 3
    x2 = np.arange(10.0, 25.0)
    y2 = np.arange(10, 25, dtype=np.int32) * 3

    with uproot.recreate(f1) as file:
        file["ntuple"] = {"x": x1, "y": y1}
    with uproot.recreate(f2) as file:
        file["ntuple"] = {"x": x2, "y": y2}

    return f1, f2, x1, y1, x2, y2


def test_rntuple_dask_numpy_reads(sample_rntuples):
    f1, f2, x1, y1, x2, y2 = sample_rntuples

    # Single-file read
    dask_arrays = uproot.dask(f"{f1}:ntuple", library="np")
    assert set(dask_arrays.keys()) == {"x", "y"}
    computed_x = dask_arrays["x"].compute()
    computed_y = dask_arrays["y"].compute()
    assert np.array_equal(computed_x, x1)
    assert np.array_equal(computed_y, y1)
    assert computed_x.dtype == x1.dtype
    assert computed_y.dtype == y1.dtype

    # Filter name
    filtered = uproot.dask(f"{f1}:ntuple", filter_name="x", library="np")
    assert list(filtered.keys()) == ["x"]
    assert np.array_equal(filtered["x"].compute(), x1)

    # Step size
    stepped = uproot.dask(f"{f1}:ntuple", step_size=3, library="np")
    assert stepped["x"].chunks == ((3, 3, 3, 1),)
    assert np.array_equal(stepped["x"].compute(), x1)

    # Multi-file read
    multi = uproot.dask([f"{f1}:ntuple", f"{f2}:ntuple"], library="np")
    assert set(multi.keys()) == {"x", "y"}
    assert np.array_equal(multi["x"].compute(), np.concatenate([x1, x2]))
    assert np.array_equal(multi["y"].compute(), np.concatenate([y1, y2]))


def test_rntuple_dask_numpy_delayed_open(sample_rntuples):
    f1, f2, x1, _, x2, _ = sample_rntuples

    dask_arrays = uproot.dask(
        [f"{f1}:ntuple", f"{f2}:ntuple"],
        open_files=False,
        library="np",
    )
    assert dask_arrays["x"].chunks == ((np.nan, np.nan),)
    assert np.array_equal(dask_arrays["x"].compute(), np.concatenate([x1, x2]))


def test_rntuple_dask_numpy_jagged_and_strings(tmp_path):
    ak = pytest.importorskip("awkward")
    filepath = os.path.join(tmp_path, "rntuple_complex.root")
    data_jagged = [[1.0, 2.0], [3.0], []]
    strings = ["a", "bc", "def"]

    with uproot.recreate(filepath) as file:
        file.mkrntuple("ntuple", ak.Array({"jagged": data_jagged, "strings": strings}))

    dask_arrays = uproot.dask(f"{filepath}:ntuple", library="np")
    assert dask_arrays["jagged"].dtype == np.dtype(object)
    assert dask_arrays["jagged"].shape == (3,)
    assert dask_arrays["strings"].dtype == np.dtype(object)
    assert dask_arrays["strings"].shape == (3,)

    computed_j = dask_arrays["jagged"].compute()
    assert computed_j.dtype == np.dtype(object)
    assert len(computed_j) == 3
    assert np.array_equal(computed_j[0], np.array([1.0, 2.0]))
    assert np.array_equal(computed_j[1], np.array([3.0]))
    assert len(computed_j[2]) == 0

    computed_s = dask_arrays["strings"].compute()
    assert computed_s.dtype == np.dtype(object)
    assert list(computed_s) == strings


def test_rntuple_dask_numpy_inconsistent_dtype(tmp_path):
    filepath1 = os.path.join(tmp_path, "rntuple_inc1.root")
    filepath2 = os.path.join(tmp_path, "rntuple_inc2.root")

    with uproot.recreate(filepath1) as file:
        file["ntuple"] = {"x": np.arange(5, dtype=np.float64)}
    with uproot.recreate(filepath2) as file:
        file["ntuple"] = {"x": np.arange(5, dtype=np.int32)}

    with pytest.raises(ValueError, match="inconsistent NumPy dtype for field"):
        uproot.dask([f"{filepath1}:ntuple", f"{filepath2}:ntuple"], library="np")


def test_rntuple_dask_numpy_fixed_array_dimension_mismatch(tmp_path):
    ak = pytest.importorskip("awkward")
    filepath1 = os.path.join(tmp_path, "rntuple_fix1.root")
    filepath2 = os.path.join(tmp_path, "rntuple_fix2.root")

    with uproot.recreate(filepath1) as file:
        file.mkrntuple("ntuple", ak.Array({"fixed": np.ones((5, 3), dtype=np.int32)}))
    with uproot.recreate(filepath2) as file:
        file.mkrntuple("ntuple", ak.Array({"fixed": np.ones((5, 4), dtype=np.int32)}))

    with pytest.raises(ValueError, match="inconsistent NumPy dtype for field"):
        uproot.dask([f"{filepath1}:ntuple", f"{filepath2}:ntuple"], library="np")


def test_ttree_dask_numpy_fixed_array_dimension_mismatch(tmp_path):
    filepath1 = os.path.join(tmp_path, "tree_fix1.root")
    filepath2 = os.path.join(tmp_path, "tree_fix2.root")

    with uproot.recreate(filepath1) as file:
        file.mktree("tree", {"arr": "3*int32"})
        file["tree"].extend({"arr": np.ones((5, 3), dtype=np.int32)})
    with uproot.recreate(filepath2) as file:
        file.mktree("tree", {"arr": "4*int32"})
        file["tree"].extend({"arr": np.ones((5, 4), dtype=np.int32)})

    with pytest.raises(ValueError, match="inconsistent NumPy dtype for branch"):
        uproot.dask([f"{filepath1}:tree", f"{filepath2}:tree"], library="np")

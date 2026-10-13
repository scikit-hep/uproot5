# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations

import os

import numpy as np
import pytest

import uproot

da = pytest.importorskip("dask.array")


def test_rntuple_dask_numpy_single_file(tmp_path):
    filepath = os.path.join(tmp_path, "rntuple_single.root")
    data_x = np.arange(10.0)
    data_y = np.arange(10, dtype=np.int32) * 3

    with uproot.recreate(filepath) as file:
        file["ntuple"] = {"x": data_x, "y": data_y}

    dask_arrays = uproot.dask(f"{filepath}:ntuple", library="np")
    assert set(dask_arrays.keys()) == {"x", "y"}

    computed_x = dask_arrays["x"].compute()
    computed_y = dask_arrays["y"].compute()

    assert np.array_equal(computed_x, data_x)
    assert np.array_equal(computed_y, data_y)
    assert computed_x.dtype == data_x.dtype
    assert computed_y.dtype == data_y.dtype


def test_rntuple_dask_numpy_multi_file(tmp_path):
    filepath1 = os.path.join(tmp_path, "rntuple_multi1.root")
    filepath2 = os.path.join(tmp_path, "rntuple_multi2.root")

    x1 = np.arange(10.0)
    y1 = np.arange(10, dtype=np.int32)
    x2 = np.arange(10.0, 25.0)
    y2 = np.arange(10, 25, dtype=np.int32)

    with uproot.recreate(filepath1) as file:
        file["ntuple"] = {"x": x1, "y": y1}
    with uproot.recreate(filepath2) as file:
        file["ntuple"] = {"x": x2, "y": y2}

    dask_arrays = uproot.dask(
        [f"{filepath1}:ntuple", f"{filepath2}:ntuple"], library="np"
    )

    assert set(dask_arrays.keys()) == {"x", "y"}
    assert np.array_equal(dask_arrays["x"].compute(), np.concatenate([x1, x2]))
    assert np.array_equal(dask_arrays["y"].compute(), np.concatenate([y1, y2]))


def test_rntuple_dask_numpy_delayed_open(tmp_path):
    filepath1 = os.path.join(tmp_path, "rntuple_del1.root")
    filepath2 = os.path.join(tmp_path, "rntuple_del2.root")

    x1 = np.arange(8.0)
    x2 = np.arange(8.0, 16.0)

    with uproot.recreate(filepath1) as file:
        file["ntuple"] = {"x": x1}
    with uproot.recreate(filepath2) as file:
        file["ntuple"] = {"x": x2}

    dask_arrays = uproot.dask(
        [f"{filepath1}:ntuple", f"{filepath2}:ntuple"],
        open_files=False,
        library="np",
    )

    assert dask_arrays["x"].chunks == ((np.nan, np.nan),)
    assert np.array_equal(dask_arrays["x"].compute(), np.concatenate([x1, x2]))


def test_rntuple_dask_numpy_step_size(tmp_path):
    filepath = os.path.join(tmp_path, "rntuple_step.root")
    data_x = np.arange(10.0)

    with uproot.recreate(filepath) as file:
        file["ntuple"] = {"x": data_x}

    dask_arrays = uproot.dask(f"{filepath}:ntuple", step_size=3, library="np")
    assert dask_arrays["x"].chunks == ((3, 3, 3, 1),)
    assert np.array_equal(dask_arrays["x"].compute(), data_x)


def test_rntuple_dask_numpy_filter_name(tmp_path):
    filepath = os.path.join(tmp_path, "rntuple_filter.root")

    with uproot.recreate(filepath) as file:
        file["ntuple"] = {"x": np.arange(5.0), "y": np.arange(5, dtype=np.int32)}

    dask_arrays = uproot.dask(f"{filepath}:ntuple", filter_name="x", library="np")
    assert list(dask_arrays.keys()) == ["x"]
    assert np.array_equal(dask_arrays["x"].compute(), np.arange(5.0))


def test_rntuple_dask_numpy_inconsistent_dtype(tmp_path):
    filepath1 = os.path.join(tmp_path, "rntuple_inc1.root")
    filepath2 = os.path.join(tmp_path, "rntuple_inc2.root")

    with uproot.recreate(filepath1) as file:
        file["ntuple"] = {"x": np.arange(5, dtype=np.float64)}
    with uproot.recreate(filepath2) as file:
        file["ntuple"] = {"x": np.arange(5, dtype=np.int32)}

    with pytest.raises(ValueError, match="inconsistent NumPy dtype for field"):
        uproot.dask([f"{filepath1}:ntuple", f"{filepath2}:ntuple"], library="np")


def test_rntuple_dask_numpy_jagged_field(tmp_path):
    ak = pytest.importorskip("awkward")
    filepath = os.path.join(tmp_path, "rntuple_jagged.root")
    data_jagged = [[1.0, 2.0], [3.0], []]

    with uproot.recreate(filepath) as file:
        file.mkrntuple("ntuple", ak.Array({"jagged": data_jagged}))

    dask_arrays = uproot.dask(f"{filepath}:ntuple", library="np")
    assert dask_arrays["jagged"].dtype == np.dtype(object)
    assert dask_arrays["jagged"].shape == (3,)

    computed = dask_arrays["jagged"].compute()
    assert computed.dtype == np.dtype(object)
    assert len(computed) == 3
    assert np.array_equal(computed[0], np.array([1.0, 2.0]))
    assert np.array_equal(computed[1], np.array([3.0]))
    assert len(computed[2]) == 0


def test_rntuple_dask_numpy_string_field(tmp_path):
    ak = pytest.importorskip("awkward")
    filepath = os.path.join(tmp_path, "rntuple_str.root")
    strings = ["a", "bc", "def"]

    with uproot.recreate(filepath) as file:
        file.mkrntuple("ntuple", ak.Array({"s": strings}))

    dask_arrays = uproot.dask(f"{filepath}:ntuple", library="np")
    assert dask_arrays["s"].dtype == np.dtype(object)
    assert dask_arrays["s"].shape == (3,)

    computed = dask_arrays["s"].compute()
    assert computed.dtype == np.dtype(object)
    assert list(computed) == strings


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

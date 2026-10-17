# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations

import os

import awkward as ak
import numpy as np
import pytest

import uproot


def test_issue1760_reproduce_iterate_numpy_and_pandas(tmp_path):
    filepath = os.path.join(tmp_path, "issue1760.root")
    data = ak.Array(
        {
            "x": [[1.0, 2.0], [3.0, 4.0], [5.0], [6.0, 7.0]],
            "s": ["a", "bc", "def", "g"],
        }
    )
    with uproot.recreate(filepath) as f:
        f.mkrntuple("ntuple", data)

    ntuple = uproot.open(filepath)["ntuple"]

    # Verify library="np" with iterate:
    # First chunk has equal-length lists [2, 2], but must still be 1D object array.
    for arrays in ntuple.iterate(step_size=2, library="np"):
        assert arrays["x"].dtype == np.dtype(object)
        assert arrays["x"].shape == (2,)
        assert isinstance(arrays["x"][0], np.ndarray)
        assert arrays["s"].dtype == np.dtype(object)
        assert arrays["s"].shape == (2,)
        assert isinstance(arrays["s"][0], str)

    # Verify library="pd" with iterate:
    # Equal-length lists in chunk 0 must not raise ValueError.
    pd = pytest.importorskip("pandas")
    dfs = list(ntuple.iterate(step_size=2, library="pd"))
    assert len(dfs) == 2
    for df in dfs:
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ["x", "s"]


def test_rfield_array_numpy_and_pandas(tmp_path):
    filepath = os.path.join(tmp_path, "field_arrays.root")
    data = ak.Array(
        {
            "x": [[1.0, 2.0], [3.0, 4.0], [5.0], [6.0, 7.0]],
            "s": ["a", "bc", "def", "g"],
        }
    )
    with uproot.recreate(filepath) as f:
        f.mkrntuple("ntuple", data)

    ntuple = uproot.open(filepath)["ntuple"]

    arr_x_np = ntuple["x"].array(library="np")
    assert arr_x_np.dtype == np.dtype(object)
    assert arr_x_np.shape == (4,)
    assert np.array_equal(arr_x_np[0], np.array([1.0, 2.0]))

    arr_s_np = ntuple["s"].array(library="np")
    assert arr_s_np.dtype == np.dtype(object)
    assert arr_s_np.shape == (4,)
    assert list(arr_s_np) == ["a", "bc", "def", "g"]

    pd = pytest.importorskip("pandas")
    ser_x_pd = ntuple["x"].array(library="pd")
    assert isinstance(ser_x_pd, pd.Series)
    assert len(ser_x_pd) == 4

    ser_s_pd = ntuple["s"].array(library="pd")
    assert isinstance(ser_s_pd, pd.Series)
    assert len(ser_s_pd) == 4


def test_rntuple_numpy_fixed_size_regular_array(tmp_path):
    filepath = os.path.join(tmp_path, "regular_array.root")
    data = ak.Array(
        {
            "reg": ak.to_regular(ak.Array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])),
        }
    )
    with uproot.recreate(filepath) as f:
        f.mkrntuple("ntuple", data)

    ntuple = uproot.open(filepath)["ntuple"]

    arr_np = ntuple["reg"].array(library="np")
    assert arr_np.dtype == np.float64
    assert arr_np.shape == (3, 2)
    assert np.array_equal(arr_np[0], np.array([1.0, 2.0]))

    for chunk in ntuple.iterate(step_size=2, library="np"):
        assert chunk["reg"].dtype == np.float64
        assert chunk["reg"].ndim == 2
        assert chunk["reg"].shape[1] == 2


def test_rntuple_numpy_struct_form_driven(tmp_path):
    filepath = os.path.join(tmp_path, "struct.root")
    data = ak.Array(
        {
            "st": [
                {"id": 1, "v": [1.0, 2.0], "tag": "alpha"},
                {"id": 2, "v": [3.0], "tag": "beta"},
            ]
        }
    )
    with uproot.recreate(filepath) as f:
        f.mkrntuple("ntuple", data)

    ntuple = uproot.open(filepath)["ntuple"]

    arr_np = ntuple["st"].array(library="np")
    expected_dtype = np.dtype([("id", "<i8"), ("v", "O"), ("tag", "O")])
    assert arr_np.dtype == expected_dtype
    assert len(arr_np) == 2
    assert arr_np[0]["id"] == 1
    assert np.array_equal(arr_np[0]["v"], np.array([1.0, 2.0]))
    assert arr_np[0]["tag"] == "alpha"


def test_rntuple_numpy_optional_fields(tmp_path):
    filepath = os.path.join(tmp_path, "optionals.root")
    data = ak.Array(
        {
            "opt_num": [1.0, None, 3.0],
            "opt_list": [None, [1.0, 2.0], None],
            "opt_str": ["first", None, "third"],
        }
    )
    with uproot.recreate(filepath) as f:
        f.mkrntuple("ntuple", data)

    ntuple = uproot.open(filepath)["ntuple"]

    opt_num_np = ntuple["opt_num"].array(library="np")
    assert isinstance(opt_num_np, np.ma.MaskedArray)
    assert opt_num_np[0] == 1.0
    assert opt_num_np.mask[1]

    opt_list_np = ntuple["opt_list"].array(library="np")
    assert opt_list_np.dtype == np.dtype(object)
    assert opt_list_np[0] is None
    assert np.array_equal(opt_list_np[1], np.array([1.0, 2.0]))

    opt_str_np = ntuple["opt_str"].array(library="np")
    assert opt_str_np.dtype == np.dtype(object)
    assert opt_str_np[0] == "first"
    assert opt_str_np[1] is None
    assert opt_str_np[2] == "third"


def test_rntuple_empty_slice(tmp_path):
    filepath = os.path.join(tmp_path, "empty.root")
    data = ak.Array(
        {
            "x": [[1.0, 2.0], [3.0, 4.0]],
            "s": ["a", "bc"],
            "r": ak.to_regular(ak.Array([[10, 20], [30, 40]])),
        }
    )
    with uproot.recreate(filepath) as f:
        f.mkrntuple("ntuple", data)

    ntuple = uproot.open(filepath)["ntuple"]

    empty_arrays = ntuple.arrays(entry_start=0, entry_stop=0, library="np")
    assert empty_arrays["x"].dtype == np.dtype(object)
    assert empty_arrays["x"].shape == (0,)
    assert empty_arrays["s"].dtype == np.dtype(object)
    assert empty_arrays["s"].shape == (0,)
    assert empty_arrays["r"].shape == (0, 2)

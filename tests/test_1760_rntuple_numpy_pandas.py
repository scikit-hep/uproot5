# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations

import os

import awkward as ak
import numpy as np
import pytest

import uproot
from uproot.behaviors.RNTuple import (
    _form_to_numpy_dtype,
    _is_string_form,
    _rntuple_to_numpy,
)


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


def test_rntuple_numpy_regular_array_of_structs(tmp_path):
    filepath = os.path.join(tmp_path, "reg_struct.root")
    s = ak.to_regular(ak.Array([[{"a": 1, "b": 2.0}] * 3] * 5), axis=1)
    with uproot.recreate(filepath) as f:
        f.mkrntuple("ntuple", ak.Array({"s": s}))

    ntuple = uproot.open(filepath)["ntuple"]

    arr = ntuple["s"].array(library="np")
    assert arr.shape == (5, 3)
    assert arr.dtype == np.dtype([("a", "<i8"), ("b", "<f8")])

    empty_arr = ntuple["s"].array(entry_start=0, entry_stop=0, library="np")
    assert empty_arr.shape == (0, 3)
    assert empty_arr.dtype == np.dtype([("a", "<i8"), ("b", "<f8")])

    res_empty = _rntuple_to_numpy(s[:0])
    assert res_empty.shape == (0, 3)
    assert res_empty.dtype == np.dtype([("a", "<i8"), ("b", "<f8")])


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
            "opt_l": [None, [1, 2]],
            "st": [{"a": 1}, {"a": 2}],
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
    assert empty_arrays["opt_l"].dtype == np.dtype(object)
    assert empty_arrays["opt_l"].shape == (0,)
    assert len(empty_arrays["st"]) == 0


def test_rntuple_numpy_cannot_convert_union(tmp_path):
    filepath = os.path.join(tmp_path, "union.root")
    data = ak.Array({"u": [1, "two"]})
    with uproot.recreate(filepath) as f:
        f.mkrntuple("ntuple", data)

    ntuple = uproot.open(filepath)["ntuple"]

    with pytest.raises(ValueError, match="Field u cannot be converted to NumPy/Pandas"):
        ntuple.arrays(library="np")

    with pytest.raises(ValueError, match="Field u cannot be converted to NumPy/Pandas"):
        ntuple["u"].array(library="np")

    pytest.importorskip("pandas")
    with pytest.raises(ValueError, match="Field u cannot be converted to NumPy/Pandas"):
        ntuple["u"].array(library="pd")


def test_rntuple_numpy_helper_branches():
    # Non ak.Array input
    assert np.array_equal(_rntuple_to_numpy([1, 2, 3]), np.array([1, 2, 3]))

    # ak.contents.Content input normalization
    cont_arr = ak.Array([None, [1, 2]]).layout
    res_cont = _rntuple_to_numpy(cont_arr)
    assert res_cont.dtype == object
    assert res_cont[0] is None
    assert np.array_equal(res_cont[1], np.array([1, 2]))

    # Bytestring
    arr_bytes = ak.Array([b"abc", b"def"])
    res_bytes = _rntuple_to_numpy(arr_bytes)
    assert res_bytes.dtype == object
    assert res_bytes[0] == b"abc"

    # ByteMasked string
    bm_str = ak.Array(
        ak.contents.ByteMaskedArray(
            ak.index.Index8([0, 1]), ak.Array(["a", "b"]).layout, True
        )
    )
    assert _is_string_form(bm_str.layout.form)
    res_bm = _rntuple_to_numpy(bm_str)
    assert res_bm[0] is None
    assert res_bm[1] == "b"

    # BitMasked string
    bit_str = ak.Array(
        ak.contents.BitMaskedArray(
            ak.index.IndexU8([1]), ak.Array(["a"]).layout, True, 1, True
        )
    )
    assert _is_string_form(bit_str.layout.form)
    assert _rntuple_to_numpy(bit_str)[0] == "a"

    # Unmasked string
    unm_str = ak.Array(ak.contents.UnmaskedArray(ak.Array(["a"]).layout))
    assert _is_string_form(unm_str.layout.form)
    assert _rntuple_to_numpy(unm_str)[0] == "a"

    # Char string form
    char_form = ak.forms.ListOffsetForm(
        "i64", ak.forms.NumpyForm("uint8", parameters={"__array__": "char"})
    )
    assert _is_string_form(char_form)

    # ListArray (ListForm)
    list_arr = ak.Array(
        ak.contents.ListArray(
            ak.index.Index64([0]),
            ak.index.Index64([2]),
            ak.Array([1.0, 2.0]).layout,
        )
    )
    res_la = _rntuple_to_numpy(list_arr)
    assert res_la.dtype == object
    assert np.array_equal(res_la[0], np.array([1.0, 2.0]))

    # Empty option array of non-numpy (struct)
    empty_opt_struct = ak.Array([None, {"a": 1}])[:0]
    res_eos = _rntuple_to_numpy(empty_opt_struct)
    assert res_eos.dtype == object
    assert len(res_eos) == 0

    # Empty list array
    empty_list = ak.Array([[1, 2]])[:0]
    res_el = _rntuple_to_numpy(empty_list)
    assert res_el.dtype == object
    assert len(res_el) == 0

    # Form to numpy dtype for RegularForm of RecordForm
    reg_rec_form = ak.to_regular(
        ak.Array([[{"a": 1, "b": 2.0}], [{"a": 3, "b": 4.0}]])
    ).layout.form
    dt = _form_to_numpy_dtype(reg_rec_form)
    assert dt == np.dtype(([("a", "<i8"), ("b", "<f8")], (1,)))

    # Form to numpy dtype for option forms returns object
    opt_form = ak.Array([1.0, None]).layout.form
    assert _form_to_numpy_dtype(opt_form) == np.dtype(object)

    opt_struct_form = ak.Array([{"a": 1}, None]).layout.form
    assert _form_to_numpy_dtype(opt_struct_form) == np.dtype(object)


def test_rntuple_struct_optional_subfield_preserves_mask(tmp_path):
    filepath = os.path.join(tmp_path, "rec_opt.root")
    data = ak.Array(
        {
            "rec": [{"a": 1, "b": None}, {"a": 2, "b": 5}],
        }
    )
    with uproot.recreate(filepath) as f:
        f.mkrntuple("ntuple", data)

    ntuple = uproot.open(filepath)["ntuple"]
    arr = ntuple["rec"].array(library="np")
    assert isinstance(arr, np.ma.MaskedArray)
    assert arr.dtype == np.dtype([("a", "<i8"), ("b", "<i8")])
    assert arr["a"][0] == 1
    assert arr["a"][1] == 2
    assert arr.mask["b"][0]
    assert not arr.mask["b"][1]
    assert arr["b"][1] == 5


def test_rntuple_struct_holding_fixed_size_array_of_lists(tmp_path):
    filepath = os.path.join(tmp_path, "recreg.root")
    regl = ak.to_regular(ak.Array([[[1], [2, 3]], [[4], [5]]]), axis=1)
    data = ak.Array({"st": {"r": regl}})
    with uproot.recreate(filepath) as f:
        f.mkrntuple("ntuple", data)

    ntuple = uproot.open(filepath)["ntuple"]
    arr = ntuple["st"].array(library="np")
    assert arr.shape == (2,)
    assert arr.dtype == np.dtype([("r", "O", (2,))])
    assert arr["r"].shape == (2, 2)
    assert np.array_equal(arr["r"][0, 0], np.array([1]))
    assert np.array_equal(arr["r"][0, 1], np.array([2, 3]))

    # Empty slice of top-level field preserves inner shape (0, 2)
    empty_slice = ntuple["st"]["r"].array(entry_start=0, entry_stop=0, library="np")
    assert empty_slice.shape == (0, 2)
    assert empty_slice.dtype == np.dtype(object)


def test_rntuple_pandas_fixed_size_arrays(tmp_path):
    pd = pytest.importorskip("pandas")
    filepath = os.path.join(tmp_path, "reg_pandas.root")
    regl = ak.to_regular(ak.Array([[[1], [2, 3]], [[4], [5]]]), axis=1)
    regs = ak.to_regular(ak.Array([["x", "yz"], ["abc", "d"]]), axis=1)
    regn = ak.to_regular(ak.Array([[1.0, 2.0], [3.0, 4.0]]), axis=1)
    data = ak.Array({"regl": regl, "regs": regs, "regn": regn})
    with uproot.recreate(filepath) as f:
        f.mkrntuple("ntuple", data)

    ntuple = uproot.open(filepath)["ntuple"]

    df = ntuple.arrays(library="pd")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df.columns) == ["regl", "regs", "regn"]

    s_regl = ntuple["regl"].array(library="pd")
    assert isinstance(s_regl, pd.Series)
    assert len(s_regl) == 2

    s_regs = ntuple["regs"].array(library="pd")
    assert isinstance(s_regs, pd.Series)
    assert len(s_regs) == 2

    s_regn = ntuple["regn"].array(library="pd")
    assert isinstance(s_regn, pd.Series)
    assert len(s_regn) == 2


def test_rntuple_dask_fixed_size_and_optionals(tmp_path):
    pytest.importorskip("dask.array")
    filepath = os.path.join(tmp_path, "dask_reg.root")
    regl = ak.to_regular(ak.Array([[[1], [2, 3]], [[4], [5]]]), axis=1)
    opt_num = ak.Array([1.0, None])
    reg_opt = ak.Array(ak.contents.RegularArray(ak.Array([1, None, None, 3]).layout, 2))
    num_opt_rec = ak.Array([{"a": 1, "b": None}, {"a": 2, "b": 3}])
    data = ak.Array(
        {
            "regl": regl,
            "opt_num": opt_num,
            "reg_opt": reg_opt,
            "num_opt_rec": num_opt_rec,
        }
    )
    with uproot.recreate(filepath) as f:
        f.mkrntuple("ntuple", data)

    dask_dict = uproot.dask(filepath + ":ntuple", library="np")

    d_regl = dask_dict["regl"]
    assert d_regl.shape == (2, 2)
    comp_regl = d_regl.compute()
    assert comp_regl.shape == (2, 2)

    d_opt = dask_dict["opt_num"]
    assert d_opt.dtype == np.dtype(object)
    comp_opt = d_opt.compute()
    assert comp_opt.dtype == np.dtype(object)
    assert comp_opt[0] == 1.0
    assert comp_opt[1] is None

    d_reg_opt = dask_dict["reg_opt"]
    assert d_reg_opt.dtype == np.dtype(object)
    assert d_reg_opt.shape == (2, 2)
    comp_reg_opt = d_reg_opt.compute()
    assert comp_reg_opt.dtype == np.dtype(object)
    assert comp_reg_opt.shape == (2, 2)
    assert comp_reg_opt[0, 0] == 1
    assert comp_reg_opt[0, 1] is None
    assert comp_reg_opt[1, 0] is None
    assert comp_reg_opt[1, 1] == 3

    d_rec = dask_dict["num_opt_rec"]
    assert d_rec.dtype == np.dtype([("a", "<i8"), ("b", "O")])
    assert d_rec.shape == (2,)
    comp_rec = d_rec.compute()
    assert comp_rec.dtype == np.dtype([("a", "<i8"), ("b", "O")])
    assert comp_rec["a"][0] == 1
    assert comp_rec["b"][0] is None
    assert comp_rec["a"][1] == 2
    assert comp_rec["b"][1] == 3


def test_rntuple_struct_jagged_and_fixed_size_optional_array(tmp_path):
    filepath = os.path.join(tmp_path, "rec_jagged_opt.root")
    r_layout = ak.contents.RegularArray(ak.Array([1, None, None, 3]).layout, 2)
    a_layout = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 2, 3], dtype=np.int64)),
        ak.Array([1, 2, 3]).layout,
    )
    rec = ak.Array(ak.contents.RecordArray([a_layout, r_layout], ["a", "r"]))
    data = ak.Array({"rec": rec})

    with uproot.recreate(filepath) as f:
        f.mkrntuple("ntuple", data)

    ntuple = uproot.open(filepath)["ntuple"]

    arr = ntuple["rec"].array(library="np")
    assert arr.dtype == np.dtype([("a", "O"), ("r", "O", (2,))])
    assert np.array_equal(arr["a"][0], np.array([1, 2]))
    assert np.array_equal(arr["a"][1], np.array([3]))
    assert arr["r"][0, 0] == 1
    assert arr["r"][0, 1] is None
    assert arr["r"][1, 0] is None
    assert arr["r"][1, 1] == 3

    pytest.importorskip("dask.array")
    d_dict = uproot.dask(filepath + ":ntuple", library="np")
    d_rec = d_dict["rec"]
    assert d_rec.dtype == np.dtype([("a", "O"), ("r", "O", (2,))])
    comp_rec = d_rec.compute()
    assert comp_rec.dtype == np.dtype([("a", "O"), ("r", "O", (2,))])
    assert comp_rec["r"][0, 0] == 1
    assert comp_rec["r"][0, 1] is None
    assert comp_rec["r"][1, 0] is None
    assert comp_rec["r"][1, 1] == 3

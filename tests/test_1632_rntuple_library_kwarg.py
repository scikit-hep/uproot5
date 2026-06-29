# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import awkward as ak
import numpy as np
import pytest
import skhep_testdata

import uproot

SAMPLE_FILE = "Run2012BC_DoubleMuParked_Muons_1000evts_rntuple_v1-0-0-0.root"
MULTI_FIELD_FILE = "test_int_float_rntuple_v1-0-0-0.root"


def _lookup_file(name):
    """Helper to look up object paths for known test files."""
    if name == "test_int_float_rntuple_v1-0-0-0.root":
        return "ntuple"
    elif name == "ntpl001_staff_rntuple_v1-0-0-0.root":
        return "Staff"
    else:
        return name.split("_")[0] if "_" in name else name


def test_arrays_numpy_returns_dict():
    filename = skhep_testdata.data_path(MULTI_FIELD_FILE)
    obj_name = "ntuple"
    with uproot.open(filename) as f:
        obj = f[obj_name]
        np_arrays = obj.arrays(library="np")
        assert isinstance(np_arrays, dict)
        assert set(np_arrays.keys()) == {"one_integers", "two_floats"}
        assert isinstance(np_arrays["one_integers"], np.ndarray)
        assert isinstance(np_arrays["two_floats"], np.ndarray)


def test_arrays_numpy_how_tuple():
    filename = skhep_testdata.data_path(MULTI_FIELD_FILE)
    obj_name = "ntuple"
    with uproot.open(filename) as f:
        obj = f[obj_name]
        result = obj.arrays(library="np", how=tuple)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(arr, np.ndarray) for arr in result)


def test_arrays_numpy_how_list():
    filename = skhep_testdata.data_path(MULTI_FIELD_FILE)
    obj_name = "ntuple"
    with uproot.open(filename) as f:
        obj = f[obj_name]
        result = obj.arrays(library="np", how=list)
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(arr, np.ndarray) for arr in result)


def test_arrays_numpy_how_dict():
    filename = skhep_testdata.data_path(MULTI_FIELD_FILE)
    obj_name = "ntuple"
    with uproot.open(filename) as f:
        obj = f[obj_name]
        result = obj.arrays(library="np", how=dict)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"one_integers", "two_floats"}


def test_arrays_pandas_returns_dataframe():
    pd = pytest.importorskip("pandas")
    filename = skhep_testdata.data_path(MULTI_FIELD_FILE)
    obj_name = "ntuple"
    with uproot.open(filename) as f:
        obj = f[obj_name]
        result = obj.arrays(library="pd")
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns.tolist()) == {"one_integers", "two_floats"}
        assert len(result) == 10


def test_arrays_pandas_how_dict():
    pd = pytest.importorskip("pandas")
    filename = skhep_testdata.data_path(MULTI_FIELD_FILE)
    obj_name = "ntuple"
    with uproot.open(filename) as f:
        obj = f[obj_name]
        result = obj.arrays(library="pd", how=dict)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"one_integers", "two_floats"}
        assert all(isinstance(v, pd.Series) for v in result.values())


def test_single_field_array_numpy():
    filename = skhep_testdata.data_path(SAMPLE_FILE)
    with uproot.open(filename) as f:
        obj = f["Events"]
        result = obj["nMuon"].array(library="np")
        assert isinstance(result, np.ndarray)
        assert len(result) == 1000


def test_single_field_array_pandas():
    pd = pytest.importorskip("pandas")
    filename = skhep_testdata.data_path(SAMPLE_FILE)
    with uproot.open(filename) as f:
        obj = f["Events"]
        result = obj["nMuon"].array(library="pd")
        assert isinstance(result, pd.Series)
        assert len(result) == 1000


def test_numpy_values_match_awkward():
    filename = skhep_testdata.data_path(SAMPLE_FILE)
    with uproot.open(filename) as f:
        obj = f["Events"]
        ak_arrays = obj.arrays(filter_name="n*")
        np_arrays = obj.arrays(library="np", filter_name="n*")

        for field in ak_arrays.fields:
            assert ak.array_equal(ak_arrays[field], np_arrays[field])


def test_iterate_numpy():
    filename = skhep_testdata.data_path(SAMPLE_FILE)
    with uproot.open(filename) as f:
        obj = f["Events"]
        for i, arrays in enumerate(obj.iterate(step_size=100, library="np")):
            assert isinstance(arrays, dict)
            assert isinstance(arrays["nMuon"], np.ndarray)
            assert len(arrays["nMuon"]) == 100
            if i == 9:
                break


def test_iterate_pandas():
    pd = pytest.importorskip("pandas")
    filename = skhep_testdata.data_path(SAMPLE_FILE)
    with uproot.open(filename) as f:
        obj = f["Events"]
        for i, arrays in enumerate(obj.iterate(step_size=100, library="pd")):
            assert isinstance(arrays, pd.DataFrame)
            assert len(arrays) == 100
            if i == 9:
                break


def test_concatenate_numpy():
    filename = skhep_testdata.data_path(SAMPLE_FILE)
    result = uproot.concatenate({filename: "Events"}, library="np")
    assert isinstance(result, dict)
    assert len(result["nMuon"]) == 1000


def test_concatenate_pandas():
    pd = pytest.importorskip("pandas")
    filename = skhep_testdata.data_path(SAMPLE_FILE)
    result = uproot.concatenate({filename: "Events"}, library="pd")
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1000


def test_nested_structs_numpy():
    filename = skhep_testdata.data_path("test_nested_structs_rntuple_v1-0-0-0.root")
    with uproot.open(filename) as f:
        obj = f["ntuple"]
        result = obj.arrays(library="np")
        assert isinstance(result, dict)
        assert "my_struct" in result
        assert isinstance(result["my_struct"], np.ndarray)


def test_jagged_arrays_numpy():
    filename = skhep_testdata.data_path(SAMPLE_FILE)
    with uproot.open(filename) as f:
        obj = f["Events"]
        ak_arrays = obj.arrays(filter_name=["Muon_pt"])
        np_arrays = obj.arrays(library="np", filter_name=["Muon_pt"])

        assert isinstance(np_arrays, dict)
        assert "Muon_pt" in np_arrays
        assert isinstance(np_arrays["Muon_pt"], np.ndarray)
        assert np_arrays["Muon_pt"].dtype == object

        for i in range(10):
            assert ak.array_equal(ak_arrays["Muon_pt"][i], np_arrays["Muon_pt"][i])

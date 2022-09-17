# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

import numpy
import pytest
import skhep_testdata

import uproot

dask = pytest.importorskip("dask")
dask_awkward = pytest.importorskip("dask_awkward")

pytest.importorskip("pyarrow")  # dask_awkward.lib.testutils needs pyarrow
from dask_awkward.lib.testutils import assert_eq


def test_dask_numpy_empty_arrays():
    test_path = skhep_testdata.data_path("uproot-issue-697.root") + ":tree"
    ttree = uproot.open(test_path)

    np_arrays = ttree.arrays(library="np")
    dask_arrays = uproot.dask(test_path, library="np")

    assert list(dask_arrays.keys()) == list(
        np_arrays.keys()
    ), "Different keys detected in dictionary of dask arrays and dictionary of numpy arrays"

    for key in np_arrays.keys():
        comp = dask_arrays[key].compute() == np_arrays[key]
        assert comp.all(), f"Incorrect array at key {key}"


def test_dask_delayed_open_numpy():
    test_path = skhep_testdata.data_path("uproot-issue-697.root") + ":tree"
    ttree = uproot.open(test_path)

    np_arrays = ttree.arrays(library="np")
    dask_arrays = uproot.dask(test_path, library="np", open_files=False)

    assert list(dask_arrays.keys()) == list(
        np_arrays.keys()
    ), "Different keys detected in dictionary of dask arrays and dictionary of numpy arrays"

    for key in np_arrays.keys():
        comp = dask_arrays[key].compute() == np_arrays[key]
        assert comp.all(), f"Incorrect array at key {key}"


def test_dask_awkward_empty_arrays():
    test_path = skhep_testdata.data_path("uproot-issue-697.root") + ":tree"
    ttree = uproot.open(test_path)

    ak_array = ttree.arrays()
    dak_array = uproot.dask(test_path, library="ak")

    assert_eq(dak_array, ak_array)


def test_dask_delayed_open_awkward():
    test_path = skhep_testdata.data_path("uproot-issue-697.root") + ":tree"
    ttree = uproot.open(test_path)

    ak_array = ttree.arrays()
    dak_array = uproot.dask(test_path, library="ak", open_files=False)

    assert_eq(dak_array, ak_array)


def test_no_common_tree_branches():
    test_path1 = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    test_path2 = skhep_testdata.data_path("uproot-issue-697.root") + ":tree"

    with pytest.raises(ValueError):
        dask_arrays = uproot.dask([test_path1, test_path2], library="np")

    with pytest.raises(ValueError):
        dask_arrays = uproot.dask([test_path1, test_path2])

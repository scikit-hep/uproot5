# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import pytest
import skhep_testdata
import awkward

import uproot

dask = pytest.importorskip("dask")
da = pytest.importorskip("dask.array")


def test_read_AsGrouped_with_dask_or_virtual_array():
    """
    ``RNTuple.iterate(report=True)`` must yield ``(arrays, Report)`` pairs,
    matching the contract documented in its docstring and already provided
    by ``TTree.iterate``.
    """
    path = skhep_testdata.data_path("uproot-issue-1502.root")
    with uproot.open(path) as f:
        arr = f["tree"].arrays()
        virtual_arr = f["tree"].arrays(virtual=True)

    assert awkward.array_equal(arr, awkward.materialize(virtual_arr))

    dask_arr = uproot.dask(path).compute()
    assert awkward.array_equal(arr, dask_arr)

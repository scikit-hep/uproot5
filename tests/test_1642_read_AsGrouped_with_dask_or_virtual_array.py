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


def test_filter_name_on_AsGrouped_eager():
    """
    tree.arrays(filter_name=<AsGrouped branch name>) must return a non-empty
    record containing the grouped sub-fields, not an empty result.

    Previously, _regularize_expressions skipped ALL AsGrouped branches when
    expressions=None, so filtering to a single AsGrouped branch produced an
    empty record.
    """
    path = skhep_testdata.data_path("uproot-FCCDelphesOutput.root")
    with uproot.open(path) as f:
        tree = f["events"]
        arr = tree.arrays(filter_name="genParticles")

    assert arr.fields == ["genParticles"]
    sub = arr["genParticles"]
    assert "genParticles.core.pdgId" in sub.fields
    assert "genParticles.core.p4.mass" in sub.fields
    # Spot-check a known value in the first event
    assert sub["genParticles.core.pdgId"][0][:3].tolist() == [11, -11, 11]


def test_filter_name_on_AsGrouped_virtual():
    """
    tree.arrays(virtual=True, filter_name=<AsGrouped branch name>) must return
    the same result as the eager path after materialisation.
    """
    path = skhep_testdata.data_path("uproot-FCCDelphesOutput.root")
    with uproot.open(path) as f:
        tree = f["events"]
        arr = tree.arrays(filter_name="genParticles")
        virtual_arr = tree.arrays(virtual=True, filter_name="genParticles")
        assert awkward.array_equal(arr, awkward.materialize(virtual_arr))


def test_filter_name_on_AsGrouped_dask():
    """
    uproot.dask(filter_name=<AsGrouped branch name>).compute() must return the
    same result as the eager path and must not crash.

    Previously, TrivialFormMappingInfo.load_buffers used how=tuple when calling
    tree.arrays(), which for AsGrouped branches returns a Python tuple of
    sub-arrays instead of an awkward RecordArray.  awkward.to_buffers() on that
    Python tuple produced a mismatched form, causing an AssertionError.
    """
    path = skhep_testdata.data_path("uproot-FCCDelphesOutput.root")
    with uproot.open(path) as f:
        arr = f["events"].arrays(filter_name="genParticles")

    dask_arr = uproot.dask(path + ":events", filter_name="genParticles")
    computed = dask_arr.compute()

    assert computed.fields == ["genParticles"]
    assert awkward.array_equal(arr, computed)

import pytest
import awkward as ak
import fsspec
import uproot

skhep_testdata = pytest.importorskip("skhep_testdata")
dask = pytest.importorskip("dask")
da = pytest.importorskip("dask.array")
dask_awkward = pytest.importorskip("dask_awkward")
math = pytest.importorskip("math")


def simple_test(partitions):
    arr = ak.Array([{"a": [1, 2, 3]}, {"a": [4, 5]}])
    dask_arr = dask_awkward.from_awkward(arr, npartitions=partitions)
    uproot.dask_write(
        dask_arr, "/Users/zobil/Documents/uproot5/src/uproot/my-output", prefix="data"
    )
    file_1 = uproot.open(
        "/Users/zobil/Documents/uproot5/src/uproot/my-output/data-part0.root"
    )
    assert len(file_1["tree"]["a"].arrays()) == math.ceil(len(arr) / partitions)
    assert ak.all(file_1["tree"]["a"].arrays()["a"][0] == arr[0]["a"])
    file_2 = uproot.open(
        "/Users/zobil/Documents/uproot5/src/uproot/my-output/data-part1.root"
    )
    assert len(file_2["tree"]["a"].arrays()) == int(len(arr) / partitions)
    assert ak.all(file_2["tree"]["a"].arrays()["a"][0] == arr[1]["a"])


def HZZ_test(partitions):
    arr = uproot.open(skhep_testdata.data_path("uproot-HZZ.root"))["events"].arrays()
    dask_arr = dask_awkward.from_awkward(ak.from_iter(arr), partitions)
    uproot.dask_write(dask_arr, TMPDIR, prefix="data")
    file_1 = uproot.open(
        "/Users/zobil/Documents/uproot5/src/uproot/my-output/data-part0.root"
    )
    assert len(file_1["tree"]["Jet_Px"].arrays()) == math.ceil(len(arr) / partitions)
    file_2 = uproot.open(
        "/Users/zobil/Documents/uproot5/src/uproot/my-output/data-part1.root"
    )
    assert len(file_2["tree"]["Jet_Px"].arrays()) == int(len(arr) / partitions)

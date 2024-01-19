import pytest
import awkward as ak
import uproot

skhep_testdata = pytest.importorskip("skhep_testdata")
dask = pytest.importorskip("dask")
da = pytest.importorskip("dask.array")
dask_awkward = pytest.importorskip("dask_awkward")
math = pytest.importorskip("math")


def simple_test():
    partitions = 2
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


def HZZ_test():
    """
    Write data from HZZ with 3 partitions.
    """
    partitions = 2
    arr = uproot.open(skhep_testdata.data_path("uproot-HZZ.root"))["events"].arrays()
    dask_arr = dask_awkward.from_awkward(ak.from_iter(arr), partitions)
    uproot.dask_write(
        dask_arr, "/Users/zobil/Documents/uproot5/src/uproot/my-output", prefix="data"
    )
    file_1 = uproot.open(
        "/Users/zobil/Documents/uproot5/src/uproot/my-output/data-part0.root"
    )
    assert len(file_1["tree"]["Jet_Px"].arrays()) == math.ceil(len(arr) / partitions)
    file_2 = uproot.open(
        "/Users/zobil/Documents/uproot5/src/uproot/my-output/data-part1.root"
    )
    assert len(file_2["tree"]["Jet_Px"].arrays()) == int(len(arr) / partitions)


def zip_test():
    """
    Trying to find a way around duplicate counter issue.
    """
    partitions = 2
    arr = uproot.open(skhep_testdata.data_path("uproot-HZZ.root"))["events"].arrays()
    uproot.open(skhep_testdata.data_path("uproot-HZZ.root"))["events"].show()
    zipped = ak.Array(
        [
            {
                "Muon_": ak.zip(
                    {
                        name[5:]: array
                        for name, array in zip(ak.fields(arr), ak.unzip(arr))
                        if name.startswith("Muon_")
                    }
                )
            },
            {
                "Jet_": ak.zip(
                    {
                        name[: name.index("_")]: array
                        for name, array in zip(ak.fields(arr), ak.unzip(arr))
                        if name.startswith("Jet_")
                    }
                )
            },
            {
                "Photon_": ak.zip(
                    {
                        name[: name.index("_")]: array
                        for name, array in zip(ak.fields(arr), ak.unzip(arr))
                        if name.startswith("Photon_")
                    }
                )
            },
            {
                "Electron_": ak.zip(
                    {
                        name[: name.index("_")]: array
                        for name, array in zip(ak.fields(arr), ak.unzip(arr))
                        if name.startswith("Electron_")
                    }
                )
            },
        ]
    )
    dask_arr = dask_awkward.from_awkward(ak.from_iter(zipped), partitions)
    uproot.dask_write(
        dask_arr, "/Users/zobil/Documents/uproot5/src/uproot/my-output", prefix="data"
    )
    file_1 = uproot.open(
        "/Users/zobil/Documents/uproot5/src/uproot/my-output/data-part0.root"
    )
    assert len(file_1["tree"]["Jet_Px"].arrays()) == math.ceil(len(arr) / partitions)
    file_2 = uproot.open(
        "/Users/zobil/Documents/uproot5/src/uproot/my-output/data-part1.root"
    )
    assert len(file_2["tree"]["Jet_Px"].arrays()) == int(len(arr) / partitions)


def HZZ_test_3_partitions():
    """
    Write data from HZZ with 3 partitions
    """
    partitions = 3
    arr = uproot.open(skhep_testdata.data_path("uproot-HZZ.root"))["events"].arrays()
    dask_arr = dask_awkward.from_awkward(ak.from_iter(arr), partitions)
    uproot.dask_write(
        dask_arr, "/Users/zobil/Documents/uproot5/src/uproot/my-output", prefix="data"
    )
    file_1 = uproot.open(
        "/Users/zobil/Documents/uproot5/src/uproot/my-output/data-part0.root"
    )
    assert len(file_1["tree"]["Jet_Px"].arrays()) == math.ceil(len(arr) / partitions)
    file_2 = uproot.open(
        "/Users/zobil/Documents/uproot5/src/uproot/my-output/data-part1.root"
    )
    assert len(file_2["tree"]["Jet_Px"].arrays()) == int(len(arr) / partitions)
    file_3 = uproot.open(
        "/Users/zobil/Documents/uproot5/src/uproot/my-output/data-part2.root"
    )
    assert len(file_3["tree"]["Jet_Px"].arrays()) == int(len(arr) / partitions)


def graph():
    """
    Test compute parameter (want to return highlevelgraph)
    """
    partitions = 2
    arr = ak.Array([{"a": [1, 2, 3]}, {"a": [4, 5]}])
    dask_arr = dask_awkward.from_awkward(arr, npartitions=partitions)
    graph = uproot.dask_write(
        dask_arr,
        "/Users/zobil/Documents/uproot5/src/uproot/my-output",
        prefix="compute",
        compute=False,
    )
    graph.compute()
    file_1 = uproot.open(
        "/Users/zobil/Documents/uproot5/src/uproot/my-output/compute-part0.root"
    )
    assert len(file_1["tree"]["a"].arrays()) == math.ceil(len(arr) / partitions)
    assert ak.all(file_1["tree"]["a"].arrays()["a"][0] == arr[0]["a"])
    file_2 = uproot.open(
        "/Users/zobil/Documents/uproot5/src/uproot/my-output/compute-part1.root"
    )
    assert len(file_2["tree"]["a"].arrays()) == int(len(arr) / partitions)
    assert ak.all(file_2["tree"]["a"].arrays()["a"][0] == arr[1]["a"])

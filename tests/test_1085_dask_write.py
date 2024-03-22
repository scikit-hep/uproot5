import math
import pytest
import awkward as ak
import skhep_testdata
import os

dask = pytest.importorskip("dask")
da = pytest.importorskip("dask.array")
dd = pytest.importorskip("dask.distributed")
dask_awkward = pytest.importorskip("dask_awkward")

from distributed import Client
import uproot


def simple_test(tmp_path):
    partitions = 2
    arr = ak.Array([{"a": [1, 2, 3]}, {"a": [4, 5]}])
    dask_arr = dask_awkward.from_awkward(arr, npartitions=partitions)
    uproot.dask_write(dask_arr, tmp_path, prefix="data")
    file_1 = uproot.open(os.path.join(tmp_path, "data-part0.root"))
    assert len(file_1["tree"]["a"].arrays()) == math.ceil(len(arr) / partitions)
    assert ak.all(file_1["tree"]["a"].arrays()["a"][0] == arr[0]["a"])
    file_2 = uproot.open(os.path.join(tmp_path, "data-part1.root"))
    assert len(file_2["tree"]["a"].arrays()) == int(len(arr) / partitions)
    assert ak.all(file_2["tree"]["a"].arrays()["a"][0] == arr[1]["a"])


def HZZ_test(tmp_path):
    """
    Write data from HZZ with 3 partitions.
    """
    partitions = 2
    arr = uproot.open(skhep_testdata.data_path("uproot-HZZ.root"))["events"].arrays()
    dask_arr = dask_awkward.from_awkward(ak.from_iter(arr), partitions)
    uproot.dask_write(dask_arr, tmp_path, prefix="data")
    file_1 = uproot.open(os.path.join(tmp_path, "data-part0.root"))
    assert len(file_1["tree"]["Jet_Px"].arrays()) == math.ceil(len(arr) / partitions)
    assert ak.all(file_1["tree"]["Jet_Px"].arrays()["Jet_Px"][0] == arr["Jet_Px"])
    file_2 = uproot.open(os.path.join(tmp_path, "data-part1.root"))
    assert len(file_2["tree"]["Jet_Px"].arrays()) == int(len(arr) / partitions)
    assert ak.all(file_2["tree"]["Jet_Px"].arrays()["Jet_Px"][0] == arr["Jet_Px"])


def test_graph(tmp_path):
    """
    Test compute parameter (want to return highlevelgraph)
    """
    partitions = 2
    arr = ak.Array([{"a": [1, 2, 3]}, {"a": [4, 5]}])
    dask_arr = dask_awkward.from_awkward(arr, npartitions=partitions)
    graph = uproot.dask_write(
        dask_arr,
        str(tmp_path),
        prefix="compute",
        compute=False,
    )
    graph.compute()
    file_1 = uproot.open(os.path.join(tmp_path, "compute-part0.root"))
    assert len(file_1["tree"]["a"].arrays()) == math.ceil(len(arr) / partitions)
    assert ak.all(file_1["tree"]["a"].arrays()["a"][0] == arr[0]["a"])
    file_2 = uproot.open(os.path.join(tmp_path, "compute-part1.root"))
    assert len(file_2["tree"]["a"].arrays()) == int(len(arr) / partitions)
    assert ak.all(file_2["tree"]["a"].arrays()["a"][0] == arr[1]["a"])


@pytest.mark.distributed
def test_compute(tmp_path):
    partitions = 2
    arr = uproot.open(skhep_testdata.data_path("uproot-HZZ.root"))["events"].arrays()
    dask_arr = dask_awkward.from_awkward(ak.from_iter(arr), partitions)
    with Client():
        graph = uproot.dask_write(
            dask_arr, str(tmp_path), prefix="distribute", compute=False
        )
        dask.compute(graph)
    dask_arr = dask_awkward.from_awkward(ak.from_iter(arr), partitions)
    file_1 = uproot.open(os.path.join(tmp_path, "distribute-part0.root"))
    assert len(file_1["tree"]["Jet_Px"].arrays()) == math.ceil(len(arr) / partitions)
    assert ak.all(file_1["tree"]["Jet_Px"].arrays()["Jet_Px"][0] == arr["Jet_Px"])
    file_2 = uproot.open(os.path.join(tmp_path, "distribute-part1.root"))
    assert len(file_2["tree"]["Jet_Px"].arrays()) == int(len(arr) / partitions)
    assert ak.all(file_2["tree"]["Jet_Px"].arrays()["Jet_Px"][0] == arr["Jet_Px"])

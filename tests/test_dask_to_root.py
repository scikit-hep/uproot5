import pytest
import uproot
import awkward as ak

skhep_testdata = pytest.importorskip("skhep_testdata")
dask = pytest.importorskip("dask")
da = pytest.importorskip("dask.array")
dask_awkward = pytest.importorskip("dask_awkward")
math = pytest.importorskip("math")

import fsspec

fs = fsspec.filesystem("file")
tmpdir = "src/uproot/my-output"
files = fs.ls(tmpdir)


def simple_test(partitions):
    a = ak.Array([{"a": [1, 2, 3]}, {"a": [4, 5]}])
    d = dask_awkward.from_awkward(a, npartitions=partitions)
    uproot.dask_write(d, tmpdir, prefix="data")
    f = uproot.open(
        "/Users/zobil/Documents/uproot5/src/uproot/my-output/data-part0.root"
    )
    assert len(f["tree"]["a"].arrays()) == math.ceil(len(a) / partitions)
    assert ak.all(f["tree"]["a"].arrays()["a"][0] == a[0]["a"])
    f1 = uproot.open(
        "/Users/zobil/Documents/uproot5/src/uproot/my-output/data-part1.root"
    )
    assert len(f1["tree"]["a"].arrays()) == int(len(a) / partitions)
    assert ak.all(f1["tree"]["a"].arrays()["a"][0] == a[1]["a"])


def HZZ_test(partitions):
    a = uproot.open(skhep_testdata.data_path("uproot-HZZ.root"))["events"].arrays()
    d = dask_awkward.from_awkward(ak.from_iter(a), partitions)
    uproot.dask_write(d, tmpdir, prefix="data")
    f = uproot.open(
        "/Users/zobil/Documents/uproot5/src/uproot/my-output/data-part0.root"
    )
    assert len(f["tree"]["Jet_Px"].arrays()) == math.ceil(len(a) / partitions)
    f1 = uproot.open(
        "/Users/zobil/Documents/uproot5/src/uproot/my-output/data-part1.root"
    )
    assert len(f1["tree"]["Jet_Px"].arrays()) == int(len(a) / partitions)


simple_test(2)

import os

assert [os.path.basename(_) for _ in sorted(files)] == [
    "data-part0.root",
    "data-part1.root",
]

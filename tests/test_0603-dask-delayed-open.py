# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import numpy
import pytest
import skhep_testdata

import uproot

dask = pytest.importorskip("dask")
da = pytest.importorskip("dask.array")


def test_single_delay_open():
    filename1 = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    ttree = uproot.open(filename1)

    assert uproot.dask(filename1, open_files=False, library="np")["px1"].chunks == (
        (numpy.nan,),
    )
    arr = uproot.dask(filename1, open_files=False, library="np")[
        "px1"
    ].compute() == ttree["px1"].array(library="np")
    assert arr.all()


def test_multiple_delay_open():
    filename1 = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    filename2 = skhep_testdata.data_path("uproot-Zmumu-uncompressed.root") + ":events"
    true_val = uproot.concatenate([filename1, filename2], library="np")

    assert uproot.dask([filename1, filename2], open_files=False, library="np")[
        "px1"
    ].chunks == ((numpy.nan, numpy.nan),)
    arr = (
        uproot.dask([filename1, filename2], open_files=False, library="np")[
            "px1"
        ].compute()
        == true_val["px1"]
    )
    assert arr.all()


def test_supplied_chunks():
    filename1 = skhep_testdata.data_path("uproot-Zmumu.root")
    filename2 = skhep_testdata.data_path("uproot-Zmumu-uncompressed.root")
    true_val = uproot.concatenate([filename1 + ":events", filename2 + ":events"])

    chunks1 = [0, 2305]
    chunks2 = [[0, 2305]]

    files = {
        filename1: {"object_path": "events", "chunks": chunks1},
        filename2: {"object_path": "events", "chunks": chunks2},
    }

    assert uproot.dask(files, open_files=False)["px1"].divisions == (None, None, None)
    arr = (
        uproot.dask([filename1, filename2], open_files=False)["px1"].compute()
        == true_val["px1"]
    )
    assert arr.to_numpy().all()

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

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

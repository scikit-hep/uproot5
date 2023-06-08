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


@pytest.mark.parametrize("open_files", [False, True])
@pytest.mark.parametrize("library", ["np", "ak"])
def test_supplied_chunks(open_files, library):
    filename1 = skhep_testdata.data_path("uproot-Zmumu.root")
    filename2 = skhep_testdata.data_path("uproot-Zmumu-uncompressed.root")
    true_val = uproot.concatenate(
        [filename1 + ":events", filename2 + ":events"], "px1", library=library
    )["px1"].tolist()

    chunks1 = [0, 1000, 2304]
    chunks2 = [[0, 1000], [1000, 2304]]

    files = {
        filename1: {"object_path": "events", "chunks": chunks1},
        filename2: {"object_path": "events", "chunks": chunks2},
    }

    daskarr = uproot.dask(files, open_files=open_files, library=library)["px1"]

    if library == "ak":
        if open_files:
            assert daskarr.divisions == (None, None, None)  # FIXME! (Jim)
        else:
            assert daskarr.divisions == (None, None, None, None, None)
    else:
        if open_files:
            assert daskarr.chunks == ((2304, 2304),)  # FIXME! (Jim)
        else:
            assert daskarr.chunks == ((1000, 1304, 1000, 1304),)

    assert daskarr.compute().tolist() == true_val

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import multiprocessing

import pytest
import skhep_testdata

import uproot


def test_empty():
    with uproot.open(skhep_testdata.data_path("uproot-empty.root")) as f:
        t = f["tree"]
        assert t["x"].array(library="np").tolist() == []
        assert t["y"].array(library="np").tolist() == []
        assert t["z"].array(library="np").tolist() == []


def read_one(filename):
    with uproot.open(filename, handler=uproot.source.file.MemmapSource) as f:
        f.decompression_executor = uproot.ThreadPoolExecutor()
        t = f["events"]
        b = t["px1"]
        b.array(library="np")


def test_multiprocessing():
    with multiprocessing.Pool(1) as pool:
        out = pool.map(
            read_one,
            [
                skhep_testdata.data_path("uproot-Zmumu.root"),
                skhep_testdata.data_path("uproot-Zmumu-zlib.root"),
            ],
        )
        list(out)

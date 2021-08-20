# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

from __future__ import absolute_import

import os

import numpy as np
import pytest
import skhep_testdata

import uproot

ROOT = pytest.importorskip("ROOT")


def test_ZLIB():
    for _ in range(2):
        with uproot.open(skhep_testdata.data_path("uproot-Zmumu-zlib.root"))[
            "events"
        ] as events:
            assert events["px1"].array(entry_stop=5).tolist() == [
                -41.1952876442,
                35.1180497674,
                35.1180497674,
                34.1444372454,
                22.7835819537,
            ]


def test_LZMA():
    pytest.importorskip("lzma")

    for _ in range(2):
        with uproot.open(skhep_testdata.data_path("uproot-Zmumu-lzma.root"))[
            "events"
        ] as events:
            assert events["px1"].array(entry_stop=5).tolist() == [
                -41.1952876442,
                35.1180497674,
                35.1180497674,
                34.1444372454,
                22.7835819537,
            ]


def test_LZ4():
    pytest.importorskip("lz4")

    for _ in range(2):
        with uproot.open(skhep_testdata.data_path("uproot-Zmumu-lz4.root"))[
            "events"
        ] as events:
            assert events["px1"].array(entry_stop=5).tolist() == [
                -41.1952876442,
                35.1180497674,
                35.1180497674,
                34.1444372454,
                22.7835819537,
            ]


def test_ZSTD():
    pytest.importorskip("zstandard")

    for _ in range(2):
        with uproot.open(skhep_testdata.data_path("uproot-Zmumu-zstd.root"))[
            "events"
        ] as events:
            assert events["px1"].array(entry_stop=5).tolist() == [
                -41.1952876442,
                35.1180497674,
                35.1180497674,
                34.1444372454,
                22.7835819537,
            ]


def test_histogram_ZLIB(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    SIZE = 2 ** 21
    histogram = (np.random.normal(0, 1, SIZE), np.linspace(0, 1, SIZE + 1))
    last = histogram[0][-1]

    with uproot.recreate(newfile, compression=uproot.ZLIB(1)) as fout:
        fout["out"] = histogram

    with uproot.open(newfile) as fin:
        content, edges = fin["out"].to_numpy()
        assert len(content) == SIZE
        assert len(edges) == SIZE + 1
        assert content[-1] == last

    f3 = ROOT.TFile(newfile)
    h3 = f3.Get("out")
    assert h3.GetNbinsX() == SIZE
    assert h3.GetBinContent(SIZE) == last
    f3.Close()


def test_histogram_LZMA(tmp_path):
    pytest.importorskip("lzma")

    newfile = os.path.join(tmp_path, "newfile.root")

    SIZE = 2 ** 20
    histogram = (np.random.normal(0, 1, SIZE), np.linspace(0, 1, SIZE + 1))
    last = histogram[0][-1]

    with uproot.recreate(newfile, compression=uproot.LZMA(1)) as fout:
        fout["out"] = histogram

    with uproot.open(newfile) as fin:
        content, edges = fin["out"].to_numpy()
        assert len(content) == SIZE
        assert len(edges) == SIZE + 1
        assert content[-1] == last

    f3 = ROOT.TFile(newfile)
    h3 = f3.Get("out")
    assert h3.GetNbinsX() == SIZE
    assert h3.GetBinContent(SIZE) == last
    f3.Close()


def test_histogram_LZ4(tmp_path):
    pytest.importorskip("lz4")

    newfile = os.path.join(tmp_path, "newfile.root")

    SIZE = 2 ** 21
    histogram = (np.random.normal(0, 1, SIZE), np.linspace(0, 1, SIZE + 1))
    last = histogram[0][-1]

    with uproot.recreate(newfile, compression=uproot.LZ4(1)) as fout:
        fout["out"] = histogram

    with uproot.open(newfile) as fin:
        content, edges = fin["out"].to_numpy()
        assert len(content) == SIZE
        assert len(edges) == SIZE + 1
        assert content[-1] == last

    f3 = ROOT.TFile(newfile)
    h3 = f3.Get("out")
    assert h3.GetNbinsX() == SIZE
    assert h3.GetBinContent(SIZE) == last
    f3.Close()


def test_histogram_ZSTD(tmp_path):
    pytest.importorskip("zstandard")

    newfile = os.path.join(tmp_path, "newfile.root")

    SIZE = 2 ** 21
    histogram = (np.random.normal(0, 1, SIZE), np.linspace(0, 1, SIZE + 1))
    last = histogram[0][-1]

    with uproot.recreate(newfile, compression=uproot.ZSTD(1)) as fout:
        fout["out"] = histogram

    with uproot.open(newfile) as fin:
        content, edges = fin["out"].to_numpy()
        assert len(content) == SIZE
        assert len(edges) == SIZE + 1
        assert content[-1] == last

    f3 = ROOT.TFile(newfile)
    h3 = f3.Get("out")
    assert h3.GetNbinsX() == SIZE
    assert h3.GetBinContent(SIZE) == last
    f3.Close()

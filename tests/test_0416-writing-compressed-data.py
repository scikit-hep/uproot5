# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

from __future__ import absolute_import

import numpy
import pytest
import skhep_testdata

import uproot


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

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import sys
import json

import numpy
import pytest
import skhep_testdata

import uproot4
import uproot4.interpretation.library
import uproot4.interpretation.jagged
import uproot4.interpretation.numerical


def test_branchname():
    with uproot4.open(
        skhep_testdata.data_path("uproot-sample-6.20.04-uncompressed.root")
    )["sample"] as sample:
        assert sample.arrays("i4", library="np")["i4"].tolist() == list(range(-15, 15))

        arrays = sample.arrays(["i4", "i8"], library="np")
        assert set(arrays.keys()) == set(["i4", "i8"])
        assert arrays["i4"].tolist() == list(range(-15, 15))
        assert arrays["i8"].tolist() == list(range(-15, 15))

        arrays = sample.arrays(filter_name="/i[48]/", library="np")
        assert set(arrays.keys()) == set(["i4", "i8"])
        assert arrays["i4"].tolist() == list(range(-15, 15))
        assert arrays["i8"].tolist() == list(range(-15, 15))

        arrays = sample.arrays(filter_name=["/i[12]/", "/i[48]/"], library="np")
        assert set(arrays.keys()) == set(["i1", "i2", "i4", "i8"])
        assert arrays["i1"].tolist() == list(range(-15, 15))
        assert arrays["i2"].tolist() == list(range(-15, 15))
        assert arrays["i4"].tolist() == list(range(-15, 15))
        assert arrays["i8"].tolist() == list(range(-15, 15))

        arrays = sample.arrays(filter_name="i*", library="np")
        assert set(arrays.keys()) == set(["i1", "i2", "i4", "i8"])
        assert arrays["i1"].tolist() == list(range(-15, 15))
        assert arrays["i2"].tolist() == list(range(-15, 15))
        assert arrays["i4"].tolist() == list(range(-15, 15))
        assert arrays["i8"].tolist() == list(range(-15, 15))

        arrays = sample.arrays(["i4", "i8"], filter_name="u*", library="np")
        assert set(arrays.keys()) == set(["i4", "i8"])
        assert arrays["i4"].tolist() == list(range(-15, 15))
        assert arrays["i8"].tolist() == list(range(-15, 15))


def test_tuple_branchname():
    with uproot4.open(
        skhep_testdata.data_path("uproot-sample-6.20.04-uncompressed.root")
    )["sample"] as sample:
        arrays = sample.arrays(["i4", "i8"], library="np", how=tuple)
        assert isinstance(arrays, tuple) and len(arrays) == 2
        assert arrays[0].tolist() == list(range(-15, 15))
        assert arrays[1].tolist() == list(range(-15, 15))

        arrays = sample.arrays(["i4", "i4"], library="np", how=tuple)
        assert isinstance(arrays, tuple) and len(arrays) == 2
        assert arrays[0].tolist() == list(range(-15, 15))
        assert arrays[1].tolist() == list(range(-15, 15))


def test_interpretation():
    with uproot4.open(
        skhep_testdata.data_path("uproot-sample-6.20.04-uncompressed.root")
    )["sample"] as sample:
        assert sample["i2"].array(">u2", library="np").tolist() == list(
            range(65521, 65536)
        ) + list(range(0, 15))

        arrays = sample.arrays({"i1": ">u1", "i2": ">u2"}, library="np")
        assert set(arrays.keys()) == set(["i1", "i2"])
        assert arrays["i1"].tolist() == list(range(241, 256)) + list(range(0, 15))
        assert arrays["i2"].tolist() == list(range(65521, 65536)) + list(range(0, 15))

        arrays = sample.arrays([("i1", ">u1"), ("i2", ">u2")], library="np", how=tuple)
        assert isinstance(arrays, tuple) and len(arrays) == 2
        assert arrays[0].tolist() == list(range(241, 256)) + list(range(0, 15))
        assert arrays[1].tolist() == list(range(65521, 65536)) + list(range(0, 15))

        arrays = sample.arrays({"i1": ">u1", "i2": None}, library="np")
        assert set(arrays.keys()) == set(["i1", "i2"])
        assert arrays["i1"].tolist() == list(range(241, 256)) + list(range(0, 15))
        assert arrays["i2"].tolist() == list(range(-15, 15))

        arrays = sample.arrays([("i1", ">u1"), ("i2", None)], library="np", how=tuple)
        assert isinstance(arrays, tuple) and len(arrays) == 2
        assert arrays[0].tolist() == list(range(241, 256)) + list(range(0, 15))
        assert arrays[1].tolist() == list(range(-15, 15))


def test_compute():
    with uproot4.open(
        skhep_testdata.data_path("uproot-sample-6.20.04-uncompressed.root")
    )["sample"] as sample:
        assert sample.arrays("i4 + 100", library="np")["i4 + 100"].tolist() == list(
            range(85, 115)
        )

        arrays = sample.arrays(["i4 + 100", "i8 + 100"], library="np")
        assert set(arrays.keys()) == set(["i4 + 100", "i8 + 100"])
        assert arrays["i4 + 100"].tolist() == list(range(85, 115))
        assert arrays["i8 + 100"].tolist() == list(range(85, 115))

        arrays = sample.arrays(["i4 + 100", "i4 + 200"], library="np")
        assert set(arrays.keys()) == set(["i4 + 100", "i4 + 200"])
        assert arrays["i4 + 100"].tolist() == list(range(85, 115))
        assert arrays["i4 + 200"].tolist() == list(range(185, 215))

        arrays = sample.arrays(["i4 + 100", "i4 + 100"], library="np")
        assert set(arrays.keys()) == set(["i4 + 100"])
        assert arrays["i4 + 100"].tolist() == list(range(85, 115))

        arrays = sample.arrays(["i4 + 100", "i4 + 100"], library="np", how=tuple)
        assert isinstance(arrays, tuple) and len(arrays) == 2
        assert arrays[0].tolist() == list(range(85, 115))
        assert arrays[1].tolist() == list(range(85, 115))


def test_cut():
    with uproot4.open(
        skhep_testdata.data_path("uproot-sample-6.20.04-uncompressed.root")
    )["sample"] as sample:
        assert sample.arrays("i4 + 100", cut="i4 > 0", library="np")["i4 + 100"].tolist() == list(
            range(101, 115)
        )

        arrays = sample.arrays(["i4 + 100", "i8 + 100"], cut="i4 > 0", library="np")
        assert set(arrays.keys()) == set(["i4 + 100", "i8 + 100"])
        assert arrays["i4 + 100"].tolist() == list(range(101, 115))
        assert arrays["i8 + 100"].tolist() == list(range(101, 115))

        arrays = sample.arrays(["i4", "i8"], cut="i4 > 0", library="np")
        assert set(arrays.keys()) == set(["i4", "i8"])
        assert arrays["i4"].tolist() == list(range(1, 15))
        assert arrays["i8"].tolist() == list(range(1, 15))

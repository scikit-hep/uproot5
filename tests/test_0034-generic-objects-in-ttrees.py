# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import sys
import json

import numpy
import pytest
import skhep_testdata

import uproot4


def test_histograms_outside_of_ttrees():
    with uproot4.open(skhep_testdata.data_path("uproot-hepdata-example.root")) as f:
        contents = numpy.asarray(f["hpx"].bases[-1])
        assert (contents.min(), contents.max()) == (0.0, 2417.0)

        contents = numpy.asarray(f["hpxpy"].bases[-1])
        assert (contents.min(), contents.max()) == (0.0, 497.0)

        contents = numpy.asarray(f["hprof"].bases[-1].bases[-1])
        assert (contents.min(), contents.max()) == (0.0, 3054.7299575805664)

        numpy.asarray(f["ntuple"])


def test_gohep_nosplit_file():
    with uproot4.open(skhep_testdata.data_path("uproot-small-evnt-tree-nosplit.root"))[
        "tree/evt"
    ] as branch:
        result = branch.array(library="np", entry_start=5, entry_stop=6)[0]
        assert result.member("Beg") == "beg-005"
        assert result.member("I16") == 5
        assert result.member("I32") == 5
        assert result.member("I64") == 5
        assert result.member("U16") == 5
        assert result.member("U32") == 5
        assert result.member("U64") == 5
        assert result.member("F32") == 5.0
        assert result.member("F64") == 5.0
        assert result.member("Str") == "evt-005"
        assert result.member("P3").member("Px") == 4
        assert result.member("P3").member("Py") == 5.0
        assert result.member("P3").member("Pz") == 4
        assert result.member("ArrayI16").tolist() == [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
        assert result.member("ArrayU16").tolist() == [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
        assert result.member("ArrayI32").tolist() == [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
        assert result.member("ArrayU32").tolist() == [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
        assert result.member("ArrayI64").tolist() == [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
        assert result.member("ArrayU64").tolist() == [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
        assert result.member("ArrayF32").tolist() == [
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
        ]
        assert result.member("ArrayF32").tolist() == [
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
        ]
        assert result.member("ArrayF64").tolist() == [
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
        ]
        assert result.member("ArrayF64").tolist() == [
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
        ]
        assert result.member("StdStr") == "std-005"
        assert result.member("SliceI16").tolist() == [5, 5, 5, 5, 5]
        assert result.member("SliceI32").tolist() == [5, 5, 5, 5, 5]
        assert result.member("SliceI64").tolist() == [5, 5, 5, 5, 5]
        assert result.member("SliceU16").tolist() == [5, 5, 5, 5, 5]
        assert result.member("SliceU32").tolist() == [5, 5, 5, 5, 5]
        assert result.member("SliceU64").tolist() == [5, 5, 5, 5, 5]
        assert result.member("StlVecI16").tolist() == [5, 5, 5, 5, 5]
        assert result.member("StlVecI32").tolist() == [5, 5, 5, 5, 5]
        assert result.member("StlVecI64").tolist() == [5, 5, 5, 5, 5]
        assert result.member("StlVecU16").tolist() == [5, 5, 5, 5, 5]
        assert result.member("StlVecU32").tolist() == [5, 5, 5, 5, 5]
        assert result.member("StlVecU64").tolist() == [5, 5, 5, 5, 5]
        assert result.member("StlVecF32").tolist() == [5.0, 5.0, 5.0, 5.0, 5.0]
        assert result.member("StlVecF64").tolist() == [5.0, 5.0, 5.0, 5.0, 5.0]
        assert result.member("StlVecStr").tolist() == [
            "vec-005",
            "vec-005",
            "vec-005",
            "vec-005",
            "vec-005",
        ]
        assert result.member("End") == "end-005"


def test_TLorentzVectors_show():
    with uproot4.open(skhep_testdata.data_path("uproot-HZZ-objects.root"))[
        "events"
    ] as tree:
        tree.show()


def test_TVector2():
    with uproot4.open(skhep_testdata.data_path("uproot-HZZ-objects.root"))[
        "events"
    ] as tree:
        result = tree["MET"].array(library="np", entry_stop=1)[0]
        assert (result.member("fX"), result.member("fY")) == (
            5.912771224975586,
            2.5636332035064697,
        )


def test_vector_TLorentzVector():
    with uproot4.open(skhep_testdata.data_path("uproot-HZZ-objects.root"))[
        "events"
    ] as tree:
        result = tree["muonp4"].array(library="np", entry_stop=1)[0]
        assert (
            result[0].member("fE"),
            result[0].member("fP").member("fX"),
            result[0].member("fP").member("fY"),
            result[0].member("fP").member("fZ"),
        ) == (
            54.77949905395508,
            -52.89945602416992,
            -11.654671669006348,
            -8.16079330444336,
        )


def test_strided():
    with uproot4.open(skhep_testdata.data_path("uproot-HZZ-objects.root"))[
        "events"
    ] as tree:
        assert tree.file.class_named("TLorentzVector", "max").strided_interpretation(
            tree.file
        ).from_dtype == numpy.dtype(
            [
                ("fUniqueID", ">u4"),
                ("fBits", ">u4"),
                ("fP/fUniqueID", ">u4"),
                ("fP/fBits", ">u4"),
                ("fP/fX", ">f8"),
                ("fP/fY", ">f8"),
                ("fP/fZ", ">f8"),
                ("fE", ">f8"),
            ]
        )

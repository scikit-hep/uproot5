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


def test_formula_with_dot():
    with uproot4.open(
        skhep_testdata.data_path("uproot-small-evnt-tree-fullsplit.root")
    )["tree"] as tree:
        assert tree.arrays("P3.Py - 50", library="np")["P3.Py - 50"].tolist() == list(
            range(-50, 50)
        )


def test_formula_with_slash():
    with uproot4.open(
        skhep_testdata.data_path("uproot-small-evnt-tree-fullsplit.root")
    )["tree"] as tree:
        assert tree.arrays("get('evt/P3/P3.Py') - 50", library="np")[
            "get('evt/P3/P3.Py') - 50"
        ].tolist() == list(range(-50, 50))


def test_formula_with_missing():
    with uproot4.open(
        skhep_testdata.data_path("uproot-small-evnt-tree-fullsplit.root")
    )["tree"] as tree:
        with pytest.raises(KeyError):
            tree.arrays("wonky", library="np")


def test_strings1():
    with uproot4.open(
        skhep_testdata.data_path("uproot-sample-6.20.04-uncompressed.root")
    )["sample/str"] as branch:
        result = branch.array(library="np")
        assert result.tolist() == ["hey-{0}".format(i) for i in range(30)]


def test_strings4():
    with uproot4.open(
        skhep_testdata.data_path("uproot-small-evnt-tree-fullsplit.root")
    )["tree/StlVecStr"] as branch:
        result = branch.array(library="np")
        assert [result.tolist() for x in result] == [
            ["vec-{0:03d}".format(i)] * (i % 10) for i in range(100)
        ]


@pytest.mark.skip(reason="FIXME: implement unsplit object")
def test_strings4():
    with uproot4.open(skhep_testdata.data_path("uproot-small-evnt-tree-nosplit.root"))[
        "tree/evt"
    ] as branch:
        result = branch.array(library="np")
        assert [result.member("StlVecStr").tolist() for x in result] == [
            ["vec-{0:03d}".format(i)] * (i % 10) for i in range(100)
        ]


def test_strings4():
    with uproot4.open(skhep_testdata.data_path("uproot-vectorVectorDouble.root"))[
        "t/x"
    ] as branch:
        result = branch.array(library="np")
        assert [x.tolist() for x in result] == [
            [],
            [[], []],
            [[10.0], [], [10.0, 20.0]],
            [[20.0, -21.0, -22.0]],
            [[200.0], [-201.0], [202.0]],
        ]


def test_double32():
    with uproot4.open(skhep_testdata.data_path("uproot-demo-double32.root"))["T"] as t:
        fD64 = t["fD64"].array(library="np")
        fF32 = t["fF32"].array(library="np")
        fI32 = t["fI32"].array(library="np")
        fI30 = t["fI30"].array(library="np")
        fI28 = t["fI28"].array(library="np")
        ratio_fF32 = fF32 / fD64
        ratio_fI32 = fI32 / fD64
        ratio_fI30 = fI30 / fD64
        ratio_fI28 = fI28 / fD64
        assert ratio_fF32.min() > 0.9999 and ratio_fF32.max() < 1.0001
        assert ratio_fI32.min() > 0.9999 and ratio_fI32.max() < 1.0001
        assert ratio_fI30.min() > 0.9999 and ratio_fI30.max() < 1.0001
        assert ratio_fI28.min() > 0.9999 and ratio_fI28.max() < 1.0001

def test_double32_2():
    with uproot4.open(skhep_testdata.data_path("uproot-issue187.root"))["fTreeV0"] as t:
        assert numpy.all(t["fMultiplicity"].array(library="np") == -1)
        assert t["V0s.fEtaPos"].array(library="np")[-3].tolist() == [-0.390625, 0.046875]

def test_double32_3():
    with uproot4.open(skhep_testdata.data_path("uproot-issue232.root"))["fTreeV0"] as t:
        assert t["V0Hyper.fNsigmaHe3Pos"].array(library="np")[-1].tolist() == [19.38658905029297, 999.0]
        assert t["V0Hyper.fDcaPos2PrimaryVertex"].array(library="np")[-1].tolist() == [0.256, 0.256]

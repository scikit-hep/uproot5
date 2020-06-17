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


@pytest.mark.skip(reason="FIXME: implement strings specified by a TStreamer")
def test_strings2():
    with uproot4.open(
        skhep_testdata.data_path("uproot-small-evnt-tree-fullsplit.root")
    )["tree/Str"] as branch:
        result = branch.array(library="np")
        assert result.tolist() == ["evt-{0:03d}".format(i) for i in range(100)]


@pytest.mark.skip(reason="FIXME: implement std::string")
def test_strings3():
    with uproot4.open(
        skhep_testdata.data_path("uproot-small-evnt-tree-fullsplit.root")
    )["tree/StdStr"] as branch:
        result = branch.array(library="np")
        assert result.tolist() == ["std-{0:03d}".format(i) for i in range(100)]


@pytest.mark.skip(reason="FIXME: implement std::vector<std::string>")
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


@pytest.mark.skip(reason="FIXME: implement std::vector<std::vector<double>>")
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

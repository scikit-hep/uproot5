# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import sys
import json

import numpy
import pytest
import skhep_testdata

import uproot4


def test_awkward_strings():
    awkward1 = pytest.importorskip("awkward1")
    with uproot4.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert awkward1.to_list(tree["string"].array(library="ak")) == [
            "one",
            "two",
            "three",
            "four",
            "five",
        ]


def test_pandas_strings():
    pandas = pytest.importorskip("pandas")
    with uproot4.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert tree["string"].array(library="pd").values.tolist() == [
            "one",
            "two",
            "three",
            "four",
            "five",
        ]


def test_leaflist_numpy():
    with uproot4.open(skhep_testdata.data_path("uproot-leaflist.root"))[
        "tree/leaflist"
    ] as branch:
        result = branch.array(library="np")
        assert result.dtype.names == ("x", "y", "z")
        assert result.tolist() == [
            (1.1, 1, 97),
            (2.2, 2, 98),
            (3.3, 3, 99),
            (4.0, 4, 100),
            (5.5, 5, 101),
        ]


def test_leaflist_awkward():
    awkward1 = pytest.importorskip("awkward1")
    with uproot4.open(skhep_testdata.data_path("uproot-leaflist.root"))[
        "tree/leaflist"
    ] as branch:
        result = branch.array(library="ak")
        assert str(awkward1.type(result)) == '5 * {"x": float64, "y": int32, "z": int8}'
        assert awkward1.to_list(result) == [
            {"x": 1.1, "y": 1, "z": 97},
            {"x": 2.2, "y": 2, "z": 98},
            {"x": 3.3, "y": 3, "z": 99},
            {"x": 4.0, "y": 4, "z": 100},
            {"x": 5.5, "y": 5, "z": 101},
        ]


def test_leaflist_pandas():
    pandas = pytest.importorskip("pandas")
    with uproot4.open(skhep_testdata.data_path("uproot-leaflist.root"))["tree"] as tree:
        result = tree["leaflist"].array(library="pd")
        assert list(result.columns) == [":x", ":y", ":z"]
        assert result[":x"].values.tolist() == [1.1, 2.2, 3.3, 4.0, 5.5]
        assert result[":y"].values.tolist() == [1, 2, 3, 4, 5]
        assert result[":z"].values.tolist() == [97, 98, 99, 100, 101]

        result = tree.arrays("leaflist", library="pd")
        assert list(result.columns) == ["leaflist:x", "leaflist:y", "leaflist:z"]
        assert result["leaflist:x"].values.tolist() == [1.1, 2.2, 3.3, 4.0, 5.5]
        assert result["leaflist:y"].values.tolist() == [1, 2, 3, 4, 5]
        assert result["leaflist:z"].values.tolist() == [97, 98, 99, 100, 101]

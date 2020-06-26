# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import sys
import json

import numpy
import pytest
import skhep_testdata

import uproot4


def test_string():
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

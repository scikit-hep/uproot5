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
        assert sample.arrays("i4", library="np")["i4"].tolist() == [-15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

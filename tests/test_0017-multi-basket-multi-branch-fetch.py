# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import sys
import json

try:
    from io import StringIO
except ImportError:
    from StringIO import StringIO

import numpy
import pytest
import skhep_testdata

import uproot4
import uproot4.interpret.numerical


def test_any_basket():
    interpretation = uproot4.interpret.numerical.AsDtype(">i4")

    with uproot4.open(
        skhep_testdata.data_path("uproot-sample-6.20.04-uncompressed.root")
    )["sample/i4"] as branch:
        assert branch.basket(0).array(interpretation).tolist() == [
            -15,
            -14,
            -13,
            -12,
            -11,
            -10,
            -9,
        ]
        assert branch.basket(1).array(interpretation).tolist() == [
            -8,
            -7,
            -6,
            -5,
            -4,
            -3,
            -2,
        ]
        assert branch.basket(2).array(interpretation).tolist() == [
            -1,
            0,
            1,
            2,
            3,
            4,
            5,
        ]
        assert branch.basket(3).array(interpretation).tolist() == [
            6,
            7,
            8,
            9,
            10,
            11,
            12,
        ]
        assert branch.basket(4).array(interpretation).tolist() == [13, 14]


def test_stitching_arrays():
    interpretation = uproot4.interpret.numerical.AsDtype("i8")
    expectation = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    basket_arrays = [[0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], [10], [11, 12, 13, 14]]
    entry_offsets = [0, 5, 7, 7, 10, 11, 15]

    for start in range(16):
        for stop in range(15, -1, -1):
            actual = interpretation.final_array(
                basket_arrays, start, stop, entry_offsets
            )
            assert expectation[start:stop] == actual.tolist()

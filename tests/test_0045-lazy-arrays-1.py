# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import numpy
import pytest
import skhep_testdata

import uproot4


def test_array_cast():
    with uproot4.open(skhep_testdata.data_path("uproot-Zmumu.root"))[
        "events"
    ] as events:
        assert numpy.array(events["px1"])[:5].tolist() == [
            -41.1952876442,
            35.1180497674,
            35.1180497674,
            34.1444372454,
            22.7835819537,
        ]


def test_awkward():
    awkward1 = pytest.importorskip("awkward1")
    files = (
        skhep_testdata.data_path("uproot-sample-6.20.04-uncompressed.root").replace(
            "6.20.04", "*"
        )
        + ":sample"
    )
    cache = {}
    array = uproot4.lazy(files, array_cache=cache)
    assert len(cache) == 0

    assert awkward1.to_list(array[:5, "i4"]) == [-15, -14, -13, -12, -11]
    assert len(cache) == 1

    assert awkward1.to_list(array[:5, "ai4"]) == [
        [-14, -13, -12],
        [-13, -12, -11],
        [-12, -11, -10],
        [-11, -10, -9],
        [-10, -9, -8],
    ]
    assert len(cache) == 2

    assert awkward1.to_list(array[:5, "Ai4"]) == [
        [],
        [-15],
        [-15, -13],
        [-15, -13, -11],
        [-15, -13, -11, -9],
    ]
    assert len(cache) == 3

    assert awkward1.to_list(array[:5, "str"]) == [
        "hey-0",
        "hey-1",
        "hey-2",
        "hey-3",
        "hey-4",
    ]
    assert len(cache) == 4

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

import numpy
import pytest
import skhep_testdata

import uproot


def test_array_cast():
    with uproot.open(skhep_testdata.data_path("uproot-Zmumu.root"))["events"] as events:
        assert numpy.array(events["px1"])[:5].tolist() == [
            -41.1952876442,
            35.1180497674,
            35.1180497674,
            34.1444372454,
            22.7835819537,
        ]


def test_branch_pluralization():
    awkward = pytest.importorskip("awkward")
    with uproot.open(skhep_testdata.data_path("uproot-Zmumu.root"))[
        "events/px1"
    ] as px1:
        assert px1.array(library="np")[:5].tolist() == [
            -41.1952876442,
            35.1180497674,
            35.1180497674,
            34.1444372454,
            22.7835819537,
        ]
        assert px1.arrays(library="np")["px1"][:5].tolist() == [
            -41.1952876442,
            35.1180497674,
            35.1180497674,
            34.1444372454,
            22.7835819537,
        ]

        for i, arrays in enumerate(px1.iterate(library="np", step_size=1000)):
            if i == 0:
                assert arrays["px1"][:5].tolist() == [
                    -41.1952876442,
                    35.1180497674,
                    35.1180497674,
                    34.1444372454,
                    22.7835819537,
                ]
            elif i == 1:
                assert arrays["px1"][:5].tolist() == [
                    26.043758785,
                    26.043758785,
                    25.9962042016,
                    -44.4626620943,
                    28.2794901505,
                ]
            elif i == 2:
                assert arrays["px1"][:5].tolist() == [
                    -43.3783782352,
                    -43.3783782352,
                    -43.2444221651,
                    -20.2126675303,
                    43.7131175076,
                ]
            else:
                assert False

    for i, arrays in enumerate(
        uproot.iterate({skhep_testdata.data_path("uproot-Zmumu.root"): "events/px1"})
    ):
        if i == 0:
            assert arrays["px1"][:5].tolist() == [
                -41.1952876442,
                35.1180497674,
                35.1180497674,
                34.1444372454,
                22.7835819537,
            ]
        elif i == 1:
            assert arrays["px1"][:5].tolist() == [
                26.043758785,
                26.043758785,
                25.9962042016,
                -44.4626620943,
                28.2794901505,
            ]
        elif i == 2:
            assert arrays["px1"][:5].tolist() == [
                -43.3783782352,
                -43.3783782352,
                -43.2444221651,
                -20.2126675303,
                43.7131175076,
            ]
        else:
            assert False


def test_awkward():
    awkward = pytest.importorskip("awkward")
    files = skhep_testdata.data_path("uproot-sample-6.20.04-uncompressed.root").replace(
        "6.20.04", "*"
    )
    cache = {}
    array = uproot.lazy({files: "sample"}, array_cache=cache)
    assert len(cache) == 0

    assert awkward.to_list(array[:5, "i4"]) == [-15, -14, -13, -12, -11]
    assert len(cache) == 1

    assert awkward.to_list(array[:5, "ai4"]) == [
        [-14, -13, -12],
        [-13, -12, -11],
        [-12, -11, -10],
        [-11, -10, -9],
        [-10, -9, -8],
    ]
    assert len(cache) == 2

    assert awkward.to_list(array[:5, "Ai4"]) == [
        [],
        [-15],
        [-15, -13],
        [-15, -13, -11],
        [-15, -13, -11, -9],
    ]
    assert len(cache) == 3

    assert awkward.to_list(array[:5, "str"]) == [
        "hey-0",
        "hey-1",
        "hey-2",
        "hey-3",
        "hey-4",
    ]
    assert len(cache) == 4


def test_awkward_pluralization():
    awkward = pytest.importorskip("awkward")
    files = skhep_testdata.data_path("uproot-sample-6.20.04-uncompressed.root").replace(
        "6.20.04", "*"
    )
    array = uproot.lazy({files: "sample"})
    assert awkward.to_list(array[:5, "i4"]) == [-15, -14, -13, -12, -11]


def test_lazy_called_on_nonexistent_file():
    awkward = pytest.importorskip("awkward")
    filename = "nonexistent_file.root"
    with pytest.raises(uproot._util._FileNotFoundError) as excinfo:
        uproot.lazy(filename)
    assert filename in str(excinfo.value)

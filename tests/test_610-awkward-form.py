import json
import sys

import numpy
import pytest
import skhep_testdata

import uproot


def test_awkward_strings():
    awkward = pytest.importorskip("awkward")
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert awkward.to_list(tree["string"].array(library="ak")) == [
            "one",
            "two",
            "three",
            "four",
            "five",
        ]


def test_awkward_tstring():
    awkward = pytest.importorskip("awkward")
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert awkward.to_list(tree["tstring"].array(library="ak")) == [
            "one",
            "two",
            "three",
            "four",
            "five",
        ]


def test_awkward_vector_int32():
    awkward = pytest.importorskip("awkward")
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert awkward.to_list(tree["vector_int32"].array(library="ak")) == [
            [1],
            [1, 2],
            [1, 2, 3],
            [1, 2, 3, 4],
            [1, 2, 3, 4, 5],
        ]


def test_awkward_vector_string():
    awkward = pytest.importorskip("awkward")
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert awkward.to_list(tree["vector_string"].array(library="ak")) == [
            ["one"],
            ["one", "two"],
            ["one", "two", "three"],
            ["one", "two", "three", "four"],
            ["one", "two", "three", "four", "five"],
        ]


def test_awkward_vector_tstring():
    awkward = pytest.importorskip("awkward")
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert awkward.to_list(tree["vector_tstring"].array(library="ak")) == [
            ["one"],
            ["one", "two"],
            ["one", "two", "three"],
            ["one", "two", "three", "four"],
            ["one", "two", "three", "four", "five"],
        ]


def test_awkward_vector_vector_int32():
    awkward = pytest.importorskip("awkward")
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert awkward.to_list(tree["vector_vector_int32"].array(library="ak")) == [
            [[1]],
            [[1], [1, 2]],
            [[1], [1, 2], [1, 2, 3]],
            [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4]],
            [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]],
        ]


def test_awkward_vector_vector_string():
    awkward = pytest.importorskip("awkward")
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert awkward.to_list(tree["vector_vector_string"].array(library="ak")) == [
            [["one"]],
            [["one"], ["one", "two"]],
            [["one"], ["one", "two"], ["one", "two", "three"]],
            [
                ["one"],
                ["one", "two"],
                ["one", "two", "three"],
                ["one", "two", "three", "four"],
            ],
            [
                ["one"],
                ["one", "two"],
                ["one", "two", "three"],
                ["one", "two", "three", "four"],
                ["one", "two", "three", "four", "five"],
            ],
        ]


def test_awkward_vector_vector_tstring():
    awkward = pytest.importorskip("awkward")
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert awkward.to_list(tree["vector_vector_tstring"].array(library="ak")) == [
            [["one"]],
            [["one"], ["one", "two"]],
            [["one"], ["one", "two"], ["one", "two", "three"]],
            [
                ["one"],
                ["one", "two"],
                ["one", "two", "three"],
                ["one", "two", "three", "four"],
            ],
            [
                ["one"],
                ["one", "two"],
                ["one", "two", "three"],
                ["one", "two", "three", "four"],
                ["one", "two", "three", "four", "five"],
            ],
        ]


def test_awkward_set_int32():
    awkward = pytest.importorskip("awkward")
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert awkward.to_list(tree["set_int32"].array(library="ak")) == [
            [1],
            [1, 2],
            [1, 2, 3],
            [1, 2, 3, 4],
            [1, 2, 3, 4, 5],
        ]


def test_awkward_vector_set_int32():
    awkward = pytest.importorskip("awkward")
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert awkward.to_list(tree["vector_set_int32"].array(library="ak")) == [
            [[1]],
            [[1], [1, 2]],
            [[1], [1, 2], [1, 2, 3]],
            [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4]],
            [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]],
        ]


def test_awkward_vector_set_string():
    awkward = pytest.importorskip("awkward")
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert awkward.to_list(tree["vector_set_string"].array(library="ak")) == [
            [["one"]],
            [["one"], ["one", "two"]],
            [["one"], ["one", "two"], ["one", "three", "two"]],
            [
                ["one"],
                ["one", "two"],
                ["one", "three", "two"],
                ["four", "one", "three", "two"],
            ],
            [
                ["one"],
                ["one", "two"],
                ["one", "three", "two"],
                ["four", "one", "three", "two"],
                ["five", "four", "one", "three", "two"],
            ],
        ]


def test_awkward_set_string():
    awkward = pytest.importorskip("awkward")
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert awkward.to_list(tree["set_string"].array(library="ak")) == [
            ["one"],
            ["one", "two"],
            ["one", "three", "two"],
            ["four", "one", "three", "two"],
            ["five", "four", "one", "three", "two"],
        ]


def test_awkward_map_int32_int16():
    awkward = pytest.importorskip("awkward")
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        keys, values = awkward.unzip(tree["map_int32_int16"].array(library="ak"))
        assert awkward.to_list(keys) == [
            [1],
            [1, 2],
            [1, 2, 3],
            [1, 2, 3, 4],
            [1, 2, 3, 4, 5],
        ]
        assert awkward.to_list(values) == [
            [1],
            [1, 2],
            [1, 2, 3],
            [1, 2, 3, 4],
            [1, 2, 3, 4, 5],
        ]


def test_awkward_map_int32_vector_int16():
    awkward = pytest.importorskip("awkward")
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        keys, values = awkward.unzip(tree["map_int32_vector_int16"].array(library="ak"))
        assert awkward.to_list(keys) == [
            [1],
            [1, 2],
            [1, 2, 3],
            [1, 2, 3, 4],
            [1, 2, 3, 4, 5],
        ]
        assert awkward.to_list(values) == [
            [[1]],
            [[1], [1, 2]],
            [[1], [1, 2], [1, 2, 3]],
            [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4]],
            [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]],
        ]


def test_awkward_map_int32_vector_string():
    awkward = pytest.importorskip("awkward")
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        keys, values = awkward.unzip(
            tree["map_int32_vector_string"].array(library="ak")
        )
        assert awkward.to_list(keys) == [
            [1],
            [1, 2],
            [1, 2, 3],
            [1, 2, 3, 4],
            [1, 2, 3, 4, 5],
        ]
        assert awkward.to_list(values) == [
            [["one"]],
            [["one"], ["one", "two"]],
            [["one"], ["one", "two"], ["one", "two", "three"]],
            [
                ["one"],
                ["one", "two"],
                ["one", "two", "three"],
                ["one", "two", "three", "four"],
            ],
            [
                ["one"],
                ["one", "two"],
                ["one", "two", "three"],
                ["one", "two", "three", "four"],
                ["one", "two", "three", "four", "five"],
            ],
        ]


def test_awkward_map_int32_set_int16():
    awkward = pytest.importorskip("awkward")
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        keys, values = awkward.unzip(tree["map_int32_set_int16"].array(library="ak"))
        assert awkward.to_list(keys) == [
            [1],
            [1, 2],
            [1, 2, 3],
            [1, 2, 3, 4],
            [1, 2, 3, 4, 5],
        ]
        assert awkward.to_list(values) == [
            [[1]],
            [[1], [1, 2]],
            [[1], [1, 2], [1, 2, 3]],
            [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4]],
            [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]],
        ]


def test_awkward_map_int32_set_string():
    awkward = pytest.importorskip("awkward")
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        keys, values = awkward.unzip(tree["map_int32_set_string"].array(library="ak"))
        assert awkward.to_list(keys) == [
            [1],
            [1, 2],
            [1, 2, 3],
            [1, 2, 3, 4],
            [1, 2, 3, 4, 5],
        ]
        assert awkward.to_list(values) == [
            [["one"]],
            [["one"], ["one", "two"]],
            [["one"], ["one", "two"], ["one", "three", "two"]],
            [
                ["one"],
                ["one", "two"],
                ["one", "three", "two"],
                ["four", "one", "three", "two"],
            ],
            [
                ["one"],
                ["one", "two"],
                ["one", "three", "two"],
                ["four", "one", "three", "two"],
                ["five", "four", "one", "three", "two"],
            ],
        ]


def test_awkward_map_string_int16():
    awkward = pytest.importorskip("awkward")
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        keys, values = awkward.unzip(tree["map_string_int16"].array(library="ak"))
        assert awkward.to_list(keys) == [
            ["one"],
            ["one", "two"],
            ["one", "three", "two"],
            ["four", "one", "three", "two"],
            ["five", "four", "one", "three", "two"],
        ]
        assert awkward.to_list(values) == [
            [1],
            [1, 2],
            [1, 3, 2],
            [4, 1, 3, 2],
            [5, 4, 1, 3, 2],
        ]


def test_awkward_map_string_vector_int16():
    awkward = pytest.importorskip("awkward")
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        keys, values = awkward.unzip(
            tree["map_string_vector_int16"].array(library="ak")
        )
        assert awkward.to_list(keys) == [
            ["one"],
            ["one", "two"],
            ["one", "three", "two"],
            ["four", "one", "three", "two"],
            ["five", "four", "one", "three", "two"],
        ]
        assert awkward.to_list(values) == [
            [[1]],
            [[1], [1, 2]],
            [[1], [1, 2, 3], [1, 2]],
            [[1, 2, 3, 4], [1], [1, 2, 3], [1, 2]],
            [[1, 2, 3, 4, 5], [1, 2, 3, 4], [1], [1, 2, 3], [1, 2]],
        ]


def test_awkward_map_string_vector_string():
    awkward = pytest.importorskip("awkward")
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        keys, values = awkward.unzip(
            tree["map_string_vector_string"].array(library="ak")
        )
        assert awkward.to_list(keys) == [
            ["one"],
            ["one", "two"],
            ["one", "three", "two"],
            ["four", "one", "three", "two"],
            ["five", "four", "one", "three", "two"],
        ]
        assert awkward.to_list(values) == [
            [["one"]],
            [["one"], ["one", "two"]],
            [["one"], ["one", "two", "three"], ["one", "two"]],
            [
                ["one", "two", "three", "four"],
                ["one"],
                ["one", "two", "three"],
                ["one", "two"],
            ],
            [
                ["one", "two", "three", "four", "five"],
                ["one", "two", "three", "four"],
                ["one"],
                ["one", "two", "three"],
                ["one", "two"],
            ],
        ]


def test_awkward_map_string_set_int16():
    awkward = pytest.importorskip("awkward")
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        keys, values = awkward.unzip(tree["map_string_set_int16"].array(library="ak"))
        assert awkward.to_list(keys) == [
            ["one"],
            ["one", "two"],
            ["one", "three", "two"],
            ["four", "one", "three", "two"],
            ["five", "four", "one", "three", "two"],
        ]
        assert awkward.to_list(values) == [
            [[1]],
            [[1], [1, 2]],
            [[1], [1, 2, 3], [1, 2]],
            [[1, 2, 3, 4], [1], [1, 2, 3], [1, 2]],
            [[1, 2, 3, 4, 5], [1, 2, 3, 4], [1], [1, 2, 3], [1, 2]],
        ]


def test_awkward_map_string_set_string():
    awkward = pytest.importorskip("awkward")
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        keys, values = awkward.unzip(tree["map_string_set_string"].array(library="ak"))
        assert awkward.to_list(keys) == [
            ["one"],
            ["one", "two"],
            ["one", "three", "two"],
            ["four", "one", "three", "two"],
            ["five", "four", "one", "three", "two"],
        ]
        assert awkward.to_list(values) == [
            [["one"]],
            [["one"], ["one", "two"]],
            [["one"], ["one", "three", "two"], ["one", "two"]],
            [
                ["four", "one", "three", "two"],
                ["one"],
                ["one", "three", "two"],
                ["one", "two"],
            ],
            [
                ["five", "four", "one", "three", "two"],
                ["four", "one", "three", "two"],
                ["one"],
                ["one", "three", "two"],
                ["one", "two"],
            ],
        ]


def test_awkward_map_int32_vector_vector_int16():
    awkward = pytest.importorskip("awkward")
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        keys, values = awkward.unzip(
            tree["map_int32_vector_vector_int16"].array(library="ak")
        )
        assert awkward.to_list(keys) == [
            [1],
            [1, 2],
            [1, 2, 3],
            [1, 2, 3, 4],
            [1, 2, 3, 4, 5],
        ]
        assert awkward.to_list(values) == [
            [[[1]]],
            [[[1]], [[1], [1, 2]]],
            [[[1]], [[1], [1, 2]], [[1], [1, 2], [1, 2, 3]]],
            [
                [[1]],
                [[1], [1, 2]],
                [[1], [1, 2], [1, 2, 3]],
                [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4]],
            ],
            [
                [[1]],
                [[1], [1, 2]],
                [[1], [1, 2], [1, 2, 3]],
                [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4]],
                [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]],
            ],
        ]


def test_awkward_map_string_string():
    awkward = pytest.importorskip("awkward")
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        keys, values = awkward.unzip(tree["map_string_string"].array(library="ak"))
        assert awkward.to_list(keys) == [
            ["one"],
            ["one", "two"],
            ["one", "three", "two"],
            ["four", "one", "three", "two"],
            ["five", "four", "one", "three", "two"],
        ]
        assert awkward.to_list(values) == [
            ["ONE"],
            ["ONE", "TWO"],
            ["ONE", "THREE", "TWO"],
            ["FOUR", "ONE", "THREE", "TWO"],
            ["FIVE", "FOUR", "ONE", "THREE", "TWO"],
        ]


def test_awkward_map_string_tstring():
    awkward = pytest.importorskip("awkward")
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        keys, values = awkward.unzip(tree["map_string_tstring"].array(library="ak"))
        assert awkward.to_list(keys) == [
            ["one"],
            ["one", "two"],
            ["one", "three", "two"],
            ["four", "one", "three", "two"],
            ["five", "four", "one", "three", "two"],
        ]
        assert awkward.to_list(values) == [
            ["ONE"],
            ["ONE", "TWO"],
            ["ONE", "THREE", "TWO"],
            ["FOUR", "ONE", "THREE", "TWO"],
            ["FIVE", "FOUR", "ONE", "THREE", "TWO"],
        ]


# get a new interpretation of branch in HZZ from identify/interpretation_of with simplify = false. Then pass is it to TBranch/array

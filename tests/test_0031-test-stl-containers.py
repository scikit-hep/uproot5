# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import sys
import json

import numpy
import pytest
import skhep_testdata

import uproot4
from uproot4.interpretation.numerical import AsDtype
from uproot4.interpretation.jagged import AsJagged
from uproot4.interpretation.objects import AsObjects
from uproot4.stl_containers import AsString
from uproot4.stl_containers import AsVector
from uproot4.stl_containers import AsSet
from uproot4.stl_containers import AsMap


def test_typename():
    with uproot4.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert tree["vector_int32"].interpretation == AsJagged(AsDtype(">i4"), 10)
        assert tree["vector_string"].interpretation == AsObjects(
            AsVector(True, AsString(False))
        )
        assert tree["vector_vector_int32"].interpretation == AsObjects(
            AsVector(True, AsVector(False, numpy.dtype(">i4")))
        )
        assert tree["vector_vector_string"].interpretation == AsObjects(
            AsVector(True, AsVector(False, AsString(False)))
        )
        assert tree["vector_set_int32"].interpretation == AsObjects(
            AsVector(True, AsSet(False, numpy.dtype(">i4")))
        )
        assert tree["vector_set_string"].interpretation == AsObjects(
            AsVector(True, AsSet(False, AsString(False)))
        )
        assert tree["set_int32"].interpretation == AsObjects(
            AsSet(True, numpy.dtype(">i4"))
        )
        assert tree["set_string"].interpretation == AsObjects(
            AsSet(True, AsString(False))
        )
        assert tree["map_int32_int16"].interpretation == AsObjects(
            AsMap(True, numpy.dtype(">i4"), numpy.dtype(">i2"))
        )
        assert tree["map_int32_vector_int16"].interpretation == AsObjects(
            AsMap(True, numpy.dtype(">i4"), AsVector(True, numpy.dtype(">i2")))
        )
        assert tree["map_int32_vector_string"].interpretation == AsObjects(
            AsMap(True, numpy.dtype(">i4"), AsVector(True, AsString(False)))
        )
        assert tree["map_int32_set_int16"].interpretation == AsObjects(
            AsMap(True, numpy.dtype(">i4"), AsSet(True, numpy.dtype(">i2")))
        )
        assert tree["map_int32_set_string"].interpretation == AsObjects(
            AsMap(True, numpy.dtype(">i4"), AsSet(True, AsString(False)))
        )
        assert tree["map_string_int16"].interpretation == AsObjects(
            AsMap(True, AsString(True), numpy.dtype(">i2"))
        )
        assert tree["map_string_vector_int16"].interpretation == AsObjects(
            AsMap(True, AsString(True), AsVector(True, numpy.dtype(">i2")))
        )
        assert tree["map_string_vector_string"].interpretation == AsObjects(
            AsMap(True, AsString(True), AsVector(True, AsString(False)))
        )
        assert tree["map_string_set_int16"].interpretation == AsObjects(
            AsMap(True, AsString(True), AsSet(True, numpy.dtype(">i2")))
        )
        assert tree["map_string_set_string"].interpretation == AsObjects(
            AsMap(True, AsString(True), AsSet(True, AsString(False)))
        )
        assert tree["map_int32_vector_vector_int16"].interpretation == AsObjects(
            AsMap(
                True,
                numpy.dtype(">i4"),
                AsVector(True, AsVector(False, numpy.dtype(">i2"))),
            )
        )
        assert tree["map_int32_vector_set_int16"].interpretation == AsObjects(
            AsMap(
                True,
                numpy.dtype(">i4"),
                AsVector(True, AsSet(False, numpy.dtype(">i2"))),
            )
        )
        assert tree["map_string_string"].interpretation == AsObjects(
            AsMap(True, AsString(True), AsString(True))
        )


def test_string():
    with uproot4.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert tree["string"].array(library="np").tolist() == [
            "one",
            "two",
            "three",
            "four",
            "five",
        ]


def test_tstring():
    with uproot4.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert tree["tstring"].array(library="np").tolist() == [
            "one",
            "two",
            "three",
            "four",
            "five",
        ]


def test_vector_int32():
    with uproot4.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert [x.tolist() for x in tree["vector_int32"].array(library="np")] == [
            [1],
            [1, 2],
            [1, 2, 3],
            [1, 2, 3, 4],
            [1, 2, 3, 4, 5],
        ]


def test_vector_string():
    with uproot4.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert [x.tolist() for x in tree["vector_string"].array(library="np")] == [
            ["one"],
            ["one", "two"],
            ["one", "two", "three"],
            ["one", "two", "three", "four"],
            ["one", "two", "three", "four", "five"],
        ]


def test_vector_tstring():
    with uproot4.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert [x.tolist() for x in tree["vector_tstring"].array(library="np")] == [
            ["one"],
            ["one", "two"],
            ["one", "two", "three"],
            ["one", "two", "three", "four"],
            ["one", "two", "three", "four", "five"],
        ]


def test_vector_vector_int32():
    with uproot4.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert [
            x.tolist() for x in tree["vector_vector_int32"].array(library="np")
        ] == [
            [[1]],
            [[1], [1, 2]],
            [[1], [1, 2], [1, 2, 3]],
            [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4]],
            [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]],
        ]


def test_vector_vector_string():
    with uproot4.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert [
            x.tolist() for x in tree["vector_vector_string"].array(library="np")
        ] == [
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


def test_vector_set_int32():
    with uproot4.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert [x.tolist() for x in tree["vector_set_int32"].array(library="np")] == [
            [set([1])],
            [set([1]), set([1, 2])],
            [set([1]), set([1, 2]), set([1, 2, 3])],
            [set([1]), set([1, 2]), set([1, 2, 3]), set([1, 2, 3, 4])],
            [
                set([1]),
                set([1, 2]),
                set([1, 2, 3]),
                set([1, 2, 3, 4]),
                set([1, 2, 3, 4, 5]),
            ],
        ]


def test_vector_set_string():
    with uproot4.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert [x.tolist() for x in tree["vector_set_string"].array(library="np")] == [
            [set(["one"])],
            [set(["one"]), set(["one", "two"])],
            [set(["one"]), set(["one", "two"]), set(["one", "two", "three"])],
            [
                set(["one"]),
                set(["one", "two"]),
                set(["one", "two", "three"]),
                set(["one", "two", "three", "four"]),
            ],
            [
                set(["one"]),
                set(["one", "two"]),
                set(["one", "two", "three"]),
                set(["one", "two", "three", "four"]),
                set(["one", "two", "three", "four", "five"]),
            ],
        ]


def test_set_int32():
    with uproot4.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert [x.tolist() for x in tree["set_int32"].array(library="np")] == [
            set([1]),
            set([1, 2]),
            set([1, 2, 3]),
            set([1, 2, 3, 4]),
            set([1, 2, 3, 4, 5]),
        ]


def test_set_string():
    with uproot4.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert [x.tolist() for x in tree["set_string"].array(library="np")] == [
            set(["one"]),
            set(["one", "two"]),
            set(["one", "two", "three"]),
            set(["one", "two", "three", "four"]),
            set(["one", "two", "three", "four", "five"]),
        ]


def test_map_int32_int16():
    with uproot4.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert [x.tolist() for x in tree["map_int32_int16"].array(library="np")] == [
            {1: 1},
            {1: 1, 2: 2},
            {1: 1, 2: 2, 3: 3},
            {1: 1, 2: 2, 3: 3, 4: 4},
            {1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
        ]


def test_map_int32_vector_int16():
    with uproot4.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert [
            x.tolist() for x in tree["map_int32_vector_int16"].array(library="np")
        ] == [
            {1: [1]},
            {1: [1], 2: [1, 2]},
            {1: [1], 2: [1, 2], 3: [1, 2, 3]},
            {1: [1], 2: [1, 2], 3: [1, 2, 3], 4: [1, 2, 3, 4]},
            {1: [1], 2: [1, 2], 3: [1, 2, 3], 4: [1, 2, 3, 4], 5: [1, 2, 3, 4, 5]},
        ]


def test_map_int32_vector_string():
    with uproot4.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert [
            x.tolist() for x in tree["map_int32_vector_string"].array(library="np")
        ] == [
            {1: ["one"]},
            {1: ["one"], 2: ["one", "two"]},
            {1: ["one"], 2: ["one", "two"], 3: ["one", "two", "three"]},
            {
                1: ["one"],
                2: ["one", "two"],
                3: ["one", "two", "three"],
                4: ["one", "two", "three", "four"],
            },
            {
                1: ["one"],
                2: ["one", "two"],
                3: ["one", "two", "three"],
                4: ["one", "two", "three", "four"],
                5: ["one", "two", "three", "four", "five"],
            },
        ]


def test_map_int32_set_int16():
    with uproot4.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert [
            x.tolist() for x in tree["map_int32_set_int16"].array(library="np")
        ] == [
            {1: set([1])},
            {1: set([1]), 2: set([1, 2])},
            {1: set([1]), 2: set([1, 2]), 3: set([1, 2, 3])},
            {1: set([1]), 2: set([1, 2]), 3: set([1, 2, 3]), 4: set([1, 2, 3, 4])},
            {
                1: set([1]),
                2: set([1, 2]),
                3: set([1, 2, 3]),
                4: set([1, 2, 3, 4]),
                5: set([1, 2, 3, 4, 5]),
            },
        ]


def test_map_int32_set_string():
    with uproot4.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert [
            x.tolist() for x in tree["map_int32_set_string"].array(library="np")
        ] == [
            {1: set(["one"])},
            {1: set(["one"]), 2: set(["one", "two"])},
            {1: set(["one"]), 2: set(["one", "two"]), 3: set(["one", "two", "three"])},
            {
                1: set(["one"]),
                2: set(["one", "two"]),
                3: set(["one", "two", "three"]),
                4: set(["one", "two", "three", "four"]),
            },
            {
                1: set(["one"]),
                2: set(["one", "two"]),
                3: set(["one", "two", "three"]),
                4: set(["one", "two", "three", "four"]),
                5: set(["one", "two", "three", "four", "five"]),
            },
        ]


def test_map_string_int16():
    with uproot4.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert [x.tolist() for x in tree["map_string_int16"].array(library="np")] == [
            {"one": 1},
            {"one": 1, "two": 2},
            {"one": 1, "two": 2, "three": 3},
            {"one": 1, "two": 2, "three": 3, "four": 4},
            {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5},
        ]


def test_map_string_vector_int16():
    with uproot4.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert [
            x.tolist() for x in tree["map_string_vector_int16"].array(library="np")
        ] == [
            {"one": [1]},
            {"one": [1], "two": [1, 2]},
            {"one": [1], "two": [1, 2], "three": [1, 2, 3]},
            {"one": [1], "two": [1, 2], "three": [1, 2, 3], "four": [1, 2, 3, 4]},
            {
                "one": [1],
                "two": [1, 2],
                "three": [1, 2, 3],
                "four": [1, 2, 3, 4],
                "five": [1, 2, 3, 4, 5],
            },
        ]


def test_map_string_vector_string():
    with uproot4.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert [
            x.tolist() for x in tree["map_string_vector_string"].array(library="np")
        ] == [
            {"one": ["one"]},
            {"one": ["one"], "two": ["one", "two"]},
            {"one": ["one"], "two": ["one", "two"], "three": ["one", "two", "three"]},
            {
                "one": ["one"],
                "two": ["one", "two"],
                "three": ["one", "two", "three"],
                "four": ["one", "two", "three", "four"],
            },
            {
                "one": ["one"],
                "two": ["one", "two"],
                "three": ["one", "two", "three"],
                "four": ["one", "two", "three", "four"],
                "five": ["one", "two", "three", "four", "five"],
            },
        ]


def test_map_string_set_int16():
    with uproot4.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert [
            x.tolist() for x in tree["map_string_set_int16"].array(library="np")
        ] == [
            {"one": set([1])},
            {"one": set([1]), "two": set([1, 2])},
            {"one": set([1]), "two": set([1, 2]), "three": set([1, 2, 3])},
            {
                "one": set([1]),
                "two": set([1, 2]),
                "three": set([1, 2, 3]),
                "four": set([1, 2, 3, 4]),
            },
            {
                "one": set([1]),
                "two": set([1, 2]),
                "three": set([1, 2, 3]),
                "four": set([1, 2, 3, 4]),
                "five": set([1, 2, 3, 4, 5]),
            },
        ]


def test_map_string_set_string():
    with uproot4.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert [
            x.tolist() for x in tree["map_string_set_string"].array(library="np")
        ] == [
            {"one": set(["one"])},
            {"one": set(["one"]), "two": set(["one", "two"])},
            {
                "one": set(["one"]),
                "two": set(["one", "two"]),
                "three": set(["one", "two", "three"]),
            },
            {
                "one": set(["one"]),
                "two": set(["one", "two"]),
                "three": set(["one", "two", "three"]),
                "four": set(["one", "two", "three", "four"]),
            },
            {
                "one": set(["one"]),
                "two": set(["one", "two"]),
                "three": set(["one", "two", "three"]),
                "four": set(["one", "two", "three", "four"]),
                "five": set(["one", "two", "three", "four", "five"]),
            },
        ]


def test_map_int32_vector_vector_int16():
    with uproot4.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert [
            x.tolist()
            for x in tree["map_int32_vector_vector_int16"].array(library="np")
        ] == [
            {1: [[1]]},
            {1: [[1]], 2: [[1], [1, 2]]},
            {1: [[1]], 2: [[1], [1, 2]], 3: [[1], [1, 2], [1, 2, 3]]},
            {
                1: [[1]],
                2: [[1], [1, 2]],
                3: [[1], [1, 2], [1, 2, 3]],
                4: [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4]],
            },
            {
                1: [[1]],
                2: [[1], [1, 2]],
                3: [[1], [1, 2], [1, 2, 3]],
                4: [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4]],
                5: [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]],
            },
        ]


def test_map_int32_vector_set_int16():
    with uproot4.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert [
            x.tolist() for x in tree["map_int32_vector_set_int16"].array(library="np")
        ] == [
            {1: [set([1])]},
            {1: [set([1])], 2: [set([1]), set([1, 2])]},
            {
                1: [set([1])],
                2: [set([1]), set([1, 2])],
                3: [set([1]), set([1, 2]), set([1, 2, 3])],
            },
            {
                1: [set([1])],
                2: [set([1]), set([1, 2])],
                3: [set([1]), set([1, 2]), set([1, 2, 3])],
                4: [set([1]), set([1, 2]), set([1, 2, 3]), set([1, 2, 3, 4])],
            },
            {
                1: [set([1])],
                2: [set([1]), set([1, 2])],
                3: [set([1]), set([1, 2]), set([1, 2, 3])],
                4: [set([1]), set([1, 2]), set([1, 2, 3]), set([1, 2, 3, 4])],
                5: [
                    set([1]),
                    set([1, 2]),
                    set([1, 2, 3]),
                    set([1, 2, 3, 4]),
                    set([1, 2, 3, 4, 5]),
                ],
            },
        ]


def test_map_string_string():
    with uproot4.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert [x.tolist() for x in tree["map_string_string"].array(library="np")] == [
            {"one": "ONE"},
            {"one": "ONE", "two": "TWO"},
            {"one": "ONE", "two": "TWO", "three": "THREE"},
            {"one": "ONE", "two": "TWO", "three": "THREE", "four": "FOUR"},
            {
                "one": "ONE",
                "two": "TWO",
                "three": "THREE",
                "four": "FOUR",
                "five": "FIVE",
            },
        ]


def test_map_string_tstring():
    with uproot4.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert [x.tolist() for x in tree["map_string_tstring"].array(library="np")] == [
            {"one": "ONE"},
            {"one": "ONE", "two": "TWO"},
            {"one": "ONE", "two": "TWO", "three": "THREE"},
            {"one": "ONE", "two": "TWO", "three": "THREE", "four": "FOUR"},
            {
                "one": "ONE",
                "two": "TWO",
                "three": "THREE",
                "four": "FOUR",
                "five": "FIVE",
            },
        ]


@pytest.mark.skip(reason="FIXME: implement map<int,struct>")
def test_map_int_struct():
    # as described here:
    #
    # https://github.com/scikit-hep/uproot/issues/468#issuecomment-646325842
    #
    # python -c 'import uproot; t = uproot.open("/home/pivarski/irishep/scikit-hep-testdata/src/skhep_testdata/data/uproot-issue468.root")["Geant4Data/Geant4Data./Geant4Data.particles"]; print(t.array(uproot.asdebug)[0][:1000])'
    pass

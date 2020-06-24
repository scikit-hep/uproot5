# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import sys
import json

import numpy
import pytest
import skhep_testdata

import uproot4
from uproot4.interpretation.objects import AsObjects
from uproot4.stl_containers import AsString
from uproot4.stl_containers import AsVector
from uproot4.stl_containers import AsSet
from uproot4.stl_containers import AsMap


@pytest.mark.skip(reason="FIXME: stl_containers.root doesn't exist yet")
def test_typename():
    with uproot4.open("stl_containers.root")["tree"] as tree:
        assert tree["vector_int32"].interpretation == AsObjects(
            AsVector(True, numpy.dtype(">i4"))
        )
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
            AsMap(True, numpy.dtype(">i4"), AsVector(False, numpy.dtype(">i2")))
        )
        assert tree["map_int32_vector_string"].interpretation == AsObjects(
            AsMap(True, numpy.dtype(">i4"), AsVector(False, AsString(False)))
        )
        assert tree["map_int32_set_int16"].interpretation == AsObjects(
            AsMap(True, numpy.dtype(">i4"), AsSet(False, numpy.dtype(">i2")))
        )
        assert tree["map_int32_set_string"].interpretation == AsObjects(
            AsMap(True, numpy.dtype(">i4"), AsSet(False, AsString(False)))
        )
        assert tree["map_string_int16"].interpretation == AsObjects(
            AsMap(True, AsString(False), numpy.dtype(">i2"))
        )
        assert tree["map_string_vector_int16"].interpretation == AsObjects(
            AsMap(True, AsString(False), AsVector(False, numpy.dtype(">i2")))
        )
        assert tree["map_string_vector_string"].interpretation == AsObjects(
            AsMap(True, AsString(False), AsVector(False, AsString(False)))
        )
        assert tree["map_string_set_int16"].interpretation == AsObjects(
            AsMap(True, AsString(False), AsSet(False, numpy.dtype(">i2")))
        )
        assert tree["map_string_set_string"].interpretation == AsObjects(
            AsMap(True, AsString(False), AsSet(False, AsString(False)))
        )
        assert tree["map_int32_vector_vector_int16"].interpretation == AsObjects(
            AsMap(
                True,
                numpy.dtype(">i4"),
                AsVector(False, AsVector(False, numpy.dtype(">i2"))),
            )
        )
        assert tree["map_int32_vector_set_int16"].interpretation == AsObjects(
            AsMap(
                True,
                numpy.dtype(">i4"),
                AsVector(False, AsSet(False, numpy.dtype(">i2"))),
            )
        )
        assert tree["vector_map_int32_int16"].interpretation == AsObjects(
            AsVector(True, AsMap(False, numpy.dtype(">i4"), numpy.dtype(">i2")))
        )
        assert tree["vector_map_int32_string"].interpretation == AsObjects(
            AsVector(True, AsMap(False, numpy.dtype(">i4"), AsString(False)))
        )
        assert tree["vector_map_string_string"].interpretation == AsObjects(
            AsVector(True, AsMap(False, AsString(False), AsString(False)))
        )
        assert tree["map_string_string"].interpretation == AsObjects(
            AsMap(True, AsString(False), AsString(False))
        )


@pytest.mark.skip(reason="FIXME: stl_containers.root doesn't exist yet")
def test_string():
    with uproot4.open("stl_containers.root")["tree"] as tree:
        assert tree["string"].array(library="np").tolist() == [
            "one",
            "two",
            "three",
            "four",
            "five",
        ]


@pytest.mark.skip(reason="FIXME: stl_containers.root doesn't exist yet")
def test_tstring():
    with uproot4.open("stl_containers.root")["tree"] as tree:
        assert tree["tstring"].array(library="np").tolist() == [
            "one",
            "two",
            "three",
            "four",
            "five",
        ]


@pytest.mark.skip(reason="FIXME: stl_containers.root doesn't exist yet")
def test_vector_int32():
    with uproot4.open("stl_containers.root")["tree"] as tree:
        assert [x.tolist() for x in tree["vector_int32"].array(library="np")] == [
            [1],
            [1, 2],
            [1, 2, 3],
            [1, 2, 3, 4],
            [1, 2, 3, 4, 5],
        ]


@pytest.mark.skip(reason="FIXME: stl_containers.root doesn't exist yet")
def test_vector_string():
    with uproot4.open("stl_containers.root")["tree"] as tree:
        assert [x.tolist() for x in tree["vector_string"].array(library="np")] == [
            ["one"],
            ["one", "two"],
            ["one", "two", "three"],
            ["one", "two", "three", "four"],
            ["one", "two", "three", "four", "five"],
        ]


@pytest.mark.skip(reason="FIXME: stl_containers.root doesn't exist yet")
def test_vector_tstring():
    with uproot4.open("stl_containers.root")["tree"] as tree:
        assert [x.tolist() for x in tree["vector_tstring"].array(library="np")] == [
            ["one"],
            ["one", "two"],
            ["one", "two", "three"],
            ["one", "two", "three", "four"],
            ["one", "two", "three", "four", "five"],
        ]


@pytest.mark.skip(reason="FIXME: stl_containers.root doesn't exist yet")
def test_vector_vector_int32():
    with uproot4.open("stl_containers.root")["tree"] as tree:
        assert [
            x.tolist() for x in tree["vector_vector_int32"].array(library="np")
        ] == [
            [[1]],
            [[1], [1, 2]],
            [[1], [1, 2], [1, 2, 3]],
            [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4]],
            [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]],
        ]


@pytest.mark.skip(reason="FIXME: stl_containers.root doesn't exist yet")
def test_vector_vector_string():
    with uproot4.open("stl_containers.root")["tree"] as tree:
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


@pytest.mark.skip(reason="FIXME: stl_containers.root doesn't exist yet")
def test_vector_set_int32():
    with uproot4.open("stl_containers.root")["tree"] as tree:
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


@pytest.mark.skip(reason="FIXME: stl_containers.root doesn't exist yet")
def test_vector_set_string():
    with uproot4.open("stl_containers.root")["tree"] as tree:
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


@pytest.mark.skip(reason="FIXME: stl_containers.root doesn't exist yet")
def test_set_int32():
    with uproot4.open("stl_containers.root")["tree"] as tree:
        assert [x.tolist() for x in tree["set_int32"].array(library="np")] == [
            set([1]),
            set([1, 2]),
            set([1, 2, 3]),
            set([1, 2, 3, 4]),
            set([1, 2, 3, 4, 5]),
        ]


# @pytest.mark.skip(reason="FIXME: stl_containers.root doesn't exist yet")
# def test_set_string():
#     with uproot4.open("stl_containers.root")["tree"] as tree:
#         assert [x.tolist() for x in tree["set_string"].array(library="np")] == []


# @pytest.mark.skip(reason="FIXME: stl_containers.root doesn't exist yet")
# def test_map_int32_int16():
#     with uproot4.open("stl_containers.root")["tree"] as tree:
#         assert [x.tolist() for x in tree["map_int32_int16"].array(library="np")] == []


# @pytest.mark.skip(reason="FIXME: stl_containers.root doesn't exist yet")
# def test_map_int32_vector_int16():
#     with uproot4.open("stl_containers.root")["tree"] as tree:
#         assert [x.tolist() for x in tree["map_int32_vector_int16"].array(library="np")] == []


# @pytest.mark.skip(reason="FIXME: stl_containers.root doesn't exist yet")
# def test_map_int32_vector_string():
#     with uproot4.open("stl_containers.root")["tree"] as tree:
#         assert [x.tolist() for x in tree["map_int32_vector_string"].array(library="np")] == []


# @pytest.mark.skip(reason="FIXME: stl_containers.root doesn't exist yet")
# def test_map_int32_set_int16():
#     with uproot4.open("stl_containers.root")["tree"] as tree:
#         assert [x.tolist() for x in tree["map_int32_set_int16"].array(library="np")] == []


# @pytest.mark.skip(reason="FIXME: stl_containers.root doesn't exist yet")
# def test_map_int32_set_string():
#     with uproot4.open("stl_containers.root")["tree"] as tree:
#         assert [x.tolist() for x in tree["map_int32_set_string"].array(library="np")] == []


# @pytest.mark.skip(reason="FIXME: stl_containers.root doesn't exist yet")
# def test_map_string_int16():
#     with uproot4.open("stl_containers.root")["tree"] as tree:
#         assert [x.tolist() for x in tree["map_string_int16"].array(library="np")] == []


# @pytest.mark.skip(reason="FIXME: stl_containers.root doesn't exist yet")
# def test_map_string_vector_int16():
#     with uproot4.open("stl_containers.root")["tree"] as tree:
#         assert [x.tolist() for x in tree["map_string_vector_int16"].array(library="np")] == []


# @pytest.mark.skip(reason="FIXME: stl_containers.root doesn't exist yet")
# def test_map_string_vector_string():
#     with uproot4.open("stl_containers.root")["tree"] as tree:
#         assert [x.tolist() for x in tree["map_string_vector_string"].array(library="np")] == []


# @pytest.mark.skip(reason="FIXME: stl_containers.root doesn't exist yet")
# def test_map_string_set_int16():
#     with uproot4.open("stl_containers.root")["tree"] as tree:
#         assert [x.tolist() for x in tree["map_string_set_int16"].array(library="np")] == []


# @pytest.mark.skip(reason="FIXME: stl_containers.root doesn't exist yet")
# def test_map_string_set_string():
#     with uproot4.open("stl_containers.root")["tree"] as tree:
#         assert [x.tolist() for x in tree["map_string_set_string"].array(library="np")] == []


# @pytest.mark.skip(reason="FIXME: stl_containers.root doesn't exist yet")
# def test_map_int32_vector_vector_int16():
#     with uproot4.open("stl_containers.root")["tree"] as tree:
#         assert [x.tolist() for x in tree["map_int32_vector_vector_int16"].array(library="np")] == []


# @pytest.mark.skip(reason="FIXME: stl_containers.root doesn't exist yet")
# def test_map_int32_vector_set_int16():
#     with uproot4.open("stl_containers.root")["tree"] as tree:
#         assert [x.tolist() for x in tree["map_int32_vector_set_int16"].array(library="np")] == []


# @pytest.mark.skip(reason="FIXME: stl_containers.root doesn't exist yet")
# def test_vector_map_int32_int16():
#     with uproot4.open("stl_containers.root")["tree"] as tree:
#         assert [x.tolist() for x in tree["vector_map_int32_int16"].array(library="np")] == []


# @pytest.mark.skip(reason="FIXME: stl_containers.root doesn't exist yet")
# def test_vector_map_int32_string():
#     with uproot4.open("stl_containers.root")["tree"] as tree:
#         assert [x.tolist() for x in tree["vector_map_int32_string"].array(library="np")] == []


# @pytest.mark.skip(reason="FIXME: stl_containers.root doesn't exist yet")
# def test_vector_map_string_string():
#     with uproot4.open("stl_containers.root")["tree"] as tree:
#         assert [x.tolist() for x in tree["vector_map_string_string"].array(library="np")] == []


# @pytest.mark.skip(reason="FIXME: stl_containers.root doesn't exist yet")
# def test_map_string_string():
#     with uproot4.open("stl_containers.root")["tree"] as tree:
#         assert [x.tolist() for x in tree["map_string_string"].array(library="np")] == []

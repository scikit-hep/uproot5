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
            AsVector(numpy.dtype(">i4"))
        )
        assert tree["vector_string"].interpretation == AsObjects(AsVector(AsString()))
        assert tree["vector_vector_int32"].interpretation == AsObjects(
            AsVector(AsVector(numpy.dtype(">i4")))
        )
        assert tree["vector_vector_string"].interpretation == AsObjects(
            AsVector(AsVector(AsString()))
        )
        assert tree["vector_set_int32"].interpretation == AsObjects(
            AsVector(AsSet(numpy.dtype(">i4")))
        )
        assert tree["vector_set_string"].interpretation == AsObjects(
            AsVector(AsSet(AsString()))
        )
        assert tree["set_int32"].interpretation == AsObjects(AsSet(numpy.dtype(">i4")))
        assert tree["set_string"].interpretation == AsObjects(AsSet(AsString()))
        assert tree["map_int32_int16"].interpretation == AsObjects(
            AsMap(numpy.dtype(">i4"), numpy.dtype(">i2"))
        )
        assert tree["map_int32_vector_int16"].interpretation == AsObjects(
            AsMap(numpy.dtype(">i4"), AsVector(numpy.dtype(">i2")))
        )
        assert tree["map_int32_vector_string"].interpretation == AsObjects(
            AsMap(numpy.dtype(">i4"), AsVector(AsString()))
        )
        assert tree["map_int32_set_int16"].interpretation == AsObjects(
            AsMap(numpy.dtype(">i4"), AsSet(numpy.dtype(">i2")))
        )
        assert tree["map_int32_set_string"].interpretation == AsObjects(
            AsMap(numpy.dtype(">i4"), AsSet(AsString()))
        )
        assert tree["map_string_int16"].interpretation == AsObjects(
            AsMap(AsString(), numpy.dtype(">i2"))
        )
        assert tree["map_string_vector_int16"].interpretation == AsObjects(
            AsMap(AsString(), AsVector(numpy.dtype(">i2")))
        )
        assert tree["map_string_vector_string"].interpretation == AsObjects(
            AsMap(AsString(), AsVector(AsString()))
        )
        assert tree["map_string_set_int16"].interpretation == AsObjects(
            AsMap(AsString(), AsSet(numpy.dtype(">i2")))
        )
        assert tree["map_string_set_string"].interpretation == AsObjects(
            AsMap(AsString(), AsSet(AsString()))
        )
        assert tree["map_int32_vector_vector_int16"].interpretation == AsObjects(
            AsMap(numpy.dtype(">i4"), AsVector(AsVector(numpy.dtype(">i2"))))
        )
        assert tree["map_int32_vector_set_int16"].interpretation == AsObjects(
            AsMap(numpy.dtype(">i4"), AsVector(AsSet(numpy.dtype(">i2"))))
        )
        assert tree["vector_map_int32_int16"].interpretation == AsObjects(
            AsVector(AsMap(numpy.dtype(">i4"), numpy.dtype(">i2")))
        )
        assert tree["vector_map_int32_string"].interpretation == AsObjects(
            AsVector(AsMap(numpy.dtype(">i4"), AsString()))
        )
        assert tree["vector_map_string_string"].interpretation == AsObjects(
            AsVector(AsMap(AsString(), AsString()))
        )
        assert tree["map_string_string"].interpretation == AsObjects(
            AsMap(AsString(), AsString())
        )


@pytest.mark.skip(reason="FIXME: stl_containers.root doesn't exist yet")
def test_vector_int32():
    with uproot4.open("stl_containers.root")["tree"] as tree:
        assert tree["vector_int32"].array().tolist() == [
            [1],
            [1, 2],
            [1, 2, 3],
            [1, 2, 3, 4],
            [1, 2, 3, 4, 5],
        ]


@pytest.mark.skip(reason="FIXME: stl_containers.root doesn't exist yet")
def test_vector_string():
    with uproot4.open("stl_containers.root")["tree"] as tree:
        assert tree["vector_string"].array().tolist() == [
            ["one"],
            ["one", "two"],
            ["one", "two", "three"],
            ["one", "two", "three", "four"],
            ["one", "two", "three", "four", "five"],
        ]


@pytest.mark.skip(reason="FIXME: stl_containers.root doesn't exist yet")
def test_vector_vector_int32():
    with uproot4.open("stl_containers.root")["tree"] as tree:
        assert tree["vector_vector_int32"].array().tolist() == [
            [[1]],
            [[1], [1, 2]],
            [[1], [1, 2], [1, 2, 3]],
            [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4]],
            [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]],
        ]


# def test_vector_vector_string():
#     with uproot4.open("stl_containers.root")["tree"] as tree:
#         assert tree["vector_vector_string"].array().tolist() == []


# def test_vector_set_int32():
#     with uproot4.open("stl_containers.root")["tree"] as tree:
#         assert tree["vector_set_int32"].array().tolist() == []


# def test_vector_set_string():
#     with uproot4.open("stl_containers.root")["tree"] as tree:
#         assert tree["vector_set_string"].array().tolist() == []


# def test_set_int32():
#     with uproot4.open("stl_containers.root")["tree"] as tree:
#         assert tree["set_int32"].array().tolist() == []


# def test_set_string():
#     with uproot4.open("stl_containers.root")["tree"] as tree:
#         assert tree["set_string"].array().tolist() == []


# def test_map_int32_int16():
#     with uproot4.open("stl_containers.root")["tree"] as tree:
#         assert tree["map_int32_int16"].array().tolist() == []


# def test_map_int32_vector_int16():
#     with uproot4.open("stl_containers.root")["tree"] as tree:
#         assert tree["map_int32_vector_int16"].array().tolist() == []


# def test_map_int32_vector_string():
#     with uproot4.open("stl_containers.root")["tree"] as tree:
#         assert tree["map_int32_vector_string"].array().tolist() == []


# def test_map_int32_set_int16():
#     with uproot4.open("stl_containers.root")["tree"] as tree:
#         assert tree["map_int32_set_int16"].array().tolist() == []


# def test_map_int32_set_string():
#     with uproot4.open("stl_containers.root")["tree"] as tree:
#         assert tree["map_int32_set_string"].array().tolist() == []


# def test_map_string_int16():
#     with uproot4.open("stl_containers.root")["tree"] as tree:
#         assert tree["map_string_int16"].array().tolist() == []


# def test_map_string_vector_int16():
#     with uproot4.open("stl_containers.root")["tree"] as tree:
#         assert tree["map_string_vector_int16"].array().tolist() == []


# def test_map_string_vector_string():
#     with uproot4.open("stl_containers.root")["tree"] as tree:
#         assert tree["map_string_vector_string"].array().tolist() == []


# def test_map_string_set_int16():
#     with uproot4.open("stl_containers.root")["tree"] as tree:
#         assert tree["map_string_set_int16"].array().tolist() == []


# def test_map_string_set_string():
#     with uproot4.open("stl_containers.root")["tree"] as tree:
#         assert tree["map_string_set_string"].array().tolist() == []


# def test_map_int32_vector_vector_int16():
#     with uproot4.open("stl_containers.root")["tree"] as tree:
#         assert tree["map_int32_vector_vector_int16"].array().tolist() == []


# def test_map_int32_vector_set_int16():
#     with uproot4.open("stl_containers.root")["tree"] as tree:
#         assert tree["map_int32_vector_set_int16"].array().tolist() == []


# def test_vector_map_int32_int16():
#     with uproot4.open("stl_containers.root")["tree"] as tree:
#         assert tree["vector_map_int32_int16"].array().tolist() == []


# def test_vector_map_int32_string():
#     with uproot4.open("stl_containers.root")["tree"] as tree:
#         assert tree["vector_map_int32_string"].array().tolist() == []


# def test_vector_map_string_string():
#     with uproot4.open("stl_containers.root")["tree"] as tree:
#         assert tree["vector_map_string_string"].array().tolist() == []


# def test_map_string_string():
#     with uproot4.open("stl_containers.root")["tree"] as tree:
#         assert tree["map_string_string"].array().tolist() == []

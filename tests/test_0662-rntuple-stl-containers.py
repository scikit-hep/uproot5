# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import json
import queue
import sys

import numpy
import pytest
import skhep_testdata

import uproot

ak = pytest.importorskip("awkward")


@pytest.mark.skip(
    reason="FIXME: skipping test_ntuple_stl_containers.root until #928 is fixed"
)
def test_rntuple_stl_containers():
    filename = skhep_testdata.data_path("test_ntuple_stl_containers.root")
    with uproot.open(filename) as f:
        R = f["ntuple"]
        assert R.keys() == [
            "string",
            "vector_int32",
            "array_float",
            "vector_vector_int32",
            "vector_string",
            "vector_vector_string",
            "variant_int32_string",
            "vector_variant_int64_string",
            "tuple_int32_string",
            "pair_int32_string",
            "vector_tuple_int32_string",
            "lorentz_vector",
            "array_lv",
        ]
        r = R.arrays()
        assert ak.all(r["string"] == ["one", "two", "three", "four", "five"])

        assert r["vector_int32"][0] == [1]
        assert ak.all(r["vector_int32"][1] == [1, 2])
        assert r["vector_vector_int32"][0] == [[1]]

        assert ak.all(r["vector_vector_int32"][1] == [[1], [2]])

        assert r["vector_string"][0] == ["one"]
        assert ak.all(r["vector_string"][1] == ["one", "two"])

        assert ak.all(r["vector_vector_string"][0] == [["one"]])
        assert ak.all(
            r["vector_vector_string"][-1]
            == [["one"], ["two"], ["three"], ["four"], ["five"]]
        )
        assert r["variant_int32_string"][0] == 1
        assert r["variant_int32_string"][1] == "two"

        assert r["vector_variant_int64_string"][0][0] == "one"
        assert r["vector_variant_int64_string"][1][0] == "one"
        assert r["vector_variant_int64_string"][1][1] == 2
        assert type(r["vector_variant_int64_string"][1][1]) == numpy.int64

        assert ak.all(r["tuple_int32_string"]._0 == [1, 2, 3, 4, 5])
        assert ak.all(
            r["tuple_int32_string"]._1 == ["one", "two", "three", "four", "five"]
        )
        assert list(r["tuple_int32_string"][0].to_list().values()) == [1, "one"]
        assert list(r["tuple_int32_string"][-1].to_list().values()) == [5, "five"]
        assert ak.all(r["pair_int32_string"]._0 == [1, 2, 3, 4, 5])
        assert ak.all(
            r["pair_int32_string"]._1 == ["one", "two", "three", "four", "five"]
        )

        assert r["vector_tuple_int32_string"][0]._0 == [1]
        assert r["vector_tuple_int32_string"][0]._1 == ["one"]
        assert ak.all(r["vector_tuple_int32_string"][1]._0 == [1, 2])
        assert ak.all(r["vector_tuple_int32_string"][1]._1 == ["one", "two"])

        assert ak.all(r["array_float"][0] == [1, 1, 1])
        assert ak.all(r["array_float"][-1] == [5, 5, 5])

        assert ak.all(r["array_lv"][0].pt == [1.0, 1.0, 1.0])
        assert ak.all(r["array_lv"][0].eta == [1.0, 1.0, 1.0])
        assert ak.all(r["array_lv"][0].phi == [1.0, 1.0, 1.0])
        assert ak.all(r["array_lv"][0].mass == [1.0, 1.0, 1.0])

        assert ak.all(r["array_lv"][-1].pt == [5.0, 5.0, 5.0])
        assert ak.all(r["array_lv"][-1].eta == [5.0, 5.0, 5.0])
        assert ak.all(r["array_lv"][-1].phi == [5.0, 5.0, 5.0])
        assert ak.all(r["array_lv"][-1].mass == [5.0, 5.0, 5.0])

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import json
import os
import queue
import sys

import numpy
import pytest
import skhep_testdata

import uproot

ak = pytest.importorskip("awkward")

data = ak.Array(
    {
        "bool": [True, False, True],
        "int": [1, 2, 3],
        "float": [1.1, 2.2, 3.3],
        "jagged_list": [[1], [2, 3], [4, 5, 6]],
        "nested_list": [[[1], []], [[2], [3, 3]], [[4, 5, 6]]],
        "string": ["one", "two", "three"],
        "regular": ak.Array(
            ak.contents.RegularArray(
                ak.contents.NumpyArray([1, 2, 3, 4, 5, 6, 7, 8, 9]), 3
            )
        ),
        "numpy_regular": numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        "struct": [{"x": 1, "y": 2}, {"x": 3, "y": 4}, {"x": 5, "y": 6}],
        "struct_list": [
            [{"x": 1}, {"x": 2}],
            [{"x": 3}, {"x": 4}],
            [{"x": 5}, {"x": 6}],
        ],
        "tuple": [(1, 2), (3, 4), (5, 6)],
        "tuple_list": [[(1,), (2,)], [(3,), (4,)], [(5,), (6,)]],
        "optional": [1, None, 2],
    }
)


def test_writing_and_reading(tmp_path):
    filepath = os.path.join(tmp_path, "test.root")

    with uproot.recreate(filepath) as file:
        obj = file.mkrntuple("ntuple", data.layout.form)
        obj.extend(data)
        obj.extend(data)  # test multiple cluster groups

    obj = uproot.open(filepath)["ntuple"]
    arrays = obj.arrays()

    for f in data.fields:
        if "tuple" in f:
            # TODO: tuples are converted to records
            [tuple(t[f] for f in t.fields) for t in arrays[f][:3]] == data[f].tolist()
            [tuple(t[f] for f in t.fields) for t in arrays[f][3:]] == data[f].tolist()
        elif "optional" in f:
            assert [t[0] if len(t) > 0 else None for t in arrays[f][:3]] == data[
                f
            ].tolist()
            assert [t[0] if len(t) > 0 else None for t in arrays[f][3:]] == data[
                f
            ].tolist()
        else:
            assert arrays[f][:3].tolist() == data[f].tolist()
            assert arrays[f][3:].tolist() == data[f].tolist()


def test_writing_then_reading_with_ROOT(tmp_path, capfd):
    ROOT = pytest.importorskip("ROOT")

    filepath = os.path.join(tmp_path, "test.root")

    with uproot.recreate(filepath) as file:
        obj = file.mkrntuple("ntuple", data.layout.form)
        obj.extend(data)
        obj.extend(data)  # test multiple cluster groups

    RT = ROOT.Experimental.RNTupleReader.Open("ntuple", filepath)
    RT.PrintInfo()
    RT.Show(0)
    RT.Show(2)
    RT.Show(4)
    out = capfd.readouterr().out
    assert "* N-Tuple : ntuple" in out
    assert "* Entries : 6" in out
    assert "* Field 1            : bool (bool)" in out
    assert "* Field 2            : int (std::int64_t)" in out
    assert "* Field 3            : float (double)" in out
    assert "* Field 4            : jagged_list (std::vector<std::int64_t>)" in out
    assert (
        "* Field 5            : nested_list (std::vector<std::vector<std::int64_t>>)"
        in out
    )
    assert "* Field 6            : string (std::string)" in out
    assert "* Field 7            : regular (std::array<std::int64_t,3>)" in out
    assert "* Field 8            : numpy_regular (std::array<std::int64_t,3>)" in out
    assert "* Field 9            : struct" in out
    assert "* Field 10           : struct_list" in out
    assert "* Field 11           : tuple" in out
    assert (
        "* Field 12           : tuple_list (std::vector<std::tuple<std::int64_t>>)"
        in out
    )

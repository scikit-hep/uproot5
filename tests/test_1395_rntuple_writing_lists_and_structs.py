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
            continue
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
    out = capfd.readouterr().out
    assert "* N-Tuple : ntuple" in out
    assert "* Entries : 3" in out
    assert "* Field 1   : one (std::int64_t)" in out
    assert "* Field 2   : two (double)" in out
    assert '  "one": 1,' in out
    assert '  "two": 1.1' in out
    assert '  "one": 3' in out
    assert '  "two": 3.3' in out

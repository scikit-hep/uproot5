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
        "utf8_string": ["こんにちは", "⚛️💫🎆😀", "ǧ̸̛̫͍̰͖̟̈͛͑͆̆̌̃̉̅̄̔̈́̀̔͆̄͋̍͐͂̎͗̈́͒͘͝ͅö̴̮̝̪̬͎͚̜̖̜͖̞̤͕̙͂̀̀̊͛͑̈́͛͐͊͂͂̇͛̾̔͐͆͑͂̓̅̀͘͘͘̕͝͠͝͝ơ̶͍̙̻̾̈́̓̈́̀̅͑ḑ̷͚̠̹̗͉͙̞͇͕̼̲̥͉̯̞͕̲̻̞͗̓̃̊̅͗͊͊́̑̈́̎͋̇̓͛̅͜͜͠͝ͅb̷̢̢̨̨̛̛̘̠̞̰̺̘̰̖̺̞̱͇̰̙̲̱̪͕͎͉̖̞͇̹̮͙͋̀͑͂̈́̇͛̐͊̀̇͆̓̋̀̿̋̂̅̀̌̑̓̽͊̂͑̈̇̚͜͝y̶̗͇̠̞͚̦̮̦͈̹̥̋̓̓̈́̐̆̀̄̋̂̀̇͋̎̚͜͝ȩ̷̢̡͇̮̩̹̥̬̰͎͔̬̩̰̯͍̲͎̭͉̬̣̻̖͍̥̟̪͕̫̟̋̔̀͆̑̈́̐̃͐͌̍͒̔̈́̃̈́̐̔̾͊̿̓͆͑̚͜͝͝͝ͅ"],
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
        "union": [1, 2, "three"],
        "optional_union": [1, None, "three"],
        "list_array": ak.contents.ListArray(
            ak.index.Index([1, 0, 3]),
            ak.index.Index([1, 2, 4]),
            ak.contents.NumpyArray([0, 1, 2, 3, 4, 5]),
        ),
        "indexed_option_array": ak.contents.IndexedOptionArray(
            ak.index.Index([3, -1, 1]),
            ak.contents.NumpyArray([0, 1, 2, 3, 4, 5]),
        ),
        "indexed_option_array32": ak.contents.IndexedOptionArray(
            ak.index.Index32([3, -1, 1]),
            ak.contents.NumpyArray([0, 1, 2, 3, 4, 5]),
        ),
        "indexed_array": ak.contents.IndexedArray(
            ak.index.Index([3, 0, 1]),
            ak.contents.NumpyArray([0, 1, 2, 3, 4, 5]),
        ),
        "numpy_regular3d": numpy.array(
            [[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]]
        ),
    }
)

data.layout.parameters["__doc__"] = "This is the top record array"
data.layout.contents[0].parameters["__doc__"] = "This is a boolean"
data.layout.contents[10].content.parameters["__doc__"] = "This is an struct record"
data.layout.contents[10].content.contents[0].parameters[
    "__doc__"
] = "This is a subfield"


def test_writing_and_reading(tmp_path):
    filepath = os.path.join(tmp_path, "test.root")

    with uproot.recreate(filepath) as file:
        obj = file.mkrntuple("ntuple", data)  # test inputting the data directly
        obj.extend(data)

    with uproot.recreate(filepath) as file:
        obj = file.mkrntuple("ntuple", data.layout.form)
        obj.extend(data)
        obj.extend(data)  # test multiple cluster groups

    obj = uproot.open(filepath)["ntuple"]
    arrays = obj.arrays()

    for f in data.fields:
        assert arrays[f][:3].tolist() == data[f].tolist()
        assert arrays[f][3:].tolist() == data[f].tolist()


def test_writing_then_reading_with_ROOT(tmp_path, capfd):
    ROOT = pytest.importorskip("ROOT")
    if ROOT.gROOT.GetVersionInt() < 63400:
        pytest.skip("ROOT version does not support RNTuple v1.0.0.0")

    filepath = os.path.join(tmp_path, "test.root")

    with uproot.recreate(filepath) as file:
        obj = file.mkrntuple("ntuple", data.layout.form)
        obj.extend(data)
        obj.extend(data)  # test multiple cluster groups

    if ROOT.gROOT.GetVersionInt() < 63600:
        RT = ROOT.Experimental.RNTupleReader.Open("ntuple", filepath)
    else:
        RT = ROOT.RNTupleReader.Open("ntuple", filepath)
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
    assert "* Field 7            : utf8_string (std::string)" in out
    assert "* Field 8            : regular (std::array<std::int64_t,3>)" in out
    assert "* Field 9            : numpy_regular (std::array<std::int64_t,3>)" in out
    assert "* Field 10           : struct" in out
    assert "* Field 11           : struct_list" in out
    assert "* Field 12           : tuple (std::tuple<std::int64_t,std::int64_t>)" in out
    assert (
        "* Field 13           : tuple_list (std::vector<std::tuple<std::int64_t>>)"
        in out
    )
    assert "* Field 14           : optional (std::optional<std::int64_t>)" in out
    assert (
        "* Field 15           : union (std::variant<std::int64_t,std::string>)" in out
    )
    assert (
        "* Field 16           : optional_union (std::variant<std::optional<std::int64..."
        in out
    )
    assert "* Field 17           : list_array (std::vector<std::int64_t>)" in out
    assert (
        "* Field 18           : indexed_option_array (std::optional<std::int64_t>)"
        in out
    )
    assert (
        "* Field 19           : indexed_option_array32 (std::optional<std::int64_t>)"
        in out
    )
    assert "* Field 20           : indexed_array (std::int64_t)" in out


def test_field_descriptions(tmp_path):
    filepath = os.path.join(tmp_path, "test.root")

    with uproot.recreate(filepath) as file:
        obj = file.mkrntuple("ntuple", data)  # test inputting the data directly
        obj.extend(data)

    with uproot.recreate(filepath) as file:
        obj = file.mkrntuple("ntuple", data.layout.form)
        obj.extend(data)
        obj.extend(data)  # test multiple cluster groups

    obj = uproot.open(filepath)["ntuple"]
    arrays = obj.arrays(ak_add_doc=True)

    assert arrays.layout.parameters["__doc__"] == "This is the top record array"
    assert arrays.layout.contents[0].parameters["__doc__"] == "This is a boolean"
    assert (
        arrays.layout.contents[10].content.parameters["__doc__"]
        == "This is an struct record"
    )
    assert (
        arrays.layout.contents[10].content.contents[0].parameters["__doc__"]
        == "This is a subfield"
    )

    arrays = obj.arrays(ak_add_doc={"typename": "typename"})

    assert arrays.layout.contents[0].parameters["typename"] == "bool"


def test_writing_dict(tmp_path):
    filepath = os.path.join(tmp_path, "test.root")

    data = {
        "bool": [True, False],
        "int": [1, 2],
        "float": [1.0, 2.0],
        "string": ["hello", "world"],
        "list": [[1, 2], [3, 4]],
        "struct": [{"a": 1, "b": 2}, {"a": 3, "b": 4}],
        "optional": [None, 1],
        "union": [1, "hello"],
        "optional_union": [None, 1],
        "list_array": [[1, 2], [3, 4]],
    }

    with uproot.recreate(filepath) as file:
        obj = file.mkrntuple("ntuple", data)
        obj.extend(data)

    obj = uproot.open(filepath)["ntuple"]
    arrays = obj.arrays()

    assert len(arrays) == 2 * len(data["bool"])
    assert arrays["bool"].tolist() == data["bool"] + data["bool"]

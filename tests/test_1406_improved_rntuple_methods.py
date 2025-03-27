# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import os

import numpy
import pytest
import skhep_testdata

import uproot

ak = pytest.importorskip("awkward")

data = ak.Array(
    {
        "struct1": [{"x": 1, "y": 2}, {"x": 3, "y": 4}],
        "struct2": [{"x": 5, "y": 6, "z": 7}, {"x": 8, "y": 9, "z": 10}],
        "struct3": [
            {"x": 11, "y": 12, "z": 13, "t": 14.0},
            {"x": 15, "y": 16, "z": 17, "t": 18.0},
        ],
        "struct4": [
            {
                "x": [{"up": 1, "down": 2}, {"up": 3, "down": 4}],
                "y": [({"left": 5, "right": 6}, {"left": 7, "right": 8.0})],
            },
            {
                "x": [{"up": 9, "down": 10}, {"up": 11, "down": 12}],
                "y": [({"left": 13, "right": 14}, {"left": 15, "right": 16.0})],
            },
        ],
        "struct5": [(1, 2, 3), (4, 5, 6)],
    }
)


def test_keys(tmp_path):
    filepath = os.path.join(tmp_path, "test.root")

    with uproot.recreate(filepath) as file:
        obj = file.mkrntuple("ntuple", data)

    obj = uproot.open(filepath)["ntuple"]

    assert len(obj) == 5
    assert len(obj.keys(recursive=False)) == 5

    assert len(obj.keys()) == 29
    assert len(obj.keys(full_paths=False)) == 29
    assert len(obj.keys(full_paths=False, ignore_duplicates=True)) == 16

    assert len(obj.keys(filter_name="x")) == 4
    assert len(obj.keys(filter_name="z")) == 2
    assert len(obj.keys(filter_name="do*")) == 1

    assert len(obj.keys(filter_typename="std::int*_t")) == 16

    assert len(obj.keys(filter_field=lambda f: f.name == "up")) == 1

    assert obj["struct1"].keys() == ["x", "y"]
    assert len(obj["struct4"].keys()) == 12


def test_getitem(tmp_path):
    filepath = os.path.join(tmp_path, "test.root")

    with uproot.recreate(filepath) as file:
        obj = file.mkrntuple("ntuple", data)

    obj = uproot.open(filepath)["ntuple"]

    assert obj["struct1"] is obj.fields[0]
    assert obj["struct2"] is obj.fields[1]
    assert obj["struct3"] is obj.fields[2]
    assert obj["struct4"] is obj.fields[3]
    assert obj["struct5"] is obj.fields[4]

    assert obj["struct1"]["x"] is obj.fields[0].fields[0]
    assert obj["struct1"]["x"] is obj["struct1.x"]
    assert obj["struct1"]["x"] is obj["struct1/x"]
    assert obj["struct1"]["x"] is obj[r"struct1\x"]


def test_to_akform(tmp_path):
    filepath = os.path.join(tmp_path, "test.root")

    with uproot.recreate(filepath) as file:
        obj = file.mkrntuple("ntuple", data)

    obj = uproot.open(filepath)["ntuple"]

    akform = obj.to_akform()
    assert akform == data.layout.form

    assert obj["struct1"].to_akform() == akform.select_columns("struct1")
    assert obj["struct2"].to_akform() == akform.select_columns("struct2")
    assert obj["struct3"].to_akform() == akform.select_columns("struct3")
    assert obj["struct4"].to_akform() == akform.select_columns("struct4")
    assert obj["struct5"].to_akform() == akform.select_columns("struct5")

    assert obj["struct1"].to_akform(filter_name="x") == akform.select_columns(
        ["struct1.x"]
    )
    assert obj["struct3"].to_akform(filter_typename="double") == akform.select_columns(
        ["struct3.t"]
    )


def test_iterate_and_concatenate(tmp_path):
    filepath1 = os.path.join(tmp_path, "test1.root")
    filepath2 = os.path.join(tmp_path, "test2.root")

    with uproot.recreate(filepath1) as file:
        file.mkrntuple("ntuple", data)

    with uproot.recreate(filepath2) as file:
        file.mkrntuple("ntuple", data)

    total_iterations = 0
    for i, array in enumerate(
        uproot.behaviors.RNTuple.iterate([f"{tmp_path}/test*.root:ntuple"], step_size=2)
    ):
        total_iterations += 1
        assert ak.array_equal(array, data)

    assert total_iterations == 2

    array = uproot.behaviors.RNTuple.concatenate([f"{tmp_path}/test*.root:ntuple"])
    true_array = ak.concatenate([data, data], axis=0)

    assert ak.array_equal(array, true_array)

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
import pytest

import uproot


@pytest.fixture
def tuple_path(tmp_path):
    path = tmp_path / "tuple.root"
    with uproot.recreate(path) as file:
        file["tuple"] = ak.Array([(1.0, 2), (3.0, 4)])
    return path


def test_top_level_tuple_round_trip(tuple_path):
    with uproot.open(tuple_path) as file:
        ntuple = file["tuple"]
        form, _ = ntuple.to_akform()

        assert form.is_tuple
        # written like nested tuples, and renamed to match Awkward when read
        assert ntuple.field_names == ["_0", "_1"]
        assert ntuple.keys() == ["0", "1"]
        assert ntuple.arrays().tolist() == [(1.0, 2), (3.0, 4)]


def test_top_level_tuple_fields(tuple_path):
    with uproot.open(tuple_path) as file:
        ntuple = file["tuple"]

        assert ntuple["0"].array().tolist() == [1.0, 3.0]
        assert ntuple["1"].array().tolist() == [2, 4]

        # a subset of the fields is a record, so that its fields match the RFields
        for name, expected in [("0", [1.0, 3.0]), ("1", [2, 4])]:
            arrays = ntuple.arrays(filter_name=name)
            assert arrays.fields == [name]
            assert arrays[name].tolist() == expected


def test_top_level_tuple_subfields(tmp_path):
    path = tmp_path / "tuple.root"
    with uproot.recreate(path) as file:
        file["tuple"] = ak.Array([([{"x": 1}, {"x": 2}], 3), ([], 4)])

    with uproot.open(path) as file:
        ntuple = file["tuple"]

        assert ntuple.keys() == ["0", "0.x", "1"]
        assert ntuple["0"]["x"].array().tolist() == [[1, 2], []]
        assert ntuple.arrays().tolist() == [([{"x": 1}, {"x": 2}], 3), ([], 4)]


def test_top_level_tuple_dask(tuple_path):
    pytest.importorskip("dask_awkward")

    array = uproot.dask(f"{tuple_path}:tuple")
    assert array.compute().tolist() == [(1.0, 2), (3.0, 4)]
    assert array["1"].compute().tolist() == [2, 4]

    array = uproot.dask(f"{tuple_path}:tuple", filter_name="1")
    assert array.compute().tolist() == [{"1": 2}, {"1": 4}]


def test_top_level_tuple_extend(tuple_path):
    with uproot.update(tuple_path) as file:
        file["tuple"].extend({"0": [5.0], "1": [6]})
        file["tuple"].extend(ak.Array([(7.0, 8)]))

    with uproot.open(tuple_path) as file:
        assert file["tuple"].arrays().tolist() == [
            (1.0, 2),
            (3.0, 4),
            (5.0, 6),
            (7.0, 8),
        ]

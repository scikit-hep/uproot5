# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import os
import pytest

import uproot

import awkward as ak
import numpy as np


def test_mktree_tuple_dtype(tmp_path):
    filepath = os.path.join(tmp_path, "test.root")

    with uproot.recreate(filepath) as file:
        dt1 = ("int32", (2,))
        t1 = file.mktree("tree1", {"x": dt1})
        t1.extend({"x": np.zeros(5, dtype=dt1)})
        dt2 = ("int32", (2, 4, 1))
        t2 = file.mktree("tree2", {"y": dt2})
        t2.extend({"y": np.zeros(5, dtype=dt2)})
        dt3 = ("int32", ("int8", 4))
        t3 = file.mktree("tree3", {"z": dt3})
        t3.extend({"z": np.zeros(5, dtype=dt3)})
        dt4 = "2i4"
        t4 = file.mktree("tree4", {"w": dt4})
        t4.extend({"w": np.zeros(5, dtype=dt4)})
        dt5 = "(2,3,4)f4"
        t5 = file.mktree("tree5", {"v": dt5})
        t5.extend({"v": np.zeros(5, dtype=dt5)})

    with uproot.open(filepath) as file:
        assert file.keys(cycle=False) == ["tree1", "tree2", "tree3", "tree4", "tree5"]
        assert file["tree1"].num_entries == 5
        assert file["tree2"].num_entries == 5
        assert file["tree3"].num_entries == 5
        assert file["tree4"].num_entries == 5
        assert file["tree5"].num_entries == 5


def test_mkrntuple_tuple_dtype(tmp_path):
    filepath = os.path.join(tmp_path, "test.root")

    with uproot.recreate(filepath) as file:
        dt1 = ("int32", (2,))
        t1 = file.mkrntuple("ntuple1", {"x": dt1})
        t1.extend({"x": np.zeros(5, dtype=dt1)})
        dt2 = ("int32", (2, 4, 1))
        t2 = file.mkrntuple("ntuple2", {"y": dt2})
        t2.extend({"y": np.zeros(5, dtype=dt2)})
        dt3 = ("int32", ("int8", 4))
        t3 = file.mkrntuple("ntuple3", {"z": dt3})
        t3.extend({"z": np.zeros(5, dtype=dt3)})
        dt4 = "2i4"
        t4 = file.mkrntuple("ntuple4", {"w": dt4})
        t4.extend({"w": np.zeros(5, dtype=dt4)})
        dt5 = "(2,3,4)f4"
        t5 = file.mkrntuple("ntuple5", {"v": dt5})
        t5.extend({"v": np.zeros(5, dtype=dt5)})

    with uproot.open(filepath) as file:
        assert file.keys(cycle=False) == [
            "ntuple1",
            "ntuple2",
            "ntuple3",
            "ntuple4",
            "ntuple5",
        ]
        assert file["ntuple1"].num_entries == 5
        assert file["ntuple2"].num_entries == 5
        assert file["ntuple3"].num_entries == 5
        assert file["ntuple4"].num_entries == 5
        assert file["ntuple5"].num_entries == 5


def test_unsupported_numpy_dtypes(tmp_path):
    filepath = os.path.join(tmp_path, "test.root")

    with uproot.recreate(filepath) as file:
        # this should work, but it is interpreted as data, not a dtype
        file.mktree("tree", {"x": ("int32", "not a type")})
        file.mkrntuple("ntuple", {"x": ("int32", "not a type")})

        # structured dtypes are not supported
        complex_dtype = np.dtype([("real", "float32"), ("imag", "float32")])
        with pytest.raises(TypeError):
            file.mktree("tree1", {"x": complex_dtype})
        with pytest.raises(TypeError):
            file.mkrntuple("ntuple1", {"x": complex_dtype})

        # structured dtypes specified as strings are also not supported
        dt_str = "i4, (2,3)f8, f4"
        with pytest.raises(TypeError):
            file.mktree("tree2", {"x": dt_str})
        with pytest.raises(TypeError):
            file.mkrntuple("ntuple2", {"x": dt_str})

        # using tuples to specify structured dtypes is not supported
        dt_tuple = ("int32", [("real", "float32"), ("imag", "float32")])
        with pytest.raises(TypeError):
            file.mktree("tree3", {"x": dt_tuple})
        # this should work, but it is interpreted as data, not a dtype
        file.mkrntuple("ntuple3", {"x": dt_tuple})

        # using some random type should not work
        dt_unknown = frozenset
        with pytest.raises(TypeError):
            file.mktree("tree4", {"x": dt_tuple})
        with pytest.raises(TypeError):
            file.mkrntuple("ntuple4", {"x": dt_unknown})

    # make sure that things were written as data instead of being interpreted as dtypes
    with uproot.open(filepath) as file:
        assert file.keys(cycle=False) == ["tree", "ntuple", "ntuple3"]
        assert file["tree"].arrays().x[0] == "int32"
        assert file["tree"].arrays().x[1] == "not a type"
        assert file["ntuple"].arrays().x[0] == "int32"
        assert file["ntuple"].arrays().x[1] == "not a type"
        assert file["ntuple3"].arrays().x[0] == "int32"
        assert ak.array_equal(
            file["ntuple3"].arrays().x[1], [("real", "float32"), ("imag", "float32")]
        )

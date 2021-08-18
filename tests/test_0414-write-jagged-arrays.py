# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

import array
import os

import numpy as np
import pytest
import skhep_testdata

import uproot
import uproot.writing

ROOT = pytest.importorskip("ROOT")
awkward = pytest.importorskip("awkward")


def test_awkward_as_numpy(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    with uproot.recreate(newfile, compression=None) as fout:
        fout.mktree(
            "tree",
            {
                "b1": awkward.types.from_datashape("int32"),
                "b2": awkward.types.from_datashape("2 * float64"),
                "b3": awkward.types.from_datashape("2 * 3 * float64"),
                "b4": awkward.Array([1.1, 2.2, 3.3]).type,
            },
        )

    with uproot.open(newfile) as fin:
        assert fin["tree/b1"].typename == "int32_t"
        assert fin["tree/b2"].typename == "double[2]"
        assert fin["tree/b3"].typename == "double[2][3]"
        assert fin["tree/b4"].typename == "double"


def test_awkward_record(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    with uproot.recreate(newfile, compression=None) as fout:
        fout.mktree(
            "tree",
            {
                "b1": "int32",
                "b2": awkward.types.from_datashape('{"x": float64, "y": 3 * float64}'),
            },
        )

    with uproot.open(newfile) as fin:
        assert fin["tree/b1"].typename == "int32_t"
        assert fin["tree/b2_x"].typename == "double"
        assert fin["tree/b2_y"].typename == "double[3]"


def test_awkward_record_data(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    with uproot.recreate(newfile, compression=None) as fout:
        b1 = np.array([1, 2, 3], np.int32)
        b2 = awkward.Array([{"x": 1.1, "y": 4}, {"x": 2.2, "y": 5}, {"x": 3.3, "y": 6}])
        fout.mktree("tree", {"b1": b1.dtype, "b2": b2.type})
        fout["tree"].extend({"b1": b1, "b2": b2})

    with uproot.open(newfile) as fin:
        assert fin["tree/b1"].typename == "int32_t"
        assert fin["tree/b2_x"].typename == "double"
        assert fin["tree/b2_y"].typename == "int64_t"
        assert fin["tree/b1"].array().tolist() == [1, 2, 3]
        assert fin["tree/b2_x"].array().tolist() == [1.1, 2.2, 3.3]
        assert fin["tree/b2_y"].array().tolist() == [4, 5, 6]

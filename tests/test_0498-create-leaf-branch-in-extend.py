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

    with uproot.recreate(newfile) as fout:
        fout.mktree("tree", {"b1": "var * float64", "b2": "var * float64"})
        fout["tree"].extend(
            {
                "b1": [[1.1, 2.2, 3.3], [], [4.4, 5.5]],
                "b2": np.array([[1.1, 2.2, 3.3], [], [4.4, 5.5]], dtype="O"),
            }
        )

    with uproot.open(newfile) as fin:
        assert fin["tree/b1"].array().tolist() == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]
        assert fin["tree/b2"].array().tolist() == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]


def test_counter_shadows_branch_1(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    with uproot.recreate(newfile) as fout:
        fout.mktree("tree", {"nb": "int32", "b": "var * float64"})
        with pytest.raises(ValueError):
            fout["tree"].extend(
                {"nb": [1, 2, 3], "b": [[1.1, 2.2, 3.3], [], [4.4, 5.5]]}
            )
        fout["tree"].extend({"nb": [3, 0, 2], "b": [[1.1, 2.2, 3.3], [], [4.4, 5.5]]})

    with uproot.open(newfile) as fin:
        assert fin["tree/nb"].array().tolist() == [3, 0, 2]
        assert fin["tree/b"].array().tolist() == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]


def test_counter_shadows_branch_2(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    with uproot.recreate(newfile) as fout:
        fout.mktree("tree", {"nb": "int32", "b": "var * float64"})
        with pytest.raises(ValueError):
            fout["tree"].extend(
                {"b": [[1.1, 2.2, 3.3], [], [4.4, 5.5]], "nb": [1, 2, 3]}
            )
        fout["tree"].extend({"b": [[1.1, 2.2, 3.3], [], [4.4, 5.5]], "nb": [3, 0, 2]})

    with uproot.open(newfile) as fin:
        assert fin["tree/nb"].array().tolist() == [3, 0, 2]
        assert fin["tree/b"].array().tolist() == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]


def test_counter_shadows_branch_3(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    with uproot.recreate(newfile) as fout:
        fout.mktree("tree", {"b": "var * float64", "nb": "int32"})
        with pytest.raises(ValueError):
            fout["tree"].extend(
                {"nb": [1, 2, 3], "b": [[1.1, 2.2, 3.3], [], [4.4, 5.5]]}
            )
        fout["tree"].extend({"nb": [3, 0, 2], "b": [[1.1, 2.2, 3.3], [], [4.4, 5.5]]})

    with uproot.open(newfile) as fin:
        assert fin["tree/nb"].array().tolist() == [3, 0, 2]
        assert fin["tree/b"].array().tolist() == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]


def test_counter_shadows_branch_4(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    with uproot.recreate(newfile) as fout:
        fout.mktree("tree", {"b": "var * float64", "nb": "int32"})
        with pytest.raises(ValueError):
            fout["tree"].extend(
                {"b": [[1.1, 2.2, 3.3], [], [4.4, 5.5]], "nb": [1, 2, 3]}
            )
        fout["tree"].extend({"b": [[1.1, 2.2, 3.3], [], [4.4, 5.5]], "nb": [3, 0, 2]})

    with uproot.open(newfile) as fin:
        assert fin["tree/nb"].array().tolist() == [3, 0, 2]
        assert fin["tree/b"].array().tolist() == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]


def test_counter_shadows_branch_5(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    with uproot.recreate(newfile) as fout:
        with pytest.raises(ValueError):
            fout["tree"] = {"nb": [1, 2, 3], "b": [[1.1, 2.2, 3.3], [], [4.4, 5.5]]}
        fout["tree"] = {"nb": [3, 0, 2], "b": [[1.1, 2.2, 3.3], [], [4.4, 5.5]]}

    with uproot.open(newfile) as fin:
        assert fin["tree/nb"].array().tolist() == [3, 0, 2]
        assert fin["tree/b"].array().tolist() == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]


def test_counter_shadows_branch_6(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    with uproot.recreate(newfile) as fout:
        with pytest.raises(ValueError):
            fout["tree"] = {"b": [[1.1, 2.2, 3.3], [], [4.4, 5.5]], "nb": [1, 2, 3]}
        fout["tree"] = {"b": [[1.1, 2.2, 3.3], [], [4.4, 5.5]], "nb": [3, 0, 2]}

    with uproot.open(newfile) as fin:
        assert fin["tree/nb"].array().tolist() == [3, 0, 2]
        assert fin["tree/b"].array().tolist() == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

import array
import os

import numpy as np
import pytest

import uproot
import uproot.writing

ROOT = pytest.importorskip("ROOT")


def test_2dim(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    with uproot.recreate(newfile, compression=None) as fout:
        tree = fout.mktree("tree", "title", {"branch": np.dtype((np.float64, (3,)))})
        tree.extend({"branch": np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])})
        with pytest.raises(ValueError):
            tree.extend({"branch": np.array([7.7, 8.8, 9.9])})
        with pytest.raises(ValueError):
            tree.extend({"branch": np.array([[7.7], [8.8], [9.9]])})

    f1 = ROOT.TFile(newfile)
    t1 = f1.Get("tree")

    assert [np.asarray(x.branch).tolist() for x in t1] == [
        [1.1, 2.2, 3.3],
        [4.4, 5.5, 6.6],
    ]

    with uproot.open(newfile) as fin:
        assert fin["tree/branch"].array().tolist() == [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]


def test_2dim_interface(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    with uproot.recreate(newfile, compression=None) as fout:
        fout["tree"] = {"branch": np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])}
        fout["tree"].extend(
            {"branch": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])}
        )
        with pytest.raises(ValueError):
            fout["tree"].extend({"branch": np.array([7.7, 8.8, 9.9])})
        with pytest.raises(ValueError):
            fout["tree"].extend({"branch": np.array([[7.7], [8.8], [9.9]])})

    f1 = ROOT.TFile(newfile)
    t1 = f1.Get("tree")

    assert [np.asarray(x.branch).tolist() for x in t1] == [
        [1.1, 2.2, 3.3],
        [4.4, 5.5, 6.6],
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ]

    with uproot.open(newfile) as fin:
        assert fin["tree/branch"].array().tolist() == [
            [1.1, 2.2, 3.3],
            [4.4, 5.5, 6.6],
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]


def test_2dim_interface_2(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    with uproot.recreate(newfile, compression=None) as fout:
        fout["tree"] = {"branch": [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]}
        fout["tree"].extend(
            {"branch": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]}
        )
        with pytest.raises(ValueError):
            fout["tree"].extend({"branch": [7.7, 8.8, 9.9]})
        with pytest.raises(ValueError):
            fout["tree"].extend({"branch": [[7.7], [8.8], [9.9]]})

    f1 = ROOT.TFile(newfile)
    t1 = f1.Get("tree")

    assert [np.asarray(x.branch).tolist() for x in t1] == [
        [1.1, 2.2, 3.3],
        [4.4, 5.5, 6.6],
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ]

    with uproot.open(newfile) as fin:
        assert fin["tree/branch"].array().tolist() == [
            [1.1, 2.2, 3.3],
            [4.4, 5.5, 6.6],
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]


def test_structured_array(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    with uproot.recreate(newfile, compression=None) as fout:
        fout["tree"] = np.array(
            [(1, 1.1), (2, 2.2), (3, 3.3)], [("x", np.int32), ("y", np.float64)]
        )
        fout["tree"].extend(
            np.array(
                [(4, 4.4), (5, 5.5), (6, 6.6)], [("x", np.int32), ("y", np.float64)]
            )
        )

    f1 = ROOT.TFile(newfile)
    t1 = f1.Get("tree")

    assert t1.GetBranch("x").GetName() == "x"
    assert t1.GetBranch("y").GetName() == "y"
    assert [np.asarray(x.x).tolist() for x in t1] == [1, 2, 3, 4, 5, 6]
    assert [np.asarray(x.y).tolist() for x in t1] == [1.1, 2.2, 3.3, 4.4, 5.5, 6.6]

    with uproot.open(newfile) as fin:
        assert fin["tree/x"].name == "x"
        assert fin["tree/y"].name == "y"
        assert fin["tree/x"].typename == "int32_t"
        assert fin["tree/y"].typename == "double"
        assert fin["tree/x"].array().tolist() == [1, 2, 3, 4, 5, 6]
        assert fin["tree/y"].array().tolist() == [1.1, 2.2, 3.3, 4.4, 5.5, 6.6]


def test_pandas(tmp_path):
    pandas = pytest.importorskip("pandas")

    newfile = os.path.join(tmp_path, "newfile.root")

    df1 = pandas.DataFrame({"x": [1, 2, 3], "y": [1.1, 2.2, 3.3]})
    df2 = pandas.DataFrame({"x": [4, 5, 6], "y": [4.4, 5.5, 6.6]})

    with uproot.recreate(newfile, compression=None) as fout:
        fout["tree"] = df1
        fout["tree"].extend(df2)

    f1 = ROOT.TFile(newfile)
    t1 = f1.Get("tree")

    assert t1.GetBranch("index").GetName() == "index"
    assert t1.GetBranch("x").GetName() == "x"
    assert t1.GetBranch("y").GetName() == "y"
    assert [np.asarray(x.x).tolist() for x in t1] == [1, 2, 3, 4, 5, 6]
    assert [np.asarray(x.y).tolist() for x in t1] == [1.1, 2.2, 3.3, 4.4, 5.5, 6.6]

    with uproot.open(newfile) as fin:
        assert fin["tree/index"].name == "index"
        assert fin["tree/x"].name == "x"
        assert fin["tree/y"].name == "y"
        assert fin["tree/index"].typename.startswith("int")
        assert fin["tree/x"].typename.startswith("int")
        assert fin["tree/y"].typename == "double"
        assert fin["tree/x"].array().tolist() == [1, 2, 3, 4, 5, 6]
        assert fin["tree/y"].array().tolist() == [1.1, 2.2, 3.3, 4.4, 5.5, 6.6]

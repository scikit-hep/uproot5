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

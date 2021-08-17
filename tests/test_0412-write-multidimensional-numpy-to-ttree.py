# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

import array
import os

import numpy as np
import pytest

import uproot
import uproot.writing

ROOT = pytest.importorskip("ROOT")


def test_2dim(tmp_path):
    newfile = "newfile.root"  # os.path.join(tmp_path, "newfile.root")

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

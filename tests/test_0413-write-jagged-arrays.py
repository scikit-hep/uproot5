# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

import array
import os

import numpy as np
import pytest
import skhep_testdata

import uproot
import uproot.writing

ROOT = pytest.importorskip("ROOT")
ak = pytest.importorskip("awkward")


def test(tmp_path):
    newfile = "newfile.root"   # os.path.join(tmp_path, "newfile.root")

    with uproot.recreate(newfile, compression=None) as fout:
        fout.mktree("tree", {"b1": "int32", "b2": "float64"})

    with uproot.open(newfile) as fin:
        fin["tree"].show()

    # raise Exception

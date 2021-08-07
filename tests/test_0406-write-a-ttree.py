# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

import array
import os

import numpy as np
import pytest

import uproot
import uproot.writing

ROOT = pytest.importorskip("ROOT")


def test(tmp_path):
    original = os.path.join(tmp_path, "original.root")
    newfile = os.path.join(tmp_path, "newfile.root")

    f1 = ROOT.TFile(original, "recreate")
    f1.SetCompressionLevel(0)
    t1 = ROOT.TTree("t1", "title")
    d1 = array.array("d", [0.0])
    d2 = array.array("d", [0.0])
    t1.Branch("branch1", d1, "branch1/D")

    t1.Write()
    f1.Close()

    with uproot.open(original) as fin:
        fin["t1"]

    with uproot.recreate(newfile, compression=None) as fout:
        tree = fout._cascading.add_tree(
            fout._file.sink, "t1", "title", {"branch1": np.float64}
        )

    f2 = ROOT.TFile(newfile)
    t2 = f2.Get("t1")

    assert t2.GetName() == "t1"
    assert t2.GetTitle() == "title"

    assert t2.GetBranch("branch1").GetName() == "branch1"
    assert t2.GetBranch("branch1").GetTitle() == "branch1/D"

    assert t2.GetBranch("branch1").GetLeaf("branch1").GetName() == "branch1"
    assert t2.GetBranch("branch1").GetLeaf("branch1").GetTitle() == "branch1"

    assert t2.GetLeaf("branch1").GetName() == "branch1"
    assert t2.GetLeaf("branch1").GetTitle() == "branch1"

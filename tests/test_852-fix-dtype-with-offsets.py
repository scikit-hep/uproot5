# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import os
import pytest
import numpy as np
import uproot

ROOT = pytest.importorskip("ROOT")


def test_fix_interpreting_dtype_with_offsets(tmp_path):
    filename = os.path.join(tmp_path, "tfile_with_tvector3_1.root")
    tfile = ROOT.TFile(filename, "RECREATE")
    tree = ROOT.TTree("tree", "tree")
    tvector3 = ROOT.TVector3()
    tree.Branch("tvector3", tvector3)

    ROOT.gInterpreter.Declare(
        "struct fCoordinates {float fX; float fY; float fZ; fCoordinates(){} public: fCoordinates(float xc, float yc, float zc) : fX(xc), fY(yc), fZ(zc) {}}; "
    )
    ROOT.gInterpreter.Declare(
        "struct ov { struct fCoordinates fCoordinates; ov(float x, float y, float z) : fCoordinates(x,y,z) {}; };"
    )
    mys = [ROOT.fCoordinates(2, 3, 4)]
    tree.Branch("xyz", mys[0])

    for i in range(10):
        tvector3.SetX(i)
        mys[0] = ROOT.fCoordinates(i + 0.1, i + 0.2, i + 0.3)
        tree.Fill()

    tree.AutoSave()
    tfile.Write()
    with uproot.open(filename + ":tree") as f:
        f["xyz"].arrays().tolist()

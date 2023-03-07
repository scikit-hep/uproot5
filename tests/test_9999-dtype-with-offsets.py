# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import os
import pytest
import numpy as np
import uproot

ROOT = pytest.importorskip("ROOT")


def test_fix_interpreting_dtype_with_offsets(tmp_path):
    ROOT.gInterpreter.Declare(
        "struct fCoordinates {float fX; float fY; float fZ; fCoordinates(){} public: fCoordinates(float xc, float yc, float zc) : fX(xc), fY(yc), fZ(zc) {}}; "
    )
    ROOT.gInterpreter.Declare(
        "struct ov { struct fCoordinates fCoordinates; ov(float x, float y, float z) : fCoordinates(x,y,z) {}; };"
    )
    mys = [ROOT.fCoordinates(2, 3, 4)]

    filename = os.path.join(tmp_path, "test.root")
    f = ROOT.TFile(filename, "recreate")
    t = ROOT.TTree("mytree", "example tree")

    t.Branch("xyz", mys[0])

    for i in range(10):
        mys[0] = ROOT.fCoordinates(i + 0.1, i + 0.2, i + 0.3)

    t.Fill()
    t.Write()

    with uproot.open(filename)["mytree"] as tt:
        assert tt["xyz"].array().to_list() == []

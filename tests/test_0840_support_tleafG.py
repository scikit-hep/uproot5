from array import array
import os
import pytest
import numpy as np
import uproot

ROOT = pytest.importorskip("ROOT")


def test_support_leafG(tmp_path):
    filename = os.path.join(tmp_path, "testy.root")
    f = ROOT.TFile(filename, "recreate")
    t = ROOT.TTree("mytree", "example tree")

    n = np.array(2, dtype=np.int32)
    t.Branch("mynum", n, "mynum/I")
    x = np.array([[1, 2, 3], [4, 5, 6]])
    t.Branch("myarrayG", x, "myarrayG[mynum][3]/G")
    t.Branch("myarrayL", x, "myarrayL[mynum][3]/L")

    nentries = 25
    for i in range(nentries):
        t.Fill()

    f.Write()

    assert t.GetLeaf("myarrayG").Class_Name() == "TLeafG"
    assert t.GetLeaf("myarrayL").Class_Name() == "TLeafL"

    with uproot.open(filename)["mytree"] as t:
        assert t["myarrayG"].array(library="np").tolist()[0].tolist() == [
            [1, 2, 3],
            [4, 5, 6],
        ]
        assert t["myarrayL"].array(library="np").tolist()[0].tolist() == [
            [1, 2, 3],
            [4, 5, 6],
        ]
        assert (
            repr(t["myarrayG"].interpretation) == "AsJagged(AsDtype(\"('>i8', (3,))\"))"
        )
        assert (
            repr(t["myarrayL"].interpretation) == "AsJagged(AsDtype(\"('>i8', (3,))\"))"
        )

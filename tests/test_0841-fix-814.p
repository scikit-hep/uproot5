# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
import os
import numpy
import pytest
import uproot

pytest.importorskip("ROOT")


def test_1(tmp_path):
    filename = os.path.join(tmp_path, "tfile_with_tvector3_1.root")
    tfile = ROOT.TFile(filename, "RECREATE")
    tree = ROOT.TTree("tree", "tree")
    tvector3 = ROOT.TVector3()
    tree.Branch("tvector3", tvector3)
    for x in range(10):
        tvector3.SetX(x)
        tree.Fill()
    tree.AutoSave()
    tfile.Close()
    uproot.open("tfile_with_tvector3_1.root:tree").arrays()

    tfile = ROOT.TFile(filename, "READ")
    tree = tfile.Get("tree")
    rdf = ROOT.RDataFrame(tree)
    branchlist = ROOT.std.vector(ROOT.std.string)()
    branchlist.push_back("tvector3")
    rdf.Snapshot("tree", "tfile_with_tvector3_2.root", branchlist)
    uproot.open("tfile_with_tvector3_2.root:tree").arrays()

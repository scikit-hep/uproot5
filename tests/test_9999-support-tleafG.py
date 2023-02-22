import os
from array import array
import numpy as np
import ROOT
import uproot

def test_support_leafG(tmp_path):
    filename = os.path.join(tmp_path, "testy.root")

    f = ROOT.TFile(filename, "recreate")
    tree = ROOT.TTree("tree","tree")
    nr_elems = 10
    myvarL = array('i', [0])
    myvarG = array('i', [0])

    tree.Branch('myvarL', myvarL, 'myvarL/L')
    tree.Branch('myvarG', myvarG, 'myvarG/G')

    for i in range(nr_elems):
        myvarL[0] = int(i * 2)
        myvarG[0] = int(i * 3)
        tree.Fill()
    f.Write()

    t2 = f.Get("tree")

    assert t2.GetName() == "tree"
    assert t2.GetLeaf("myvarL").Class_Name() == "TLeafL"
    assert t2.GetLeaf("myvarG").Class_Name() == "TLeafG"

    with uproot.open(filename) as f2:
        assert len(f2["tree"]["myvarL"].array(library="np").tolist()) == nr_elems
        assert len(f2["tree"]["myvarG"].array(library="np").tolist() )== nr_elems
   
    f.Close()

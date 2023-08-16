import pytest
import os
import awkward as ak
import uproot

ROOT = pytest.importorskip("ROOT")


# def test_check_read_string(tmp_path):
#     filename = os.path.join(tmp_path, "tleafc_test_read.root")

#     tfile = ROOT.TFile(filename, "RECREATE")
#     t = ROOT.TTree("tree", "tree")
#     string_vec = ROOT.std.vector("string")(["one"])
#     # fng_vec.push_back(str(i))
#     t.Branch("string", string_vec, "string/C")
#     t.Fill()
#     assert t.GetLeaf("string").Class_Name()  == 'TLeafC'
#     tfile.Write()
   
#     file = uproot.open("tleafc_test.root")["tree"]
#     print(str(file["string"].array()[0]))
#     assert file["string"].array() == ["one"]
#     assert file["string"].member("fLeaves")[0].classname == 'TLeafC'
#     assert file["string"].interpretation == uproot.interpretation.strings.AsStrings()
    
def test_write_tfleac_uproot(tmp_path):
    filename = os.path.join(tmp_path, "tleafc_test_write.root")

    file = uproot.recreate(filename)
    array = ak.Array(["one", "two", "three"])
    file["tree"] = {"branch": array}

    rf = ROOT.TFile(filename)

    data = rf.Get("tree")

    assert data.GetLeaf("branch").Class_Name() == "TLeafC"   

    with uproot.open(filename) as g:
        assert g['tree']['branch'].array().tolist() == ["one", "two", "three"]


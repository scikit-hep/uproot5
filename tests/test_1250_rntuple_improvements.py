# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import pytest
import skhep_testdata

import uproot


def test_field_class():
    filename = skhep_testdata.data_path("DAOD_TRUTH3_RC2.root")
    with uproot.open(filename) as f:
        obj = f["RNT:CollectionTree"]
        jets = obj["AntiKt4TruthDressedWZJetsAux:"]
        assert len(jets) == 6

        pt = jets["pt"]
        assert len(pt) == 0


def test_array_methods():
    filename = skhep_testdata.data_path(
        "Run2012BC_DoubleMuParked_Muons_rntuple_1000evts.root"
    )
    with uproot.open(filename) as f:
        obj = f["Events"]
        nMuon_array = obj["nMuon"].array()
        Muon_pt_array = obj["Muon_pt"].array()
        assert nMuon_array.tolist() == [len(l) for l in Muon_pt_array]

        nMuon_arrays = obj["nMuon"].arrays()
        assert len(nMuon_arrays.fields) == 1
        assert len(nMuon_arrays) == 1000
        assert nMuon_arrays["nMuon"].tolist() == nMuon_array.tolist()

    filename = skhep_testdata.data_path("DAOD_TRUTH3_RC2.root")
    with uproot.open(filename) as f:
        obj = f["RNT:CollectionTree"]
        jets = obj["AntiKt4TruthDressedWZJetsAux:"].arrays()
        assert len(jets[0].pt) == 5

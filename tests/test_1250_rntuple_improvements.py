# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import pytest
import skhep_testdata

import uproot

pytest.skip(
    "Skipping until test files are available with RNTuple v1.0", allow_module_level=True
)


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


def test_iterate():
    filename = skhep_testdata.data_path(
        "Run2012BC_DoubleMuParked_Muons_rntuple_1000evts.root"
    )
    with uproot.open(filename) as f:
        obj = f["Events"]
        for i, arrays in enumerate(obj.iterate(step_size=100)):
            assert len(arrays) == 100
            if i == 0:
                expected_pt = [10.763696670532227, 15.736522674560547]
                expected_charge = [-1, -1]
                assert arrays["Muon_pt"][0].tolist() == expected_pt
                assert arrays["Muon_charge"][0].tolist() == expected_charge

        for i, arrays in enumerate(obj.iterate(step_size="10 kB")):
            if i == 0:
                assert len(arrays) == 363
                expected_pt = [10.763696670532227, 15.736522674560547]
                expected_charge = [-1, -1]
                assert arrays["Muon_pt"][0].tolist() == expected_pt
                assert arrays["Muon_charge"][0].tolist() == expected_charge
            elif i == 1:
                assert len(arrays) == 363
            elif i == 2:
                assert len(arrays) == 274
            else:
                assert False

        Muon_pt = obj["Muon_pt"]
        for i, arrays in enumerate(Muon_pt.iterate(step_size=100)):
            assert len(arrays) == 100
            if i == 0:
                expected_pt = [10.763696670532227, 15.736522674560547]
                assert arrays[0].tolist() == expected_pt

        for i, arrays in enumerate(Muon_pt.iterate(step_size="5 kB")):
            if i == 0:
                assert len(arrays) == 611
                expected_pt = [10.763696670532227, 15.736522674560547]
                assert arrays[0].tolist() == expected_pt
            elif i == 1:
                assert len(arrays) == 389
            else:
                assert False

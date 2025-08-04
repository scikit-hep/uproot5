# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import pytest
import skhep_testdata

import uproot


def test_field_class():
    filename = skhep_testdata.data_path("test_nested_structs_rntuple_v1-0-0-0.root")
    with uproot.open(filename) as f:
        obj = f["ntuple"]
        my_struct = obj["my_struct"]
        assert len(my_struct) == 2
        assert my_struct is f["ntuple/my_struct"]
        assert my_struct is f["ntuple"]["my_struct"]

        sub_struct = my_struct["sub_struct"]
        assert len(my_struct) == 2
        assert sub_struct is f["ntuple/my_struct/sub_struct"]
        assert sub_struct is f["ntuple"]["my_struct"]["sub_struct"]

        sub_sub_struct = sub_struct["sub_sub_struct"]
        assert len(sub_sub_struct) == 2
        assert sub_sub_struct is f["ntuple/my_struct/sub_struct/sub_sub_struct"]
        assert (
            sub_sub_struct is f["ntuple"]["my_struct"]["sub_struct"]["sub_sub_struct"]
        )

        v = sub_sub_struct["v"]
        assert len(v) == 0


def test_array_methods():
    filename = skhep_testdata.data_path(
        "Run2012BC_DoubleMuParked_Muons_1000evts_rntuple_v1-0-0-0.root"
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


def test_iterate():
    filename = skhep_testdata.data_path(
        "Run2012BC_DoubleMuParked_Muons_1000evts_rntuple_v1-0-0-0.root"
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
                assert len(arrays) == 384
                expected_pt = [10.763696670532227, 15.736522674560547]
                expected_charge = [-1, -1]
                assert arrays["Muon_pt"][0].tolist() == expected_pt
                assert arrays["Muon_charge"][0].tolist() == expected_charge
            elif i == 1:
                assert len(arrays) == 384
            elif i == 2:
                assert len(arrays) == 232
            else:
                assert False

        Muon_pt = obj["Muon_pt"]
        for i, arrays in enumerate(Muon_pt.iterate(step_size=100)):
            assert len(arrays) == 100
            if i == 0:
                expected_pt = [10.763696670532227, 15.736522674560547]
                assert arrays["Muon_pt"][0].tolist() == expected_pt

        for i, arrays in enumerate(Muon_pt.iterate(step_size="5 kB")):
            if i == 0:
                assert len(arrays) == 611
                expected_pt = [10.763696670532227, 15.736522674560547]
                assert arrays["Muon_pt"][0].tolist() == expected_pt
            elif i == 1:
                assert len(arrays) == 389
            else:
                assert False

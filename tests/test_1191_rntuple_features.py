# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import skhep_testdata

import uproot


def test_schema_extension():
    filename = skhep_testdata.data_path("test_ntuple_extension_columns.root")
    with uproot.open(filename) as f:
        obj = f["EventData"]

        assert len(obj.column_records) > len(obj.header.column_records)
        assert len(obj.column_records) == 936
        assert obj.column_records[903].first_ele_index == 36

        arrays = obj.arrays()

        pbs = arrays[
            "HLT_AntiKt4EMPFlowJets_subresjesgscIS_ftf_TLAAux::fastDIPS20211215_pb"
        ]
        assert len(pbs) == 40
        assert all(len(l) == 0 for l in pbs[:36])
        assert next(i for i, l in enumerate(pbs) if len(l) != 0) == 36

        jets = arrays["HLT_AntiKt4EMPFlowJets_subresjesgscIS_ftf_TLAAux:"]
        assert len(jets.pt) == len(pbs)


def test_rntuple_cardinality():
    filename = skhep_testdata.data_path(
        "Run2012BC_DoubleMuParked_Muons_rntuple_1000evts.root"
    )
    with uproot.open(filename) as f:
        obj = f["Events"]
        arrays = obj.arrays()
        assert arrays["nMuon"].tolist() == [len(l) for l in arrays["Muon_pt"]]


def test_skip_recursively_empty_structs():
    filename = skhep_testdata.data_path("DAOD_TRUTH3_RC2.root")
    with uproot.open(filename) as f:
        obj = uproot.open(filename)["RNT:CollectionTree"]
        arrays = obj.arrays()
        jets = arrays["AntiKt4TruthDressedWZJetsAux:"]
        assert len(jets[0].pt) == 5


def test_split_encoding():
    filename = skhep_testdata.data_path(
        "Run2012BC_DoubleMuParked_Muons_rntuple_1000evts.root"
    )
    with uproot.open(filename) as f:
        obj = f["Events"]
        arrays = obj.arrays()

        expected_pt = [10.763696670532227, 15.736522674560547]
        expected_charge = [-1, -1]
        assert arrays["Muon_pt"][0].tolist() == expected_pt
        assert arrays["Muon_charge"][0].tolist() == expected_charge
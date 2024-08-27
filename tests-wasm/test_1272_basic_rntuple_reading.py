# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
from __future__ import annotations

from utils import run_test_in_pyodide


@run_test_in_pyodide(test_file="test_ntuple_extension_columns.root")
def test_schema_extension(selenium):
    import uproot

    with uproot.open("test_ntuple_extension_columns.root") as f:
        obj = f["EventData"]

        assert len(obj.column_records) > len(obj.header.column_records)
        assert len(obj.column_records) == 936
        assert obj.column_records[903].first_ele_index == 36

        arrays = obj.arrays()

        pbs = arrays[
            "HLT_AntiKt4EMPFlowJets_subresjesgscIS_ftf_TLAAux::fastDIPS20211215_pb"
        ]
        assert len(pbs) == 40
        assert all(len(a) == 0 for a in pbs[:36])
        assert next(i for i, a in enumerate(pbs) if len(a) != 0) == 36

        jets = arrays["HLT_AntiKt4EMPFlowJets_subresjesgscIS_ftf_TLAAux:"]
        assert len(jets.pt) == len(pbs)


@run_test_in_pyodide(test_file="Run2012BC_DoubleMuParked_Muons_rntuple_1000evts.root")
def test_split_encoding():
    import uproot

    with uproot.open("Run2012BC_DoubleMuParked_Muons_rntuple_1000evts.root") as f:
        obj = f["Events"]
        arrays = obj.arrays()

        expected_pt = [10.763696670532227, 15.736522674560547]
        expected_charge = [-1, -1]
        assert arrays["Muon_pt"][0].tolist() == expected_pt
        assert arrays["Muon_charge"][0].tolist() == expected_charge

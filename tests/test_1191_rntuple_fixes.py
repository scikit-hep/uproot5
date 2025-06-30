# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import pytest
import skhep_testdata

import uproot


def test_schema_extension():
    filename = skhep_testdata.data_path("test_extension_columns_rntuple_v1-0-0-0.root")
    with uproot.open(filename) as f:
        obj = f["ntuple"]

        assert len(obj.page_link_list[0]) < len(obj.page_link_list[1])

        assert len(obj.column_records) > len(obj.header.column_records)
        assert len(obj.column_records) == 4
        assert obj.column_records[1].first_element_index == 200
        assert obj.column_records[2].first_element_index == 400

        arrays = obj.arrays()

        assert len(arrays.float_field) == 600
        assert len(arrays.intvec_field) == 600

        assert all(arrays.float_field[:200] == 0)
        assert all(len(l) == 0 for l in arrays.intvec_field[:400])

        assert next(i for i, l in enumerate(arrays.float_field) if l != 0) == 200
        assert next(i for i, l in enumerate(arrays.intvec_field) if len(l) != 0) == 400


def test_rntuple_cardinality():
    filename = skhep_testdata.data_path(
        "Run2012BC_DoubleMuParked_Muons_1000evts_rntuple_v1-0-0-0.root"
    )
    with uproot.open(filename) as f:
        obj = f["Events"]
        arrays = obj.arrays()
        assert arrays["nMuon"].tolist() == [len(l) for l in arrays["Muon_pt"]]


def test_multiple_page_delta_encoding():
    filename = skhep_testdata.data_path("test_index_multicluster_rntuple_v1-0-0-0.root")
    with uproot.open(filename) as f:
        obj = f["ntuple"]
        data = obj.read_col_page(0, 0)
        # first page has 64 elements, so this checks that data was stitched together correctly
        assert data[64] - data[63] == 2


def test_split_encoding():
    filename = skhep_testdata.data_path(
        "Run2012BC_DoubleMuParked_Muons_1000evts_rntuple_v1-0-0-0.root"
    )
    with uproot.open(filename) as f:
        obj = f["Events"]
        arrays = obj.arrays()

        expected_pt = [10.763696670532227, 15.736522674560547]
        expected_charge = [-1, -1]
        assert arrays["Muon_pt"][0].tolist() == expected_pt
        assert arrays["Muon_charge"][0].tolist() == expected_charge

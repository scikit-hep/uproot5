# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
from __future__ import annotations

import pytest

try:
    from utils import run_test_in_pyodide
except ImportError:
    pytest.skip("Pyodide is not available", allow_module_level=True)


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

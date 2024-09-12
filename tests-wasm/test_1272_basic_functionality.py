# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
from __future__ import annotations

import pytest
from utils import run_test_in_pyodide


# Taken from test_0034_generic_objects_in_ttrees.py
@run_test_in_pyodide(test_file="uproot-HZZ-objects.root", packages=["pytest", "xxhash"])
def test_read_ttree(selenium):
    import pytest

    import uproot

    awkward = pytest.importorskip("awkward")

    with uproot.open("uproot-HZZ-objects.root")["events"] as tree:
        result = tree["muonp4"].array(library="ak")

        assert (
            str(awkward.type(result))
            == "2421 * var * TLorentzVector[fP: TVector3[fX: float64, "
            "fY: float64, fZ: float64], fE: float64]"
        )

        assert result[0, 0, "fE"] == 54.77949905395508
        assert result[0, 0, "fP", "fX"] == -52.89945602416992
        assert result[0, 0, "fP", "fY"] == -11.654671669006348
        assert result[0, 0, "fP", "fZ"] == -8.16079330444336


# Taken from test_0406_write_a_tree.py
@run_test_in_pyodide()
def test_write_ttree(selenium):
    import numpy as np

    import uproot

    newfile = "newfile.root"

    b1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    b2 = [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]

    with uproot.recreate(newfile, compression=None) as fout:
        tree = fout.mktree("t", {"b1": np.int32, "b2": np.float64}, "title")

        assert tree._cascading._basket_capacity == 10

        for _ in range(5):
            fout["t"].extend({"b1": b1, "b2": b2})

        assert tree._cascading._basket_capacity == 10

        for _ in range(10):
            fout["t"].extend({"b1": b1, "b2": b2})

        assert tree._cascading._basket_capacity == 100

        for _ in range(90):
            fout["t"].extend({"b1": b1, "b2": b2})

        assert tree._cascading._basket_capacity == 1000

    with uproot.open(newfile) as fin:
        assert fin.keys() == ["t;1"]  # same cycle number
        t2 = fin["t"]
        assert t2.num_entries == len(b1) * 105
        assert t2["b1"].array(library="np").tolist() == b1 * 105
        assert t2["b2"].array(library="np").tolist() == b2 * 105


# Taken from test_1191_rntuple_fixes.py
@run_test_in_pyodide(test_file="test_ntuple_extension_columns.root")
def test_read_rntuple(selenium):
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


# Taken from test_0088_read_with_http.py
@pytest.mark.network
@run_test_in_pyodide(packages=["requests"])
def test_read_ttree_http(selenium):
    import uproot

    with uproot.open(
        "http://starterkit.web.cern.ch/starterkit/data/advanced-python-2019/dalitzdata.root",
        handler=uproot.source.http.HTTPSource,
    ) as f:
        data = f["tree"].arrays(["Y1", "Y2"], library="np")
        assert len(data["Y1"]) == 100000
        assert len(data["Y2"]) == 100000


# Taken from test_1191_rntuple_fixes.py
@pytest.mark.network
@run_test_in_pyodide(packages=["requests"])
def test_read_rntuple_http(selenium):
    import uproot

    with uproot.open(
        "https://github.com/scikit-hep/scikit-hep-testdata/raw/main/src/skhep_testdata/data/Run2012BC_DoubleMuParked_Muons_rntuple_1000evts.root",
        handler=uproot.source.http.HTTPSource,
    ) as f:
        obj = f["Events"]
        arrays = obj.arrays()
        assert arrays["nMuon"].tolist() == [len(a) for a in arrays["Muon_pt"]]

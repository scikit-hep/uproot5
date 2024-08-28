# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
from __future__ import annotations

import pytest
from utils import run_test_in_pyodide


# Taken from test_0088_read_with_http.py
@pytest.mark.network
@run_test_in_pyodide(packages=["requests"])
def test_ttree(selenium):
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
def test_rntuple(selenium):
    import uproot

    with uproot.open(
        "https://github.com/scikit-hep/scikit-hep-testdata/raw/main/src/skhep_testdata/data/Run2012BC_DoubleMuParked_Muons_rntuple_1000evts.root",
        handler=uproot.source.http.HTTPSource,
    ) as f:
        obj = f["Events"]
        arrays = obj.arrays()
        assert arrays["nMuon"].tolist() == [len(a) for a in arrays["Muon_pt"]]

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import pytest
import skhep_testdata

import uproot4


def test_keys():
    with uproot4.open(skhep_testdata.data_path("uproot-small-evnt-tree-fullsplit.root"))["tree"] as t:
        assert t.keys(filter_name="P3.Px") == ["evt/P3/P3.Px"]
        assert t.keys(filter_name="/P3.Px") == []
        assert t.keys(filter_name="P3/P3.Px") == []
        assert t.keys(filter_name="evt/P3/P3.Px") == ["evt/P3/P3.Px"]
        assert t.keys(filter_name="/evt/P3/P3.Px") == ["evt/P3/P3.Px"]
        assert t["evt"].keys(filter_name="P3.Px") == ["P3/P3.Px"]
        assert t["evt"].keys(filter_name="/P3.Px") == []
        assert t["evt"].keys(filter_name="P3/P3.Px") == ["P3/P3.Px"]
        assert t["evt"].keys(filter_name="/P3/P3.Px") == ["P3/P3.Px"]
        assert t["evt"].keys(filter_name="evt/P3/P3.Px") == []
        assert t["evt/P3"].keys(filter_name="P3.Px") == ["P3.Px"]
        assert t["evt/P3"].keys(filter_name="/P3.Px") == ["P3.Px"]
        assert t["evt/P3"].keys(filter_name="P3/P3.Px") == []


def test_numpy():
    with uproot4.open(skhep_testdata.data_path("uproot-small-evnt-tree-fullsplit.root"))["tree"] as t:
        assert list(t.arrays(filter_name="P3.Px", library="np").keys()) == ["P3.Px"]
        assert list(t.arrays(filter_name="/P3.Px", library="np").keys()) == []
        assert list(t.arrays(filter_name="P3/P3.Px", library="np").keys()) == []
        assert list(t.arrays(filter_name="evt/P3/P3.Px", library="np").keys()) == ["P3.Px"]
        assert list(t.arrays(filter_name="/evt/P3/P3.Px", library="np").keys()) == ["P3.Px"]
        assert list(t["evt"].arrays(filter_name="P3.Px", library="np").keys()) == ["P3.Px"]
        assert list(t["evt"].arrays(filter_name="/P3.Px", library="np").keys()) == []
        assert list(t["evt"].arrays(filter_name="P3/P3.Px", library="np").keys()) == ["P3.Px"]
        assert list(t["evt"].arrays(filter_name="/P3/P3.Px", library="np").keys()) == ["P3.Px"]
        assert list(t["evt"].arrays(filter_name="evt/P3/P3.Px", library="np").keys()) == []
        assert list(t["evt/P3"].arrays(filter_name="P3.Px", library="np").keys()) == ["P3.Px"]
        assert list(t["evt/P3"].arrays(filter_name="/P3.Px", library="np").keys()) == ["P3.Px"]
        assert list(t["evt/P3"].arrays(filter_name="P3/P3.Px", library="np").keys()) == []


def test_awkward():
    awkward1 = pytest.importorskip("awkward1")
    with uproot4.open(skhep_testdata.data_path("uproot-small-evnt-tree-fullsplit.root"))["tree"] as t:
        assert t.arrays(filter_name="P3.Px", library="ak").fields == ["P3.Px"]
        assert t.arrays(filter_name="/P3.Px", library="ak").fields == []
        assert t.arrays(filter_name="P3/P3.Px", library="ak").fields == []
        assert t.arrays(filter_name="evt/P3/P3.Px", library="ak").fields == ["P3.Px"]
        assert t.arrays(filter_name="/evt/P3/P3.Px", library="ak").fields == ["P3.Px"]
        assert t["evt"].arrays(filter_name="P3.Px", library="ak").fields == ["P3.Px"]
        assert t["evt"].arrays(filter_name="/P3.Px", library="ak").fields == []
        assert t["evt"].arrays(filter_name="P3/P3.Px", library="ak").fields == ["P3.Px"]
        assert t["evt"].arrays(filter_name="/P3/P3.Px", library="ak").fields == ["P3.Px"]
        assert t["evt"].arrays(filter_name="evt/P3/P3.Px", library="ak").fields == []
        assert t["evt/P3"].arrays(filter_name="P3.Px", library="ak").fields == ["P3.Px"]
        assert t["evt/P3"].arrays(filter_name="/P3.Px", library="ak").fields == ["P3.Px"]
        assert t["evt/P3"].arrays(filter_name="P3/P3.Px", library="ak").fields == []


def test_pandas():
    pandas = pytest.importorskip("pandas")
    with uproot4.open(skhep_testdata.data_path("uproot-small-evnt-tree-fullsplit.root"))["tree"] as t:
        assert t.arrays(filter_name="P3.Px", library="pd").columns.tolist() == ["P3.Px"]
        assert t.arrays(filter_name="/P3.Px", library="pd").columns.tolist() == []
        assert t.arrays(filter_name="P3/P3.Px", library="pd").columns.tolist() == []
        assert t.arrays(filter_name="evt/P3/P3.Px", library="pd").columns.tolist() == ["P3.Px"]
        assert t.arrays(filter_name="/evt/P3/P3.Px", library="pd").columns.tolist() == ["P3.Px"]
        assert t["evt"].arrays(filter_name="P3.Px", library="pd").columns.tolist() == ["P3.Px"]
        assert t["evt"].arrays(filter_name="/P3.Px", library="pd").columns.tolist() == []
        assert t["evt"].arrays(filter_name="P3/P3.Px", library="pd").columns.tolist() == ["P3.Px"]
        assert t["evt"].arrays(filter_name="/P3/P3.Px", library="pd").columns.tolist() == ["P3.Px"]
        assert t["evt"].arrays(filter_name="evt/P3/P3.Px", library="pd").columns.tolist() == []
        assert t["evt/P3"].arrays(filter_name="P3.Px", library="pd").columns.tolist() == ["P3.Px"]
        assert t["evt/P3"].arrays(filter_name="/P3.Px", library="pd").columns.tolist() == ["P3.Px"]
        assert t["evt/P3"].arrays(filter_name="P3/P3.Px", library="pd").columns.tolist() == []

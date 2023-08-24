import pytest
import os
import awkward as ak
import uproot

ROOT = pytest.importorskip("ROOT")


def test_write_tfleac_uproot_1(tmp_path):
    filename = os.path.join(tmp_path, "tleafc_test_write_1.root")

    with uproot.recreate(filename) as f:
        array = ak.Array(["one", "two", "three"])
        f["tree"] = {"branch": array}

    rf = ROOT.TFile(filename)
    data = rf.Get("tree")
    assert data.GetLeaf("branch").Class_Name() == "TLeafC"
    rf.Close()

    with uproot.open(filename) as g:
        assert g["tree"]["branch"].array().tolist() == ["one", "two", "three"]


def test_write_tfleac_uproot_2(tmp_path):
    filename = os.path.join(tmp_path, "tleafc_test_write_2.root")

    with uproot.recreate(filename) as f:
        array = ak.Array(
            ["unu", "doi", "trei", "patru", "cinci", "sase", "sapte", "opt"]
        )
        f["tree"] = {"branch": array}

    rf = ROOT.TFile(filename)
    data = rf.Get("tree")
    assert data.GetLeaf("branch").Class_Name() == "TLeafC"
    rf.Close()

    with uproot.open(filename) as g:
        assert g["tree"]["branch"].array().tolist() == [
            "unu",
            "doi",
            "trei",
            "patru",
            "cinci",
            "sase",
            "sapte",
            "opt",
        ]


def test_write_tfleac_uproot_3(tmp_path):
    filename = os.path.join(tmp_path, "tleafc_test_write_3.root")

    with uproot.recreate(filename) as f:
        array = ak.Array(["zero", "one" * 100, "two", "three" * 100, "four", "five"])
        f["tree"] = {"branch": array}

    rf = ROOT.TFile(filename)
    data = rf.Get("tree")
    assert data.GetLeaf("branch").Class_Name() == "TLeafC"
    rf.Close()

    with uproot.open(filename) as g:
        assert g["tree"]["branch"].array().tolist() == [
            "zero",
            "one" * 100,
            "two",
            "three" * 100,
            "four",
            "five",
        ]

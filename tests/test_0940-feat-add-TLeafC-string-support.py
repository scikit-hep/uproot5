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
    assert [entry.branch for entry in data] == ["one", "two", "three"]
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
    assert [entry.branch for entry in data] == [
        "unu",
        "doi",
        "trei",
        "patru",
        "cinci",
        "sase",
        "sapte",
        "opt",
    ]
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
    assert [entry.branch for entry in data] == [
        "zero",
        "one" * 100,
        "two",
        "three" * 100,
        "four",
        "five",
    ]
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


def test_empty_array(tmp_path):
    filename = os.path.join(tmp_path, "empty-array.root")

    with uproot.recreate(filename) as outfile:
        array = ak.Array(["one", "two", "three"])[0:0]  # type=string but len=0
        outfile["tree"] = {"branch": array}

    root_infile = ROOT.TFile(filename)
    root_tree = root_infile.Get("tree")
    assert root_tree.GetLeaf("branch").Class_Name() == "TLeafC"
    assert [entry.branch for entry in root_tree] == []
    root_infile.Close()

    with uproot.open(filename) as infile:
        array = infile["tree"]["branch"].array()
        assert array.tolist() == []
        assert str(array.type) == "0 * string"


def test_empty_array_2baskets(tmp_path):
    filename = os.path.join(tmp_path, "empty-array.root")

    with uproot.recreate(filename) as outfile:
        array = ak.Array(["one", "two", "three"])[0:0]  # type=string but len=0
        outfile["tree"] = {"branch": array}
        outfile["tree"].extend({"branch": array})

    root_infile = ROOT.TFile(filename)
    root_tree = root_infile.Get("tree")
    assert root_tree.GetLeaf("branch").Class_Name() == "TLeafC"
    assert [entry.branch for entry in root_tree] == []
    root_infile.Close()

    with uproot.open(filename) as infile:
        array = infile["tree"]["branch"].array()
        assert array.tolist() == []
        assert str(array.type) == "0 * string"


def test_mutating_fLen(tmp_path):
    filename = os.path.join(tmp_path, "mutating-fLen.root")

    with uproot.recreate(filename) as outfile:
        aslist = ["x" * 1]
        array = ak.Array(aslist)
        outfile["tree"] = {"branch": array}
        num_baskets = 1

        with uproot.open(filename) as infile:
            assert len(infile.keys()) == 1
            branch = infile["tree"]["branch"]
            assert branch.array().tolist() == aslist
            assert branch.member("fLeaves")[0].member("fLen") == 2
            assert len(branch.member("fBasketEntry")) == 10
            assert branch.num_baskets == num_baskets

        for i in range(2, 10):
            aslist.extend(["x" * i])
            outfile["tree"].extend({"branch": ak.Array(["x" * i])})
            num_baskets += 1

            with uproot.open(filename) as infile:
                assert len(infile.keys()) == 1
                branch = infile["tree"]["branch"]
                assert branch.array().tolist() == aslist
                # verify that fLen is mutated in-place as we add TBaskets
                assert branch.member("fLeaves")[0].member("fLen") == i + 1
                assert len(branch.member("fBasketEntry")) == 10
                assert branch.num_baskets == num_baskets

        for i in range(10, 100):
            aslist.extend(["x" * i])
            outfile["tree"].extend({"branch": ak.Array(["x" * i])})
            num_baskets += 1

            with uproot.open(filename) as infile:
                # verify that this is still the case after write_anew
                # (increasing fBasketEntry capacity means rewriting metadata)
                assert len(infile.keys()) == 1
                branch = infile["tree"]["branch"]
                assert branch.array().tolist() == aslist
                assert branch.member("fLeaves")[0].member("fLen") == i + 1
                assert len(branch.member("fBasketEntry")) == 100
                assert branch.num_baskets == num_baskets

        for i in range(100, 255, 5):
            aslist.extend(["x" * i])
            outfile["tree"].extend({"branch": ak.Array(["x" * i])})
            num_baskets += 1

            with uproot.open(filename) as infile:
                # same, but now in the 100 -> 1000 TBasket range
                assert len(infile.keys()) == 1
                branch = infile["tree"]["branch"]
                assert branch.array().tolist() == aslist
                assert branch.member("fLeaves")[0].member("fLen") == i + 1
                assert len(branch.member("fBasketEntry")) == 1000
                assert branch.num_baskets == num_baskets

        for i in range(255, 265, 5):
            aslist.extend(["x" * i])
            outfile["tree"].extend({"branch": ak.Array(["x" * i])})
            num_baskets += 1

            with uproot.open(filename) as infile:
                # oh, but when the string length exceeds 255, fLen == i + 5
                assert len(infile.keys()) == 1
                branch = infile["tree"]["branch"]
                assert branch.array().tolist() == aslist
                assert branch.member("fLeaves")[0].member("fLen") == i + 5
                assert len(branch.member("fBasketEntry")) == 1000
                assert branch.num_baskets == num_baskets

    # verify that ROOT is still happy with all of this
    root_infile = ROOT.TFile(filename)
    root_tree = root_infile.Get("tree")
    assert [entry.branch for entry in root_tree] == aslist
    root_infile.Close()

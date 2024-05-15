import uproot
from skhep_testdata import data_path
import uproot.serialization
import uproot.writing.writable
import os
import pytest

ROOT = pytest.importorskip("ROOT")
import numpy as np

import awkward as ak


def test_vector():
    with uproot.update(
        "/Users/zobil/Documents/samples/uproot-vectorVectorDouble-work.root"
    ) as write:
        write.add_branches("t", {"branch": int})

    with uproot.open(
        "/Users/zobil/Documents/samples/uproot-vectorVectorDouble.root",
        minimal_ttree_metadata=False,
    ) as read:
        print(read["t"]["x"].arrays())

    with uproot.open(
        "/Users/zobil/Documents/samples/uproot-vectorVectorDouble-work.root",
        minimal_ttree_metadata=False,
    ) as read:
        print(read["tree1"])
        # print(read["tree1"].all_members)
        # print(read["tree1"]["x"].all_members)
        # print(read["tree1"]["x"].member("fLeaves")[0])


def simple_test(tmp_path):
    data = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    data1 = np.array(
        [
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
        ],
        dtype=np.int32,
    )

    with uproot.recreate(os.path.join(tmp_path, "arrays1.root")) as f:
        f["whatever"] = {"b1": data}

    with uproot.recreate(os.path.join(tmp_path, "arrays2.root")) as f:
        f["whatever"] = {"b1": data, "b2": data1}

    with uproot.update(os.path.join(tmp_path, "arrays2.root")) as f:
        f.add_branches("whatever", {"b3": data, "b4": data1})

    with uproot.open(
        os.path.join(tmp_path, "arrays1.root"), minimal_ttree_metadata=False
    ) as check:
        with uproot.open(
            os.path.join(tmp_path, "arrays2.root"), minimal_ttree_metadata=False
        ) as new:
            inFile = ROOT.TFile.Open(os.path.join(tmp_path, "arrays2.root"), "READ")
            tree = inFile.Get("whatever;1")
            print(tree)
            for x in tree:
                print(getattr(x, "b1"))


def test_subbranches(tmp_path):
    data = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    data1 = np.array(
        [
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
        ],
        dtype=np.int32,
    )

    with uproot.recreate(os.path.join(tmp_path, "arrays2.root")) as f:
        f["whatever"] = {"b1": data, "b2": data1}

    with uproot.update(os.path.join(tmp_path, "arrays2.root")) as f:
        f.add_branches("whatever", {"b3": data, "b4": data1})

    with uproot.open(
        os.path.join(tmp_path, "tree_tester.root"), minimal_ttree_metadata=False
    ) as check:
        # check["tree"].show()
        print(check.keys())
        with uproot.open(
            os.path.join(tmp_path, "arrays2.root"), minimal_ttree_metadata=False
        ) as new:
            print(new["whatever"].all_members)
            inFile = ROOT.TFile.Open(os.path.join(tmp_path, "arrays2.root"), "READ")
            tree = inFile.Get("whatever;1")
            print(tree)
            for x in tree:
                print(getattr(x, "b1"))


def test_different_fEntries(tmp_path):
    data = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    data1 = np.array([2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.int32)

    with uproot.recreate(os.path.join(tmp_path, "arrays2.root")) as f:
        with pytest.raises(ValueError):
            f["whatever"] = {"b1": data, "b2": data1}
            f.add_branches(
                "whatever",
                {
                    "b3": data,
                    "b4": np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.int32),
                },
            )


def test_ak_arrays(tmp_path):

    data = np.array([1, 2, 3], dtype=np.int64)
    data1 = np.array([2, 3, 4], dtype=np.int64)
    data2 = np.array([3, 4, 5], dtype=np.int64)
    with uproot.recreate(os.path.join(tmp_path, "control.root")) as file:
        file["tree"] = {
            "b1": ak.Array([data, data1, data2]),
            "b2": ak.Array([data1, data2, data]),
            "b3": ak.Array([data2, data, data1]),
        }

    with uproot.recreate(os.path.join(tmp_path, "ak_test.root")) as file:
        file["tree"] = {
            "b1": ak.Array([data, data1, data2]),
        }

    with uproot.update(os.path.join(tmp_path, "ak_test.root")) as write:
        write.add_branches(
            "tree",
            {
                "b2": ak.Array([data1, data2, data]),
                "b3": ak.Array([data2, data, data1]),
            },
        )

    with uproot.open(
        os.path.join(tmp_path, "control.root"), minimal_ttree_metadata=False
    ) as correct:
        with uproot.open(
            os.path.join(tmp_path, "ak_test.root"), minimal_ttree_metadata=False
        ) as new:
            inFile = ROOT.TFile.Open(os.path.join(tmp_path, "ak_test.root"), "READ")
            tree = inFile.Get("tree")
            for x in tree:
                print(getattr(x, "b1"))
            print(tree.Scan())
            # ak.Array()
            # for x in tree:
            #     print(getattr(x, "b2").GetArray())


def test_streamers_same_dtypes(tmp_path):
    from ROOT import TTree
    from array import array

    N = 4
    data = array("f", N * [0.0])
    data1 = array("f", [2.0, 3.0, 4.0, 5.0])

    inFile = root.TFile(
        "/Users/zobil/Desktop/directory/root_streamers_F.root", "RECREATE"
    )
    tree = root.TTree("tree1", "tree")
    import numpy as np

    # Basic type branch (float) - use array of length 1
    # n = array('f', [ 1.5 ])
    # tree.Branch('b1', n, 'b1/F')

    # Array branch - use array of length N
    N = 4
    # a = array('d', N*[ 0. ])
    # tree.Branch('b1', a, 'b1[' + str(N) + ']/D')

    # # Array branch - use NumPy array of length N
    npa = np.zeros(4, dtype=np.float32)
    tree.Branch("b1", npa, "b1/F")
    for i in range(4):
        npa[0] = i**0
        tree.Fill()
    inFile.Write()
    inFile.Close()

    inFile = root.TFile.Open(os.path.join(tmp_path, "root_streamers_F.root"), "OPEN")
    tree = inFile.Get("tree1")
    tree.Scan()
    data = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)

    with uproot.update(os.path.join(tmp_path, "root_streamers_F.root")) as file:
        file.add_branches("tree1", {"b2": data})

    with uproot.open(
        os.path.join(tmp_path, "root_streamers_F.root"), minimal_ttree_metadata=False
    ) as file:
        inFile = ROOT.TFile.Open(
            os.path.join(tmp_path, "root_streamers_F.root"), "READ"
        )
        tree = inFile.Get("tree1;1")
        indx = 0
        for x in tree:
            assert getattr(x, "b1") == file["tree1"]["b1"].array()[indx]
            assert getattr(x, "b2") == file["tree1"]["b2"].array()[indx]
            indx += 1

        tree.Scan()
        check = [
            "TBranch",
            "TAttLine",
            "TCollection",
            "TLeafF",
            "listOfRules",
            "TString",
            "TObjArray",
            "TAttFill",
            "TBranchRef",
            "TList",
            "ROOT::TIOFeatures",
            "TSeqCollection",
            "TAttMarker",
            "TTree",
            "TNamed",
            "TObject",
            "TAttLine",
            "TLeaf",
            "TRefTable",
        ]
        for i in set(file.file.streamers):
            assert i in check


def test_streamers_diff_dtypes(tmp_path):

    inFile = ROOT.TFile(
        "/Users/zobil/Desktop/directory/root_diff_dtypes.root", "RECREATE"
    )
    tree = ROOT.TTree("tree1", "tree")

    # Basic type branch (float) - use array of length 1
    # n = array('f', [ 1.5 ])
    # tree.Branch('b1', n, 'b1/F')

    # Array branch - use array of length N
    N = 4
    # a = array('d', N*[ 0. ])
    # tree.Branch('b1', a, 'b1[' + str(N) + ']/D')

    # # Array branch - use NumPy array of length N
    npa = np.zeros(4, dtype=float)
    tree.Branch("b1", npa, "b1F")
    for i in range(4):
        npa[0] = i**0
        tree.Fill()
    inFile.Write()
    inFile.Close()

    inFile = ROOT.TFile.Open(os.path.join(tmp_path, "root_diff_dtypes.root"), "OPEN")
    tree = inFile.Get("tree1")
    tree.Scan()
    data = np.array([5, 6, 7, 8], dtype=np.int64)
    with uproot.update(os.path.join(tmp_path, "root_diff_dtypes.root")) as file:
        file.add_branches("tree1", {"b2": data})

    with uproot.open(
        os.path.join(tmp_path, "root_diff_dtypes.root"), minimal_ttree_metadata=False
    ) as file:
        file["tree1"]["b2"].member("fLeaves")[0].all_members

        inFile = ROOT.TFile.Open(
            os.path.join(tmp_path, "root_diff_dtypes.root"), "READ"
        )
        tree = inFile.Get("tree1;1")
        indx = 0
        for x in tree:
            assert getattr(x, "b1") == file["tree1"]["b1"].array()[indx]
            assert getattr(x, "b2") == file["tree1"]["b2"].array()[indx]
            indx += 1

        # tree.Scan()
        check = [
            "TBranch",
            "TAttLine",
            "TCollection",
            "TLeafF",
            "listOfRules",
            "TString",
            "TObjArray",
            "TAttFill",
            "TBranchRef",
            "TList",
            "ROOT::TIOFeatures",
            "TSeqCollection",
            "TAttMarker",
            "TTree",
            "TNamed",
            "TObject",
            "TAttLine",
            "TLeaf",
            "TRefTable",
            "TLeafL",
        ]
        for i in set(file.file.streamers):
            assert i in check


def HZZ_test(tmp_path):
    with uproot.open(
        data_path("uproot-HZZ.root"), minimal_ttree_metadata=False
    ) as test:
        with uproot.update(os.path.join(tmp_path, "uproot-HZZ.root copy")) as new:
            # data = np.arange(0, 2420, 1)
            data = []
            for i in range(2421):
                data.append(np.arange(0, 3, 1))
            data = ak.Array(data)
            new.add_branches("events", {"data": data})

        with uproot.open(
            os.path.join(tmp_path, "uproot-HZZ.root copy"),
            minimal_ttree_metadata=False,
        ) as check:
            for key in test["events"].keys():
                assert key in test["events"].keys()
                assert ak.all(
                    check["events"][key].array() == test["events"][key].array()
                )

            inFile = ROOT.TFile.Open(
                os.path.join(tmp_path, "uproot-HZZ.root copy"), "READ"
            )
            tree = inFile.Get("events")

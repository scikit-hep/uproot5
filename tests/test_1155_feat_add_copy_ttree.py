import uproot
from skhep_testdata import data_path, known_files
import uproot.model
import uproot.models
import uproot.serialization
import uproot.writing.writable
import os
import pytest

ROOT = pytest.importorskip("ROOT")
import numpy as np

import awkward as ak


def test_vector(tmp_path):
    # with uproot.open(
    #     os.path.join(tmp_path, "uproot-vectorVectorDouble.root"),
    #     minimal_ttree_metadata=False,
    # ) as read:
    #     print(read['t']['x'].debug(1))
        # print(read.cursor.debug(read.file.chunk(start=0, stop=5852)))
        # print(read.file.chunk(start=0, stop=5852).raw_data.tobytes())

    # with uproot.update(
    #     os.path.join(tmp_path, "cp-vectorVectorDouble.root"),
    # ) as write:
    #     write.add_branches("t", {"branch": [1,2,3,4,5]})

    # with uproot.open(
    #     os.path.join(tmp_path, "cp-vectorVectorDouble.root"),
    #     minimal_ttree_metadata=False,
    # ) as read:
    #     print(read['t'])

    inFile = ROOT.TFile.Open(os.path.join(tmp_path, "vectorVectorDouble.root"), "READ")
    tree = inFile.Get("t")
    for x in tree:
        print(getattr(x, "x"))
    ROOT.TClass.TBranchElement
    # inFile.Close()

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
        f["whatever"] = {"b1": data, "b2": data1}

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
            indx = 0
            for x in tree:
                assert getattr(x, "b1") == data[indx]
                assert getattr(x, "b2") == data1[indx]
                indx += 1
        print(check.file.chunk(start=0, stop=8000).raw_data.tobytes())

def test_multiple_trees(tmp_path):
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
        f["whatever"] = {"b1": data, "b2": data1, "b3": data, "b4": data1}
        f["whatever1"] = {"b1": data, "b2": data1, "b3": data, "b4": data1}

    with uproot.recreate(os.path.join(tmp_path, "arrays2.root")) as f:
        f["whatever"] = {"b1": data, "b2": data1}
        f["whatever1"] = {"b1": data, "b2": data1, "b3": data}

    with uproot.update(os.path.join(tmp_path, "arrays2.root")) as f:
        f.add_branches("whatever", {"b3": data, "b4": data1})
        f.add_branches("whatever1", {"b4": data1})

    with uproot.open(
        os.path.join(tmp_path, "arrays1.root"), minimal_ttree_metadata=False
    ) as check:
        with uproot.open(
            os.path.join(tmp_path, "arrays2.root"), minimal_ttree_metadata=False
        ) as new:
            assert ak.all(new['whatever']['b1'].array() == data)
            assert ak.all(new['whatever1']['b4'].array() == data1)
            assert ak.all(new['whatever1']['b2'].array() == data1)
            assert ak.all(new['whatever1']['b4'].array() == data1)
            inFile = ROOT.TFile.Open(os.path.join(tmp_path, "arrays2.root"), "READ")
            tree = inFile.Get("whatever;1")
            indx = 0
            for x in tree:
                assert getattr(x, "b1") == data[indx]
                assert getattr(x, "b2") == data1[indx]
                indx += 1

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

def test_all_dtypes(tmp_path):
    print("to-do")

def test_ak_arrays(tmp_path):

    data = np.array([1, 2, 3], dtype=np.int64)
    data1 = np.array([2, 3, 4], dtype=np.int64)
    data2 = np.array([3, 4, 5], dtype=np.int64)
    with uproot.recreate(os.path.join(tmp_path, "control.root")) as file:
        file["whatever"] = {
            "b1": ak.Array([data, data1, data2]),
            "b2": ak.Array([data1, data2, data]),
            "b3": ak.Array([data2, data, data1]),
        }

    with uproot.recreate(os.path.join(tmp_path, "ak_test.root")) as file:
        file["whatever"] = {
            "b1": ak.Array([data, data1, data2]),
        }

    with uproot.update(os.path.join(tmp_path, "ak_test.root")) as write:
        write.add_branches(
            "whatever",
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
            tree = inFile.Get("whatever")
            for x in tree:
                getattr(x, "b1")
            inFile.Close()
            df3 = ROOT.RDataFrame("whatever", os.path.join(tmp_path, "ak_test.root"))
            npy3 = ak.from_rdataframe(df3, columns=("b1", "b2", "b3"), keep_order=True)
            assert ak.all(npy3["b1"] == [data, data1, data2])
            assert ak.all(npy3["b2"] == [data1, data2, data])
            assert ak.all(npy3["b3"] == [data2, data, data1])

def test_streamers_same_dtypes(tmp_path):
    # Make file with ROOT
    inFile = ROOT.TFile(os.path.join("root_same_dtypes.root"), "RECREATE")
    tree = ROOT.TTree("tree1", "tree")
    npa = np.zeros(4, dtype=np.float32)
    tree.Branch("b1", npa, "b1/F")
    for i in range(4):
        npa[0] = i**0
        tree.Fill()
    inFile.Write()
    inFile.Close()

    inFile = ROOT.TFile.Open(os.path.join(tmp_path, "root_same_dtypes.root"), "OPEN")
    tree = inFile.Get("tree1")
    data = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)

    with uproot.update(os.path.join(tmp_path, "root_same_dtypes.root")) as file:
        file.add_branches("tree1", {"b2": data})

    with uproot.open(
        os.path.join(tmp_path, "root_same_dtypes.root"), minimal_ttree_metadata=False
    ) as file:
        inFile = ROOT.TFile.Open(
            os.path.join(tmp_path, "root_same_dtypes.root"), "READ"
        )
        inFile.ShowStreamerInfo()
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
        inFile.Close()

def test_streamers_diff_dtypes(tmp_path):
    # Make file with ROOT
    inFile = ROOT.TFile(
        "/Users/zobil/Desktop/directory/root_diff_dtypes.root", "RECREATE"
    )
    tree = ROOT.TTree("tree1", "tree")
    npa = np.zeros(4, dtype=float)
    tree.Branch("b1", npa, "b1F")
    for i in range(4):
        npa[0] = i**0
        tree.Fill()
    inFile.Write()
    inFile.Close()

    inFile = ROOT.TFile.Open(os.path.join(tmp_path, "root_diff_dtypes.root"), "OPEN")
    tree = inFile.Get("tree1")
    data = np.array([5, 6, 7, 8], dtype=np.int64)
    data1 = np.array([5.2, 6.3, 7.4, 8.5], dtype=np.float64)
    with uproot.update(os.path.join(tmp_path, "root_diff_dtypes.root")) as file:
        file.add_branches("tree1", {"b2": data, "b3": data1})

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
            "TLeafD",
        ]
        for i in set(file.file.streamers):
            assert i in check
        inFile.Close()

def HZZ_test(tmp_path):
    with uproot.open(
        data_path("uproot-HZZ.root"), minimal_ttree_metadata=False
    ) as control:
        with uproot.update(os.path.join(tmp_path, "uproot-HZZ.root copy")) as new:
            data = []
            for i in range(2421):
                data.append(np.arange(0, 3, 1))
            data = ak.Array(data)
            new.add_branches("events", {"data": data})

        with uproot.open(
            os.path.join(tmp_path, "uproot-HZZ.root copy"),
            minimal_ttree_metadata=False,
        ) as new:
            for key in control["events"].keys():
                assert key in new["events"].keys()
                assert ak.all(
                    new["events"][key].array() == control["events"][key].array()
                )
            inFile = ROOT.TFile.Open(
                os.path.join(tmp_path, "uproot-HZZ.root copy"), "READ"
            )
            tree = inFile.Get("events")
            indx = 0
            inFile.Close()
            df3 = ROOT.RDataFrame(
                "events", os.path.join(tmp_path, "uproot-HZZ.root copy")
            )
            npy3 = ak.from_rdataframe(df3, columns=("data"), keep_order=True)
            # for key in npy3.keys():
            #     assert ak.all(npy3[key] == control['events'][key].array())
            assert ak.all(npy3 == data)
            inFile.Close()

def test_nested_branches(tmp_path):
    # Make example
    with uproot.open(data_path("uproot-HZZ-objects.root")):
        print("examine this ")
    
def HZZ_test(tmp_path):
    with uproot.open(
        data_path("uproot-HZZ.root"), minimal_ttree_metadata=False
    ) as control:
        with uproot.update(os.path.join(tmp_path, "uproot-HZZ.root copy")) as new:
            data = []
            for i in range(2421):
                data.append(np.arange(0, 3, 1))
            data = ak.Array(data)
            new.add_branches("events", {"data": data})

        with uproot.open(
            os.path.join(tmp_path, "uproot-HZZ.root copy"),
            minimal_ttree_metadata=False,
        ) as new:
            for key in control["events"].keys():
                assert key in new["events"].keys()
                assert ak.all(
                    new["events"][key].array() == control["events"][key].array()
                )
            inFile = ROOT.TFile.Open(
                os.path.join(tmp_path, "uproot-HZZ.root copy"), "READ"
            )
            tree = inFile.Get("events")
            indx = 0
            inFile.Close()
            df3 = ROOT.RDataFrame(
                "events", os.path.join(tmp_path, "uproot-HZZ.root copy")
            )
            npy3 = ak.from_rdataframe(df3, columns=("data"), keep_order=True)
            # for key in npy3.keys():
            #     assert ak.all(npy3[key] == control['events'][key].array())
            assert ak.all(npy3 == data)
            inFile.Close()

def test_branch_v8(tmp_path):
    with uproot.open(os.path.join(tmp_path, 'uproot-issue-250.root')) as control:
        with uproot.update(os.path.join(tmp_path, 'uproot-issue-250.root')) as new:
            print("hi")
        # with uproot.open(os.path.join("uproot-from-geant4.root copy")) as new:

# test_vector("/Users/zobil/Desktop/directory/vectorVector")

files = ['uproot-from-geant4.root'] # Values in fBaskets, can't open with uproot.update()

# Try uproot unit tests to generate uproot-events maybe?


import uproot
import os
import pytest

ROOT = pytest.importorskip("ROOT")

import numpy as np

import awkward as ak
from skhep_testdata import data_path


def test_vector(tmp_path):
    data = np.array([1, 2], dtype=np.int64)
    data1 = np.array([2, 3, 4, 5], dtype=np.int64)
    data2 = np.array([3, 4, 5], dtype=np.int64)
    new_branch = np.array([1, 2, 3], dtype=np.int64)

    with uproot.recreate(os.path.join(tmp_path, "vector_test.root")) as file:
        file.mktree(
            "t",
            {
                "b1": ak.Array([data, data1, data2]),
                "b2": ak.Array([data1, data2, data]),
            },
        )
    with uproot.update(os.path.join(tmp_path, "vector_test.root")) as write:
        write.add_branches("t", {"b3": new_branch})

    with uproot.open(
        os.path.join(tmp_path, "vector_test.root"), minimal_ttree_metadata=False
    ) as new:
        assert ak.all(new["t"]["b1"].array()[0] == data)
        assert ak.all(new["t"]["b3"].array() == new_branch)

    inFile = ROOT.TFile.Open(os.path.join(tmp_path, "vector_test.root"), "READ")
    tree = inFile.Get("t;1")
    for i, x in enumerate(tree):
        assert getattr(x, "b3") == new_branch[i]
    inFile.Close()


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
        f.mktree("whatever", {"b1": data, "b2": data1, "b3": data, "b4": data1})

    with uproot.recreate(os.path.join(tmp_path, "arrays2.root")) as f:
        f.mktree("whatever", {"b1": data, "b2": data1})

    with uproot.update(os.path.join(tmp_path, "arrays2.root")) as f:
        f.add_branches("whatever", {"b3": data, "b4": data1})

    with uproot.open(
        os.path.join(tmp_path, "arrays1.root"), minimal_ttree_metadata=False
    ) as check:
        with uproot.open(
            os.path.join(tmp_path, "arrays2.root"), minimal_ttree_metadata=False
        ) as new:
            print(new["whatever"].arrays())
            for key in new["whatever"].keys():
                assert ak.all(
                    new["whatever"].arrays()[key] == check["whatever"].arrays()[key]
                )
            assert ak.all(new["whatever"]["b1"].array() == data)
            assert ak.all(new["whatever"]["b2"].array() == data1)
            assert ak.all(new["whatever"]["b3"].array() == data)
            assert ak.all(new["whatever"]["b4"].array() == data1)
            inFile = ROOT.TFile.Open(os.path.join(tmp_path, "arrays2.root"), "READ")
            tree = inFile.Get("whatever;1")
            indx = 0
            for x in tree:
                assert getattr(x, "b1") == data[indx]
                assert getattr(x, "b2") == data1[indx]
                indx += 1


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

    with uproot.recreate(os.path.join(tmp_path, "mult_trees.root")) as f:
        f.mktree("whatever", {"b1": data, "b2": data1})
        f.mktree("whatever1", {"b1": data, "b2": data1, "b3": data})

    with uproot.update(os.path.join(tmp_path, "mult_trees.root")) as f:
        f.add_branches("whatever", {"b3": data, "b4": data1})
        f.add_branches("whatever1", {"b4": data1})

    with uproot.open(
        os.path.join(tmp_path, "mult_trees.root"), minimal_ttree_metadata=False
    ) as new:
        assert ak.all(new["whatever"]["b1"].array() == data)
        assert ak.all(new["whatever1"]["b4"].array() == data1)
        assert ak.all(new["whatever1"]["b2"].array() == data1)
        assert ak.all(new["whatever1"]["b4"].array() == data1)
        inFile = ROOT.TFile.Open(os.path.join(tmp_path, "mult_trees.root"), "READ")
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
            f.mktree("whatever", {"b1": data, "b2": data1})
            f.add_branches(
                "whatever",
                {
                    "b3": data,
                    "b4": np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.int32),
                },
            )


def test_dtypes(tmp_path):  # tleaf types?
    data = [
        np.array(
            [
                1,
                2,
                3,
                4,
            ],
            dtype=np.int64,
        ),
        np.array(
            [
                1,
                2,
                3,
                4,
            ],
            dtype=np.int32,
        ),
        np.array(
            [
                1,
                2,
                3,
                4,
            ],
            dtype=np.int8,
        ),
        np.array(
            [
                1.0,
                2.0,
                3.0,
                4.0,
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                1.0,
                2.0,
                3.0,
                4.0,
            ],
            dtype=np.float64,
        ),
        np.array(
            [
                1,
                2,
                3,
                4,
            ],
            dtype=np.double,
        ),
        np.array([True, False, True, False], dtype=bool),
    ]

    with uproot.recreate(os.path.join(tmp_path, "all_dtypes.root")) as f:
        f.mktree(
            "whatever",
            {
                "b1": data[0],
                "b2": data[1],
                "b3": data[2],
                "b4": data[3],
                "b5": data[4],
                "b6": data[5],
                "b7": data[6],
            },
        )

    with uproot.update(os.path.join(tmp_path, "all_dtypes.root")) as write:
        write.add_branches(
            "whatever",
            {
                "b8": data[0],
                "b9": data[1],
                "b10": data[2],
                "b12": data[3],
                "b13": data[4],
                "b14": data[5],
                "b15": data[6],
            },
        )

    with uproot.open(os.path.join(tmp_path, "all_dtypes.root")) as read:

        read["whatever"]


def test_ak_arrays(tmp_path):
    data = np.array(
        [
            1,
            2,
        ],
        dtype=np.int64,
    )
    data1 = np.array([2, 3, 4, 5], dtype=np.int64)
    data2 = np.array([3, 4, 5], dtype=np.int64)

    with uproot.recreate(os.path.join(tmp_path, "ak_test.root")) as file:
        file.mktree(
            "whatever",
            {
                "b1": ak.Array([data, data1, data2]),
                "b2": ak.Array([data1, data2, data]),
            },
        )

    with uproot.update(os.path.join(tmp_path, "ak_test.root")) as write:
        write.add_branches(
            "whatever",
            {
                "b3": ak.Array([data2, data, data1]),
            },
        )

    with uproot.open(
        os.path.join(tmp_path, "ak_test.root"), minimal_ttree_metadata=False
    ) as new:
        new["whatever"].arrays()
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
    # Make an example file with ROOT
    inFile = ROOT.TFile(os.path.join(tmp_path, "root_same_dtypes.root"), "RECREATE")
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
        # inFile.ShowStreamerInfo()
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
        ]
        for i in set(file.file.streamers):
            assert i in check
        inFile.Close()


def test_streamers_diff_dtypes(tmp_path):
    # Make an example file with ROOT
    inFile = ROOT.TFile(os.path.join(tmp_path, "root_diff_dtypes.root"), "RECREATE")
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


def test_old_versions(tmp_path):
    with pytest.raises(TypeError):
        with uproot.update(data_path("uproot-HZZ.root")) as file:
            file.add_branches("events", {"b2": [1, 2, 3]})


def test_TreeEventSimple0(tmp_path):
    with uproot.recreate(os.path.join(tmp_path, "TreeEventTreeSimple0.root")) as file:
        file.mktree(
            "TreeEventTreeSimple0",
            {"b1": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], np.int64)},
        )

    with uproot.update(os.path.join(tmp_path, "TreeEventTreeSimple0.root")) as file:
        file.add_branches(
            "TreeEventTreeSimple0",
            {"b2": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], np.int64)},
        )

    with uproot.open(
        os.path.join(tmp_path, "TreeEventTreeSimple0.root"),
        minimal_ttree_metadata=False,
    ) as new:
        assert ak.all(
            new["TreeEventTreeSimple0"]["b2"].array()
            == np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], np.int64)
        )

    inFile = ROOT.TFile.Open(
        os.path.join(tmp_path, "TreeEventTreeSimple0.root"), "READ"
    )
    tree = inFile.Get("TreeEventTreeSimple0;1")
    for i, x in enumerate(tree):
        assert getattr(x, "b2") == np.int64(i + 1)
    inFile.Close()

    df = ROOT.RDataFrame(
        "TreeEventTreeSimple0", os.path.join(tmp_path, "TreeEventTreeSimple0.root")
    )
    npy = ak.from_rdataframe(df, columns=("b1", "b2"))
    assert ak.all(npy["b1"] == np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], np.int64))
    assert ak.all(npy["b2"] == np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], np.int64))


def test_TreeEventSimple1(tmp_path):
    with uproot.recreate(os.path.join(tmp_path, "TreeEventTreeSimple1.root")) as f:
        f.mktree(
            "TreeEventTreeSimple1",
            {"existing_branch": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], np.float32)},
        )

    with uproot.update(os.path.join(tmp_path, "TreeEventTreeSimple1.root")) as file:
        file.add_branches(
            "TreeEventTreeSimple1",
            {"new_v": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], np.float32)},
        )

    with uproot.open(
        os.path.join(tmp_path, "TreeEventTreeSimple1.root"),
        minimal_ttree_metadata=False,
    ) as new:
        assert ak.all(
            new["TreeEventTreeSimple1"]["new_v"].array()
            == np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], np.float32)
        )

    inFile = ROOT.TFile.Open(
        os.path.join(tmp_path, "TreeEventTreeSimple1.root"), "READ"
    )
    tree = inFile.Get("TreeEventTreeSimple1;1")
    for i, x in enumerate(tree):
        assert getattr(x, "new_v") == np.float32(i + 1)
    inFile.Close()

    df = ROOT.RDataFrame(
        "TreeEventTreeSimple1", os.path.join(tmp_path, "TreeEventTreeSimple1.root")
    )
    npy = ak.from_rdataframe(df, columns=("existing_branch", "new_v"))
    assert ak.all(
        npy["existing_branch"] == np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], np.float32)
    )
    assert ak.all(npy["new_v"] == np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], np.float32))


@pytest.mark.xfail(
    reason="ROOT cannot read files with TBranchElement branches modified by uproot - class tag back-references not yet implemented"
)
def test_TreeClass0(tmp_path):
    import shutil

    shutil.copy(
        data_path("uproot-HZZ-objects.root"), os.path.join(tmp_path, "HZZ-objects.root")
    )

    with uproot.update(os.path.join(tmp_path, "HZZ-objects.root")) as f:
        f.add_branches("events", {"new_branch": np.ones(2421, dtype=np.float32)})

    with uproot.open(
        os.path.join(tmp_path, "HZZ-objects.root"), minimal_ttree_metadata=False
    ) as f:
        assert ak.all(
            f["events"]["new_branch"].array() == np.ones(2421, dtype=np.float32)
        )

    inFile = ROOT.TFile.Open(os.path.join(tmp_path, "HZZ-objects.root"), "READ")
    tree = inFile.Get("events;1")
    for i, x in enumerate(tree):
        assert getattr(x, "new_branch") == np.float32(1.0)
    inFile.Close()

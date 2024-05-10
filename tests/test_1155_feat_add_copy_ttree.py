import uproot
from skhep_testdata import data_path
import uproot.serialization
import uproot.writing.writable
import os
import pytest

ROOT = pytest.importorskip("ROOT")

# import ROOT
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
    import ROOT

    data = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    data1 = np.array([2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.int32)

    with uproot.recreate(os.path.join(tmp_path, "arrays1.root")) as f:
        f["whatever"] = {"b1": data}

    with uproot.recreate(os.path.join(tmp_path, "arrays2.root")) as f:
        f["whatever"] = {"b1": data, "b2": data1}

    with uproot.update(os.path.join(tmp_path, "arrays2.root")) as f:
        f.add_branches("whatever", {"b3": data, "b4": data1})

    with uproot.open(
        os.path.join(tmp_path, "arrays1.root"), minimal_ttree_metadata=False
    ) as check:
        # check["tree"].show()
        with uproot.open(
            os.path.join(tmp_path, "arrays2.root"), minimal_ttree_metadata=False
        ) as new:
            print(new.file.chunk(1358, 2677).raw_data.tobytes(), "\n")
            print(check.file.chunk(1358, 2677).raw_data.tobytes())

            inFile = ROOT.TFile.Open(os.path.join(tmp_path, "arrays2.root"), "READ")
            tree = inFile.Get("whatever;1")
            print(tree)
            for x in tree:
                print(getattr(x, "b1"))


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
            "b2": ak.Array([data1, data2, data]),
        }

    with uproot.update(os.path.join(tmp_path, "ak_test.root")) as write:
        write.add_branches(
            "tree",
            {
                "b3": ak.Array([data2, data, data1]),
            },
        )

    with uproot.open(
        os.path.join(tmp_path, "control.root"), minimal_ttree_metadata=False
    ) as correct:
        with uproot.open(
            os.path.join(tmp_path, "ak_test.root"), minimal_ttree_metadata=False
        ) as new:

            import ROOT

            inFile = ROOT.TFile.Open(os.path.join(tmp_path, "ak_test.root"), "READ")
            tree = inFile.Get("tree")
            print(tree.GetBranch("b1"))
            for x in tree:

                print(getattr(x, "b1"))
            print(tree.GetBranch("b2"))
            for x in tree:
                print(getattr(x, "b2"))


def HZZ_test(tmp_path):
    with uproot.open(
        data_path("uproot-HZZ.root"), minimal_ttree_metadata=False
    ) as test:

        # print(test["events"]["NMuon"].typename)
        # print(test["events"])
        # print(test['events'].all_members)

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
            print(check.keys(cycle=False))
            print(check["events"]["data"].array())
            print(check["events"].arrays())
            print(test["events"].arrays())

            for key in test["events"].keys():
                assert key in test["events"].keys()
                assert ak.all(
                    check["events"][key].array() == test["events"][key].array()
                )

            # print(check.file.chunk.start, check['events'].chunk.stop)
            # print(check.file.chunk.get(1000, 2000, check['events'].cursor, context=None).tobytes())

            import ROOT

            inFile = ROOT.TFile.Open(
                os.path.join(tmp_path, "uproot-HZZ.root copy"), "READ"
            )
            tree = inFile.Get("events")

        # print(check["events"]["Photon_Px"].member("fLeaves")[0].member("fLeafCount"))


simple_test("/Users/zobil/Desktop/directory")
# HZZ_test("/Users/zobil/Desktop/directory")
# test_ak_arrays("/Users/zobil/Desktop/directory")

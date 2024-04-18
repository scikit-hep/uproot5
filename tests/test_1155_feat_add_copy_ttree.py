import uproot
from skhep_testdata import data_path
import uproot.writing.writable
from pathlib import Path

# import ROOT
import numpy as np

import awkward as ak


def test_vector():
    with uproot.update(
        "/Users/zobil/Documents/samples/uproot-vectorVectorDouble-work.root"
    ) as write:
        write.add_branches("tree1", {"branch": int}, source="t")

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


def simple_test():
    with uproot.recreate("arrays.root") as file:
        file["tree"] = {"b1": [1, 2, 3], "b2": [2, 3, 4]}

    with uproot.update("arrays.root") as write:
        write.add("tree", {"b3": [5, 6, 7], "b4": [7, 8, 9]}, source="tree")

    with uproot.open("arrays.root", minimal_ttree_metadata=False) as new:
        print(new["tree"].all_members)
        print(new["tree"]["b4"].all_members)
        assert new["tree"].keys() == ["b1", "b2", "b3", "b4"]
        assert ak.all(new["tree"].arrays()["b1"] == [1, 2, 3])
        assert ak.all(new["tree"].arrays()["b2"] == [2, 3, 4])
        assert ak.all(new["tree"].arrays()["b3"] == [5, 6, 7])
        assert ak.all(new["tree"].arrays()["b4"] == [7, 8, 9])


def test_ak_arrays():
    with uproot.recreate("ak_arrays.root") as file:
        file["tree"] = {
            "b1": ak.Array([[1, 2, 3], [1, 2], [6, 7]]),
            "b2": ak.Array([[1, 2, 3], [1, 2], [6, 7, 8]]),
        }
    with uproot.recreate("ak_test.root") as file:
        file["tree"] = {
            "b1": ak.Array([[1, 2, 3], [1, 2], [6, 7]]),
            "b2": ak.Array([[1, 2, 3], [1, 2], [6, 7, 8]]),
            "b3": ak.Array([[5, 4, 5], [6], [7]]),
            "b4": ak.Array([[7], [8], [9]]),
        }
    with uproot.open("ak_test.root", minimal_ttree_metadata=False) as correct:
        with uproot.update("ak_arrays.root") as write:
            write.add(
                "tree",
                {
                    "b3": ak.Array([[5, 4, 5], [6], [7]]),
                    "b4": ak.Array([[7], [8], [9]]),
                },
                source="tree",
            )

        with uproot.open("ak_arrays.root", minimal_ttree_metadata=False) as new:
            print(new["tree"].member("fLeaves")[1])
            print(new["tree"]["b1"].member("fLeaves")[0])
            print(correct["tree"].member("fLeaves")[1])
            print(correct["tree"]["b1"].member("fLeaves")[0])

            assert new["tree"].keys() == correct["tree"].keys()
            assert ak.all(new["tree"]["b1"].array() == correct["tree"]["b1"].array())
            assert ak.all(new["tree"]["b2"].array() == correct["tree"]["b2"].array())
            assert ak.all(new["tree"]["b3"].array() == correct["tree"]["b3"].array())
            assert ak.all(new["tree"]["b4"].array() == correct["tree"]["b4"].array())


def HZZ_test():
    with uproot.open(
        "/Users/zobil/Documents/samples/uproot-HZZ.root", minimal_ttree_metadata=False
    ) as test:

        # print(test["events"]["NMuon"].typename)
        # print(test["events"])
        # print(test['events'].all_members)

        # with uproot.update("/Users/zobil/Documents/samples/uproot-HZZ.root copy") as new:
        #     data = np.arange(0, 2421, 1)
        #     new.add("events", {"data": data}, source="events")

        with uproot.open(
            "/Users/zobil/Documents/samples/uproot-HZZ.root copy",
            minimal_ttree_metadata=False,
        ) as check:
            print(check["events"].arrays())
            print(test["events"].arrays())

        # print(check["events"]["Photon_Px"].member("fLeaves")[0].member("fLeafCount"))


# simple_test()
HZZ_test()
test_ak_arrays()

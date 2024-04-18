import uproot
from skhep_testdata import data_path
import uproot.writing.writable

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


def ak_test():
    with uproot.recreate("ak_arrays.root") as file:
        file["tree"] = {
            "b1": ak.Array([[1, 2, 3], [1, 2], [6, 7]]),
            "b2": ak.Array([[1, 2, 3], [1, 2], [6, 7, 8]]),
        }
        # file.mktree("tree", )

    with uproot.open("ak_arrays.root") as check:
        print(
            "counter",
            check["tree"]["b1"].member("fLeaves")[0].member("fLeafCount").all_members,
        )

    with uproot.update("ak_arrays.root") as write:
        write.add(
            "tree",
            {"b3": ak.Array([[5, 4, 5], [6], [7]]), "b4": ak.Array([[7], [8], [9]])},
            source="tree",
        )

    with uproot.open("ak_arrays.root", minimal_ttree_metadata=False) as new:
        assert new["tree"].keys() == [
            "nb1",
            "b1",
            "nb2",
            "b2",
            "nb3",
            "b3",
            "nb4",
            "b4",
        ]
        assert ak.all(
            new["tree"]["b1"].array() == ak.Array([[1, 2, 3], [1, 2], [6, 7]])
        )
        assert ak.all(
            new["tree"]["b2"].array() == ak.Array([[1, 2, 3], [1, 2], [6, 7, 8]])
        )
        assert ak.all(new["tree"]["b3"].array() == ak.Array([[5, 4, 5], [6], [7]]))
        assert ak.all(new["tree"]["b4"].array() == ak.Array([[7], [8], [9]]))


with uproot.open(
    "/Users/zobil/Documents/samples/uproot-HZZ.root", minimal_ttree_metadata=False
) as test:
    # print(test['events']["Jet_Px"].all_members)
    print(test["events"])
    # print(test['events'].all_members)


# with uproot.update("/Users/zobil/Documents/samples/uproot-HZZ2.root") as test:
#     data = np.arange(0, 2421, 1)
#     test.add("events", {"data": data}, source="events")

with uproot.open("/Users/zobil/Documents/samples/uproot-HZZ2.root") as check:
    print(check["events"].arrays())

# simple_test()
ak_test()

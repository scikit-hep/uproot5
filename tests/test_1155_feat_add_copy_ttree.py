import uproot
from skhep_testdata import data_path
import uproot.serialization
import uproot.writing.writable
import os

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
    data = np.array([1, 2, 3, 4, 5], dtype=np.int8)
    data1 = np.array([2, 3, 4, 5, 6], dtype=np.int8)

    with uproot.recreate(os.path.join(tmp_path, "arrays1.root")) as f:
        f["tree"] = {"b1": data, "b2": data1}

    with uproot.recreate(os.path.join(tmp_path, "arrays2.root")) as f:
        f["tree"] = {"b1": data}

    with uproot.update(os.path.join(tmp_path, "arrays2.root")) as f:
        f.add("tree", {"b2": data1})

    with uproot.open(
        os.path.join(tmp_path, "arrays1.root"), minimal_ttree_metadata=False
    ) as check:
        # check["tree"].show()
        with uproot.open(
            os.path.join(tmp_path, "arrays2.root"), minimal_ttree_metadata=False
        ) as new:
            new_chunk, new_cursor = new.key("tree").get_uncompressed_chunk_cursor()
            check_chenk, check_cursor = check.key(
                "tree"
            ).get_uncompressed_chunk_cursor()
            print("begin", new.file.chunk(22002, new.file.fEND).raw_data.tobytes())
            print(check.file.fEND)
            print(new.file.fEND)
            # print(check['tree'].chunk.raw_data.tobytes(), "\n")
            # print(new['tree'].chunk.raw_data.tobytes())
            # cursor = uproot.source.cursor.Cursor(0)
            # print(check.cursor)
            # print(check.file.source.chunk(
            #     0, 160
            # ).raw_data.tobytes())
            # print(new.file.chunk(0, 100).raw_data.tobytes())
            # print("?",len('root\x00\x00\xf3\xc0\x00\x00\x00d\x00\x00V\x06\x00\x00U\xc4\x00\x00\x00B\x00\x00\x00\x02\x00\x00\x00<\x04\x00\x00\x00e\x00\x00\tn\x00\x00LV\x00\x01\xcd)\xb8|\x06\x0f\x11\xef\x82\xae\xfe\xb1\xc7\x122b\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x'), "\n")

            # print(new['tree'].cursor)
            # print(check['tree'].chunk.start)

            # key = new.key("tree")
            # chunk, cursor = key.get_uncompressed_chunk_cursor()

            # new.file.chunk(new.cursor.index, new.cursor.index+500)
            # new.cursor.debug(new.file.chunk(new.cursor.index, new.cursor.index+500), limit_bytes=1000)

            # print(new["tree"]["b4"].member("fLeaves")[0].all_members)
            assert new["tree"].keys() == ["b1", "b2"]

            print(new.keys())
            print(check["tree"]["b1"].all_members)
            assert ak.all(new["tree"].arrays()["b1"] == [1, 2, 3, 4, 5])

            assert ak.all(new["tree"].arrays()["b2"] == [2, 3, 4, 5, 6])
            # assert ak.all(new["tree"].arrays()["b3"] == [5, 6, 7])
            # assert ak.all(new["tree"].arrays()["b4"] == [7, 8, 9])

            import ROOT

            inFile = ROOT.TFile.Open(os.path.join(tmp_path, "arrays2.root"), "READ")
            tree = inFile.Get("tree")
            # print(tree.GetBranch("b2"))
            # for x in tree:
            #     print(getattr(x, 'b2'))


def test_ak_arrays(tmp_path):
    with uproot.recreate(os.path.join(tmp_path, "control.root")) as file:
        file["tree"] = {
            "b1": ak.Array([[1, 2, 3], [1, 2], [6, 7]]),
            "b2": ak.Array([[1, 2, 3], [1, 2], [6, 7, 8]]),
            "b3": ak.Array([[5, 4, 5], [6], [7]]),
            "b4": ak.Array([[7], [8], [9]]),
        }

    with uproot.recreate(os.path.join(tmp_path, "ak_test.root")) as file:
        file["tree"] = {
            "b1": ak.Array([[1, 2, 3], [1, 2], [6, 7]]),
            "b2": ak.Array([[1, 2, 3], [1, 2], [6, 7, 8]]),
        }

    with uproot.update(os.path.join(tmp_path, "ak_test.root")) as write:
        write.add(
            "tree",
            {
                "b3": ak.Array([[5, 4, 5], [6], [7]]),
                "b4": ak.Array([[7], [8], [9]]),
            },
        )

    with uproot.open(
        os.path.join(tmp_path, "control.root"), minimal_ttree_metadata=False
    ) as correct:
        with uproot.open(
            os.path.join(tmp_path, "ak_test.root"), minimal_ttree_metadata=False
        ) as new:
            print(new.file.show_streamers("TLeafL"))
            # print(new['tree']['b1'].member("fLeaves")[0].member("fName"))
            # print(correct["tree"].show())
            # print(new['tree'].chunk.raw_data.tobytes())
            # print(correct["tree"]["b1"].member("fLeaves")[0])
            # print(correct.file.chunk(correct.file.fSeekInfo, correct.file.fEND).raw_data.tobytes())
            # correct.file.show_streamers()

            # key = new.key("tree")
            # chunk, cursor = key.get_uncompressed_chunk_cursor()
            # cursor.debug(chunk, limit_bytes=1000)
            # print("...")
            # key = correct.key("tree")
            # chunk, cursor = key.get_uncompressed_chunk_cursor()
            # cursor.debug(chunk, limit_bytes=1000)

            # assert new["tree"].keys() == correct["tree"].keys()
            # assert ak.all(new["tree"]["b1"].array() == correct["tree"]["b1"].array())
            # assert ak.all(new["tree"]["b2"].array() == correct["tree"]["b2"].array())
            # assert ak.all(new["tree"]["b3"].array() == correct["tree"]["b3"].array())
            # assert ak.all(new["tree"]["b4"].array() == correct["tree"]["b4"].array())
            import ROOT

            inFile = ROOT.TFile.Open(os.path.join(tmp_path, "ak_test.root"), "READ")
            tree = inFile.Get("tree")


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
            new.add("events", {"data": data})

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

            # print(check['events'].chunk.start, check['events'].chunk.stop)
            # print(check['events'].chunk.get(1000, 2000, check['events'].cursor, context=None).tobytes())

            import ROOT

            inFile = ROOT.TFile.Open(
                os.path.join(tmp_path, "uproot-HZZ.root copy"), "READ"
            )
            tree = inFile.Get("events")

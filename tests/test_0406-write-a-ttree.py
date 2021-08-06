# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

import array
import os

import numpy as np
import pytest

import uproot
import uproot.writing

ROOT = pytest.importorskip("ROOT")


# def test(tmp_path):
#     # original = os.path.join(tmp_path, "original.root")
#     # newfile = os.path.join(tmp_path, "newfile.root")
#     original = "original.root"
#     newfile = "newfile.root"

#     f1 = ROOT.TFile(original, "recreate")
#     t1 = ROOT.TTree("t1", "title")
#     d1 = array.array("d", [0.0])
#     t1.Branch("branch", d1, "leaf/D")

#     # d1[0] = 0.0
#     # t1.Fill()
#     # d1[0] = 1.1
#     # t1.Fill()
#     # d1[0] = 2.2
#     # t1.Fill()
#     # d1[0] = 3.3
#     # t1.Fill()
#     # d1[0] = 4.4
#     # t1.Fill()
#     # d1[0] = 5.5
#     # t1.Fill()
#     # d1[0] = 6.6
#     # t1.Fill()
#     # d1[0] = 7.7
#     # t1.Fill()
#     # d1[0] = 8.8
#     # t1.Fill()
#     # d1[0] = 9.9
#     # t1.Fill()

#     t1.Write()
#     f1.Close()


def test(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    with uproot.recreate(newfile) as fout:
        tree = fout._cascading.add_tree(fout._file.sink, "name", "title", {})

        print(tree)

        print(tree.key.location + tree.key.num_bytes + 17)
        data = fout._file.sink.read(
            tree.key.location + tree.key.num_bytes, tree.key.compressed_bytes
        )

        print(data)
        print(np.frombuffer(data, "u1"))

        tree.change(fout._file.sink, b"whatever")
        data = fout._file.sink.read(
            tree.key.location + tree.key.num_bytes, tree.key.compressed_bytes
        )

        print()
        print(data)
        print(np.frombuffer(data, "u1"))

        raw_data = uproot.writing.to_TObjString("even longer string thing").serialize(
            "name"
        )

        new_key = fout._cascading.add_object(
            fout._file.sink,
            "TObjString",
            "name",
            "title",
            raw_data,
            len(raw_data),
            replaces=tree.key,
        )

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import os

import uproot


def test_mktree_repeated_same_subdir(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    with uproot.recreate(newfile) as f:
        f.mktree("tree/x", {"x": [1, 2, 3]})
        f.mktree("tree/y", {"x": [1, 2, 3]})
        f.mktree("tree/z", {"x": [1, 2, 3]})
        f.mktree("tree/w", {"x": [1, 2, 3]})

        assert set(f.keys(cycle=False)) == {
            "tree",
            "tree/x",
            "tree/y",
            "tree/z",
            "tree/w",
        }

    with uproot.open(newfile) as fin:
        assert set(fin.keys(cycle=False)) == {
            "tree",
            "tree/x",
            "tree/y",
            "tree/z",
            "tree/w",
        }


def test_mkrntuple_repeated_same_subdir(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    with uproot.recreate(newfile) as f:
        f.mkrntuple("ntpl/x", {"x": [1, 2, 3]})
        f.mkrntuple("ntpl/y", {"x": [1, 2, 3]})
        f.mkrntuple("ntpl/z", {"x": [1, 2, 3]})
        f.mkrntuple("ntpl/w", {"x": [1, 2, 3]})

        assert set(f.keys(cycle=False)) == {
            "ntpl",
            "ntpl/x",
            "ntpl/y",
            "ntpl/z",
            "ntpl/w",
        }

    with uproot.open(newfile) as fin:
        assert set(fin.keys(cycle=False)) == {
            "ntpl",
            "ntpl/x",
            "ntpl/y",
            "ntpl/z",
            "ntpl/w",
        }

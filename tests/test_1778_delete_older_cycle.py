# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import os

import numpy as np

import uproot


def hist(values):
    return (
        np.array(values, dtype=np.float64),
        np.arange(len(values) + 1, dtype=np.float64),
    )


def test_delete_older_cycle(tmp_path):
    filename = os.path.join(tmp_path, "cycles.root")
    with uproot.recreate(filename) as f:
        f["h"] = hist([1.0, 2.0, 3.0])
        f["h"] = hist([4.0, 5.0, 6.0])

    with uproot.update(filename) as f:
        del f["h;1"]
        assert f.keys() == ["h;2"]
        assert "h" in f

    with uproot.open(filename) as f:
        assert f.keys() == ["h;2"]
        assert f["h;2"].values().tolist() == [4.0, 5.0, 6.0]


def test_delete_only_cycle(tmp_path):
    filename = os.path.join(tmp_path, "single.root")
    with uproot.recreate(filename) as f:
        f["h"] = hist([1.0])
        del f["h"]
        assert f.keys() == []
        assert "h" not in f

    with uproot.open(filename) as f:
        assert f.keys() == []


def test_extend_older_tree_cycle(tmp_path):
    filename = os.path.join(tmp_path, "tree.root")
    with uproot.recreate(filename) as f:
        f.mktree("t", {"x": np.int32})
        f.mktree("t", {"x": np.int32})
        t1 = f["t;1"]
        for _ in range(20):
            t1.extend({"x": np.arange(3, dtype=np.int32)})

    with uproot.open(filename) as f:
        assert f["t;1"].num_entries == 60
        assert f["t;2"].num_entries == 0

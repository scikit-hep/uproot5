# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations

import os
import shutil

import awkward as ak
import numpy as np
import pytest
import skhep_testdata

import uproot


def test_add_branch_to_root_written_divergent_tree(tmp_path):
    orig_path = skhep_testdata.data_path("uproot-HZZ.root")
    path = os.path.join(tmp_path, "hzz.root")
    shutil.copyfile(orig_path, path)

    with uproot.update(path) as f:
        events = f["events"]
        assert events._cascading._has_divergent_baskets
        n = events.num_entries
        events.add_branches({"new_scalar": np.arange(n, dtype=np.float32)})

    orig_f = uproot.open(orig_path)
    mod_f = uproot.open(path)
    orig_events = orig_f["events"]
    mod_events = mod_f["events"]

    assert mod_events.num_entries == 2421
    assert np.array_equal(
        mod_events["new_scalar"].array(library="np"),
        np.arange(2421, dtype=np.float32),
    )

    for key in orig_events.keys():
        orig_arr = orig_events[key].array()
        mod_arr = mod_events[key].array()
        if isinstance(orig_arr, ak.Array):
            assert ak.all(orig_arr == mod_arr)
        else:
            assert np.array_equal(orig_arr, mod_arr)


def test_add_jagged_branch_to_root_written_tree(tmp_path):
    orig_path = skhep_testdata.data_path("uproot-HZZ.root")
    path = os.path.join(tmp_path, "hzz_jagged.root")
    shutil.copyfile(orig_path, path)

    data = ak.Array(
        [[float(i), float(i + 1)] if i % 2 == 0 else [] for i in range(2421)]
    )

    with uproot.update(path) as f:
        events = f["events"]
        events.add_branches({"new_jagged": data})

    with uproot.open(path) as f:
        events = f["events"]
        assert "new_jagged" in events
        assert "nnew_jagged" in events
        assert ak.all(events["new_jagged"].array() == data)
        assert np.array_equal(
            events["nnew_jagged"].array(library="np"),
            ak.num(data, axis=1).to_numpy(),
        )
        orig_muon_px = uproot.open(orig_path)["events"]["Muon_Px"].array()
        assert ak.all(events["Muon_Px"].array() == orig_muon_px)


def test_add_jagged_branch_uproot_written_tree(tmp_path):
    path = os.path.join(tmp_path, "test.root")
    with uproot.recreate(path) as f:
        f.mktree("tree", {"x": "int32"})
        f["tree"].extend({"x": np.array([10, 20, 30], dtype="int32")})

    with uproot.update(path) as f:
        f["tree"].add_branches({"j": ak.Array([[1.0], [], [2.0, 3.0]])})

    with uproot.open(path) as f:
        tree = f["tree"]
        assert np.array_equal(tree["x"].array(library="np"), np.array([10, 20, 30]))
        assert np.array_equal(tree["nj"].array(library="np"), np.array([1, 0, 2]))
        assert ak.all(tree["j"].array() == ak.Array([[1.0], [], [2.0, 3.0]]))


def test_add_jagged_branch_with_custom_counter_name(tmp_path):
    path = os.path.join(tmp_path, "test_custom_counter.root")
    with uproot.recreate(path) as f:
        f.mktree("tree", {"x": "int32"})
        f["tree"].extend({"x": np.array([1, 2, 3], dtype="int32")})

    with uproot.update(path) as f:
        f["tree"].add_branches(
            {"j": ak.Array([[1.0], [], [2.0, 3.0]])},
            counter_name=lambda b: f"num_{b}",
        )

    with uproot.open(path) as f:
        tree = f["tree"]
        assert "num_j" in tree
        assert np.array_equal(tree["num_j"].array(library="np"), np.array([1, 0, 2]))
        assert ak.all(tree["j"].array() == ak.Array([[1.0], [], [2.0, 3.0]]))


def test_add_jagged_branch_with_explicit_matching_counter(tmp_path):
    path = os.path.join(tmp_path, "test_explicit.root")
    with uproot.recreate(path) as f:
        f.mktree("tree", {"x": "int32"})
        f["tree"].extend({"x": np.array([1, 2, 3], dtype="int32")})

    with uproot.update(path) as f:
        f["tree"].add_branches(
            {
                "nj": np.array([1, 0, 2], dtype=np.int32),
                "j": ak.Array([[10.0], [], [20.0, 30.0]]),
            }
        )

    with uproot.open(path) as f:
        tree = f["tree"]
        assert np.array_equal(tree["nj"].array(library="np"), np.array([1, 0, 2]))
        assert ak.all(tree["j"].array() == ak.Array([[10.0], [], [20.0, 30.0]]))


def test_add_jagged_branch_with_explicit_disagreeing_counter_raises(tmp_path):
    path = os.path.join(tmp_path, "test_disagree.root")
    with uproot.recreate(path) as f:
        f.mktree("tree", {"x": "int32"})
        f["tree"].extend({"x": np.array([1, 2, 3], dtype="int32")})

    with uproot.update(path) as f, pytest.raises(ValueError, match="disagree"):
        f["tree"].add_branches(
            {
                "nj": np.array([9, 9, 9], dtype=np.int32),
                "j": ak.Array([[10.0], [], [20.0, 30.0]]),
            }
        )


def test_add_jagged_branch_wrong_length_raises(tmp_path):
    path = os.path.join(tmp_path, "test_wrong_len.root")
    with uproot.recreate(path) as f:
        f.mktree("tree", {"x": "int32"})
        f["tree"].extend({"x": np.array([1, 2, 3], dtype="int32")})

    with (
        uproot.update(path) as f,
        pytest.raises(ValueError, match="all new branches must match the tree length"),
    ):
        f["tree"].add_branches({"j": ak.Array([[1.0], [2.0]])})


def test_add_multiple_jagged_and_flat_branches(tmp_path):
    path = os.path.join(tmp_path, "test_mult.root")
    with uproot.recreate(path) as f:
        f.mktree("tree", {"x": "int32"})
        f["tree"].extend({"x": np.array([1, 2, 3], dtype="int32")})

    with uproot.update(path) as f:
        f["tree"].add_branches(
            {
                "y": np.array([4.0, 5.0, 6.0], dtype=np.float64),
                "j": ak.Array([[1.5], [], [2.5, 3.5]]),
            }
        )

    with uproot.open(path) as f:
        tree = f["tree"]
        assert np.array_equal(tree["x"].array(library="np"), np.array([1, 2, 3]))
        assert np.array_equal(tree["y"].array(library="np"), np.array([4.0, 5.0, 6.0]))
        assert np.array_equal(tree["nj"].array(library="np"), np.array([1, 0, 2]))
        assert ak.all(tree["j"].array() == ak.Array([[1.5], [], [2.5, 3.5]]))


def test_add_branches_sequential_sessions(tmp_path):
    orig_path = skhep_testdata.data_path("uproot-HZZ.root")
    path = os.path.join(tmp_path, "hzz_seq.root")
    shutil.copyfile(orig_path, path)

    with uproot.update(path) as f:
        f["events"].add_branches({"first": np.zeros(2421, dtype=np.float32)})

    with uproot.update(path) as f:
        f["events"].add_branches({"second": np.ones(2421, dtype=np.int64)})

    with uproot.open(path) as f:
        events = f["events"]
        assert "first" in events
        assert "second" in events
        assert np.array_equal(
            events["first"].array(library="np"), np.zeros(2421, dtype=np.float32)
        )
        assert np.array_equal(
            events["second"].array(library="np"), np.ones(2421, dtype=np.int64)
        )
        assert ak.all(
            events["Muon_Px"].array()
            == uproot.open(orig_path)["events"]["Muon_Px"].array()
        )

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations

import os
import shutil

import numpy as np
import pytest
import skhep_testdata

import uproot

ak = pytest.importorskip("awkward")


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


def test_empty_tree_add_jagged_branch_then_extend(tmp_path):
    path = os.path.join(tmp_path, "test_empty_jagged.root")
    with uproot.recreate(path) as f:
        f.mktree("t", {"a": np.int64})
    with uproot.update(path) as f:
        f["t"].add_branches({"j": ak.Array([[1.0]])[:0]})
        f["t"].extend(
            {
                "a": np.array([1, 2], dtype=np.int64),
                "j": ak.Array([[10.0, 20.0], [30.0]]),
            }
        )
    with uproot.open(path) as f:
        tree = f["t"]
        assert np.array_equal(tree["a"].array(library="np"), np.array([1, 2]))
        assert np.array_equal(tree["nj"].array(library="np"), np.array([2, 1]))
        assert ak.all(tree["j"].array() == ak.Array([[10.0, 20.0], [30.0]]))


def test_add_branches_sequential_sessions(tmp_path):
    orig_path = skhep_testdata.data_path("uproot-HZZ.root")
    path = os.path.join(tmp_path, "hzz_seq.root")
    shutil.copyfile(orig_path, path)

    with uproot.update(path) as f:
        f["events"].add_branches({"first": np.zeros(2421, dtype=np.float32)})

    with uproot.update(path) as f:
        f["events"].add_branches(
            {
                "second_jagged": ak.Array(
                    [[float(i)] if i % 2 == 0 else [] for i in range(2421)]
                )
            }
        )

    with uproot.open(path) as f:
        events = f["events"]
        assert "first" in events
        assert "second_jagged" in events
        assert "nsecond_jagged" in events
        assert np.array_equal(
            events["first"].array(library="np"), np.zeros(2421, dtype=np.float32)
        )
        assert np.array_equal(
            events["nsecond_jagged"].array(library="np"),
            [1 if i % 2 == 0 else 0 for i in range(2421)],
        )
        assert ak.all(
            events["Muon_Px"].array()
            == uproot.open(orig_path)["events"]["Muon_Px"].array()
        )


def test_add_jagged_branch_with_string_counter_name_raises(tmp_path):
    path = os.path.join(tmp_path, "test_str_counter.root")
    with uproot.recreate(path) as f:
        f.mktree("tree", {"x": "int32"})
        f["tree"].extend({"x": np.array([1, 2, 3], dtype="int32")})

    with (
        uproot.update(path) as f,
        pytest.raises(TypeError, match="counter_name must be a callable or None"),
    ):
        f["tree"].add_branches(
            {"j": ak.Array([[1.0], [], [2.0, 3.0]])},
            counter_name="count_{}",
        )

    with (
        uproot.update(path) as f,
        pytest.raises(TypeError, match="counter_name must be a callable or None"),
    ):
        f["tree"].add_branches(
            {"j": ak.Array([[1.0], [], [2.0, 3.0]])},
            counter_name="len_",
        )


def test_add_jagged_branch_invalid_counter_name_raises(tmp_path):
    path = os.path.join(tmp_path, "test_bad_counter.root")
    with uproot.recreate(path) as f:
        f.mktree("tree", {"x": "int32"})
        f["tree"].extend({"x": np.array([1, 2, 3], dtype="int32")})

    with uproot.update(path) as f, pytest.raises(TypeError, match="counter_name"):
        f["tree"].add_branches(
            {"j": ak.Array([[1.0], [], [2.0, 3.0]])},
            counter_name=12345,
        )


def test_add_jagged_branch_counter_collides_with_normal_branch_raises(tmp_path):
    path = os.path.join(tmp_path, "test_collision.root")
    with uproot.recreate(path) as f:
        f.mktree("tree", {"x": "int32", "nj": "float32"})
        f["tree"].extend(
            {
                "x": np.array([1, 2, 3], dtype="int32"),
                "nj": np.array([1.1, 2.2, 3.3], dtype="float32"),
            }
        )

    with uproot.update(path) as f, pytest.raises(ValueError, match="collides"):
        f["tree"].add_branches({"j": ak.Array([[1.0], [], [2.0, 3.0]])})


def test_add_multiple_jagged_branches_sharing_counter(tmp_path):
    path = os.path.join(tmp_path, "test_shared_counter.root")
    with uproot.recreate(path) as f:
        f.mktree("tree", {"x": "int32"})
        f["tree"].extend({"x": np.array([1, 2, 3], dtype="int32")})

    with uproot.update(path) as f:
        f["tree"].add_branches(
            {
                "j1": ak.Array([[1.0], [], [2.0, 3.0]]),
                "j2": ak.Array([[10.0], [], [20.0, 30.0]]),
            },
            counter_name=lambda _: "n_particles",
        )

    with uproot.open(path) as f:
        tree = f["tree"]
        assert "n_particles" in tree
        assert np.array_equal(
            tree["n_particles"].array(library="np"), np.array([1, 0, 2])
        )
        assert ak.all(tree["j1"].array() == ak.Array([[1.0], [], [2.0, 3.0]]))
        assert ak.all(tree["j2"].array() == ak.Array([[10.0], [], [20.0, 30.0]]))


def test_add_multiple_jagged_branches_sharing_counter_disagree_raises(tmp_path):
    path = os.path.join(tmp_path, "test_shared_disagree.root")
    with uproot.recreate(path) as f:
        f.mktree("tree", {"x": "int32"})
        f["tree"].extend({"x": np.array([1, 2, 3], dtype="int32")})

    with uproot.update(path) as f, pytest.raises(ValueError, match="disagrees"):
        f["tree"].add_branches(
            {
                "j1": ak.Array([[1.0], [], [2.0, 3.0]]),
                "j2": ak.Array([[10.0, 20.0], [], [30.0]]),
            },
            counter_name=lambda _: "n_particles",
        )


def test_add_jagged_branch_sliced_indexed_array(tmp_path):
    path = os.path.join(tmp_path, "test_sliced_indexed.root")
    with uproot.recreate(path) as f:
        f.mktree("tree", {"x": "int32"})
        f["tree"].extend({"x": np.array([1, 2], dtype="int32")})

    big_arr = ak.Array([[1.0, 2.0], [], [3.0], [4.0, 5.0]])[::2]
    with uproot.update(path) as f:
        f["tree"].add_branches({"j": big_arr})

    with uproot.open(path) as f:
        tree = f["tree"]
        assert ak.all(tree["j"].array() == big_arr)
        assert np.array_equal(tree["nj"].array(library="np"), np.array([2, 1]))


def test_add_branch_flat_object_dtype_raises(tmp_path):
    path = os.path.join(tmp_path, "test_obj_dtype.root")
    with uproot.recreate(path) as f:
        f.mktree("tree", {"x": "int32"})
        f["tree"].extend({"x": np.array([1, 2, 3], dtype="int32")})

    with uproot.update(path) as f, pytest.raises(TypeError, match="object dtype"):
        f["tree"].add_branches({"bad": np.array(["a", object(), 123], dtype=object)})


def test_add_nested_jagged_branch_raises(tmp_path):
    path = os.path.join(tmp_path, "test_nested.root")
    with uproot.recreate(path) as f:
        f.mktree("tree", {"x": "int32"})
        f["tree"].extend({"x": np.array([1, 2, 3], dtype="int32")})

    with (
        uproot.update(path) as f,
        pytest.raises(TypeError, match="cannot write Awkward Array type to ROOT file"),
    ):
        f["tree"].add_branches({"j": ak.Array([[[1.0]], [], [[2.0], [3.0]]])})

    with uproot.open(path) as f:
        tree = f["tree"]
        assert tree.keys() == ["x"]
        assert np.array_equal(tree["x"].array(library="np"), np.array([1, 2, 3]))


def test_add_jagged_string_branch_raises(tmp_path):
    path = os.path.join(tmp_path, "test_jagged_str.root")
    with uproot.recreate(path) as f:
        f.mktree("tree", {"x": "int32"})
        f["tree"].extend({"x": np.array([1, 2, 3], dtype="int32")})

    with (
        uproot.update(path) as f,
        pytest.raises(TypeError, match="cannot write Awkward Array type to ROOT file"),
    ):
        f["tree"].add_branches({"j": ak.Array([["a"], [], ["b", "c"]])})

    with uproot.open(path) as f:
        tree = f["tree"]
        assert tree.keys() == ["x"]
        assert np.array_equal(tree["x"].array(library="np"), np.array([1, 2, 3]))


def test_add_branches_failure_does_not_corrupt_casc(tmp_path):
    path = os.path.join(tmp_path, "test_rollback.root")
    with uproot.recreate(path) as f:
        f.mktree("tree", {"x": "int32"})
        f["tree"].extend({"x": np.array([1, 2, 3], dtype="int32")})

    with uproot.update(path) as f:
        tree = f["tree"]
        with pytest.raises(TypeError, match="cannot write NumPy dtype"):
            tree.add_branches(
                {
                    "j": ak.Array([[1.0], [], [2.0]]),
                    "s": np.array(["x", "y", "z"]),
                }
            )
        # casc must be cleanly rolled back: phantom branches must not linger
        tree.extend({"x": np.array([4, 5], dtype="int32")})

    with uproot.open(path) as f:
        tree = f["tree"]
        assert tree.keys() == ["x"]
        assert np.array_equal(tree["x"].array(library="np"), np.array([1, 2, 3, 4, 5]))


def test_add_jagged_branch_shares_existing_counter_mismatch_raises(tmp_path):
    path = os.path.join(tmp_path, "test_existing_counter.root")
    with uproot.recreate(path) as f:
        f.mktree("tree", {"x": "int32"})
        f["tree"].extend({"x": np.array([1, 2, 3], dtype="int32")})

    with uproot.update(path) as f:
        f["tree"].add_branches({"x_jag": ak.Array([[1], [], [2, 3]])})

    # sharing existing counter nx_jag with mismatching lengths must raise ValueError
    with (
        uproot.update(path) as f,
        pytest.raises(
            ValueError, match="disagrees with the existing counter branch's lengths"
        ),
    ):
        f["tree"].add_branches(
            {"y_jag": ak.Array([[1, 2, 3], [4], []])},
            counter_name=lambda n: "nx_jag",
        )

    # sharing existing counter nx_jag with matching lengths must succeed
    with uproot.update(path) as f:
        f["tree"].add_branches(
            {"y_jag": ak.Array([[10.0], [], [20.0, 30.0]])},
            counter_name=lambda n: "nx_jag",
        )

    with uproot.open(path) as f:
        tree = f["tree"]
        assert "y_jag" in tree
        assert ak.all(tree["y_jag"].array() == ak.Array([[10.0], [], [20.0, 30.0]]))


def test_add_branches_custom_counter_name_does_not_leak_to_extend(tmp_path):
    path = os.path.join(tmp_path, "test_counter_leak.root")
    with uproot.recreate(path) as f:
        f.mktree("tree", {"x": "int32"})
        f["tree"].extend({"x": np.array([1, 2, 3], dtype="int32")})

    with uproot.update(path) as f:
        tree = f["tree"]
        tree.add_branches({"x_jag": ak.Array([[1.0], [], [2.0, 3.0]])})
        tree.add_branches(
            {"y_jag": ak.Array([[10.0], [], [20.0, 30.0]])},
            counter_name=lambda n: f"{n}_cnt",
        )
        # extending existing jagged branch must still work with its own counter
        tree.extend(
            {
                "x": np.array([4], dtype="int32"),
                "x_jag": ak.Array([[4.0]]),
                "y_jag": ak.Array([[40.0]]),
            }
        )

    with uproot.open(path) as f:
        tree = f["tree"]
        assert np.array_equal(tree["x"].array(library="np"), np.array([1, 2, 3, 4]))
        assert ak.all(tree["x_jag"].array() == ak.Array([[1.0], [], [2.0, 3.0], [4.0]]))
        assert ak.all(
            tree["y_jag"].array() == ak.Array([[10.0], [], [20.0, 30.0], [40.0]])
        )


def test_add_jagged_branch_explicit_float_counter_converted_to_int32(tmp_path):
    path = os.path.join(tmp_path, "test_float_counter.root")
    with uproot.recreate(path) as f:
        f.mktree("tree", {"x": "int32"})
        f["tree"].extend({"x": np.array([1, 2, 3], dtype="int32")})

    with uproot.update(path) as f:
        f["tree"].add_branches(
            {
                "j": ak.Array([[1.0], [], [2.0, 3.0]]),
                "nj": np.array([1.0, 0.0, 2.0]),
            }
        )

    with uproot.open(path) as f:
        tree = f["tree"]
        assert tree["nj"].interpretation.to_dtype == np.dtype("int32")
        assert tree["nj"].array(library="np").dtype == np.dtype("int32")
        assert tree["nj"].title == "nj/I"
        assert np.array_equal(tree["nj"].array(library="np"), np.array([1, 0, 2]))
        assert ak.all(tree["j"].array() == ak.Array([[1.0], [], [2.0, 3.0]]))


def test_add_multiple_explicit_counters_deterministic_order(tmp_path):
    path = os.path.join(tmp_path, "test_det_order.root")
    with uproot.recreate(path) as f:
        f.mktree("tree", {"x": "int32"})
        f["tree"].extend({"x": np.array([1, 2], dtype="int32")})

    with uproot.update(path) as f:
        f["tree"].add_branches(
            {
                "c1": np.array([1, 2], dtype=np.int32),
                "j1": ak.Array([[1.0], [2.0, 3.0]]),
                "c2": np.array([2, 1], dtype=np.int32),
                "j2": ak.Array([[10.0, 20.0], [30.0]]),
            },
            counter_name=lambda b: "c1" if b == "j1" else "c2",
        )

    with uproot.open(path) as f:
        tree = f["tree"]
        assert tree.keys() == ["x", "c1", "c2", "j1", "j2"]

import os
import shutil

import numpy as np
import pytest

import uproot
from skhep_testdata import data_path

try:
    import ROOT

    has_root = True
except ImportError:
    has_root = False

skip_no_root = pytest.mark.skipif(not has_root, reason="ROOT is not installed")


# ── add_branches tests ────────────────────────────────────────────────────────


def test_add_branch_simple(tmp_path):
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f.mktree("tree", {"x": np.float32})
        f["tree"].extend({"x": np.ones(100, dtype=np.float32)})

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        f["tree"].add_branches({"new_branch": np.ones(100, dtype=np.float32)})

    with uproot.open(os.path.join(tmp_path, "test.root")) as f:
        assert len(f["tree"].branches) == 2
        assert "new_branch" in [b.name for b in f["tree"].branches]
        assert np.all(f["tree"]["new_branch"].array() == 1.0)


def test_add_branch_multiple_branches(tmp_path):
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f.mktree("tree", {"x": np.float32})
        f["tree"].extend({"x": np.ones(100, dtype=np.float32)})

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        f["tree"].add_branches(
            {
                "branch_a": np.ones(100, dtype=np.float32),
                "branch_b": np.zeros(100, dtype=np.int32),
            }
        )

    with uproot.open(os.path.join(tmp_path, "test.root")) as f:
        assert len(f["tree"].branches) == 3
        assert np.all(f["tree"]["branch_a"].array() == 1.0)
        assert np.all(f["tree"]["branch_b"].array() == 0)


def test_add_branch_int32(tmp_path):
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f.mktree("tree", {"x": np.float32})
        f["tree"].extend({"x": np.ones(100, dtype=np.float32)})

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        f["tree"].add_branches({"new_int": np.arange(100, dtype=np.int32)})

    with uproot.open(os.path.join(tmp_path, "test.root")) as f:
        assert np.all(f["tree"]["new_int"].array() == np.arange(100, dtype=np.int32))


def test_add_branch_preserves_existing(tmp_path):
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f.mktree("tree", {"x": np.float32, "y": np.int32})
        f["tree"].extend(
            {
                "x": np.arange(100, dtype=np.float32),
                "y": np.arange(100, dtype=np.int32),
            }
        )

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        f["tree"].add_branches({"new_branch": np.ones(100, dtype=np.float32)})

    with uproot.open(os.path.join(tmp_path, "test.root")) as f:
        assert np.all(f["tree"]["x"].array() == np.arange(100, dtype=np.float32))
        assert np.all(f["tree"]["y"].array() == np.arange(100, dtype=np.int32))
        assert np.all(f["tree"]["new_branch"].array() == 1.0)


def test_add_branch_tbranchelement(tmp_path):
    # add_branches for TBranchElement files is not supported
    # due to internal reference numbers that break when blob is rewritten
    shutil.copy(
        data_path("uproot-HZZ-objects.root"), os.path.join(tmp_path, "HZZ.root")
    )

    with uproot.update(os.path.join(tmp_path, "HZZ.root")) as f:
        with pytest.raises(Exception):
            f["events"].add_branches({"new_branch": np.ones(2421, dtype=np.float32)})


def test_add_branch_wrong_length(tmp_path):
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f.mktree("tree", {"x": np.float32})
        f["tree"].extend({"x": np.ones(100, dtype=np.float32)})

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        with pytest.raises(ValueError, match="entries"):
            f["tree"].add_branches({"new_branch": np.ones(50, dtype=np.float32)})


def test_add_branch_nonexistent_tree(tmp_path):
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f.mktree("tree", {"x": np.float32})
        f["tree"].extend({"x": np.ones(100, dtype=np.float32)})

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        with pytest.raises(Exception):
            f["nonexistent"].add_branches(
                {"new_branch": np.ones(100, dtype=np.float32)}
            )


@skip_no_root
def test_add_branch_root_readable(tmp_path):
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f.mktree("tree", {"x": np.float32})
        f["tree"].extend({"x": np.ones(100, dtype=np.float32)})

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        f["tree"].add_branches({"new_branch": np.ones(100, dtype=np.float32)})

    f = ROOT.TFile.Open(str(os.path.join(tmp_path, "test.root")), "READ")
    tree = f.Get("tree;1")
    tree.SetCacheSize(0)
    tree.GetEntry(0)
    assert tree.new_branch == pytest.approx(1.0)
    f.Close()


@skip_no_root
def test_add_branch_tbranchelement_root_readable(tmp_path):
    shutil.copy(
        data_path("uproot-HZZ-objects.root"), os.path.join(tmp_path, "HZZ.root")
    )

    with uproot.update(os.path.join(tmp_path, "HZZ.root")) as f:
        with pytest.raises((NotImplementedError, TypeError, KeyError)):
            f["events"].add_branches({"new_branch": np.ones(2421, dtype=np.float32)})


# ── extend tests ──────────────────────────────────────────────────────────────


def test_extend_simple(tmp_path):
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f.mktree("tree", {"x": np.float32, "y": np.int32})
        f["tree"].extend(
            {"x": np.ones(100, dtype=np.float32), "y": np.zeros(100, dtype=np.int32)}
        )

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        f["tree"].extend(
            {
                "x": np.ones(50, dtype=np.float32) * 2,
                "y": np.ones(50, dtype=np.int32) * 3,
            }
        )

    with uproot.open(os.path.join(tmp_path, "test.root")) as f:
        assert f["tree"].member("fEntries") == 150
        assert np.all(f["tree"]["x"].array()[:100] == 1.0)
        assert np.all(f["tree"]["x"].array()[100:] == 2.0)
        assert np.all(f["tree"]["y"].array()[:100] == 0)
        assert np.all(f["tree"]["y"].array()[100:] == 3)


def test_extend_preserves_existing(tmp_path):
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f.mktree("tree", {"x": np.float32})
        f["tree"].extend({"x": np.arange(100, dtype=np.float32)})

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        f["tree"].extend({"x": np.arange(100, dtype=np.float32) + 100})

    with uproot.open(os.path.join(tmp_path, "test.root")) as f:
        arr = f["tree"]["x"].array()
        assert len(arr) == 200
        assert np.all(arr[:100] == np.arange(100, dtype=np.float32))
        assert np.all(arr[100:] == np.arange(100, dtype=np.float32) + 100)


def test_extend_zero_basket_tree(tmp_path):
    """Extending a freshly-mktree'd tree (no baskets written yet) via uproot.update().

    Regression test: _load_existing_ttree used to locate the TTree's own
    fEntries/fTotBytes/fZipBytes metadata, and each branch's
    fBasketBytes/fBasketEntry/fBasketSeek arrays, by searching for the raw
    bytes of their current (0, for a just-created tree) values. A search for
    a run of zero bytes matches arbitrary unrelated data elsewhere in the
    tree, corrupting the rewritten file instead of raising.
    """
    path = os.path.join(tmp_path, "test.root")
    with uproot.recreate(path) as f:
        f.mktree("tree", {"x": np.float32, "y": np.int32})

    with uproot.update(path) as f:
        f["tree"].extend(
            {
                "x": np.arange(10, dtype=np.float32),
                "y": np.arange(10, dtype=np.int32) * 2,
            }
        )

    with uproot.open(path) as f:
        assert f["tree"].num_entries == 10
        assert f["tree"]["x"].array().tolist() == list(range(10))
        assert f["tree"]["y"].array().tolist() == [i * 2 for i in range(10)]


def test_extend_missing_branch(tmp_path):
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f.mktree("tree", {"x": np.float32, "y": np.int32})
        f["tree"].extend(
            {"x": np.ones(100, dtype=np.float32), "y": np.zeros(100, dtype=np.int32)}
        )

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        with pytest.raises(ValueError, match="missing"):
            f["tree"].extend({"x": np.ones(50, dtype=np.float32)})


def test_extend_mismatched_lengths(tmp_path):
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f.mktree("tree", {"x": np.float32, "y": np.int32})
        f["tree"].extend(
            {"x": np.ones(100, dtype=np.float32), "y": np.zeros(100, dtype=np.int32)}
        )

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        with pytest.raises(ValueError):
            f["tree"].extend(
                {"x": np.ones(50, dtype=np.float32), "y": np.ones(30, dtype=np.int32)}
            )


def test_extend_nonexistent_branch(tmp_path):
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f.mktree("tree", {"x": np.float32})
        f["tree"].extend({"x": np.ones(100, dtype=np.float32)})

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        with pytest.raises(Exception):
            f["tree"].extend({"nonexistent": np.ones(100, dtype=np.float32)})


def test_extend_accept_new_fields(tmp_path):
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f.mktree("tree", {"x": np.float32})
        f["tree"].extend({"x": np.ones(100, dtype=np.float32)})

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        f["tree"].extend(
            {
                "x": np.ones(50, dtype=np.float32) * 2,
                "new_branch": np.ones(50, dtype=np.float32) * 99,
            },
            accept_new_fields=True,
        )

    with uproot.open(os.path.join(tmp_path, "test.root")) as f:
        assert f["tree"].member("fEntries") == 150
        assert "new_branch" in [b.name for b in f["tree"].branches]
        assert np.all(f["tree"]["new_branch"].array()[:100] == 0.0)
        assert np.all(f["tree"]["new_branch"].array()[100:] == 99.0)
        assert np.all(f["tree"]["x"].array()[100:] == 2.0)


def test_extend_new_fields_error_without_flag(tmp_path):
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f.mktree("tree", {"x": np.float32})
        f["tree"].extend({"x": np.ones(100, dtype=np.float32)})

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        with pytest.raises(ValueError):
            f["tree"].extend(
                {
                    "x": np.ones(50, dtype=np.float32),
                    "new_branch": np.ones(50, dtype=np.float32),
                }
            )


@skip_no_root
def test_extend_root_readable(tmp_path):
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f.mktree("tree", {"x": np.float32})
        f["tree"].extend({"x": np.ones(100, dtype=np.float32)})

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        f["tree"].extend({"x": np.ones(50, dtype=np.float32) * 2})

    ROOT.gROOT.ProcessLine("gErrorIgnoreLevel = kError;")
    f = ROOT.TFile.Open(str(os.path.join(tmp_path, "test.root")), "READ")
    tree = f.Get("tree")
    assert tree.GetEntries() == 150
    tree.SetCacheSize(0)
    tree.GetEntry(149)
    assert tree.x == pytest.approx(2.0)
    f.Close()


def test_add_branch_sequential(tmp_path):
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f.mktree("tree", {"x": np.float32})
        f["tree"].extend({"x": np.ones(100, dtype=np.float32)})

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        f["tree"].add_branches({"branch_a": np.ones(100, dtype=np.float32) * 2})

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        f["tree"].add_branches({"branch_b": np.ones(100, dtype=np.int32) * 3})

    with uproot.open(os.path.join(tmp_path, "test.root")) as f:
        assert len(f["tree"].branches) == 3
        assert np.all(f["tree"]["branch_a"].array() == 2.0)
        assert np.all(f["tree"]["branch_b"].array() == 3)


def test_add_branch_then_extend_same_session(tmp_path):
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f.mktree("tree", {"x": np.float32})
        f["tree"].extend({"x": np.ones(100, dtype=np.float32)})

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        t = f["tree"]
        t.add_branches({"new_branch": np.zeros(100, dtype=np.float32)})
        t.extend(
            {
                "x": np.ones(50, dtype=np.float32) * 2,
                "new_branch": np.ones(50, dtype=np.float32) * 99,
            }
        )

    with uproot.open(os.path.join(tmp_path, "test.root")) as f:
        assert f["tree"].member("fEntries") == 150
        assert np.all(f["tree"]["new_branch"].array()[:100] == 0.0)
        assert np.all(f["tree"]["new_branch"].array()[100:] == 99.0)
        assert np.all(f["tree"]["x"].array()[100:] == 2.0)


def test_extend_multiple_sessions(tmp_path):
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f.mktree("tree", {"x": np.float32})
        f["tree"].extend({"x": np.ones(100, dtype=np.float32)})

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        f["tree"].extend({"x": np.ones(50, dtype=np.float32) * 2})

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        f["tree"].extend({"x": np.ones(50, dtype=np.float32) * 3})

    with uproot.open(os.path.join(tmp_path, "test.root")) as f:
        assert f["tree"].member("fEntries") == 200
        assert np.all(f["tree"]["x"].array()[:100] == 1.0)
        assert np.all(f["tree"]["x"].array()[100:150] == 2.0)
        assert np.all(f["tree"]["x"].array()[150:] == 3.0)


def test_extend_after_add_branch_new_session(tmp_path):
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f.mktree("tree", {"x": np.float32})
        f["tree"].extend({"x": np.ones(100, dtype=np.float32)})

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        f["tree"].add_branches({"new_branch": np.zeros(100, dtype=np.float32)})

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        f["tree"].extend(
            {
                "x": np.ones(50, dtype=np.float32) * 2,
                "new_branch": np.ones(50, dtype=np.float32) * 99,
            }
        )

    with uproot.open(os.path.join(tmp_path, "test.root")) as f:
        assert f["tree"].member("fEntries") == 150
        assert np.all(f["tree"]["new_branch"].array()[:100] == 0.0)
        assert np.all(f["tree"]["new_branch"].array()[100:] == 99.0)


def test_extend_jagged_array(tmp_path):
    """Counter branches should not be required from the user when extending."""
    ak = pytest.importorskip("awkward")
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f.mktree("tree", {"jets": "var * float32", "x": np.float32})
        f["tree"].extend(
            {
                "jets": ak.Array([[1.0, 2.0], [3.0], [4.0, 5.0, 6.0]]),
                "x": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            }
        )

    with uproot.open(os.path.join(tmp_path, "test.root")) as f:
        assert f["tree"].num_entries == 3
        assert f["tree"]["jets"].array().tolist() == [
            [1.0, 2.0],
            [3.0],
            [4.0, 5.0, 6.0],
        ]


def test_extend_jagged_array_new_session(tmp_path):
    """Extending a jagged branch via uproot.update() must not corrupt data.

    Regression test: uproot.update() reconstructs branch metadata from disk
    (_load_existing_ttree), which used to derive the on-disk dtype of a
    jagged branch's content from the AsJagged interpretation's numpy_dtype
    (always dtype('O')) instead of its content dtype, silently writing
    garbage instead of raising.
    """
    ak = pytest.importorskip("awkward")
    path = os.path.join(tmp_path, "test.root")
    with uproot.recreate(path) as f:
        f.mktree("tree", {"jets": "var * float32"})
        f["tree"].extend({"jets": ak.Array([[1.0, 2.0], [3.0]])})

    with uproot.update(path) as f:
        f["tree"].extend({"jets": ak.Array([[7.0, 8.0, 9.0], [10.0]])})

    with uproot.open(path) as f:
        assert f["tree"]["jets"].array().tolist() == [
            [1.0, 2.0],
            [3.0],
            [7.0, 8.0, 9.0],
            [10.0],
        ]


def test_extend_string_branch_new_session(tmp_path):
    """Extending a string branch via uproot.update() must work, not raise/corrupt.

    Regression test: _load_existing_ttree had no handling for the AsStrings
    interpretation (numpy_dtype is dtype('O'), same as AsJagged), so
    extending a string branch after reopening with uproot.update() failed.
    """
    path = os.path.join(tmp_path, "test.root")
    with uproot.recreate(path) as f:
        f.mktree("tree", {"s": "string"})
        f["tree"].extend({"s": ["a_very_long_string_here"]})

    with uproot.update(path) as f:
        f["tree"].extend({"s": ["x", "yy"]})

    with uproot.open(path) as f:
        assert f["tree"]["s"].array().tolist() == [
            "a_very_long_string_here",
            "x",
            "yy",
        ]


def test_extend_string_branch_zero_basket(tmp_path):
    """Extending a string branch that has never been extended (zero baskets)."""
    path = os.path.join(tmp_path, "test.root")
    with uproot.recreate(path) as f:
        f.mktree("tree", {"s": "string", "x": np.float32})

    with uproot.update(path) as f:
        f["tree"].extend(
            {"s": ["hi", "there"], "x": np.array([1.0, 2.0], dtype=np.float32)}
        )

    with uproot.open(path) as f:
        assert f["tree"]["s"].array().tolist() == ["hi", "there"]
        assert f["tree"]["x"].array().tolist() == [1.0, 2.0]


def test_extend_after_many_extends(tmp_path):
    """Extending a tree that already has more than 10 baskets (fMaxBaskets expansion)."""
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f.mktree("tree", {"x": np.float32})
        for i in range(12):
            f["tree"].extend({"x": np.full(5, i, dtype=np.float32)})

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        f["tree"].extend({"x": np.full(5, 99, dtype=np.float32)})

    with uproot.open(os.path.join(tmp_path, "test.root")) as f:
        assert f["tree"].num_entries == 65
        assert f["tree"]["x"].array()[-5:].tolist() == [99.0] * 5

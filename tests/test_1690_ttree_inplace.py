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
    shutil.copy(
        data_path("uproot-HZZ-objects.root"), os.path.join(tmp_path, "HZZ.root")
    )

    with uproot.update(os.path.join(tmp_path, "HZZ.root")) as f:
        f["events"].add_branches({"new_branch": np.ones(2421, dtype=np.float32)})

    with uproot.open(os.path.join(tmp_path, "HZZ.root")) as f:
        assert len(f["events"].branches) == 23
        assert np.all(f["events"]["new_branch"].array() == 1.0)


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
        f["events"].add_branches({"new_branch": np.ones(2421, dtype=np.float32)})

    ROOT.gROOT.ProcessLine("gErrorIgnoreLevel = kError;")
    f = ROOT.TFile.Open(str(os.path.join(tmp_path, "HZZ.root")), "READ")
    tree = f.Get("events")
    assert tree.GetNbranches() == 23
    tree.SetCacheSize(0)
    tree.GetEntry(0)
    assert tree.new_branch == pytest.approx(1.0)
    f.Close()


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

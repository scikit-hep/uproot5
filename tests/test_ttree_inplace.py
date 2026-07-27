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
        f["tree"].add_branches({
            "branch_a": np.ones(100, dtype=np.float32),
            "branch_b": np.zeros(100, dtype=np.int32),
        })

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
        f["tree"].extend({
            "x": np.arange(100, dtype=np.float32),
            "y": np.arange(100, dtype=np.int32),
        })

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        f["tree"].add_branches({"new_branch": np.ones(100, dtype=np.float32)})

    with uproot.open(os.path.join(tmp_path, "test.root")) as f:
        assert np.all(f["tree"]["x"].array() == np.arange(100, dtype=np.float32))
        assert np.all(f["tree"]["y"].array() == np.arange(100, dtype=np.int32))
        assert np.all(f["tree"]["new_branch"].array() == 1.0)


def test_add_branch_tbranchelement(tmp_path):
    shutil.copy(
        data_path("uproot-HZZ-objects.root"),
        os.path.join(tmp_path, "HZZ.root"),
    )

    with uproot.update(os.path.join(tmp_path, "HZZ.root")) as f:
        f["events"].add_branches({"new_branch": np.ones(2421, dtype=np.float32)})

    with uproot.open(os.path.join(tmp_path, "HZZ.root")) as f:
        assert len(f["events"].branches) == 23
        assert np.all(f["events"]["new_branch"].array() == 1.0)


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
        data_path("uproot-HZZ-objects.root"),
        os.path.join(tmp_path, "HZZ.root"),
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

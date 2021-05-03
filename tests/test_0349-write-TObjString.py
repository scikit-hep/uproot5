# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

import os

import numpy as np
import pytest

import uproot
import uproot.writing

ROOT = pytest.importorskip("ROOT")


def test_recreate(tmp_path):
    filename = os.path.join(tmp_path, "whatever.root")

    with uproot.recreate(filename) as f1:
        f1["hey"] = "you"
        f1["subdir/there"] = "you guys"

    with uproot.open(filename) as f2:
        assert f2["hey"] == "you"
        assert f2["subdir/there"] == "you guys"
        assert list(f2.file.streamers) == ["TObjString"]

    f3 = ROOT.TFile(filename, "update")
    assert [x.GetName() for x in f3.GetStreamerInfoList()] == ["TObjString"]
    assert f3.Get("hey") == "you"
    assert f3.Get("subdir/there") == "you guys"
    f3.cd("subdir")
    x = ROOT.TObjString("wowie")
    x.Write()
    f3.Close()

    with uproot.open(filename) as f4:
        assert f4["hey"] == "you"
        assert f4["subdir/there"] == "you guys"
        assert f4["subdir/wowie"] == "wowie"
        assert list(f4.file.streamers) == ["TObjString"]


def test_update(tmp_path):
    filename = os.path.join(tmp_path, "whatever.root")

    f1 = ROOT.TFile(filename, "recreate")
    f1.mkdir("subdir")
    f1.cd("subdir")
    x = ROOT.TObjString("wowie")
    x.Write()
    f1.Close()

    with uproot.open(filename) as f2:
        assert f2["subdir/wowie"] == "wowie"
        assert list(f2.file.streamers) == ["TObjString"]

    with uproot.update(filename) as f3:
        f3["hey"] = "you"
        f3["subdir/there"] = "you guys"

    with uproot.open(filename) as f4:
        assert f4["hey"] == "you"
        assert f4["subdir/there"] == "you guys"
        assert f4["subdir/wowie"] == "wowie"
        assert list(f4.file.streamers) == ["TObjString"]

    f5 = ROOT.TFile(filename, "update")
    assert [x.GetName() for x in f5.GetStreamerInfoList()] == ["TObjString"]
    assert f5.Get("hey") == "you"
    assert f5.Get("subdir/there") == "you guys"
    f5.cd("subdir")
    y = ROOT.TObjString("zowie")
    y.Write()
    f5.Close()

    with uproot.open(filename) as f6:
        assert f6["hey"] == "you"
        assert f6["subdir/there"] == "you guys"
        assert f6["subdir/wowie"] == "wowie"
        assert f6["subdir/zowie"] == "zowie"
        assert list(f6.file.streamers) == ["TObjString"]

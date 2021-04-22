# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

import os

import numpy as np
import pytest

import uproot
import uproot.writing

ROOT = pytest.importorskip("ROOT")


def test_get_subdir(tmp_path):
    empty_filename = os.path.join(tmp_path, "empty.root")
    nonempty_filename = os.path.join(tmp_path, "nonempty.root")

    empty = ROOT.TFile(empty_filename, "recreate")
    empty.mkdir("subdir")
    empty.Close()
    del empty

    nonempty = ROOT.TFile(nonempty_filename, "recreate")
    nonempty.mkdir("subdir")
    nonempty.cd("subdir")
    x = ROOT.TObjString("hello")
    x.Write()
    nonempty.Close()
    del nonempty

    with uproot.update(empty_filename) as f1:
        f1._get("subdir", 1).mkdir("another")

    with uproot.update(nonempty_filename) as f2:
        f2._get("subdir", 1).mkdir("another")

    assert uproot.open(empty_filename).keys() == ["subdir;1", "subdir/another;1"]
    assert uproot.open(nonempty_filename).keys() == [
        "subdir;1",
        "subdir/hello;1",
        "subdir/another;1",
    ]

    f3 = ROOT.TFile(empty_filename, "update")
    f3.cd("subdir/another")
    y = ROOT.TObjString("there")
    y.Write()
    f3.Close()
    del f3

    f4 = ROOT.TFile(nonempty_filename, "update")
    f4.cd("subdir/another")
    z = ROOT.TObjString("you")
    z.Write()
    f4.Close()
    del f4

    assert uproot.open(empty_filename).keys() == [
        "subdir;1",
        "subdir/another;1",
        "subdir/another/there;1",
    ]
    assert uproot.open(nonempty_filename).keys() == [
        "subdir;1",
        "subdir/hello;1",
        "subdir/another;1",
        "subdir/another/you;1",
    ]

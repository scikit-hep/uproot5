# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

import os

import numpy as np
import pytest
import skhep_testdata

import uproot
import uproot.writing

ROOT = pytest.importorskip("ROOT")


def test_subdirs(tmp_path):
    filename = os.path.join(tmp_path, "testy.root")

    with uproot.writing.recreate(filename) as root_directory:
        subdir1 = root_directory.mkdir("wowzers")
        subdir2 = root_directory.mkdir("yikes")

    assert uproot.open(filename).classnames() == {
        "wowzers": "TDirectory",
        "yikes": "TDirectory",
    }

    f = ROOT.TFile(filename, "update")
    x = ROOT.TObjString("hello")
    x.Write()
    f.cd("wowzers")
    y = ROOT.TObjString("there")
    y.Write()
    f.cd("/yikes")
    z = ROOT.TObjString("you")
    z.Write()
    f.Close()

    assert uproot.open(filename).classnames() == {
        "wowzers": "TDirectory",
        "yikes": "TDirectory",
        "hello": "TObjString",
        "wowzers/there": "TObjString",
        "yikes/you": "TObjString",
    }

    g = ROOT.TFile(filename, "update")
    w = ROOT.TObjString("one_more")
    w.Write()
    g.Close()

    assert uproot.open(filename).classnames() == {
        "wowzers": "TDirectory",
        "yikes": "TDirectory",
        "hello": "TObjString",
        "one_more": "TObjString",
        "wowzers/there": "TObjString",
        "yikes/you": "TObjString",
    }


def test_subsubdir(tmp_path):
    filename = os.path.join(tmp_path, "testy.root")

    with uproot.writing.recreate(filename) as root_directory:
        subdir1 = root_directory.mkdir("wowzers")
        subdir2 = subdir1.mkdir("yikes")

    assert uproot.open(filename).classnames() == {
        "wowzers": "TDirectory",
        "wowzers/yikes": "TDirectory",
    }

    f = ROOT.TFile(filename, "update")
    x = ROOT.TObjString("hello")
    x.Write()
    f.cd("/wowzers")
    y = ROOT.TObjString("there")
    y.Write()
    f.cd("/wowzers/yikes")
    z = ROOT.TObjString("you")
    z.Write()
    f.Close()

    assert uproot.open(filename).classnames() == {
        "wowzers": "TDirectory",
        "wowzers/yikes": "TDirectory",
        "hello": "TObjString",
        "wowzers/there": "TObjString",
        "wowzers/yikes/you": "TObjString",
    }

    g = ROOT.TFile(filename, "update")
    g.cd("/wowzers/yikes")
    w = ROOT.TObjString("one_more")
    w.Write()
    g.Close()

    assert uproot.open(filename).classnames() == {
        "wowzers": "TDirectory",
        "wowzers/yikes": "TDirectory",
        "hello": "TObjString",
        "wowzers/there": "TObjString",
        "wowzers/yikes/you": "TObjString",
        "wowzers/yikes/one_more": "TObjString",
    }

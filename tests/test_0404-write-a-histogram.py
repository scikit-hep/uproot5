# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

import os

import numpy as np
import pytest

import uproot
import uproot.writing

ROOT = pytest.importorskip("ROOT")


def test(tmp_path):
    original = os.path.join(tmp_path, "original.root")
    newfile = os.path.join(tmp_path, "newfile.root")

    f1 = ROOT.TFile(original, "recreate")
    h1 = ROOT.TH1F("h1", "title", 8, -3.14, 2.71)
    h1.SetBinContent(0, 0.0)
    h1.SetBinContent(1, 1.1)
    h1.SetBinContent(2, 2.2)
    h1.SetBinContent(3, 3.3)
    h1.SetBinContent(4, 4.4)
    h1.SetBinContent(5, 5.5)
    h1.SetBinContent(6, 6.6)
    h1.SetBinContent(7, 7.7)
    h1.SetBinContent(8, 8.8)
    h1.SetBinContent(9, 9.9)
    h1.Write()
    f1.Close()

    with uproot.open(original) as fin:
        h2 = fin["h1"]

        with uproot.recreate(newfile) as fout:
            fout["h1"] = h2

    f3 = ROOT.TFile(newfile)
    h3 = f3.Get("h1")
    assert h3.GetBinContent(0) == pytest.approx(0.0)
    assert h3.GetBinContent(1) == pytest.approx(1.1)
    assert h3.GetBinContent(2) == pytest.approx(2.2)
    assert h3.GetBinContent(3) == pytest.approx(3.3)
    assert h3.GetBinContent(4) == pytest.approx(4.4)
    assert h3.GetBinContent(5) == pytest.approx(5.5)
    assert h3.GetBinContent(6) == pytest.approx(6.6)
    assert h3.GetBinContent(7) == pytest.approx(7.7)
    assert h3.GetBinContent(8) == pytest.approx(8.8)
    assert h3.GetBinContent(9) == pytest.approx(9.9)

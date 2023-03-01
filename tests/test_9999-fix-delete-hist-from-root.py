# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE
import os
import numpy
import pytest
import uproot
import ROOT

# pytest.importorskip("ROOT")


def test_delete_from_file_with_deleted_histogram_at_the_end(tmp_path):
    c1 = ROOT.TCanvas("c1", "The FillRandom example", 200, 10, 700, 900)
    c1.SetFillColor(18)

    pad1 = ROOT.TPad("pad1", "The pad with the function", 0.05, 0.50, 0.95, 0.95, 21)
    pad2 = ROOT.TPad("pad2", "The pad with the histogram", 0.05, 0.05, 0.95, 0.45, 21)
    pad1.Draw()
    pad2.Draw()
    pad1.cd()

    form1 = ROOT.TFormula("form1", "abs(sin(x)/x)")
    sqroot = ROOT.TF1("sqroot", "x*gaus(0) + [3]*form1", 0, 10)
    sqroot.SetParameters(10, 4, 1, 20)
    pad1.SetGridx()
    pad1.SetGridy()
    pad1.GetFrame().SetFillColor(42)
    pad1.GetFrame().SetBorderMode(-1)
    pad1.GetFrame().SetBorderSize(5)
    sqroot.SetLineColor(4)
    sqroot.SetLineWidth(6)
    sqroot.Draw()
    lfunction = ROOT.TPaveLabel(5, 39, 9.8, 46, "The sqroot function")
    lfunction.SetFillColor(41)
    lfunction.Draw()
    c1.Update()

    pad2.cd()
    pad2.GetFrame().SetFillColor(42)
    pad2.GetFrame().SetBorderMode(-1)
    pad2.GetFrame().SetBorderSize(5)
    h1f = ROOT.TH1F("h1f", "Random numbers", 200, 0, 10)
    h1f.SetFillColor(45)
    h1f.FillRandom("sqroot", 10000)
    h1f.Draw()
    c1.Update()

    filename = os.path.join(tmp_path, "hist_del_test.root")
    tfile = ROOT.TFile(filename, "RECREATE")
    form1.Write()
    sqroot.Write()
    h1f.Write()
    tfile.Write()

    with uproot.update(filename) as f:
        assert f.keys() == ["form1;1", "sqroot;1", "h1f;1"]
        del f["h1f"]

        assert f.keys() == ["form1;1", "sqroot;1"]
        del f["sqroot"]

        assert f.keys() == ["form1;1"]

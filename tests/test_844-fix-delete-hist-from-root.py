# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE
import os
import numpy
import pytest
import uproot

ROOT = pytest.importorskip("ROOT")


def test_delete_from_file_with_deleted_histogram_at_the_end(tmp_path):
    c1 = ROOT.TCanvas("c1", "The FillRandom example", 200, 10, 700, 900)
    c1.SetFillColor(18)
    
    pad0 = ROOT.TPad("pad0", "The pad with the histogram", 0.05, 0.05, 0.95, 0.45, 21)
    pad1 = ROOT.TPad("pad1", "The pad with the histogram", 0.05, 0.05, 0.95, 0.45, 21)
    pad2 = ROOT.TPad("pad2", "The pad with the histogram", 0.05, 0.05, 0.95, 0.45, 21)
    pad3 = ROOT.TPad("pad1", "The pad with the function", 0.05, 0.50, 0.95, 0.95, 21)

    form1 = ROOT.TFormula("form1", "abs(sin(x)/x)")
    sqroot = ROOT.TF1("sqroot", "x*gaus(0) + [3]*form1", 0, 10)
    sqroot.SetParameters(10, 4, 1, 20)
    pad3.SetGridx()
    pad3.SetGridy()
    pad3.GetFrame().SetFillColor(42)
    pad3.GetFrame().SetBorderMode(-1)
    pad3.GetFrame().SetBorderSize(5)
    sqroot.SetLineColor(4)
    sqroot.SetLineWidth(6)
    sqroot.Draw()
    lfunction = ROOT.TPaveLabel(5, 39, 9.8, 46, "The sqroot function")
    lfunction.SetFillColor(41)
    lfunction.Draw()
    c1.Update()

    pad0.cd()
    pad0.GetFrame().SetFillColor(42)
    pad0.GetFrame().SetBorderMode(-1)
    pad0.GetFrame().SetBorderSize(5)
    h0f = ROOT.TH1F("h0f", "Random numbers", 200, 0, 10)
    h0f.SetFillColor(45)
    h0f.FillRandom("sqroot", 10000)
    h0f.Draw()
    c1.Update()


    pad1.cd()
    pad1.GetFrame().SetFillColor(42)
    pad1.GetFrame().SetBorderMode(-1)
    pad1.GetFrame().SetBorderSize(5)
    h1f = ROOT.TH1F("h1f", "Random numbers", 200, 0, 10)
    h1f.SetFillColor(45)
    h1f.FillRandom("sqroot", 10000)
    h1f.Draw()
    c1.Update()

    pad2.cd()
    pad2.GetFrame().SetFillColor(42)
    pad2.GetFrame().SetBorderMode(-1)
    pad2.GetFrame().SetBorderSize(5)
    h2f = ROOT.TH1F("h2f", "Random numbers", 200, 0, 10)
    h2f.SetFillColor(45)
    h2f.FillRandom("sqroot", 10000)
    h2f.Draw()
    c1.Update()

    # file with 3 equal histograms
    filename = os.path.join(tmp_path, "hist_del_test_equal.root")
    tfile = ROOT.TFile(filename, "RECREATE")

    form1.Write()
    sqroot.Write()
    h0f.Write()
    h1f.Write()
    h2f.Write()
    tfile.Write()

    file_size_equal_hists = os.path.getsize(filename)

    # file with 2 equal and one larger histogram
    filename2= os.path.join(tmp_path, "hist_del_test_non_equal.root")
    tfile2 = ROOT.TFile(filename2, "RECREATE")

    pad2.cd()
    pad2.GetFrame().SetFillColor(42)
    pad2.GetFrame().SetBorderMode(-1)
    pad2.GetFrame().SetBorderSize(5)
    h2f = ROOT.TH1F("h2fgreaterinthisfile", "Random numbers", 200, 0, 10)
    h2f.SetFillColor(45)
    h2f.FillRandom("sqroot", 40000)
    h2f.Draw()
    c1.Update()

    form1.Write()
    sqroot.Write()
    h0f.Write()
    h1f.Write()
    h2f.Write()
    tfile2.Write()

    file_size_non_equal_hists = os.path.getsize(filename2)

    assert file_size_equal_hists < file_size_non_equal_hists

    with uproot.update(filename) as f:
        initial_size = os.path.getsize(filename)

        assert f.keys() == ["form1;1", "sqroot;1", "h0f;1", "h1f;1", "h2f;1"]
        del f["h0f;1"]

        updated_size = os.path.getsize(filename)

        # assert initial_size > updated_size

        f["hnf"] = numpy.histogram(numpy.random.normal(0, 1, 50000))

        updated_size2 = os.path.getsize(filename)
        assert updated_size < updated_size2
        
        assert f.keys() == ['form1;1', 'sqroot;1', 'h1f;1', 'h2f;1', 'hnf;1']
        del f['form1;1']
        del f["hnf;1"]
        del f["sqroot;1"]

        updated_size3 = os.path.getsize(filename)
        assert updated_size2 > updated_size3
        
        assert f.keys() == ["h1f;1", "h2f;1"]
        del f["h1f;1"]
        
        assert f.keys() == ["h2f;1"]
        del f["h2f;1"]
        

    
    
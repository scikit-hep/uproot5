# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE
import os
import numpy
import pytest
import uproot
import struct

ROOT = pytest.importorskip("ROOT")


def test_delete_from_file_with_deleted_histogram_at_the_end(tmp_path):
    c1 = ROOT.TCanvas("c1", "The FillRandom example", 200, 10, 700, 900)

    form1 = ROOT.TFormula("form1", "abs(sin(x)/x)")
    sqroot = ROOT.TF1("sqroot", "x*gaus(0) + [3]*form1", 0, 10)
    sqroot.Draw()

    h0f = ROOT.TH1F("h0f", "Random numbers", 200, 0, 10)
    h0f.FillRandom("sqroot", 10000)
    h0f.Draw()

    h1f = ROOT.TH1F("h1f", "Random numbers", 200, 0, 10)
    h1f.FillRandom("sqroot", 10000)
    h1f.Draw()

    h2f = ROOT.TH1F("h2f", "Random numbers", 200, 0, 10)
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

    tfile.Close()

    with uproot.update(filename) as f:
        assert f.keys() == ["form1;1", "sqroot;1", "h0f;1", "h1f;1", "h2f;1"]
        del f["h2f;1"]
        del f["h1f;1"]
        del f["h0f;1"]
        assert f.keys() == ["form1;1", "sqroot;1"]


def test_hist_del_uproot_equal(tmp_path):
    filename = os.path.join(tmp_path, "uproot_hist_del_test_equal.root")
    with uproot.recreate(filename) as f:
        file_size = os.path.getsize(filename)
        f["hnf1"] = numpy.histogram(numpy.random.normal(0, 1, 10000))
        f["hnf2"] = numpy.histogram(numpy.random.normal(0, 1, 10000))
        f["hnf3"] = numpy.histogram(numpy.random.normal(0, 1, 10000))
        assert f.keys() == ['hnf1;1', 'hnf2;1', 'hnf3;1']

        del f["hnf1;1"]
        assert f.keys() == ['hnf2;1', 'hnf3;1']
        assert file_size < os.path.getsize(filename)
        file_size = os.path.getsize(filename)

        f["hnf4"] = numpy.histogram(numpy.random.normal(0, 1, 10000))

        assert f.keys() == ['hnf2;1', 'hnf3;1', 'hnf4;1']
        assert file_size == os.path.getsize(filename)
        file_size = os.path.getsize(filename)

        del f['hnf4;1']
        f["hnf5"] = numpy.histogram(numpy.random.normal(0, 1, 50000))
        
        assert f.keys() == ['hnf2;1', 'hnf3;1', 'hnf5;1']
        assert file_size < os.path.getsize(filename)
        file_size = os.path.getsize(filename)



    
# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

import os

import numpy as np
import pytest

import uproot
import uproot.writing

ROOT = pytest.importorskip("ROOT")
hist = pytest.importorskip("hist")


def test_regular_1d(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    with uproot.recreate(newfile) as fout:
        fout["h1"] = (
            hist.Hist.new.Reg(10, -5, 5, name="wow", label="wee")
            .Weight()
            .fill([-2, 3, 3, 1, 99], weight=[1, 1, 5, 5, 3])
        )

    with uproot.open(newfile) as fin:
        h1 = fin["h1"]
        assert h1.member("fEntries") == 15
        assert h1.values(flow=True).tolist() == pytest.approx(
            [0, 0, 0, 0, 1, 0, 0, 5, 0, 6, 0, 3]
        )
        assert h1.axis().member("fName") == "wow"
        assert h1.axis().member("fTitle") == "wee"
        assert h1.axis().member("fXmin") == -5
        assert h1.axis().member("fXmax") == 5
        assert len(h1.axis().member("fXbins")) == 0

    f = ROOT.TFile(newfile)
    h2 = f.Get("h1")
    assert h2.GetEntries() == 15
    assert h2.GetBinContent(9) == 6
    assert h2.GetBinContent(11) == 3
    assert h2.GetXaxis().GetName() == "wow"
    assert h2.GetXaxis().GetTitle() == "wee"
    assert h2.GetBinLowEdge(1) == pytest.approx(-5)
    assert h2.GetBinWidth(1) == pytest.approx(1)
    f.Close()


def test_variable_1d(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    with uproot.recreate(newfile) as fout:
        fout["h1"] = (
            hist.Hist.new.Var([-5, -3, 0, 1, 2, 10], name="wow", label="wee")
            .Weight()
            .fill([-2, 3, 3, 1, 99], weight=[1, 1, 5, 5, 3])
        )

    with uproot.open(newfile) as fin:
        h1 = fin["h1"]
        assert h1.member("fEntries") == 15
        assert h1.values(flow=True).tolist() == pytest.approx([0, 0, 1, 0, 5, 6, 3])
        assert h1.axis().member("fName") == "wow"
        assert h1.axis().member("fTitle") == "wee"
        assert h1.axis().member("fXmin") == -5
        assert h1.axis().member("fXmax") == 10
        assert list(h1.axis().member("fXbins")) == pytest.approx([-5, -3, 0, 1, 2, 10])

    f = ROOT.TFile(newfile)
    h2 = f.Get("h1")
    assert h2.GetEntries() == 15
    assert h2.GetBinContent(5) == 6
    assert h2.GetBinContent(6) == 3
    assert h2.GetXaxis().GetName() == "wow"
    assert h2.GetXaxis().GetTitle() == "wee"
    assert h2.GetBinLowEdge(1) == pytest.approx(-5)
    assert h2.GetBinWidth(1) == pytest.approx(2)
    f.Close()


def test_regular_2d(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    with uproot.recreate(newfile) as fout:
        tmp = (
            hist.Hist.new.Reg(10, -5, 5, name="wow", label="wee")
            .Reg(8, 2, 10)
            .Weight()
            .fill([-2, 3, 3, 1, 99], [9, 9, 9, 4, 4], weight=[1, 1, 5, 5, 3])
        )
        asarray = tmp.values(flow=True)
        assert asarray[9, 8] == 6
        assert asarray[8, 9] == 0
        fout["h1"] = tmp

    with uproot.open(newfile) as fin:
        h1 = fin["h1"]
        assert h1.member("fEntries") == 15
        assert h1.axis(0).member("fName") == "wow"
        assert h1.axis(0).member("fTitle") == "wee"
        assert h1.axis(0).member("fXmin") == -5
        assert h1.axis(0).member("fXmax") == 5
        assert h1.axis(1).member("fXmin") == 2
        assert h1.axis(1).member("fXmax") == 10
        assert np.allclose(asarray, h1.values(flow=True))
        assert np.allclose(asarray, h1.to_hist().values(flow=True))

    f = ROOT.TFile(newfile)
    h2 = f.Get("h1")
    assert h2.GetEntries() == 15
    assert h2.GetBinContent(9, 8) == 6
    assert h2.GetBinContent(8, 9) == 0
    f.Close()


def test_regular_3d(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    with uproot.recreate(newfile) as fout:
        tmp = (
            hist.Hist.new.Reg(10, -5, 5, name="wow", label="wee")
            .Reg(8, 2, 10)
            .Reg(2, -2, 2)
            .Weight()
            .fill(
                [-2, 3, 3, 1, 99],
                [9, 9, 9, 4, 4],
                [1, 1, 1, -1, -1],
                weight=[1, 1, 5, 5, 3],
            )
        )
        asarray = tmp.values(flow=True)
        assert asarray[9, 8, 2] == 6
        assert asarray[8, 9, 2] == 0
        assert asarray[9, 8, 1] == 0
        fout["h1"] = tmp

    with uproot.open(newfile) as fin:
        h1 = fin["h1"]
        assert h1.member("fEntries") == 15
        assert h1.axis(0).member("fName") == "wow"
        assert h1.axis(0).member("fTitle") == "wee"
        assert h1.axis(0).member("fXmin") == -5
        assert h1.axis(0).member("fXmax") == 5
        assert h1.axis(1).member("fXmin") == 2
        assert h1.axis(1).member("fXmax") == 10
        assert h1.axis(2).member("fXmin") == -2
        assert h1.axis(2).member("fXmax") == 2
        assert np.allclose(asarray, h1.values(flow=True))
        assert np.allclose(asarray, h1.to_hist().values(flow=True))

    f = ROOT.TFile(newfile)
    h2 = f.Get("h1")
    assert h2.GetEntries() == 15
    assert h2.GetBinContent(9, 8, 2) == 6
    assert h2.GetBinContent(8, 9, 2) == 0
    assert h2.GetBinContent(9, 8, 1) == 0
    f.Close()


def test_issue_0659(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    cat_axis = hist.axis.IntCategory([10, 11, 12], label="Category")
    reg_axis = hist.axis.Regular(100, 0, 100, label="Random")
    h = hist.Hist(cat_axis)
    h2 = hist.Hist(cat_axis, reg_axis)
    h.fill(np.random.randint(1, 4, 1000))
    h2.fill(np.random.randint(1, 4, 1000), np.random.normal(20, 5, 1000))

    with uproot.recreate(newfile) as fout:
        fout["h"] = h
        fout["h2"] = h2

    with uproot.open(newfile) as fin:
        h_opened = fin["h"]
        assert h_opened.values(flow=False).shape == (3,)
        assert h_opened.values(flow=True).shape == (5,)
        assert h_opened.axis(0).edges().tolist() == [0.0, 1.0, 2.0, 3.0]

        h2_opened = fin["h2"]
        assert h2_opened.values(flow=False).shape == (3, 100)
        assert h2_opened.values(flow=True).shape == (5, 102)
        assert h2_opened.axis(0).edges().tolist() == [0.0, 1.0, 2.0, 3.0]
        assert h2_opened.axis(1).edges().tolist() == list(map(float, range(101)))

        h_opened.to_hist()
        h2_opened.to_hist()

    f = ROOT.TFile(newfile)
    h_opened2 = f.Get("h")
    h2_opened2 = f.Get("h2")
    assert h_opened2.GetBinContent(0) == 0.0
    assert h2_opened2.GetBinContent(0) == 0.0
    f.Close()


def test_issue_722(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    h = ROOT.TH1D("h", "h", 10, 0.0, 1.0)
    h.FillRandom("gaus", 10000)

    assert len(h.GetSumw2()) == 0

    fout = ROOT.TFile(newfile, "RECREATE")
    h.Write()
    fout.Close()

    # open with uproot
    with uproot.open(newfile) as fin:
        h1 = fin["h"]

    assert len(h1.axes) == 1
    assert h1.axis(0).edges().tolist() == pytest.approx(
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    )
    assert len(h1.member("fSumw2")) == 0

    # convert to hist
    h2 = h1.to_hist()
    assert str(h2.storage_type) == "<class 'boost_histogram.storage.Double'>"

    # write and read again
    with uproot.recreate(newfile) as fout2:
        fout2["h"] = h2

    with uproot.open(newfile) as fin2:
        h3 = fin2["h"]

    assert len(h3.member("fSumw2")) == 0


def test_hist_weights_from_root(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    h = ROOT.TH1D("h", "h", 20, 0.0, 5.0)
    for _ in range(1000):
        # fill with random values and random weights
        h.Fill(5.0 * np.random.random(), np.random.random())

    assert len(h.GetSumw2()) == 22  # 20 bins + 2, should not be 0 since we have weights

    fout = ROOT.TFile(newfile, "RECREATE")
    h.Write()
    fout.Close()

    with uproot.open(newfile) as fin:
        h1 = fin["h"]

    assert len(h1.member("fSumw2")) == 22

    h2 = h1.to_hist()
    assert str(h2.storage_type) == "<class 'boost_histogram.storage.Weight'>"

    # write and read again
    with uproot.recreate(newfile) as fout2:
        fout2["h"] = h2

    with uproot.open(newfile) as fin2:
        h3 = fin2["h"]

    assert len(h3.member("fSumw2")) == 22

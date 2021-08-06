# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

import os

import numpy as np
import pytest
import skhep_testdata

import uproot
import uproot.writing

ROOT = pytest.importorskip("ROOT")


def test_copy(tmp_path):
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


def test_from_old(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    with uproot.open(skhep_testdata.data_path("uproot-histograms.root")) as fin:
        one = fin["one"]

        with uproot.recreate(newfile) as fout:
            fout["one"] = one

    f1 = ROOT.TFile(newfile)
    h1 = f1.Get("one")
    assert h1.GetBinContent(0) == 0
    assert h1.GetBinContent(1) == 68
    assert h1.GetBinContent(2) == 285
    assert h1.GetBinContent(3) == 755
    assert h1.GetBinContent(4) == 1580
    assert h1.GetBinContent(5) == 2296
    assert h1.GetBinContent(6) == 2286
    assert h1.GetBinContent(7) == 1570
    assert h1.GetBinContent(8) == 795
    assert h1.GetBinContent(9) == 289
    assert h1.GetBinContent(10) == 76
    assert h1.GetBinContent(11) == 0


def test_new_name(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    with uproot.open(skhep_testdata.data_path("uproot-histograms.root")) as fin:
        one = fin["one"]

        with uproot.recreate(newfile) as fout:
            fout["whatever"] = one

    f1 = ROOT.TFile(newfile)
    h1 = f1.Get("whatever")
    assert h1.GetBinContent(0) == 0
    assert h1.GetBinContent(1) == 68
    assert h1.GetBinContent(2) == 285
    assert h1.GetBinContent(3) == 755
    assert h1.GetBinContent(4) == 1580
    assert h1.GetBinContent(5) == 2296
    assert h1.GetBinContent(6) == 2286
    assert h1.GetBinContent(7) == 1570
    assert h1.GetBinContent(8) == 795
    assert h1.GetBinContent(9) == 289
    assert h1.GetBinContent(10) == 76
    assert h1.GetBinContent(11) == 0


@pytest.mark.parametrize("cls", [ROOT.TH1C, ROOT.TH1D, ROOT.TH1F, ROOT.TH1I, ROOT.TH1S])
def test_all_TH1(tmp_path, cls):
    original = os.path.join(tmp_path, "original.root")
    newfile = os.path.join(tmp_path, "newfile.root")

    f1 = ROOT.TFile(original, "recreate")
    h1 = cls("h1", "title", 2, -3.14, 2.71)
    h1.Fill(-4)
    h1.Fill(-3.1)
    h1.Fill(-3.1)
    h1.Fill(2.7, 5)
    h1.Fill(3, 4)
    h1.Write()
    f1.Close()

    with uproot.open(original) as fin:
        h2 = fin["h1"]

        with uproot.recreate(newfile) as fout:
            fout["out"] = h2

    f3 = ROOT.TFile(newfile)
    h3 = f3.Get("out")
    assert h3.GetEntries() == 5
    assert h3.GetSumOfWeights() == 7
    assert h3.GetBinLowEdge(1) == pytest.approx(-3.14)
    assert h3.GetBinWidth(1) == pytest.approx((2.71 - -3.14) / 2)
    assert h3.GetBinContent(0) == pytest.approx(1)
    assert h3.GetBinContent(1) == pytest.approx(2)
    assert h3.GetBinContent(2) == pytest.approx(5)
    assert h3.GetBinContent(3) == pytest.approx(4)
    assert h3.GetBinError(0) == pytest.approx(1)
    assert h3.GetBinError(1) == pytest.approx(1.4142135623730951)
    assert h3.GetBinError(2) == pytest.approx(5)
    assert h3.GetBinError(3) == pytest.approx(4)


@pytest.mark.parametrize("cls", [ROOT.TH2C, ROOT.TH2D, ROOT.TH2F, ROOT.TH2I, ROOT.TH2S])
def test_all_TH2(tmp_path, cls):
    original = os.path.join(tmp_path, "original.root")
    newfile = os.path.join(tmp_path, "newfile.root")

    f1 = ROOT.TFile(original, "recreate")
    h1 = cls("h1", "title", 2, -3.14, 2.71, 3, -5, 10)
    h1.Fill(-4, 9)
    h1.Fill(-3.1, 9)
    h1.Fill(-3.1, 9)
    h1.Fill(2.7, -4, 5)
    h1.Fill(3, 9, 4)
    h1.Write()
    f1.Close()

    with uproot.open(original) as fin:
        h2 = fin["h1"]

        with uproot.recreate(newfile) as fout:
            fout["out"] = h2

    f3 = ROOT.TFile(newfile)
    h3 = f3.Get("out")
    assert h3.GetEntries() == 5
    assert h3.GetSumOfWeights() == 7
    assert h3.GetNbinsX() == 2
    assert h3.GetNbinsY() == 3
    assert h3.GetXaxis().GetBinLowEdge(1) == pytest.approx(-3.14)
    assert h3.GetXaxis().GetBinUpEdge(2) == pytest.approx(2.71)
    assert h3.GetYaxis().GetBinLowEdge(1) == pytest.approx(-5)
    assert h3.GetYaxis().GetBinUpEdge(3) == pytest.approx(10)
    assert [[h3.GetBinContent(i, j) for j in range(5)] for i in range(4)] == [
        pytest.approx([0, 0, 0, 1, 0]),
        pytest.approx([0, 0, 0, 2, 0]),
        pytest.approx([0, 5, 0, 0, 0]),
        pytest.approx([0, 0, 0, 4, 0]),
    ]


@pytest.mark.parametrize("cls", [ROOT.TH3C, ROOT.TH3D, ROOT.TH3F, ROOT.TH3I, ROOT.TH3S])
def test_all_TH3(tmp_path, cls):
    original = os.path.join(tmp_path, "original.root")
    newfile = os.path.join(tmp_path, "newfile.root")

    f1 = ROOT.TFile(original, "recreate")
    h1 = cls("h1", "title", 2, -3.14, 2.71, 3, -5, 10, 1, 100, 200)
    h1.Fill(-4, 9, 150)
    h1.Fill(-3.1, 9, 150)
    h1.Fill(-3.1, 9, 150)
    h1.Fill(2.7, -4, 150, 5)
    h1.Fill(3, 9, 150, 4)
    h1.Write()
    f1.Close()

    with uproot.open(original) as fin:
        h2 = fin["h1"]

        with uproot.recreate(newfile) as fout:
            fout["out"] = h2

    f3 = ROOT.TFile(newfile)
    h3 = f3.Get("out")
    assert h3.GetEntries() == 5
    assert h3.GetSumOfWeights() == 7
    assert h3.GetNbinsX() == 2
    assert h3.GetNbinsY() == 3
    assert h3.GetNbinsZ() == 1
    assert h3.GetXaxis().GetBinLowEdge(1) == pytest.approx(-3.14)
    assert h3.GetXaxis().GetBinUpEdge(2) == pytest.approx(2.71)
    assert h3.GetYaxis().GetBinLowEdge(1) == pytest.approx(-5)
    assert h3.GetYaxis().GetBinUpEdge(3) == pytest.approx(10)
    assert h3.GetZaxis().GetBinLowEdge(1) == pytest.approx(100)
    assert h3.GetZaxis().GetBinUpEdge(1) == pytest.approx(200)
    approx = pytest.approx
    assert [
        [[h3.GetBinContent(i, j, k) for k in range(3)] for j in range(5)]
        for i in range(4)
    ] == [
        [[0, 0, 0], approx([0, 0, 0]), [0, 0, 0], approx([0, 1, 0]), [0, 0, 0]],
        [[0, 0, 0], approx([0, 0, 0]), [0, 0, 0], approx([0, 2, 0]), [0, 0, 0]],
        [[0, 0, 0], approx([0, 5, 0]), [0, 0, 0], approx([0, 0, 0]), [0, 0, 0]],
        [[0, 0, 0], approx([0, 0, 0]), [0, 0, 0], approx([0, 4, 0]), [0, 0, 0]],
    ]


def test_TProfile(tmp_path):
    original = os.path.join(tmp_path, "original.root")
    newfile = os.path.join(tmp_path, "newfile.root")

    f1 = ROOT.TFile(original, "recreate")
    h1 = ROOT.TProfile("h1", "title", 2, -3.14, 2.71)
    h1.Fill(-4, 10)
    h1.Fill(-3.1, 10)
    h1.Fill(-3.1, 20)
    h1.Fill(2.7, 20)
    h1.Fill(3, 20)
    h1.Write()
    f1.Close()

    with uproot.open(original) as fin:
        h2 = fin["h1"]

        with uproot.recreate(newfile) as fout:
            fout["out"] = h2

    f3 = ROOT.TFile(newfile)
    h3 = f3.Get("out")
    assert h3.GetEntries() == 5
    assert h3.GetSumOfWeights() == 35
    assert h3.GetBinLowEdge(1) == pytest.approx(-3.14)
    assert h3.GetBinWidth(1) == pytest.approx((2.71 - -3.14) / 2)
    assert h3.GetBinContent(0) == pytest.approx(10)
    assert h3.GetBinContent(1) == pytest.approx(15)
    assert h3.GetBinContent(2) == pytest.approx(20)
    assert h3.GetBinContent(3) == pytest.approx(20)
    assert h3.GetBinError(0) == pytest.approx(0)
    assert h3.GetBinError(1) == pytest.approx(np.sqrt(12.5))
    assert h3.GetBinError(2) == pytest.approx(0)
    assert h3.GetBinError(3) == pytest.approx(0)


def test_TProfile2D(tmp_path):
    original = os.path.join(tmp_path, "original.root")
    newfile = os.path.join(tmp_path, "newfile.root")

    f1 = ROOT.TFile(original, "recreate")
    h1 = ROOT.TProfile2D("h1", "title", 2, -3.14, 2.71, 3, -5, 10)
    h1.Fill(-4, 9, 10)
    h1.Fill(-3.1, 9, 10)
    h1.Fill(-3.1, 9, 20)
    h1.Fill(2.7, -4, 20)
    h1.Fill(3, 9, 20)
    h1.Write()
    f1.Close()

    with uproot.open(original) as fin:
        h2 = fin["h1"]

        with uproot.recreate(newfile) as fout:
            fout["out"] = h2

    f3 = ROOT.TFile(newfile)
    h3 = f3.Get("out")
    assert h3.GetEntries() == 5
    assert h3.GetSumOfWeights() == 35
    assert h3.GetNbinsX() == 2
    assert h3.GetNbinsY() == 3
    assert h3.GetXaxis().GetBinLowEdge(1) == pytest.approx(-3.14)
    assert h3.GetXaxis().GetBinUpEdge(2) == pytest.approx(2.71)
    assert h3.GetYaxis().GetBinLowEdge(1) == pytest.approx(-5)
    assert h3.GetYaxis().GetBinUpEdge(3) == pytest.approx(10)
    assert [[h3.GetBinContent(i, j) for j in range(5)] for i in range(4)] == [
        pytest.approx([0, 0, 0, 10, 0]),
        pytest.approx([0, 0, 0, 15, 0]),
        pytest.approx([0, 20, 0, 0, 0]),
        pytest.approx([0, 0, 0, 20, 0]),
    ]
    assert [[h3.GetBinError(i, j) for j in range(5)] for i in range(4)] == [
        pytest.approx([0, 0, 0, 0, 0]),
        pytest.approx([0, 0, 0, np.sqrt(12.5), 0]),
        pytest.approx([0, 0, 0, 0, 0]),
        pytest.approx([0, 0, 0, 0, 0]),
    ]


def test_TProfile3D(tmp_path):
    original = os.path.join(tmp_path, "original.root")
    newfile = os.path.join(tmp_path, "newfile.root")

    f1 = ROOT.TFile(original, "recreate")
    h1 = ROOT.TProfile3D("h1", "title", 2, -3.14, 2.71, 3, -5, 10, 1, 100, 200)
    h1.Fill(-4, 9, 150, 10)
    h1.Fill(-3.1, 9, 150, 10)
    h1.Fill(-3.1, 9, 150, 20)
    h1.Fill(2.7, -4, 150, 20)
    h1.Fill(3, 9, 150, 20)
    h1.Write()
    f1.Close()

    with uproot.open(original) as fin:
        h2 = fin["h1"]

        with uproot.recreate(newfile) as fout:
            fout["out"] = h2

    f3 = ROOT.TFile(newfile)
    h3 = f3.Get("out")
    assert h3.GetEntries() == 5
    assert h3.GetSumOfWeights() == 35
    assert h3.GetNbinsX() == 2
    assert h3.GetNbinsY() == 3
    assert h3.GetNbinsZ() == 1
    assert h3.GetXaxis().GetBinLowEdge(1) == pytest.approx(-3.14)
    assert h3.GetXaxis().GetBinUpEdge(2) == pytest.approx(2.71)
    assert h3.GetYaxis().GetBinLowEdge(1) == pytest.approx(-5)
    assert h3.GetYaxis().GetBinUpEdge(3) == pytest.approx(10)
    assert h3.GetZaxis().GetBinLowEdge(1) == pytest.approx(100)
    assert h3.GetZaxis().GetBinUpEdge(1) == pytest.approx(200)
    approx = pytest.approx
    assert [
        [[h3.GetBinContent(i, j, k) for k in range(3)] for j in range(5)]
        for i in range(4)
    ] == [
        [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 10, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 15, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 20, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 20, 0], [0, 0, 0]],
    ]
    assert [
        [[h3.GetBinError(i, j, k) for k in range(3)] for j in range(5)]
        for i in range(4)
    ] == [
        [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, approx(np.sqrt(12.5)), 0], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
    ]


def test_ex_nihilo(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    h1 = uproot.writing.to_TH1x(
        fName="h1",
        fTitle="title",
        data=np.array([1.0, 2.0, 5.0, 4.0], np.float64),
        fEntries=5.0,
        fTsumw=7.0,
        fTsumw2=27.0,
        fTsumwx=7.3,
        fTsumwx2=55.67,
        fSumw2=np.array([1.0, 2.0, 25.0, 16.0], np.float64),
        fXaxis=uproot.writing.to_TAxis(
            fName="xaxis",
            fTitle="",
            fNbins=2,
            fXmin=-3.14,
            fXmax=2.71,
        ),
    )

    with uproot.recreate(newfile) as fout:
        fout["h1"] = h1

    f3 = ROOT.TFile(newfile)
    h3 = f3.Get("h1")
    assert h3.GetEntries() == 5
    assert h3.GetSumOfWeights() == 7
    assert h3.GetBinLowEdge(1) == pytest.approx(-3.14)
    assert h3.GetBinWidth(1) == pytest.approx((2.71 - -3.14) / 2)
    assert h3.GetBinContent(0) == pytest.approx(1)
    assert h3.GetBinContent(1) == pytest.approx(2)
    assert h3.GetBinContent(2) == pytest.approx(5)
    assert h3.GetBinContent(3) == pytest.approx(4)
    assert h3.GetBinError(0) == pytest.approx(1)
    assert h3.GetBinError(1) == pytest.approx(1.4142135623730951)
    assert h3.GetBinError(2) == pytest.approx(5)
    assert h3.GetBinError(3) == pytest.approx(4)

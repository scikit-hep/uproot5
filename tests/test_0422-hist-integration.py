# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

import array
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


@pytest.mark.parametrize("overflow", [True, False])
@pytest.mark.parametrize("underflow", [True, False])
def test_flow_bin_writing(tmp_path, underflow, overflow):
    newfile = os.path.join(tmp_path, "newfile.root")
    tmp = hist.new.Reg(3, 1, 4, name='x', underflow=underflow,
                       overflow=overflow).Weight().fill([0, 1, 2, 3, 4],
                                                        weight=[1, 1, np.nan, 1, 1])

    with uproot.recreate(newfile) as fout:
        fout["h1"] = tmp

    with uproot.open(newfile) as fin:
        h1 = fin["h1"]

    assert np.allclose(tmp.values(), h1.values(), equal_nan=True)
    assert np.allclose(tmp.values(flow=True), h1.values(flow=True), equal_nan=True)


@pytest.mark.parametrize("under1", [True, False])
@pytest.mark.parametrize("under2", [True, False])
@pytest.mark.parametrize("under3", [True, False])
@pytest.mark.parametrize("over1", [True, False])
@pytest.mark.parametrize("over2", [True, False])
@pytest.mark.parametrize("over3", [True, False])
def test_flow_bin_writing_3d(tmp_path, under1, under2, under3, over1, over2, over3):
    newfile = os.path.join(tmp_path, "newfile.root")
    tmp = (
        hist.Hist.new
        .Reg(3, 1, 4, name='x', underflow=under1, overflow=over1)
        .Reg(3, 1, 4, name='y', underflow=under2, overflow=over2)
        .Reg(3, 1, 4, name='z', underflow=under3, overflow=over3)
        .Weight()
        .fill(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )
    )

    with uproot.recreate(newfile) as fout:
        fout["h1"] = tmp

    with uproot.open(newfile) as fin:
        h1 = fin["h1"]

    assert np.allclose(tmp.values(), h1.values())
    assert np.allclose(tmp.values(flow=True), h1.values(flow=True))

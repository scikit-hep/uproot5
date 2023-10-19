# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import pytest
import uproot
import os
import numpy as np
import hist
import ROOT


def test_write_TProfile(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    h1 = ROOT.TProfile("h1", "title", 2, -3.14, 2.71)
    h1.Fill(-4, 10)
    h1.Fill(-3.1, 10)
    h1.Fill(-3.1, 20)
    h1.Fill(2.7, 20)
    h1.Fill(3, 20)

    hhist = uproot.from_pyroot(h1).to_hist()

    uhist = uproot.writing.identify.to_TProfile(
        fName="h1",
        fTitle="title",
        data=np.array([10, 30, 20, 20], np.float64),
        fEntries=5.0,
        fTsumw=3.0,
        fTsumw2=3.0,
        fTsumwx=-3.5,
        fTsumwx2=26.51,
        fTsumwy=50.0,
        fTsumwy2=900.0,
        fSumw2=np.array([100, 500, 400, 400], np.float64),
        fBinEntries=np.array([1, 2, 1, 1], np.float64),
        fBinSumw2=np.array([], np.float64),
        fXaxis=uproot.writing.identify.to_TAxis(
            fName="xaxis",
            fTitle="",
            fNbins=2,
            fXmin=-3.14,
            fXmax=2.71,
        ),
    )

    with uproot.recreate(newfile) as fin:
        fin["hhist"] = hhist
        fin["uhist"] = uhist

    f3 = ROOT.TFile(newfile)
    h3 = f3.Get("hhist")
    h4 = f3.Get("uhist")

    assert h3.GetEntries() == h4.GetEntries() == 5
    assert h3.GetSumOfWeights() == h4.GetSumOfWeights() == 35
    assert h3.GetBinLowEdge(1) == h4.GetBinLowEdge(1) == pytest.approx(-3.14)
    assert h3.GetBinWidth(1) == h4.GetBinWidth(1) == pytest.approx((2.71 + 3.14) / 2)
    assert h3.GetBinContent(0) == h4.GetBinContent(0) == pytest.approx(10)
    assert h3.GetBinContent(1) == h4.GetBinContent(1) == pytest.approx(15)
    assert h3.GetBinContent(2) == h4.GetBinContent(2) == pytest.approx(20)
    assert h3.GetBinContent(3) == h4.GetBinContent(3) == pytest.approx(20)
    assert h3.GetBinError(0) == h4.GetBinError(0) == pytest.approx(0)
    assert h3.GetBinError(1) == h4.GetBinError(1) == pytest.approx(np.sqrt(12.5))
    assert h3.GetBinError(2) == h4.GetBinError(2) == pytest.approx(0)
    assert h3.GetBinError(3) == h4.GetBinError(3) == pytest.approx(0)

    f3.Close()

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import os

import numpy as np
import pytest

import uproot
from uproot.writing.identify import (
    to_TAxis,
    to_TH1x,
    to_THashList,
    to_TObjString,
    to_TProfile,
)

pytest.importorskip("hist")


def _axis(nbins, labels):
    # labels: list of (bin number, label) in the order they were set,
    # as ROOT's TAxis::SetBinLabel stores them
    fLabels = to_THashList([to_TObjString(label) for _, label in labels])
    for (bin_number, _), label in zip(labels, fLabels):
        label._bases[0]._members["@fUniqueID"] = bin_number
    return to_TAxis(
        fName="xaxis", fTitle="", fNbins=nbins, fXmin=0, fXmax=nbins, fLabels=fLabels
    )


def _th1(nbins, labels, data):
    return to_TH1x(
        fName="h",
        fTitle="",
        data=np.array(data, np.float64),
        fEntries=2.0,
        fTsumw=2.0,
        fTsumw2=2.0,
        fTsumwx=0.0,
        fTsumwx2=0.0,
        fSumw2=np.array(data, np.float64) ** 2,
        fXaxis=_axis(nbins, labels),
    )


def _tprofile(nbins, labels, data):
    data = np.array(data, np.float64)
    entries = (data != 0).astype(np.float64)
    return to_TProfile(
        fName="p",
        fTitle="",
        data=data,
        fEntries=2.0,
        fTsumw=2.0,
        fTsumw2=2.0,
        fTsumwx=0.0,
        fTsumwx2=0.0,
        fTsumwy=data.sum(),
        fTsumwy2=(data**2).sum(),
        fSumw2=data**2,
        fBinEntries=entries,
        fBinSumw2=np.array([], np.float64),
        fXaxis=_axis(nbins, labels),
    )


@pytest.mark.parametrize("make", [_th1, _tprofile])
def test_labels_out_of_order(tmp_path, make):
    filepath = os.path.join(tmp_path, "test.root")
    with uproot.recreate(filepath) as f:
        f["h"] = make(3, [(3, "c"), (1, "a"), (2, "b")], [0, 10, 0, 30, 0])

    with uproot.open(filepath) as f:
        h = f["h"]
        axis = h.member("fXaxis")
        assert axis.labels() == ["a", "b", "c"]
        assert list(axis) == ["a", "b", "c"]
        assert axis[2] == "c"
        assert axis.traits.discrete

        out = h.to_hist()

    assert list(out.axes[0]) == ["a", "b", "c"]
    assert out.values().tolist() == [10, 0, 30]
    assert out[{0: "c"}].value == 30


@pytest.mark.parametrize("make", [_th1, _tprofile])
def test_labels_partial(tmp_path, make):
    filepath = os.path.join(tmp_path, "test.root")
    with uproot.recreate(filepath) as f:
        f["h"] = make(6, [(1, "a"), (2, "b")], [0, 10, 0, 0, 0, 50, 0, 0])

    with uproot.open(filepath) as f:
        h = f["h"]
        axis = h.member("fXaxis")
        assert axis.labels() is None
        assert not axis.traits.discrete

        out = h.to_hist()

    assert out.axes[0].edges.tolist() == [0, 1, 2, 3, 4, 5, 6]
    assert out.values().tolist() == [10, 0, 0, 0, 50, 0]


def test_labels_pyroot(tmp_path):
    ROOT = pytest.importorskip("ROOT")

    filepath = os.path.join(tmp_path, "test.root")
    f = ROOT.TFile(filepath, "RECREATE")
    h = ROOT.TH2D("h2_1766_labels", "", 2, 0, 2, 3, 0, 3)
    h.SetDirectory(f)
    for bin_number, label in [(2, "y"), (1, "x")]:
        h.GetXaxis().SetBinLabel(bin_number, label)
    for bin_number, label in [(3, "w"), (2, "v"), (1, "u")]:
        h.GetYaxis().SetBinLabel(bin_number, label)
    h.Fill(1.5, 2.5, 7)
    p = ROOT.TProfile("p_1766_labels", "", 6, 0, 6)
    p.SetDirectory(f)
    p.GetXaxis().SetBinLabel(2, "b")
    p.Fill(4.5, 5)
    f.Write()
    f.Close()

    with uproot.open(filepath) as f:
        h = f["h2_1766_labels"].to_hist()
        p = f["p_1766_labels"].to_hist()

    assert list(h.axes[0]) == ["x", "y"]
    assert list(h.axes[1]) == ["u", "v", "w"]
    assert h[{0: "y", 1: "w"}].value == 7
    assert h.values().sum() == 7

    assert p.axes[0].edges.tolist() == [0, 1, 2, 3, 4, 5, 6]
    assert p.values().tolist() == [0, 0, 0, 0, 5, 0]

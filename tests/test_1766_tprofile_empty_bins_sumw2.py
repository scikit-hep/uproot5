# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import os

import numpy as np
import pytest

import uproot

hist = pytest.importorskip("hist")


def test_weighted_empty_bins_roundtrip(tmp_path):
    h = hist.Hist(hist.axis.Regular(3, 0, 3), storage=hist.storage.WeightedMean())
    h.fill([0.5, 0.5], sample=[1.0, 3.0], weight=[2.0, 1.0])

    filepath = os.path.join(tmp_path, "test.root")
    with uproot.recreate(filepath) as f:
        f["h"] = h

    with uproot.open(filepath) as f:
        p = f["h"]
        out = p.to_hist()

    view = out.view(flow=True)
    assert not np.isnan(view["sum_of_weights_squared"]).any()
    assert view["sum_of_weights_squared"].tolist() == [0.0, 5.0, 0.0, 0.0, 0.0]
    assert out.counts() == pytest.approx(p.counts())
    assert out.counts() == pytest.approx([9.0 / 5.0, 0.0, 0.0])


def test_weighted_empty_bins_labels(tmp_path):
    h = hist.Hist(
        hist.axis.StrCategory(["a", "b", "c"]), storage=hist.storage.WeightedMean()
    )
    h.fill(["a", "a"], sample=[1.0, 3.0], weight=[2.0, 1.0])

    filepath = os.path.join(tmp_path, "test.root")
    with uproot.recreate(filepath) as f:
        f["h"] = h

    with uproot.open(filepath) as f:
        out = f["h"].to_hist()

    assert not np.isnan(out.view(flow=True)["sum_of_weights_squared"]).any()
    assert out.counts() == pytest.approx([9.0 / 5.0, 0.0, 0.0])


def test_weighted_empty_bins_pyroot(tmp_path):
    ROOT = pytest.importorskip("ROOT")

    filepath = os.path.join(tmp_path, "root.root")
    p = ROOT.TProfile("p_1766", "", 3, 0, 3)
    p.Sumw2()
    p.Fill(0.5, 1, 2)
    p.Fill(0.5, 3, 1)
    f = ROOT.TFile(filepath, "RECREATE")
    p.Write()
    f.Close()

    with uproot.open(filepath) as f:
        h = f["p_1766"].to_hist()

    assert not np.isnan(h.view(flow=True)["sum_of_weights_squared"]).any()

    roundtrip = os.path.join(tmp_path, "roundtrip.root")
    with uproot.recreate(roundtrip) as f:
        f["p_1766"] = h

    f = ROOT.TFile(roundtrip)
    r = f.Get("p_1766")
    other = ROOT.TProfile("other_1766", "", 3, 0, 3)
    other.Sumw2()
    other.Fill(1.5, 2, 1.5)
    other.Fill(1.5, 4, 0.5)
    r.Add(other)

    for i in range(5):
        assert np.isfinite(r.GetBinError(i))
    assert r.GetBinEffectiveEntries(1) == pytest.approx(9.0 / 5.0)
    assert r.GetBinEffectiveEntries(2) == pytest.approx(4.0 / 2.5)
    f.Close()

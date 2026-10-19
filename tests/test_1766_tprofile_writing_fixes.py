# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import os

import numpy as np
import pytest

import uproot

hist = pytest.importorskip("hist")

# (axis, has underflow, has overflow, bin index -> fill value)
AXES = {
    "Regular": (hist.axis.Regular(4, 0, 4), True, True, lambda i: i + 0.5),
    "Variable": (hist.axis.Variable([0, 1, 3, 6]), True, True, None),
    "Integer": (hist.axis.Integer(0, 4), True, True, lambda i: i),
    "Regular-noflow": (
        hist.axis.Regular(4, 0, 4, underflow=False, overflow=False),
        False,
        False,
        lambda i: i + 0.5,
    ),
    "Regular-nounderflow": (
        hist.axis.Regular(4, 0, 4, underflow=False),
        False,
        True,
        lambda i: i + 0.5,
    ),
    "IntCategory": (
        hist.axis.IntCategory([10, 20, 30]),
        False,
        True,
        lambda i: [10, 20, 30, 99][i],
    ),
    "IntCategory-noflow": (
        hist.axis.IntCategory([10, 20, 30], overflow=False),
        False,
        False,
        lambda i: [10, 20, 30][i],
    ),
    "StrCategory": (
        hist.axis.StrCategory(["a", "b", "c"]),
        False,
        True,
        lambda i: ["a", "b", "c", "zz"][i],
    ),
    "Boolean": (hist.axis.Boolean(), False, False, bool),
}


def _fill(name, storage, seed=0):
    axis, underflow, overflow, to_value = AXES[name]
    rng = np.random.default_rng(seed)
    n = len(axis)
    edges = axis.edges
    index = rng.integers(-1 if underflow else 0, (n if overflow else n - 1) + 1, 200)
    y = rng.normal(5, 2, len(index))
    if storage == "WeightedMean":
        w = rng.uniform(-0.5, 2, len(index))
    else:
        w = np.ones(len(index))

    # ROOT x-coordinates: bin centers, with flow bins just outside the edges
    centers = np.concatenate(
        [[edges[0] - 0.5], (edges[:-1] + edges[1:]) / 2, [edges[-1] + 0.5]]
    )
    x = centers[index + 1]

    h = hist.Hist(axis, storage=getattr(hist.storage, storage)())
    values = x if to_value is None else [to_value(i) for i in index]
    if storage == "WeightedMean":
        h.fill(values, sample=y, weight=w)
    else:
        h.fill(values, sample=y)
    return h, x, y, w, index


@pytest.mark.parametrize("storage", ["Mean", "WeightedMean"])
@pytest.mark.parametrize("name", list(AXES))
def test_axes(tmp_path, name, storage):
    h, x, y, w, index = _fill(name, storage)
    n = len(h.axes[0])

    filepath = os.path.join(tmp_path, "test.root")
    with uproot.recreate(filepath) as f:
        f["p"] = h

    with uproot.open(filepath) as f:
        p = f["p"]
        bin_entries = np.asarray(p.member("fBinEntries"))

        expected = np.zeros(n + 2)
        np.add.at(expected, index + 1, w)
        assert bin_entries == pytest.approx(expected)

        if storage == "Mean":
            assert len(p.member("fBinSumw2")) == 0
            assert p.member("fEntries") == len(index)
        else:
            assert len(p.member("fBinSumw2")) == n + 2
            assert p.member("fEntries") == pytest.approx(w.sum())

        inside = (index >= 0) & (index < n)
        assert p.member("fTsumw") == pytest.approx(w[inside].sum())
        assert p.member("fTsumwx") == pytest.approx((w * x)[inside].sum())
        assert p.member("fTsumwx2") == pytest.approx((w * x**2)[inside].sum())
        assert p.member("fTsumwy") == pytest.approx((w * y)[inside].sum())
        assert p.member("fTsumwy2") == pytest.approx((w * y**2)[inside].sum())

        out = p.to_hist()

    assert out.storage_type is h.storage_type
    assert out.values() == pytest.approx(h.values(), nan_ok=True)


@pytest.mark.parametrize("storage", ["Mean", "WeightedMean"])
@pytest.mark.parametrize("name", list(AXES))
def test_axes_pyroot(tmp_path, name, storage):
    ROOT = pytest.importorskip("ROOT")

    h, x, y, w, _ = _fill(name, storage)
    n = len(h.axes[0])

    expected = ROOT.TProfile(
        f"expected_1766_write_{name}_{storage}",
        "",
        n,
        np.asarray(h.axes[0].edges, "f8"),
    )
    expected.SetDirectory(ROOT.nullptr)
    if storage == "WeightedMean":
        expected.Sumw2()
    for xi, yi, wi in zip(x, y, w):
        expected.Fill(xi, yi, wi)

    filepath = os.path.join(tmp_path, "test.root")
    with uproot.recreate(filepath) as f:
        f["p"] = h

    f = ROOT.TFile(filepath)
    p = f.Get("p")
    for i in range(n + 2):
        assert p.GetBinContent(i) == pytest.approx(expected.GetBinContent(i))
        assert p.GetBinError(i) == pytest.approx(expected.GetBinError(i))
        assert p.GetBinEntries(i) == pytest.approx(expected.GetBinEntries(i))
        assert p.GetBinEffectiveEntries(i) == pytest.approx(
            expected.GetBinEffectiveEntries(i)
        )
    for axis in (1, 2):
        assert p.GetMean(axis) == pytest.approx(expected.GetMean(axis))
        assert p.GetStdDev(axis) == pytest.approx(expected.GetStdDev(axis))
        assert p.GetMeanError(axis) == pytest.approx(expected.GetMeanError(axis))
    assert p.GetEffectiveEntries() == pytest.approx(expected.GetEffectiveEntries())
    assert p.GetBinSumw2().GetSize() == expected.GetBinSumw2().GetSize()
    if storage == "Mean":
        assert p.GetEntries() == expected.GetEntries()
    f.Close()


def test_non_dict_metadata(tmp_path):
    h = hist.Hist(
        hist.axis.Regular(3, 0, 3), storage=hist.storage.Mean(), metadata="info"
    )
    h.fill([1, 5], sample=[1.0, 2.0])

    filepath = os.path.join(tmp_path, "test.root")
    with uproot.recreate(filepath) as f:
        f["p"] = h

    with uproot.open(filepath) as f:
        assert f["p"].member("fEntries") == 2


def test_non_boost_mean_raises(tmp_path):
    h = hist.Hist(hist.axis.Regular(3, 0, 3), storage=hist.storage.Mean())

    class MeanHistogram:
        kind = "MEAN"
        axes = h.axes

        def values(self, flow=False):
            return h.values(flow=flow)

        def variances(self, flow=False):
            return h.variances(flow=flow)

        def counts(self, flow=False):
            return h.counts(flow=flow)

    filepath = os.path.join(tmp_path, "test.root")
    with uproot.recreate(filepath) as f:
        with pytest.raises(TypeError, match="Mean or WeightedMean storage"):
            f["p"] = MeanHistogram()


@pytest.mark.parametrize("error_mode", [0, 1, 2, 3])
def test_error_mode_roundtrip(tmp_path, error_mode):
    filepath = os.path.join(tmp_path, "test.root")
    with uproot.recreate(filepath) as f:
        f["p"] = uproot.writing.identify.to_TProfile(
            fName="p",
            fTitle="",
            data=np.array([0.0, 3.0, 0.0], np.float64),
            fEntries=2.0,
            fTsumw=2.0,
            fTsumw2=2.0,
            fTsumwx=1.0,
            fTsumwx2=0.5,
            fTsumwy=3.0,
            fTsumwy2=5.0,
            fSumw2=np.array([0.0, 5.0, 0.0], np.float64),
            fBinEntries=np.array([0.0, 2.0, 0.0], np.float64),
            fBinSumw2=np.array([], np.float64),
            fXaxis=uproot.writing.identify.to_TAxis(
                fName="xaxis", fTitle="", fNbins=1, fXmin=0, fXmax=1
            ),
            fErrorMode=error_mode,
        )

    with uproot.open(filepath) as f:
        h = f["p"].to_hist()

    assert h.metadata["fErrorMode"] == error_mode

    roundtrip = os.path.join(tmp_path, "roundtrip.root")
    with uproot.recreate(roundtrip) as f:
        f["p"] = h

    with uproot.open(roundtrip) as f:
        assert f["p"].member("fErrorMode") == error_mode

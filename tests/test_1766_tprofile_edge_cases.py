# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import os

import numpy as np
import pytest
import skhep_testdata

import uproot


def _weighted_profile(fSumw2):
    return uproot.writing.identify.to_TProfile(
        fName="p",
        fTitle="",
        data=np.array([0.0, 3.0, 0.0], np.float64),
        fEntries=3.0,
        fTsumw=1.0,
        fTsumw2=5.0,
        fTsumwx=0.5,
        fTsumwx2=0.25,
        fTsumwy=3.0,
        fTsumwy2=fSumw2,
        fSumw2=np.array([0.0, fSumw2, 0.0], np.float64),
        fBinEntries=np.array([0.0, 1.0, 0.0], np.float64),
        fBinSumw2=np.array([0.0, 5.0, 0.0], np.float64),
        fXaxis=uproot.writing.identify.to_TAxis(
            fName="xaxis", fTitle="", fNbins=1, fXmin=0, fXmax=1
        ),
    )


def test_negative_sum_of_squared_deviations(tmp_path):
    pytest.importorskip("hist")

    # two fills with weights 2 and -1 and y = 3 +- sqrt(2.5), 3 +- 2 sqrt(2.5):
    # sum(w) = 1, sum(w^2) = 5, mean = 3, sum(w*y^2) = 4 < sum(w) * mean^2 = 9
    filepath = os.path.join(tmp_path, "test.root")
    with uproot.recreate(filepath) as f:
        f["p"] = _weighted_profile(4.0)

    with uproot.open(filepath) as f:
        p = f["p"]
        errors = p.errors(flow=True)
        h = p.to_hist()

    assert h.view(flow=True)["_sum_of_weighted_deltas_squared"][1] == pytest.approx(
        4.0 - 9.0
    )

    roundtrip = os.path.join(tmp_path, "roundtrip.root")
    with uproot.recreate(roundtrip) as f:
        f["p"] = h

    with uproot.open(roundtrip) as f:
        assert np.asarray(f["p"].member("fSumw2")) == pytest.approx([0, 4, 0])
        assert f["p"].errors(flow=True) == pytest.approx(errors)


def test_round_off_is_clamped(tmp_path):
    pytest.importorskip("hist")

    filepath = os.path.join(tmp_path, "test.root")
    with uproot.recreate(filepath) as f:
        f["p"] = _weighted_profile(9.0 * (1 - 1e-15))

    with uproot.open(filepath) as f:
        h = f["p"].to_hist()

    assert h.view(flow=True)["_sum_of_weighted_deltas_squared"][1] == 0


def test_negative_weights_pyroot(tmp_path):
    ROOT = pytest.importorskip("ROOT")
    pytest.importorskip("hist")

    rng = np.random.default_rng(0)
    p = ROOT.TProfile("p_1766_edge", "", 4, 0, 4)
    p.SetDirectory(ROOT.nullptr)
    p.Sumw2()
    for x, y, w in zip(
        rng.uniform(0, 4, 40), rng.normal(5, 2, 40), rng.uniform(-1, 1.2, 40)
    ):
        p.Fill(x, y, w)

    filepath = os.path.join(tmp_path, "test.root")
    f = ROOT.TFile(filepath, "RECREATE")
    p.Write()
    f.Close()

    with uproot.open(filepath) as f:
        h = f["p_1766_edge"].to_hist()

    roundtrip = os.path.join(tmp_path, "roundtrip.root")
    with uproot.recreate(roundtrip) as f:
        f["p_1766_edge"] = h

    f = ROOT.TFile(roundtrip)
    q = f.Get("p_1766_edge")
    for i in range(6):
        assert q.GetBinContent(i) == pytest.approx(p.GetBinContent(i))
        assert q.GetBinError(i) == pytest.approx(p.GetBinError(i))
    f.Close()


def test_equality():
    with uproot.open(skhep_testdata.data_path("uproot-hepdata-example.root")) as f:
        assert f["hprof"] == f["hprof"]
        assert not (f["hprof"] == f["hpx"])

    with uproot.open(skhep_testdata.data_path("uproot-issue-227a.root")) as f:
        key = next(k for k, v in f.classnames().items() if v == "TProfile2D")
        assert f[key] == f[key]


@pytest.mark.parametrize(
    "filename, classname",
    [
        ("uproot-hepdata-example.root", "TProfile"),
        ("uproot-issue-227a.root", "TProfile2D"),
        ("uproot-issue-227b.root", "TProfile3D"),
    ],
)
def test_missing_fBinSumw2(filename, classname):
    # TProfile class versions before 6 (ROOT < 5.26) have no fBinSumw2
    filepath = skhep_testdata.data_path(filename)
    with uproot.open(filepath) as f:
        key = next(k for k, v in f.classnames().items() if v == classname)
        p = f[key]
        expected_counts = np.asarray(p.member("fBinEntries"))
        values = p.values(flow=True)
        errors = p.errors(flow=True)
        assert not p.weighted

    # reopen to get an object without cached values and errors
    with uproot.open(filepath, object_cache=None) as f:
        p = f[key]
        del p._members["fBinSumw2"]

        assert p.counts(flow=True).reshape(-1, order="F") == pytest.approx(
            expected_counts
        )
        assert p.values(flow=True) == pytest.approx(values)
        assert p.errors(flow=True) == pytest.approx(errors)
        assert p.variances(flow=True) == pytest.approx(errors**2)
        if classname == "TProfile":
            pytest.importorskip("hist")
            assert p.to_hist().values(flow=True) == pytest.approx(values)

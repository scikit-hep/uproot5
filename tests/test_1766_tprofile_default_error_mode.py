# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import os

import numpy as np
import pytest

import uproot


def _write_profile(filepath, error_mode):
    with uproot.recreate(filepath) as f:
        f["p"] = uproot.writing.identify.to_TProfile(
            fName="p",
            fTitle="",
            data=np.array([0.0, 4.0, 6.0, 0.0], np.float64),
            fEntries=4.0,
            fTsumw=4.0,
            fTsumw2=4.0,
            fTsumwx=4.0,
            fTsumwx2=5.0,
            fTsumwy=10.0,
            fTsumwy2=28.0,
            fSumw2=np.array([0.0, 10.0, 18.0, 0.0], np.float64),
            fBinEntries=np.array([0.0, 2.0, 2.0, 0.0], np.float64),
            fBinSumw2=np.array([], np.float64),
            fXaxis=uproot.writing.identify.to_TAxis(
                fName="xaxis", fTitle="", fNbins=2, fXmin=0, fXmax=2
            ),
            fErrorMode=error_mode,
        )


@pytest.mark.parametrize("error_mode, name", [(0, ""), (1, "s"), (2, "i"), (3, "g")])
def test_default_uses_fErrorMode(tmp_path, error_mode, name):
    filepath = os.path.join(tmp_path, "test.root")
    _write_profile(filepath, error_mode)

    with uproot.open(filepath) as f:
        p = f["p"]
        assert p.errors() == pytest.approx(p.errors(error_mode=name))
        assert p.errors(flow=True) == pytest.approx(
            p.errors(flow=True, error_mode=name)
        )
        assert p.variances() == pytest.approx(p.variances(error_mode=name))


def test_explicit_error_mode(tmp_path):
    filepath = os.path.join(tmp_path, "test.root")
    _write_profile(filepath, 1)

    with uproot.open(filepath) as f:
        p = f["p"]
        # bin 1: y = {1, 3}; bin 2: y = {3, 3}
        assert p.errors() == pytest.approx([1, 0])
        assert p.errors(error_mode="s") == pytest.approx([1, 0])
        assert p.errors(error_mode="") == pytest.approx([1 / np.sqrt(2), 0])
        assert p.errors(error_mode="i") == pytest.approx(
            [1 / np.sqrt(2), 1 / np.sqrt(12 * 2)]
        )


@pytest.mark.parametrize("option", ["", "s", "i", "g"])
@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_default_error_mode_pyroot(tmp_path, option, ndim):
    ROOT = pytest.importorskip("ROOT")

    rng = np.random.default_rng(ndim)
    name = f"p_1766_errmode_{ndim}_{option or 'mean'}"
    if ndim == 1:
        p = ROOT.TProfile(name, "", 3, 0, 3, option)
    elif ndim == 2:
        p = ROOT.TProfile2D(name, "", 3, 0, 3, 2, 0, 2, option)
    else:
        p = ROOT.TProfile3D(name, "", 3, 0, 3, 2, 0, 2, 2, 0, 2, option)
    p.SetDirectory(ROOT.nullptr)
    for _ in range(2000):
        x = rng.uniform(-0.5, 3.5, 4)
        p.Fill(*x[:ndim], rng.normal(5, 2))

    filepath = os.path.join(tmp_path, "test.root")
    f = ROOT.TFile(filepath, "RECREATE")
    p.Write()
    f.Close()

    with uproot.open(filepath) as f:
        errors = f[name].errors(flow=True)

    for index in np.ndindex(errors.shape):
        expected = p.GetBinError(p.GetBin(*index))
        assert errors[index] == pytest.approx(expected, rel=1e-6, abs=1e-6)

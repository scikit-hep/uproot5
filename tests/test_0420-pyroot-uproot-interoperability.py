# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

import array
import os

import numpy as np
import pytest

import uproot

ROOT = pytest.importorskip("ROOT")


def test_from_pyroot():
    pyroot_hist = ROOT.TH1F("hist", "title", 10, -5, 5)
    uproot_hist = uproot.from_pyroot(pyroot_hist)
    assert isinstance(uproot_hist, uproot.Model)
    assert isinstance(uproot_hist, uproot.behaviors.TH1.TH1)
    assert isinstance(uproot_hist.values(), np.ndarray)
    assert isinstance(uproot_hist.axis().edges(), np.ndarray)

    pyroot_vec = ROOT.TLorentzVector(1, 2, 3, 4)
    uproot_vec = uproot.from_pyroot(pyroot_vec)
    assert isinstance(uproot_vec, uproot.Model)

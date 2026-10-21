# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import os

import numpy as np
import pytest

import uproot

hist = pytest.importorskip("hist")


def _write_and_read(tmp_path, obj):
    filepath = os.path.join(tmp_path, "test.root")
    with uproot.recreate(filepath) as f:
        f["h"] = obj
    return uproot.open(filepath)["h"]



@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_weighted_global_statistics(tmp_path, ndim):
    rng = np.random.default_rng(ndim)
    axes = [hist.axis.Regular(3, 0, 3)] * ndim
    h = hist.Hist(*axes, storage=hist.storage.Weight())
    x = rng.uniform(-0.5, 3.5, (ndim, 100))
    w = rng.uniform(0.5, 2, 100)
    h.fill(*x, weight=w)

    out = _write_and_read(tmp_path, h)
    inside = ((x >= 0) & (x < 3)).all(axis=0)
    assert out.member("fTsumw") == pytest.approx(w[inside].sum())
    # the sum of squared weights, which gives ROOT's effective entries
    assert out.member("fTsumw2") == pytest.approx((w[inside] ** 2).sum())

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import uproot
import pytest
import numpy as np
import os

hist = pytest.importorskip("hist")


def test_writing_tprofile(tmp_path):
    axis_x = hist.axis.IntCategory(range(10))
    h = hist.Hist(axis_x, storage=hist.storage.Mean())

    sample0 = np.array([10, 20, 30, 10])
    sample1 = np.array([10, 20, 10, 10, 0])
    h.fill([0, 0, 0, 0], sample=sample0)
    h.fill([1, 1, 1, 1, 1], sample=sample1)

    expected_count0 = len(sample0)
    expected_count1 = len(sample1)
    expected_mean0 = np.mean(sample0)
    expected_mean1 = np.mean(sample1)
    expected_variance0 = np.var(sample0) / (len(sample0) - 1)
    expected_variance1 = np.var(sample1) / (len(sample1) - 1)

    hist_count0 = h[0].count
    hist_count1 = h[1].count
    hist_mean0 = h[0].value
    hist_mean1 = h[1].value
    hist_variance0 = h.variances()[0]
    hist_variance1 = h.variances()[1]

    filepath = os.path.join(tmp_path, "test.root")

    with uproot.recreate(filepath) as f:
        f["h"] = h

    with uproot.open(filepath) as f:
        h = f["h"].to_hist()

        uproot_mean0 = h[0].value
        uproot_mean1 = h[1].value
        uproot_variance0 = h.variances()[0]
        uproot_variance1 = h.variances()[1]

    assert np.isclose(expected_count0, hist_count0)
    assert np.isclose(expected_count1, hist_count1)
    assert np.isclose(expected_mean0, hist_mean0)
    assert np.isclose(expected_mean1, hist_mean1)
    assert np.isclose(expected_variance0, hist_variance0)
    assert np.isclose(expected_variance1, hist_variance1)


def test_tprofile_pyroot_comparison(tmp_path):
    ROOT = pytest.importorskip("ROOT")

    axis_x = hist.axis.IntCategory(range(10))
    h = hist.Hist(axis_x, storage=hist.storage.Mean())

    sample0 = np.array([10, 20, 30, 10])
    sample1 = np.array([10, 20, 10, 10, 0])
    h.fill([0, 0, 0, 0], sample=sample0)
    h.fill([1, 1, 1, 1, 1], sample=sample1)

    expected_count0 = len(sample0)
    expected_count1 = len(sample1)
    expected_mean0 = np.mean(sample0)
    expected_mean1 = np.mean(sample1)
    expected_variance0 = np.var(sample0) / (len(sample0) - 1)
    expected_variance1 = np.var(sample1) / (len(sample1) - 1)

    hist_count0 = h[0].count
    hist_count1 = h[1].count
    hist_mean0 = h[0].value
    hist_mean1 = h[1].value
    hist_variance0 = h.variances()[0]
    hist_variance1 = h.variances()[1]

    filepath = os.path.join(tmp_path, "test.root")

    with uproot.recreate(filepath) as f:
        f["h"] = h

    with uproot.open(filepath) as f:
        h = f["h"].to_hist()

        uproot_mean0 = h[0].value
        uproot_mean1 = h[1].value
        uproot_variance0 = h.variances()[0]
        uproot_variance1 = h.variances()[1]

    # Create an equivalent TProfile in ROOT using a regular axis.
    # Bins span [-0.5, 9.5] so that filling at x=0,1 maps to ROOT bins 1,2.
    root_tp = ROOT.TProfile("root_tp", "", 10, -0.5, 9.5)
    for v in sample0:
        root_tp.Fill(0, v)
    for v in sample1:
        root_tp.Fill(1, v)

    # Read the ROOT TProfile back with uproot and compare
    h_from_root = uproot.from_pyroot(root_tp).to_hist()
    root_mean0 = h_from_root[0].value
    root_mean1 = h_from_root[1].value
    root_variance0 = h_from_root.variances()[0]
    root_variance1 = h_from_root.variances()[1]

    assert np.isclose(expected_mean0, root_mean0)
    assert np.isclose(expected_mean1, root_mean1)
    assert np.isclose(expected_variance0, root_variance0)
    assert np.isclose(expected_variance1, root_variance1)

    # Also verify M2 consistency: ROOT's kERRORMEAN gives GetBinError = sqrt(pop_var/n),
    # so M2 = GetBinError^2 * n^2.  uproot's variances() = M2 / (n*(n-1)),
    # so M2 = variances * n*(n-1).  Both should give the same M2.
    for root_bin, (samples, from_uproot_var) in enumerate(
        [(sample0, uproot_variance0), (sample1, uproot_variance1)], start=1
    ):
        n = len(samples)
        root_error = root_tp.GetBinError(root_bin)
        m2_from_root_error = root_error**2 * n**2  # GetBinError^2 = pop_var/n = M2/n^2
        m2_from_uproot = from_uproot_var * n * (n - 1)  # variances = M2/(n*(n-1))
        assert np.isclose(m2_from_root_error, m2_from_uproot)


@pytest.mark.parametrize("storage", ["WeightedMean", "Mean"])
def test_roundtrip(tmp_path, storage):
    axis = hist.axis.Regular(10, 0, 10)

    rng = np.random.default_rng(seed=42)
    x = rng.uniform(0, 10, size=100)
    sample = rng.normal(5, 2, size=100)

    if storage == "WeightedMean":
        h = hist.Hist(axis, storage=hist.storage.WeightedMean())
        weight = rng.uniform(0.5, 2.0, size=100)
        h.fill(x, sample=sample, weight=weight)
    else:
        h = hist.Hist(axis, storage=hist.storage.Mean())
        h.fill(x, sample=sample)

    expected_counts = h.counts()
    expected_density = h.density()
    expected_values = h.values()
    expected_variances = h.variances()

    filepath = os.path.join(tmp_path, "test.root")
    with uproot.recreate(filepath) as f:
        f["h"] = h

    with uproot.open(filepath) as f:
        h = f["h"].to_hist()

        uproot_counts = h.counts()
        uproot_density = h.density()
        uproot_values = h.values()
        uproot_variances = h.variances()

    assert np.allclose(expected_counts, uproot_counts, equal_nan=True)
    assert np.allclose(expected_density, uproot_density, equal_nan=True)
    assert np.allclose(expected_values, uproot_values, equal_nan=True)
    assert np.allclose(expected_variances, uproot_variances, equal_nan=True)


def test_tprofile_strcategory(tmp_path):
    axis_x = hist.axis.StrCategory(["A", "B", "C"])
    h = hist.Hist(axis_x, storage=hist.storage.Mean())

    sample_A = np.array([10, 20, 30])
    sample_B = np.array([5, 15])
    h.fill(["A", "A", "A"], sample=sample_A)
    h.fill(["B", "B"], sample=sample_B)

    expected_count_A = len(sample_A)
    expected_count_B = len(sample_B)
    expected_mean_A = np.mean(sample_A)
    expected_mean_B = np.mean(sample_B)
    expected_variance_A = np.var(sample_A, ddof=1)
    expected_variance_B = np.var(sample_B, ddof=1)

    # boost-histogram Mean storage variances() returns variance of the mean (SEM squared)
    assert np.isclose(expected_variance_A / expected_count_A, h.variances()[0])
    assert np.isclose(expected_variance_B / expected_count_B, h.variances()[1])

    filepath = os.path.join(tmp_path, "test_str.root")

    with uproot.recreate(filepath) as f:
        f["h"] = h

    with uproot.open(filepath) as f:
        h_read = f["h"].to_hist()

        # h_read will have WeightedMean storage (as all TProfiles do)
        uproot_mean_A = h_read["A"].value
        uproot_mean_B = h_read["B"].value
        # boost-histogram WeightedMean storage variances() returns s^2 / n (SEM squared)
        uproot_variance_A = h_read.variances()[0]
        uproot_variance_B = h_read.variances()[1]

    assert np.isclose(expected_count_A, h_read["A"].sum_of_weights)
    assert np.isclose(expected_count_B, h_read["B"].sum_of_weights)
    assert np.isclose(expected_mean_A, uproot_mean_A)
    assert np.isclose(expected_mean_B, uproot_mean_B)
    assert np.isclose(expected_variance_A / expected_count_A, uproot_variance_A)
    assert np.isclose(expected_variance_B / expected_count_B, uproot_variance_B)

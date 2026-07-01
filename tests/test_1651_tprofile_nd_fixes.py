# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
Regression tests for PR #1651: fix TProfile2D/3D weighted property and counts() shape.

Test files:
  uproot-issue-227a.root  — contains a TProfile2D named "hprof2d"  (2 x 3 bins)
  uproot-issue-227b.root  — contains a TProfile3D named "hprof3d"  (2 x 3 x 4 bins)
"""

import numpy
import pytest
import skhep_testdata

import uproot


@pytest.fixture(scope="module")
def h2():
    with uproot.open(skhep_testdata.data_path("uproot-issue-227a.root")) as f:
        yield f["hprof2d"]


@pytest.fixture(scope="module")
def h3():
    with uproot.open(skhep_testdata.data_path("uproot-issue-227b.root")) as f:
        yield f["hprof3d"]


# ---------------------------------------------------------------------------
# TProfile2D
# ---------------------------------------------------------------------------


def test_tprofile2d_weighted_no_raise(h2):
    """weighted must not raise TypeError (fNcells is int, not iterable)."""
    _ = h2.weighted  # previously raised "object of type 'int' has no len()"


def test_tprofile2d_weighted_value(h2):
    """Histogram was not filled with weights, so weighted should be False."""
    assert h2.weighted is False


def test_tprofile2d_counts_flow_true_shape(h2):
    """counts(flow=True) must have the same shape as values(flow=True)."""
    assert h2.counts(flow=True).shape == h2.values(flow=True).shape


def test_tprofile2d_counts_flow_false_shape(h2):
    """counts(flow=False) must have the same shape as values(flow=False)."""
    assert h2.counts(flow=False).shape == h2.values(flow=False).shape


def test_tprofile2d_counts_flow_false_no_index_error(h2):
    """counts() with default flow=False previously raised IndexError."""
    c = h2.counts()  # flow=False by default
    assert c.shape == (2, 3)


def test_tprofile2d_counts_flow_true_shape_value(h2):
    """counts(flow=True) shape should be (nx+2, ny+2)."""
    assert h2.counts(flow=True).shape == (4, 5)


def test_tprofile2d_counts_is_subset_of_flow(h2):
    """counts() with and without flow must agree on the inner region."""
    c_flow = h2.counts(flow=True)
    c_noflow = h2.counts(flow=False)
    numpy.testing.assert_array_equal(c_flow[1:-1, 1:-1], c_noflow)


def test_tprofile2d_counts_dtype(h2):
    """counts() must return a float64 array."""
    assert h2.counts().dtype == numpy.float64


# ---------------------------------------------------------------------------
# TProfile3D
# ---------------------------------------------------------------------------


def test_tprofile3d_weighted_no_raise(h3):
    """weighted must not raise TypeError (fNcells is int, not iterable)."""
    _ = h3.weighted  # previously raised "object of type 'int' has no len()"


def test_tprofile3d_weighted_value(h3):
    """Histogram was not filled with weights, so weighted should be False."""
    assert h3.weighted is False


def test_tprofile3d_counts_flow_true_shape(h3):
    """counts(flow=True) must have the same shape as values(flow=True)."""
    assert h3.counts(flow=True).shape == h3.values(flow=True).shape


def test_tprofile3d_counts_flow_false_shape(h3):
    """counts(flow=False) must have the same shape as values(flow=False)."""
    assert h3.counts(flow=False).shape == h3.values(flow=False).shape


def test_tprofile3d_counts_flow_false_no_index_error(h3):
    """counts() with default flow=False previously raised IndexError."""
    c = h3.counts()  # flow=False by default
    assert c.shape == (2, 3, 4)


def test_tprofile3d_counts_flow_true_shape_value(h3):
    """counts(flow=True) shape should be (nx+2, ny+2, nz+2)."""
    assert h3.counts(flow=True).shape == (4, 5, 6)


def test_tprofile3d_counts_is_subset_of_flow(h3):
    """counts() with and without flow must agree on the inner region."""
    c_flow = h3.counts(flow=True)
    c_noflow = h3.counts(flow=False)
    numpy.testing.assert_array_equal(c_flow[1:-1, 1:-1, 1:-1], c_noflow)


def test_tprofile3d_counts_dtype(h3):
    """counts() must return a float64 array."""
    assert h3.counts().dtype == numpy.float64

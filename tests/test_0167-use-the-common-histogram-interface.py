# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import numpy
import pytest
import skhep_testdata

import uproot4


def test_axis():
    with uproot4.open(skhep_testdata.data_path("uproot-hepdata-example.root")) as f:
        f["hpx"].axes[0] == f["hpx"].axis(0) == f["hpx"].axis(-1) == f["hpx"].axis("x")
        axis = f["hpx"].axis()
        assert len(axis) == 100
        assert axis[0] == (-4.0, -3.92)
        assert axis[1] == (-3.92, -3.84)
        assert list(axis)[:3] == [(-4.0, -3.92), (-3.92, -3.84), (-3.84, -3.76)]
        assert axis == axis
        assert f["hpxpy"].axis(0) == f["hpxpy"].axis(1)
        assert axis.circular is False
        assert axis.discrete is False
        assert axis.low == -4
        assert axis.high == 4
        assert axis.width == 0.08
        assert axis.labels() is None
        assert axis.edges()[:5].tolist() == [-4.0, -3.92, -3.84, -3.76, -3.68]
        assert axis.edges(True)[:4].tolist() == [-numpy.inf, -4.0, -3.92, -3.84]
        assert axis.intervals()[:2].tolist() == [[-4.0, -3.92], [-3.92, -3.84]]
        assert axis.intervals(True)[:2].tolist() == [[-numpy.inf, -4.0], [-4.0, -3.92]]
        assert axis.centers()[:4].tolist() == [-3.96, -3.88, -3.8, -3.7199999999999998]
        assert axis.centers()[:3].tolist() == [-3.96, -3.88, -3.8]
        assert axis.centers(True)[:3].tolist() == [-numpy.inf, -3.96, -3.88]
        assert axis.widths()[:3].tolist() == [0.08, 0.08, 0.08]
        assert axis.widths(True)[:2].tolist() == [numpy.inf, 0.08000000000000007]

    with uproot4.open(skhep_testdata.data_path("uproot-issue33.root")) as f:
        f["cutflow"].axes[0] == f["cutflow"].axis(0) == f["cutflow"].axis("x")
        axis = f["cutflow"].axis()
        assert len(axis) == 7
        assert axis[0] == "Dijet"
        assert axis[1] == "MET"
        assert list(axis)[:3] == ["Dijet", "MET", "MuonVeto"]
        assert axis == axis
        assert axis.circular is False
        assert axis.discrete is True
        assert axis.low == 0.0
        assert axis.high == 7.0
        assert axis.width == 1.0
        assert axis.labels()[:3] == ["Dijet", "MET", "MuonVeto"]
        assert list(axis) == axis.labels()
        assert axis.edges()[:5].tolist() == [0, 1, 2, 3, 4]
        assert axis.edges(True)[:4].tolist() == [-numpy.inf, 0, 1, 2]
        assert axis.intervals()[:2].tolist() == [[0, 1], [1, 2]]
        assert axis.intervals(True)[:2].tolist() == [[-numpy.inf, 0], [0, 1]]
        assert axis.centers()[:4].tolist() == [0.5, 1.5, 2.5, 3.5]
        assert axis.centers(True)[:3].tolist() == [-numpy.inf, 0.5, 1.5]
        assert axis.widths()[:3].tolist() == [1, 1, 1]
        assert axis.widths(True)[:3].tolist() == [numpy.inf, 1, 1]

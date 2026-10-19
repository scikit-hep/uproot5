# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations

import numpy
import pytest
import skhep_testdata

import uproot


def test_regular_axis_negative_index():
    with uproot.open(skhep_testdata.data_path("uproot-hepdata-example.root")) as f:
        axis = f["hpx"].axis()
        assert axis[-1] == pytest.approx((3.92, 4.0))
        assert axis[-100] == axis[0]
        assert axis[-1] == tuple(list(axis)[-1])
        with pytest.raises(IndexError):
            axis[100]
        with pytest.raises(IndexError):
            axis[-101]


def test_variable_axis_negative_index(tmp_path):
    edges = numpy.array([0.0, 1.0, 3.0, 7.0])
    with uproot.recreate(tmp_path / "variable.root") as f:
        f["h"] = (numpy.array([1.0, 2.0, 3.0]), edges)

    with uproot.open(tmp_path / "variable.root") as f:
        axis = f["h"].axis()
        assert axis[-1] == (3.0, 7.0)
        assert axis[-3] == axis[0] == (0.0, 1.0)
        with pytest.raises(IndexError):
            axis[3]
        with pytest.raises(IndexError):
            axis[-4]

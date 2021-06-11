# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

"""
This module defines the behaviors of ``TH3`` and its subclasses (not including
``TProfile3D``).
"""

from __future__ import absolute_import

import numpy

import uproot
from uproot.behaviors.TH1 import boost_axis_metadata, boost_metadata


class TH3(uproot.behaviors.TH1.Histogram):
    """
    Behaviors for three-dimensional histograms: descendants of ROOT's
    ``TH3``, not including ``TProfile3D``.
    """

    no_inherit = (uproot.behaviors.TH1.TH1,)

    @property
    def axes(self):
        return (self.member("fXaxis"), self.member("fYaxis"), self.member("fZaxis"))

    def axis(self, axis):
        if axis == 0 or axis == -3 or axis == "x":
            return self.member("fXaxis")
        elif axis == 1 or axis == -2 or axis == "y":
            return self.member("fYaxis")
        elif axis == 2 or axis == -1 or axis == "z":
            return self.member("fZaxis")
        else:
            raise ValueError(
                "axis must be 0 (-3), 1 (-2), 2 (-1) or 'x', 'y', 'z' for a TH3"
            )

    @property
    def weighted(self):
        sumw2 = self.member("fSumw2", none_if_missing=True)
        return sumw2 is not None and len(sumw2) == self.member("fNcells")

    @property
    def kind(self):
        return "COUNT"

    def values(self, flow=False):
        if hasattr(self, "_values"):
            values = self._values
        else:
            (values,) = self.base(uproot.models.TArray.Model_TArray)
            values = numpy.asarray(values, dtype=values.dtype.newbyteorder("="))

            xaxis_fNbins = self.member("fXaxis").member("fNbins")
            yaxis_fNbins = self.member("fYaxis").member("fNbins")
            zaxis_fNbins = self.member("fZaxis").member("fNbins")
            values = numpy.transpose(
                values.reshape(zaxis_fNbins + 2, yaxis_fNbins + 2, xaxis_fNbins + 2)
            )
            self._values = values

        if flow:
            return values
        else:
            return values[1:-1, 1:-1, 1:-1]

    def _values_variances(self, flow):
        values = self.values(flow=True)

        if hasattr(self, "_variances"):
            variances = self._variances
        else:
            variances = numpy.zeros(values.shape, dtype=numpy.float64)
            sumw2 = self.member("fSumw2", none_if_missing=True)
            if sumw2 is not None and len(sumw2) == self.member("fNcells"):
                sumw2 = numpy.asarray(sumw2, dtype=sumw2.dtype.newbyteorder("="))
                sumw2 = numpy.transpose(numpy.reshape(sumw2, values.shape[::-1]))
                positive = sumw2 > 0
                variances[positive] = sumw2[positive]
            else:
                positive = values > 0
                variances[positive] = values[positive]
            self._variances = variances

        if flow:
            return values, variances
        else:
            return values[1:-1, 1:-1, 1:-1], variances[1:-1, 1:-1, 1:-1]

    def to_numpy(self, flow=False, dd=False):
        """
        Args:
            flow (bool): If True, include underflow and overflow bins; otherwise,
                only normal (finite-width) bins are included.
            dd (bool): If True, the return type follows
                `numpy.histogramdd <https://numpy.org/doc/stable/reference/generated/numpy.histogramdd.html>`__;
                otherwise, it follows `numpy.histogram <https://numpy.org/doc/stable/reference/generated/numpy.histogram.html>`__
                and `numpy.histogram2d <https://numpy.org/doc/stable/reference/generated/numpy.histogram2d.html>`__.

        Converts the histogram into a form like the ones produced by the NumPy
        histogram functions.
        """
        values = self.values(flow=flow)
        xedges = self.axis(0).edges(flow=flow)
        yedges = self.axis(1).edges(flow=flow)
        zedges = self.axis(2).edges(flow=flow)
        if dd:
            return values, (xedges, yedges, zedges)
        else:
            return values, xedges, yedges, zedges

    def to_boost(self, metadata=boost_metadata, axis_metadata=boost_axis_metadata):
        boost_histogram = uproot.extras.boost_histogram()

        values = self.values(flow=True)

        sumw2 = self.member("fSumw2", none_if_missing=True)

        if sumw2 is not None and len(sumw2) == self.member("fNcells"):
            sumw2 = numpy.asarray(sumw2, dtype=sumw2.dtype.newbyteorder("="))
            sumw2 = numpy.transpose(numpy.reshape(sumw2, values.shape[::-1]))
            storage = boost_histogram.storage.Weight()
        else:
            if issubclass(values.dtype.type, numpy.integer):
                storage = boost_histogram.storage.Int64()
            else:
                storage = boost_histogram.storage.Double()

        xaxis = uproot.behaviors.TH1._boost_axis(self.member("fXaxis"), axis_metadata)
        yaxis = uproot.behaviors.TH1._boost_axis(self.member("fYaxis"), axis_metadata)
        zaxis = uproot.behaviors.TH1._boost_axis(self.member("fZaxis"), axis_metadata)
        out = boost_histogram.Histogram(xaxis, yaxis, zaxis, storage=storage)
        for k, v in metadata.items():
            setattr(out, k, self.member(v))

        if isinstance(xaxis, boost_histogram.axis.StrCategory):
            values = values[1:, :, :]
        if isinstance(yaxis, boost_histogram.axis.StrCategory):
            values = values[:, 1:, :]
        if isinstance(zaxis, boost_histogram.axis.StrCategory):
            values = values[:, :, 1:]

        view = out.view(flow=True)
        if sumw2 is not None and len(sumw2) == len(values):
            view.value = values
            view.variance = sumw2
        else:
            view[...] = values

        return out

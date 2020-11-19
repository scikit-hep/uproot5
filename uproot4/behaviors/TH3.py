# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

"""
Defines the behaviors of ``TH3`` and its subclasses (not including
``TProfile3D``).
"""

from __future__ import absolute_import

import numpy

import uproot4.models.TArray
import uproot4.behaviors.TH1


class TH3(uproot4.behaviors.TH1.Histogram):
    """
    Behaviors for three-dimensional histograms: descendants of ROOT's
    ``TH3``, not including ``TProfile3D``.
    """

    no_inherit = (uproot4.behaviors.TH1.TH1,)

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

    def values(self, flow=False):
        (values,) = self.base(uproot4.models.TArray.Model_TArray)
        values = numpy.array(values, dtype=values.dtype.newbyteorder("="))

        xaxis_fNbins = self.member("fXaxis").member("fNbins")
        yaxis_fNbins = self.member("fYaxis").member("fNbins")
        zaxis_fNbins = self.member("fZaxis").member("fNbins")
        out = numpy.transpose(
            values.reshape(zaxis_fNbins + 2, yaxis_fNbins + 2, xaxis_fNbins + 2)
        )
        if flow:
            return out
        else:
            return out[1:-1, 1:-1, 1:-1]

    def values_errors(self, flow=False):
        values = self.values(flow=True)
        errors = numpy.transpose(numpy.zeros(values.shape[::-1], dtype=numpy.float64))

        sumw2 = self.member("fSumw2", none_if_missing=True)
        if sumw2 is not None and len(sumw2) == self.member("fNcells"):
            sumw2 = numpy.transpose(numpy.reshape(sumw2, values.shape[::-1]))
            positive = sumw2 > 0
            errors[positive] = numpy.sqrt(sumw2[positive])
        else:
            positive = values > 0
            errors[positive] = numpy.sqrt(values[positive])

        if flow:
            return values, errors
        else:
            return values[1:-1, 1:-1, 1:-1], errors[1:-1, 1:-1, 1:-1]

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
        xedges = self.edges(0)
        yedges = self.edges(1)
        zedges = self.edges(2)
        if dd:
            return values, (xedges, yedges, zedges)
        else:
            return values, xedges, yedges, zedges

    def to_boost(self):
        boost_histogram = uproot4.extras.boost_histogram()

        values = self.values(flow=True)

        sumw2 = self.member("fSumw2", none_if_missing=True)

        if sumw2 is not None and len(sumw2) == self.member("fNcells"):
            sumw2 = numpy.array(sumw2, dtype=sumw2.dtype.newbyteorder("="))
            sumw2 = sumw2.reshape(values.shape)
            storage = boost_histogram.storage.Weight()
        else:
            if issubclass(values.dtype.type, numpy.integer):
                storage = boost_histogram.storage.Int64()
            else:
                storage = boost_histogram.storage.Double()

        xaxis = uproot4.behaviors.TH1._boost_axis(self.member("fXaxis"))
        yaxis = uproot4.behaviors.TH1._boost_axis(self.member("fYaxis"))
        zaxis = uproot4.behaviors.TH1._boost_axis(self.member("fZaxis"))
        out = boost_histogram.Histogram(xaxis, yaxis, zaxis, storage=storage)

        if isinstance(xaxis, boost_histogram.axis.StrCategory):
            values = values[1:, :, :]
        if isinstance(yaxis, boost_histogram.axis.StrCategory):
            values = values[:, 1:, :]
        if isinstance(zaxis, boost_histogram.axis.StrCategory):
            values = values[:, :, 1:]

        view = out.view(flow=True)
        if sumw2 is not None and len(sumw2) == len(values):
            view.value[:] = values
            view.variance[:] = sumw2
        else:
            view[:] = values

        return out

    def to_hist(self):
        return uproot4.extras.hist().Hist(self.to_boost())

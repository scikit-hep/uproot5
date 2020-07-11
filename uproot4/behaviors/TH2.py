# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import numpy

import uproot4.models.TArray
import uproot4.behaviors.TH1


class TH2(object):
    no_inherit = (uproot4.behaviors.TH1.TH1,)

    def edges(self, axis):
        if axis == 0 or axis == "x":
            return uproot4.behaviors.TH1._edges(self.member("fXaxis"))
        elif axis == 1 or axis == "y":
            return uproot4.behaviors.TH1._edges(self.member("fYaxis"))
        else:
            raise ValueError("axis must be 0, 1 or 'x', 'y' for a TH2")

    def values(self):
        (values,) = self.base(uproot4.models.TArray.Model_TArray)
        values = numpy.array(values, dtype=values.dtype.newbyteorder("="))

        xaxis_fNbins = self.member("fXaxis").member("fNbins")
        yaxis_fNbins = self.member("fYaxis").member("fNbins")
        return values.reshape(xaxis_fNbins + 2, yaxis_fNbins + 2)

    def values_errors(self):
        values = self.values()
        errors = numpy.zeros(values.shape, dtype=numpy.float64)

        sumw2 = self.member("fSumw2", none_if_missing=True)
        if sumw2 is not None and len(sumw2) == self.member("fNcells"):
            sumw2 = sumw2.reshape(values.shape)
            positive = sumw2 > 0
            errors[positive] = numpy.sqrt(sumw2[positive])
        else:
            positive = values > 0
            errors[positive] = numpy.sqrt(values[positive])

        return values, errors

    def to_numpy(self, flow=True, dd=False, errors=False):
        if errors:
            values, errs = self.values_errors()
        else:
            values, errs = self.values(), None

        xedges = self.edges(0)
        yedges = self.edges(1)
        if not flow:
            values = values[1:-1, 1:-1]
            if errors:
                errs = errs[1:-1, 1:-1]
            xedges = xedges[1:-1]
            yedges = yedges[1:-1]

        if errors:
            values_errors = values, errs
        else:
            values_errors = values

        if dd:
            return values_errors, (xedges, yedges)
        else:
            return values_errors, xedges, yedges

    def to_boost(self):
        boost_histogram = uproot4.extras.boost_histogram()

        values = self.values()

        sumw2 = self.member("fSumw2", none_if_missing=True)

        if sumw2 is not None and len(sumw2) == self.member("fNcells"):
            sumw2 = numpy.array(sumw2, dtype=sumw2.dtype.newbyteorder("="))
            sumw2.reshape(values.shape)
            storage = boost_histogram.storage.Weight()
        else:
            if issubclass(values.dtype.type, numpy.integer):
                storage = boost_histogram.storage.Int64()
            else:
                storage = boost_histogram.storage.Double()

        xaxis = uproot4.behaviors.TH1._boost_axis(self.member("fXaxis"))
        yaxis = uproot4.behaviors.TH1._boost_axis(self.member("fYaxis"))
        out = boost_histogram.Histogram(xaxis, yaxis, storage=storage)

        if isinstance(xaxis, boost_histogram.axis.StrCategory):
            values = values[1:, :]
        if isinstance(yaxis, boost_histogram.axis.StrCategory):
            values = values[:, 1:]

        view = out.view(flow=True)
        if sumw2 is not None and len(sumw2) == len(values):
            view.value[:] = values
            view.variance[:] = sumw2
        else:
            view[:] = values

        return out

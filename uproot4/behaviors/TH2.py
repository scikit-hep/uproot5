# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import numpy

import uproot4.models.TArray
import uproot4.behaviors.TH1


class TH2(object):
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

    # values_errors "inherited" from TH1

    @property
    def np(self):
        return self.values(), self.edges(0), self.edges(1)

    @property
    def bh(self):
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

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import numpy

import uproot4.models.TArray
from uproot4.behaviors.TH1 import _edges, _boost_axis


class TH3(object):
    @property
    def np(self):
        (values,) = self.base(uproot4.models.TArray.Model_TArray)
        values = numpy.array(values, dtype=values.dtype.newbyteorder("="))

        xaxis_fNbins = self.member("fXaxis").member("fNbins")
        yaxis_fNbins = self.member("fYaxis").member("fNbins")
        zaxis_fNbins = self.member("fZaxis").member("fNbins")
        values = values.reshape(xaxis_fNbins + 2, yaxis_fNbins + 2, zaxis_fNbins + 2)

        return (
            values,
            _edges(self.member("fXaxis")),
            _edges(self.member("fYaxis")),
            _edges(self.member("fZaxis")),
        )

    @property
    def bh(self):
        boost_histogram = uproot4.extras.boost_histogram()

        (values,) = self.base(uproot4.models.TArray.Model_TArray)
        values = numpy.array(values, dtype=values.dtype.newbyteorder("="))

        xaxis_fNbins = self.member("fXaxis").member("fNbins")
        yaxis_fNbins = self.member("fYaxis").member("fNbins")
        zaxis_fNbins = self.member("fZaxis").member("fNbins")
        values = values.reshape(xaxis_fNbins + 2, yaxis_fNbins + 2, zaxis_fNbins + 2)

        sumw2 = self.member("fSumw2", none_if_missing=True)

        if sumw2 is not None and len(sumw2) == len(values):
            sumw2 = numpy.array(sumw2, dtype=sumw2.dtype.newbyteorder("="))
            sumw2.reshape(xaxis_fNbins + 2, yaxis_fNbins + 2, zaxis_fNbins + 2)
            storage = boost_histogram.storage.Weight()
        else:
            if issubclass(values.dtype.type, numpy.integer):
                storage = boost_histogram.storage.Int64()
            else:
                storage = boost_histogram.storage.Double()

        xaxis = _boost_axis(boost_histogram, self.member("fXaxis"))
        yaxis = _boost_axis(boost_histogram, self.member("fYaxis"))
        zaxis = _boost_axis(boost_histogram, self.member("fZaxis"))
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

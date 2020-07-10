# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import numpy

import uproot4.models.TArray
import uproot4.extras


def _edges(axis):
    fNbins = axis.member("fNbins")
    out = numpy.empty(fNbins + 3, dtype=numpy.float64)
    out[0] = -numpy.inf
    out[-1] = numpy.inf

    axis_fXbins = axis.member("fXbins", none_if_missing=True)
    if axis_fXbins is None or len(axis_fXbins) == 0:
        out[1:-1] = numpy.linspace(
            axis.member("fXmin"), axis.member("fXmax"), fNbins + 1
        )
    else:
        out[1:-1] = axis_fXbins

    return out


def _boost_axis(axis):
    boost_histogram = uproot4.extras.boost_histogram()

    fNbins = axis.member("fNbins")
    fXbins = axis.member("fXbins", none_if_missing=True)

    metadata = axis.all_members
    metadata.pop("fXbins", None)
    metadata.pop("fLabels", None)

    if axis.member("fLabels") is not None:
        return boost_histogram.axis.StrCategory(
            [str(x) for x in axis.member("fLabels")], metadata=metadata,
        )

    elif fXbins is None or len(fXbins) == 0:
        return boost_histogram.axis.Regular(
            fNbins,
            axis.member("fXmin"),
            axis.member("fXmax"),
            underflow=True,
            overflow=True,
            metadata=metadata,
        )

    else:
        return boost_histogram.axis.Variable(
            fXbins, underflow=True, overflow=True, metadata=metadata,
        )


class TH1(object):
    def edges(self, axis=0):
        if axis == 0 or axis == "x":
            return uproot4.behaviors.TH1._edges(self.member("fXaxis"))
        else:
            raise ValueError("axis must be 0 or 'x' for a TH1")

    def values(self):
        (values,) = self.base(uproot4.models.TArray.Model_TArray)
        return numpy.array(values, dtype=values.dtype.newbyteorder("="))

    def values_errors(self):
        # this should work equally well for TH2 and TH3

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

    @property
    def np(self):
        return self.values(), self.edges(0)

    @property
    def bh(self):
        boost_histogram = uproot4.extras.boost_histogram()

        values = self.values()

        sumw2 = self.member("fSumw2", none_if_missing=True)

        if sumw2 is not None and len(sumw2) == self.member("fNcells"):
            storage = boost_histogram.storage.Weight()
        else:
            if issubclass(values.dtype.type, numpy.integer):
                storage = boost_histogram.storage.Int64()
            else:
                storage = boost_histogram.storage.Double()

        xaxis = _boost_axis(self.member("fXaxis"))
        out = boost_histogram.Histogram(xaxis, storage=storage)

        metadata = self.all_members
        metadata.pop("fXaxis", None)
        metadata.pop("fYaxis", None)
        metadata.pop("fZaxis", None)
        metadata.pop("fContour", None)
        metadata.pop("fSumw2", None)
        metadata.pop("fBuffer", None)
        out.metadata = metadata

        if isinstance(xaxis, boost_histogram.axis.StrCategory):
            values = values[1:]

        view = out.view(flow=True)
        if sumw2 is not None and len(sumw2) == len(values):
            view.value[:] = values
            view.variance[:] = sumw2
        else:
            view[:] = values

        return out

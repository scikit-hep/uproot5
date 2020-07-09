# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import numpy

import uproot4.models.TArray
import uproot4.extras


class TH1(object):
    @property
    def np(self):
        xaxis = self.member("fXaxis")

        xaxis_fNbins = xaxis.member("fNbins")
        xedges = numpy.empty(xaxis_fNbins + 3, dtype=numpy.float64)
        xedges[0] = -numpy.inf
        xedges[-1] = numpy.inf

        xaxis_fXbins = xaxis.member("fXbins", none_if_missing=True)
        if xaxis_fXbins is None or len(xaxis_fXbins) == 0:
            xedges[1:-1] = numpy.linspace(
                xaxis.member("fXmin"), xaxis.member("fXmax"), xaxis_fNbins + 1
            )
        else:
            xedges[1:-1] = xaxis_fXbins

        for base in self.bases:
            if isinstance(base, uproot4.models.TArray.Model_TArray):
                values = numpy.array(base, dtype=base.dtype.newbyteorder("="))
                break

        return values, xedges

    def metadata(self, axis):
        if axis == "x":
            axis = self.member("fXaxis")
        return {
            "name": self.member("fName"),
            "title": self.member("fTitle"),
            "entries": self.member("fEntries"),
            "total-sumw": self.member("fTsumw"),
            "total-sumw2": self.member("fTsumw2"),
            "total-sumwx": self.member("fTsumwx"),
            "total-sumwx2": self.member("fTsumwx2"),
            "norm-factor": self.member("fNormFactor"),
            "option": self.member("fOption"),
            "line-color": self.member("fLineColor"),
            "line-style": self.member("fLineStyle"),
            "line-width": self.member("fLineWidth"),
            "fill-color": self.member("fFillColor"),
            "fill-style": self.member("fFillStyle"),
            "marker-color": self.member("fMarkerColor"),
            "marker-style": self.member("fMarkerStyle"),
            "marker-size": self.member("fMarkerSize"),
            "bar-offset": self.member("fBarOffset"),
            "bar-width": self.member("fBarWidth"),
            "labels": axis.member("fLabels"),
            "axis-color": axis.member("fAxisColor"),
            "label-color": axis.member("fLabelColor"),
            "label-font": axis.member("fLabelFont"),
            "label-offset": axis.member("fLabelOffset"),
            "label-size": axis.member("fLabelSize"),
            "tick-length": axis.member("fTickLength"),
            "title-offset": axis.member("fTitleOffset"),
            "title-size": axis.member("fTitleSize"),
            "title-color": axis.member("fTitleColor"),
            "title-font": axis.member("fTitleFont"),
            "time-display": axis.member("fTimeDisplay"),
            "time-format": str(axis.member("fTimeFormat")),
            "contour": numpy.asarray(self.member("fContour")),
            "functions": list(self.member("fFunctions")),
        }

    @property
    def bh(self):
        boost_histogram = uproot4.extras.boost_histogram()

        xaxis = self.member("fXaxis")

        xaxis_fNbins = xaxis.member("fNbins")
        xaxis_fXbins = xaxis.member("fXbins", none_if_missing=True)
        if xaxis_fXbins is None or len(xaxis_fXbins) == 0:
            boost_xaxis = boost_histogram.axis.Regular(
                xaxis_fNbins,
                xaxis.member("fXmin"),
                xaxis.member("fXmax"),
                underflow=True,
                overflow=True,
                metadata=self.metadata(xaxis),
            )
        else:
            boost_xaxis = boost_histogram.axis.Variable(
                xaxis_fXbins,
                underflow=True,
                overflow=True,
                metadata=self.metadata(xaxis),
            )

        for base in self.bases:
            if isinstance(base, uproot4.models.TArray.Model_TArray):
                values = numpy.asarray(base)
                break

        sumw2 = self.member("fSumw2", none_if_missing=True)

        if sumw2 is not None and len(sumw2) == len(values):
            storage = boost_histogram.storage.Weight()
        else:
            if issubclass(values.dtype.type, numpy.integer):
                storage = boost_histogram.storage.Int64()
            else:
                storage = boost_histogram.storage.Double()

        out = boost_histogram.Histogram(boost_xaxis, storage=storage)
        view = out.view(flow=True)

        if sumw2 is not None and len(sumw2) == len(values):
            view.value[:] = values
            view.variance[:] = sumw2
        else:
            view[:] = values

        return out

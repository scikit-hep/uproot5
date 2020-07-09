# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import numpy

import uproot4.models.TArray


class TH1(object):
    @property
    def np(self):
        axis = self.member("fXaxis")

        fNbins = axis.member("fNbins")
        edges = numpy.empty(fNbins + 3, dtype=numpy.float64)
        edges[0] = -numpy.inf
        edges[-1] = numpy.inf

        fXbins = axis.member("fXbins", none_if_missing=True)
        if fXbins is None or len(fXbins) == 0:
            edges[1:-1] = numpy.linspace(axis.member("fXmin"), axis.member("fXmax"), fNbins + 1)
        else:
            edges[1:-1] = fXbins

        for base in self.bases:
            if isinstance(base, uproot4.models.TArray.Model_TArray):
                values = numpy.array(base, dtype=base.dtype.newbyteorder("="))
                break

        return values, edges

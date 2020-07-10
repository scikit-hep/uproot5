# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import numpy

import uproot4.models.TArray


_kERRORMEAN = 0
_kERRORSPREAD = 1
_kERRORSPREADI = 2
_kERRORSPREADG = 3


class TProfile(object):
    @property
    def effective_entries(self):
        # https://github.com/root-project/root/blob/ffc7c588ac91aca30e75d356ea971129ee6a836a/hist/hist/src/TProfileHelper.h#L141-L163

        root_sumOfWeights = self.member("fBinEntries")
        root_sumOfWeights = numpy.array(root_sumOfWeights, dtype=numpy.float64)

        root_sumOfWeightSquare = self.member("fBinSumw2")
        root_sumOfWeightSquare = numpy.array(
            root_sumOfWeightSquare, dtype=numpy.float64
        )

        if len(root_sumOfWeightSquare) == 0 or len(
            root_sumOfWeightSquare
        ) != self.member("fNcells"):
            return root_sumOfWeights

        positive = root_sumOfWeightSquare > 0

        out = numpy.zeros(len(root_sumOfWeights), dtype=numpy.float64)
        out[positive] = (
            root_sumOfWeights[positive] ** 2 / root_sumOfWeightSquare[positive]
        )
        return out

    @property
    def np(self):
        # https://github.com/root-project/root/blob/ffc7c588ac91aca30e75d356ea971129ee6a836a/hist/hist/src/TProfileHelper.h#L660-L721

        root_sum = self.member("fBinEntries")
        root_sum = numpy.array(root_sum, dtype=numpy.float64)
        nonzero = root_sum != 0

        (root_cont,) = self.base(uproot4.models.TArray.Model_TArray)
        root_cont = numpy.array(root_cont, dtype=numpy.float64)

        root_contsum = numpy.zeros(len(root_cont), dtype=numpy.float64)
        root_contsum[nonzero] = root_cont[nonzero] / root_sum[nonzero]

        if self.member("fErrorMode") == _kERRORSPREADG:
            out = numpy.zeros(len(root_cont), dtype=numpy.float64)
            out[nonzero] = 1.0 / numpy.sqrt(root_sum[nonzero])
            return root_contsum, out

        root_err2 = self.member("fSumw2", none_if_missing=True)
        if root_err2 is None or len(root_err2) != len(root_cont):
            root_err2 = numpy.zeros(len(root_cont), dtype=numpy.float64)
        else:
            root_err2 = numpy.array(root_err2, dtype=numpy.float64)

        root_neff = self.effective_entries

        root_eprim2 = numpy.zeros(len(root_cont), dtype=numpy.float64)
        root_eprim2[nonzero] = abs(
            root_err2[nonzero] / root_sum[nonzero] - root_contsum[nonzero] ** 2
        )
        root_eprim = numpy.sqrt(root_eprim2)

        if self.member("fErrorMode") == _kERRORSPREADI:
            numer = numpy.ones(len(root_cont), dtype=numpy.float64)
            denom = numpy.full(len(root_cont), numpy.sqrt(12 * root_neff))

            eprim_nonzero = root_eprim != 0
            numer[eprim_nonzero] = root_eprim[eprim_nonzero]
            denom[eprim_nonzero] = numpy.sqrt(root_neff[eprim_nonzero])

            out = numpy.zeros(len(root_cont), dtype=numpy.float64)
            out[nonzero] = numer[nonzero] / denom[nonzero]
            return root_contsum, out

        if self.member("fErrorMode") == _kERRORSPREAD:
            root_eprim[~nonzero] = 0.0
            return root_contsum, root_eprim

        out = numpy.zeros(len(root_cont), dtype=numpy.float64)
        out[nonzero] = root_eprim[nonzero] / numpy.sqrt(root_neff[nonzero])
        return root_contsum, out

    @property
    def bh(self):
        raise NotImplementedError(repr(self))

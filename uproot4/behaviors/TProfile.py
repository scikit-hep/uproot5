# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

"""
Defines the behavior of ``TProfile``.
"""

from __future__ import absolute_import

import numpy

import uproot4.models.TArray
import uproot4.behaviors.TH1


_kERRORMEAN = 0
_kERRORSPREAD = 1
_kERRORSPREADI = 2
_kERRORSPREADG = 3


# closely follows the ROOT function, using the same names (with 'root_' prepended)
# https://github.com/root-project/root/blob/ffc7c588ac91aca30e75d356ea971129ee6a836a/hist/hist/src/TProfileHelper.h#L141-L163
def _effective_entries_1d(fBinEntries, fBinSumw2, fNcells):
    root_sumOfWeights = fBinEntries
    root_sumOfWeights = numpy.array(root_sumOfWeights, dtype=numpy.float64)

    root_sumOfWeightSquare = fBinSumw2
    root_sumOfWeightSquare = numpy.array(root_sumOfWeightSquare, dtype=numpy.float64)

    if len(root_sumOfWeightSquare) == 0 or len(root_sumOfWeightSquare) != fNcells:
        return root_sumOfWeights

    positive = root_sumOfWeightSquare > 0

    out = numpy.zeros(len(root_sumOfWeights), dtype=numpy.float64)
    out[positive] = root_sumOfWeights[positive] ** 2 / root_sumOfWeightSquare[positive]
    return out


# duplicates the first part of '_values_errors_1d'
def _values_1d(fBinEntries, root_cont):
    root_sum = fBinEntries
    root_sum = numpy.array(root_sum, dtype=numpy.float64)
    nonzero = root_sum != 0

    root_contsum = numpy.zeros(len(root_cont), dtype=numpy.float64)
    root_contsum[nonzero] = root_cont[nonzero] / root_sum[nonzero]

    return root_contsum


# closely follows the ROOT function, using the same names (with 'root_' prepended)
# https://github.com/root-project/root/blob/ffc7c588ac91aca30e75d356ea971129ee6a836a/hist/hist/src/TProfileHelper.h#L660-L721
def _values_errors_1d(error_mode, fBinEntries, root_cont, fSumw2, fNcells, fBinSumw2):
    if error_mode is None or error_mode == _kERRORMEAN or error_mode == "":
        error_mode = _kERRORMEAN
    elif error_mode == _kERRORSPREAD or error_mode == "s":
        error_mode = _kERRORSPREAD
    elif error_mode == _kERRORSPREADI or error_mode == "i":
        error_mode = _kERRORSPREADI
    elif error_mode == _kERRORSPREADG or error_mode == "g":
        error_mode = _kERRORSPREADG
    else:
        raise ValueError(
            "error_mode must be None/0/'' for error-on-mean,\n"
            "                        1/'s' for spread (variance),\n"
            "                        2/'i' for integer spread (using sqrt(12)),\n"
            "                     or 3/'g' for Gaussian spread\n"
            "                    not "
            + repr(error_mode)
            + "see https://root.cern.ch/doc/master/classTProfile.html"
        )

    root_sum = fBinEntries
    root_sum = numpy.array(root_sum, dtype=numpy.float64)
    nonzero = root_sum != 0

    root_contsum = numpy.zeros(len(root_cont), dtype=numpy.float64)
    root_contsum[nonzero] = root_cont[nonzero] / root_sum[nonzero]

    if error_mode == _kERRORSPREADG:
        out = numpy.zeros(len(root_cont), dtype=numpy.float64)
        out[nonzero] = 1.0 / numpy.sqrt(root_sum[nonzero])
        return root_contsum, out

    root_err2 = fSumw2
    if root_err2 is None or len(root_err2) != fNcells:
        root_err2 = numpy.zeros(len(root_cont), dtype=numpy.float64)
    else:
        root_err2 = numpy.array(root_err2, dtype=numpy.float64)

    root_neff = _effective_entries_1d(fBinEntries, fBinSumw2, fNcells)

    root_eprim2 = numpy.zeros(len(root_cont), dtype=numpy.float64)
    root_eprim2[nonzero] = abs(
        root_err2[nonzero] / root_sum[nonzero] - root_contsum[nonzero] ** 2
    )
    root_eprim = numpy.sqrt(root_eprim2)

    if error_mode == _kERRORSPREADI:
        numer = numpy.ones(len(root_cont), dtype=numpy.float64)
        denom = numpy.full(len(root_cont), numpy.sqrt(12 * root_neff))

        eprim_nonzero = root_eprim != 0
        numer[eprim_nonzero] = root_eprim[eprim_nonzero]
        denom[eprim_nonzero] = numpy.sqrt(root_neff[eprim_nonzero])

        out = numpy.zeros(len(root_cont), dtype=numpy.float64)
        out[nonzero] = numer[nonzero] / denom[nonzero]
        return root_contsum, out

    if error_mode == _kERRORSPREAD:
        root_eprim[~nonzero] = 0.0
        return root_contsum, root_eprim

    out = numpy.zeros(len(root_cont), dtype=numpy.float64)
    out[nonzero] = root_eprim[nonzero] / numpy.sqrt(root_neff[nonzero])
    return root_contsum, out


class Profile(uproot4.behaviors.TH1.Histogram):
    """
    Abstract class for profile plots.
    """

    def effective_entries(self):
        """
        The effective number of entries, which is a step in the calculation of
        :py:meth:`~uproot4.behaviors.TProfile.Profile.values`.
        """
        pass

    def values_errors(self, error_mode=""):
        """
        The :py:meth:`~uproot4.behaviors.TH1.Histogram.values` and their associated
        errors (uncertainties) as a 2-tuple of arrays. The two arrays have the
        same ``shape``.

        The calculation of profile errors exactly follows ROOT's function, but
        in a vectorized NumPy form.

        The ``error_mode`` may be

        * ``""`` for standard error on the mean
        * ``"s"`` for spread
        * ``"i"`` for integer data
        * ``"g"`` for Gaussian

        following `ROOT's profile documentation <https://root.cern.ch/doc/master/classTProfile.html>`__.
        """
        pass


class TProfile(Profile):
    """
    Behaviors for one-dimensional profiles: ROOT's ``TProfile``.
    """

    no_inherit = (uproot4.behaviors.TH1.TH1,)

    def edges(self, axis=0):
        if axis == 0 or axis == -1 or axis == "x":
            return uproot4.behaviors.TH1._edges(self.member("fXaxis"))
        else:
            raise ValueError("axis must be 0 or 'x' for a TH1")

    def effective_entries(self):
        return _effective_entries_1d(
            self.member("fBinEntries"),
            self.member("fBinSumw2"),
            self.member("fNcells"),
        )

    def values(self):
        (root_cont,) = self.base(uproot4.models.TArray.Model_TArray)
        root_cont = numpy.asarray(root_cont, dtype=numpy.float64)
        return _values_1d(
            self.member("fBinEntries"),
            root_cont,
        )

    def values_errors(self, error_mode=""):
        (root_cont,) = self.base(uproot4.models.TArray.Model_TArray)
        root_cont = numpy.asarray(root_cont, dtype=numpy.float64)
        fSumw2 = self.member("fSumw2", none_if_missing=True)
        if fSumw2 is not None:
            fSumw2 = numpy.asarray(fSumw2)
        return _values_errors_1d(
            error_mode,
            numpy.asarray(self.member("fBinEntries")),
            root_cont,
            fSumw2,
            self.member("fNcells"),
            numpy.asarray(self.member("fBinSumw2")),
        )

    def to_numpy(self, flow=False, dd=False, errors=False, error_mode=0):
        if errors:
            values, errs = self.values_errors(error_mode=error_mode)
        else:
            values, errs = self.values(), None

        xedges = self.edges(0)
        if not flow:
            values = values[1:-1]
            if errors:
                errs = errs[1:-1]
            xedges = xedges[1:-1]

        if errors:
            values_errors = values, errs
        else:
            values_errors = values

        if dd:
            return values_errors, (xedges,)
        else:
            return values_errors, xedges

    def to_boost(self):
        boost_histogram = uproot4.extras.boost_histogram()

        storage = boost_histogram.storage.WeightedMean()

        xaxis = uproot4.behaviors.TH1._boost_axis(self.member("fXaxis"))
        out = boost_histogram.Histogram(xaxis, storage=storage)

        values, errors = self.values_errors(self.member("fErrorMode"))

        if isinstance(xaxis, boost_histogram.axis.StrCategory):
            values = values[1:]
            errors = errors[1:]

        view = out.view(flow=True)

        view.sum_of_weights
        view.sum_of_weights_squared
        view.value = values
        view.sum_of_weighted_deltas_squared

        raise NotImplementedError(repr(self))

    def to_hist(self):
        return uproot4.extras.hist().Hist(self.to_boost())

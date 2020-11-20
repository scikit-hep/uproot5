# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

"""
Defines the behaviors of ``TH1`` and its subclasses (not including ``TH2``,
``TH3``, or ``TProfile``).
"""

from __future__ import absolute_import

import numpy

import uproot4.models.TArray
import uproot4.extras


def _boost_axis(axis, metadata):
    boost_histogram = uproot4.extras.boost_histogram()

    fNbins = axis.member("fNbins")
    fXbins = axis.member("fXbins", none_if_missing=True)

    metadata = dict((k, axis.member(v)) for k, v in metadata.items())

    if axis.member("fLabels") is not None:
        return boost_histogram.axis.StrCategory(
            [str(x) for x in axis.member("fLabels")], metadata=metadata,
        )

    elif fXbins is None or len(fXbins) != fNbins:
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


class Histogram(object):
    """
    Abstract class for histograms.
    """

    def __eq__(self, other):
        """
        Two histograms are equal if their axes are equal, their values are equal,
        and their variances are equal.
        """
        if type(self) != type(other):
            return False
        if self.axes != other.axes:
            return False
        self_values, self_variances = self.values_variances(flow=True)
        other_values, other_variances = other.values_variances(flow=True)
        values_equal = numpy.array_equal(self_values, other_values)
        variances_equal = numpy.array_equal(self_variances, other_variances)
        return values_equal and variances_equal

    def __ne__(self, other):
        """
        Some versions of Python don't automatically negate __eq__.
        """
        return not self.__eq__(other)

    @property
    def axes(self):
        """
        A tuple of all :py:class:`~uproot4.behaviors.TAxis.TAxis` objects.
        """
        raise NotImplementedError(repr(self))

    def axis(self, axis):
        """
        Returns a specified :py:class:`~uproot4.behaviors.TAxis.TAxis` object.

        The ``axis`` can be specified as

        * a non-negative integer: ``0`` is the first axis, ``1`` is the second,
          and ``2`` is the third.
        * a negative integer: ``-1`` is the last axis, ``-2`` is the
          second-to-last, and ``-3`` is the third-to-last.
        * a string: ``"x"`` is the first axis, ``"y"`` is the second, and ``"z"``
          is the third

        (assuming that the histogram dimension supports a given ``axis``).
        """
        raise NotImplementedError(repr(self))

    def values(self, flow=False):
        """
        Args:
            flow (bool): If True, include underflow and overflow bins before and
                after the normal (finite-width) bins.

        Bin contents as a 1, 2, or 3 dimensional ``numpy.ndarray``. The
        ``numpy.dtype`` of this array depends on the histogram type.

        Setting ``flow=True`` increases the length of each dimension by two.
        """
        raise NotImplementedError(repr(self))

    def errors(self, flow=False):
        """
        Args:
            flow (bool): If True, include underflow and overflow bins before and
                after the normal (finite-width) bins.

        Errors (uncertainties) in the :py:meth:`~uproot4.behaviors.TH1.Histogram.values`
        as a 1, 2, or 3 dimensional ``numpy.ndarray`` of ``numpy.float64``.

        If ``fSumw2`` (weights) are available, they will be used in the
        calculation of the errors. If not, errors are assumed to be the square
        root of the values.

        Setting ``flow=True`` increases the length of each dimension by two.
        """
        values, variances = self.values_variances(flow=flow)
        return numpy.sqrt(variances)

    def variances(self, flow=False):
        """
        Args:
            flow (bool): If True, include underflow and overflow bins before and
                after the normal (finite-width) bins.

        Variances (uncertainties squared) in the
        :py:meth:`~uproot4.behaviors.TH1.Histogram.values` as a 1, 2, or 3
        dimensional ``numpy.ndarray`` of ``numpy.float64``.

        If ``fSumw2`` (weights) are available, they will be used in the
        calculation of the variances. If not, variances are assumed to be equal
        to the values.

        Setting ``flow=True`` increases the length of each dimension by two.
        """
        values, variances = self.values_variances(flow=flow)
        return variances

    def values_errors(self, flow=False):
        """
        Args:
            flow (bool): If True, include underflow and overflow bins before and
                after the normal (finite-width) bins.

        The :py:meth:`~uproot4.behaviors.TH1.Histogram.values` and their associated
        :py:meth:`~uproot4.behaviors.TH1.Histogram.errors` (uncertainties) as a
        2-tuple of arrays. The two arrays have the same ``shape``.

        If ``fSumw2`` (weights) are available, they will be used in the
        calculation of the errors. If not, errors are assumed to be the square
        root of the values.

        Setting ``flow=True`` increases the length of each dimension by two.
        """
        values, variances = self.values_variances(flow=flow)
        return values, numpy.sqrt(variances)

    def values_variances(self, flow=False):
        """
        Args:
            flow (bool): If True, include underflow and overflow bins before and
                after the normal (finite-width) bins.

        The :py:meth:`~uproot4.behaviors.TH1.Histogram.values` and their associated
        :py:meth:`~uproot4.behaviors.TH1.Histogram.variances` (uncertainties squared)
        as a 2-tuple of arrays. The two arrays have the same ``shape``.

        If ``fSumw2`` (weights) are available, they will be used in the
        calculation of the variances. If not, variances are assumed to be equal
        to the values.

        Setting ``flow=True`` increases the length of each dimension by two.
        """
        raise NotImplementedError(repr(self))

    def counts(self, flow=False):
        """
        Args:
            flow (bool): If True, include underflow and overflow bins before and
                after the normal (finite-width) bins.

        Count returns the number of values in a mean accumulator (also known as
        a Profile histogram), or is None for normal storages.
        """
        return None

    def to_boost(
        self,
        metadata={"name": "fName", "title": "fTitle"},
        axis_metadata={"name": "fName", "title": "fTitle"},
    ):
        u"""
        Args:
            metadata (dict of str \u2192 str): Metadata to collect (keys) and
                their C++ class member names (values).
            axis_metadata (dict of str \u2192 str): Metadata to collect from
                each axis.

        Converts the histogram into a ``boost-histogram`` object.
        """
        raise NotImplementedError(repr(self))

    def to_hist(self, metadata={"name": "fName", "title": "fTitle"}):
        u"""
        Args:
            metadata (dict of str \u2192 str): metadata to collect (keys) and
                their C++ class member names (values).

        Converts the histogram into a ``hist`` object.
        """
        return uproot4.extras.hist().Hist(self.to_boost(metadata=metadata))


class TH1(Histogram):
    """
    Behaviors for one-dimensional histograms: descendants of ROOT's
    ``TH1``, not including ``TProfile``, ``TH2``, ``TH3``, or their descendants.
    """

    @property
    def axes(self):
        return (self.member("fXaxis"),)

    def axis(self, axis=0):  # default axis for one-dimensional is intentional
        if axis == 0 or axis == -1 or axis == "x":
            return self.member("fXaxis")
        else:
            raise ValueError("axis must be 0 (-1) or 'x' for a TH1")

    def values(self, flow=False):
        (values,) = self.base(uproot4.models.TArray.Model_TArray)
        out = numpy.array(values, dtype=values.dtype.newbyteorder("="))
        if flow:
            return out
        else:
            return out[1:-1]

    def values_variances(self, flow=False):
        values = self.values(flow=True)
        errors = numpy.zeros(values.shape, dtype=numpy.float64)

        sumw2 = self.member("fSumw2", none_if_missing=True)
        if sumw2 is not None and len(sumw2) == self.member("fNcells"):
            sumw2 = numpy.array(sumw2, dtype=sumw2.dtype.newbyteorder("="))
            sumw2 = numpy.reshape(sumw2, values.shape)
            positive = sumw2 > 0
            errors[positive] = sumw2[positive]
        else:
            positive = values > 0
            errors[positive] = values[positive]

        if flow:
            return values, errors
        else:
            return values[1:-1], errors[1:-1]

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
        if dd:
            return values, (xedges,)
        else:
            return values, xedges

    def to_boost(
        self,
        metadata={"name": "fName", "title": "fTitle"},
        axis_metadata={"name": "fName", "title": "fTitle"},
    ):
        boost_histogram = uproot4.extras.boost_histogram()

        values = self.values(flow=True)

        sumw2 = self.member("fSumw2", none_if_missing=True)

        if sumw2 is not None and len(sumw2) == self.member("fNcells"):
            sumw2 = numpy.array(sumw2, dtype=sumw2.dtype.newbyteorder("="))
            sumw2 = numpy.reshape(sumw2, values.shape)
            storage = boost_histogram.storage.Weight()
        else:
            if issubclass(values.dtype.type, numpy.integer):
                storage = boost_histogram.storage.Int64()
            else:
                storage = boost_histogram.storage.Double()

        xaxis = _boost_axis(self.member("fXaxis"), axis_metadata)
        out = boost_histogram.Histogram(xaxis, storage=storage)
        out.metadata = dict((k, self.member(v)) for k, v in metadata.items())

        if isinstance(xaxis, boost_histogram.axis.StrCategory):
            values = values[1:]

        view = out.view(flow=True)
        if sumw2 is not None and len(sumw2) == len(values):
            view.value[:] = values
            view.variance[:] = sumw2
        else:
            view[:] = values

        return out

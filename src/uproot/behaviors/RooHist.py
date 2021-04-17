# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

"""
This module defines the behaviors of ``RooHist``.
"""

from __future__ import absolute_import

import numpy

import uproot
from uproot.behaviors.TH1 import boost_metadata, boost_axis_metadata

# ['@fUniqueID', '@fBits', 'fName', 'fTitle', 'fLineColor', 'fLineStyle',
#  'fLineWidth', 'fFillColor', 'fFillStyle', 'fMarkerColor', 'fMarkerStyle',
#  'fMarkerSize', 'fNpoints', 'fX', 'fY', 'fFunctions', 'fHistogram',
#  'fMinimum', 'fMaximum', 'fEXlow', 'fEXhigh', 'fEYlow', 'fEYhigh']


class AxisTraits(object):
    """
    Axis traits
    """

    @property
    def circular(self):
        return False

    @property
    def discrete(self):
        return False


class RooHistAxis(object):
    """
    `RooHist`'s axis
    """

    def __init__(self, low_edges, high_edges):
        self._traits = AxisTraits()
        self._low_edges = low_edges
        self._high_edges = high_edges

    def __repr__(self):
        return f"<RooHistAxis({list(zip(low_edges, high_edges))})>"

    def __eq__(self, other):
        return (self._low_edges == other._low_edges) and (
            self._high_edges == other._high_edges
        )

    def __len__(self):
        return len(self._low_edges)

    def __getitem__(self, key):
        return (self._low_edges[key], self._high_edges[key])

    @property
    def traits(self):
        return self._traits


class RooHist(uproot.behaviors.TGraphAsymmErrors.TGraphAsymmErrors):
    """
    Behavior for ``RooHist``

    A minimal Histogram-like implementation is also provided.
    """

    @property
    def name(self):
        """
        The name of the histogram.
        """
        return self.member("fName")

    @property
    def title(self):
        """
        The title of the histogram.
        """
        return self.member("fTitle")

    def __eq__(self, other):
        """
        Two histograms are equal if their axes are equal, their values are equal,
        and their variances are equal.
        """
        if type(self) != type(other):
            return False
        if self.axes != other.axes:
            return False
        values_equal = numpy.array_equal(
            super(RooHist, self).values("y"), super(RooHist, other).values("y")
        )
        variances_equal = numpy.array_equal(
            super(RooHist, self).errors("mean", "x"),
            super(RooHist, other).errors("mean", "y"),
        )
        return values_equal and variances_equal

    def __ne__(self, other):
        """
        Some versions of Python don't automatically negate __eq__.
        """
        return not self.__eq__(other)

    @property
    def axes(self):
        """
        A tuple containing the X axis
        """
        low_edges = self._normalize_array("fX") - self._normalize_array("fEXLow")
        high_edges = self._normalize_array("fX") - self._normalize_array("fEXHigh")
        return (RooHistAxis(low_edges, high_edges),)

    def axis(self, axis):
        """
        Return the X axis.

        ``axis`` *must* be ``0``, ``-1``, or ``"x"``.
        """
        if axis not in (0, -1, "x"):
            raise ValueError("axis must be 0 (-1) or 'x' for a RooHist")
        low_edges = self._normalize_array("fX") - self._normalize_array("fEXLow")
        high_edges = self._normalize_array("fX") - self._normalize_array("fEXHigh")
        return RooHistAxis(low_edges, high_edges)

    @property
    def weighted(self):
        """
        Always return weighted since values are doubles
        """
        return True

    @property
    def kind(self):
        """
        ``RooHist``s are "COUNT" histograms.
        """
        return "COUNT"

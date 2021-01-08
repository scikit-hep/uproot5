# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

"""
This module defines the behaviors of ``TGraphAsymmErrors``, not including ``TGraph``.
"""

from __future__ import absolute_import

import numpy

# dict_keys(['@fUniqueID', '@fBits', 'fName', 'fTitle', 'fLineColor', 'fLineStyle', 'fLineWidth', 'fFillColor', 'fFillStyle', 'fMarkerColor', 'fMarkerStyle', 'fMarkerSize', 'fNpoints', 'fX', 'fY', 'fFunctions', 'fHistogram', 'fMinimum', 'fMaximum', 'fEXlow', 'fEXhigh', 'fEYlow', 'fEYhigh'])


class TGraphAsymmErrors(object):
    """
    Behaviors for TGraphAsymmErrors, not including ``TGraph``
    """

    def _normalize_array(self, key):
        values = self.member(key)
        return numpy.asarray(values, dtype=values.dtype.newbyteorder("="))

    def _reduce_errors(self, which, lowkey, highkey):
        if which == "low":
            return self._normalize_array(lowkey)
        elif which == "high":
            return self._normalize_array(highkey)
        elif which == "mean":
            return (
                self._normalize_array(highkey) + self._normalize_array(lowkey)
            ) / 2.0
        elif which == "diff":
            return self._normalize_array(highkey) - self._normalize_array(lowkey)

    def values(self, axis="both"):
        if axis in [0, -2, "x"]:
            return self._normalize_array("fX")
        elif axis in [1, -1, "y"]:
            return self._normalize_array("fY")
        elif axis == "both":
            return (self.values("x"), self.values("y"))
        else:
            raise ValueError(
                "axis must be 0 (-2), 1 (-1) or 'x', 'y' or 'both' for a TGraphAsymmErrors"
            )

    def errors(self, which, axis="both"):
        if which not in ["low", "high", "mean", "diff"]:
            raise ValueError(
                "which must be 'low', 'high', 'mean', or 'diff' for a TGraphAsymmErrors"
            )

        if axis in [0, -2, "x"]:
            return self._reduce_errors(which, "fEXlow", "fEXhigh")
        elif axis in [1, -1, "y"]:
            return self._reduce_errors(which, "fEYlow", "fEYhigh")
        elif axis == "both":
            return (self.errors(which, "x"), self.errors(which, "y"))
        else:
            raise ValueError(
                "axis must be 0 (-2), 1 (-1) or 'x', 'y' or 'both' for a TGraphAsymmErrors"
            )

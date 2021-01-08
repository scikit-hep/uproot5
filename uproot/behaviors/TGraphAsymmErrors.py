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

    def values(self, axis="both"):
        if axis in [0, -2, "x"]:
            return self._normalize_array("fX")
        elif axis in [1, -1, "y"]:
            return self._normalize_array("fY")
        elif axis == "both":
            return (self._normalize_array("fX"), self._normalize_array("fY"))
        else:
            raise ValueError(
                "axis must be 0 (-2), 1 (-1) or 'x', 'y' or 'both' for a TGraphAsymmErrors"
            )

    def errors(self, axis="both", which="both"):
        """
        ( array[XLow, YLow], array[XHigh, YHigh] )
        """
        axes = []
        if axis in [0, -2, "x"]:
            axes = ["X"]
        elif axis in [1, -1, "y"]:
            axes = ["Y"]
        elif axis == "both":
            axes = ["X", "Y"]
        else:
            raise ValueError(
                "axis must be 0 (-2), 1 (-1) or 'x', 'y' or 'both' for a TGraphAsymmErrors"
            )

        dirs = []
        if which == "low":
            dirs = ["low"]
        elif which == "high":
            dirs = ["high"]
        elif which == "both":
            dirs = ["low", "high"]
        else:
            raise ValueError(
                "which must be 'low', 'high', or 'both' for a TGraphAsymmErrors"
            )

        return tuple(
            self._normalize_array(f"fE{axis_str}{dir_str}")
            for axis_str in axes
            for dir_str in dirs
        )

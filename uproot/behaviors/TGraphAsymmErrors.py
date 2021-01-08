# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

"""
This module defines the behaviors of ``TGraphAsymmErrors``, not including ``TGraph``.
"""

from __future__ import absolute_import

import numpy

import uproot

# dict_keys(['@fUniqueID', '@fBits', 'fName', 'fTitle', 'fLineColor', 'fLineStyle', 'fLineWidth', 'fFillColor', 'fFillStyle', 'fMarkerColor', 'fMarkerStyle', 'fMarkerSize', 'fNpoints', 'fX', 'fY', 'fFunctions', 'fHistogram', 'fMinimum', 'fMaximum', 'fEXlow', 'fEXhigh', 'fEYlow', 'fEYhigh'])

class TGraphAsymmErrors(object):
    """
    Behaviors for TGraphAsymmErrors, not including ``TGraph``
    """

    def values(self):
        """
        array[X, Y]
        """
        if hasattr(self, "_values"):
            values = self._values
        else:
            _x, _y = self.member("fX"), self.member("fY")
            values = numpy.asarray([_x, _y], dtype=_x.dtype.newbyteorder("="))
            self._values = values

        return values

    def errors(self):
        """
        ( array[XLow, YLow], array[XHigh, YHigh] )
        """
        if hasattr(self, "_errors"):
            errors = self._errors
        else:
            _exlow, _eylow = self.member("fEXlow"), self.member("fEYlow")
            _exhigh, _eyhigh = self.member("fEXhigh"), self.member("fEYhigh")

            _low = numpy.asarray([_exlow, _eylow], dtype=_exlow.dtype.newbyteorder("="))
            _high = numpy.asarray([_exhigh, _eyhigh], dtype=_exhigh.dtype.newbyteorder("="))
            errors = (_low, _high)
            self._errors = errors

        return errors

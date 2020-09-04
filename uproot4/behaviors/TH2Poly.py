# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

"""
Defines the behavior of ``TH2Poly``.
"""

from __future__ import absolute_import

import uproot4.behaviors.TH2


class TH2Poly(uproot4.behaviors.TH1.Histogram):
    """
    Behaviors for two-dimensional polygon histograms: ROOT's ``TH2Poly``.
    """
    no_inherit = (uproot4.behaviors.TH2.TH2,)

    def edges(self, axis):
        raise NotImplementedError(repr(self))

    def values(self):
        raise NotImplementedError(repr(self))

    def values_errors(self, error_mode=0):
        raise NotImplementedError(repr(self))

    def to_numpy(self, flow=False, dd=False, errors=False):
        raise NotImplementedError(repr(self))

    def to_boost(self):
        raise NotImplementedError(repr(self))

    def to_hist(self):
        return uproot4.extras.hist().Hist(self.to_boost())

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

"""
Defines the behavior of ``TProfile3D``.
"""

from __future__ import absolute_import

import uproot4.behaviors.TH1
import uproot4.behaviors.TH3
import uproot4.behaviors.TProfile


class TProfile3D(uproot4.behaviors.TProfile.Profile):
    """
    Behaviors for three-dimensional profiles: ROOT's ``TProfile3D``.
    """

    no_inherit = (uproot4.behaviors.TH3.TH3,)

    def axis(self, axis):
        if axis == 0 or axis == -3 or axis == "x":
            return self.member("fXaxis")
        elif axis == 1 or axis == -2 or axis == "y":
            return self.member("fYaxis")
        elif axis == 2 or axis == -1 or axis == "z":
            return self.member("fZaxis")
        else:
            raise ValueError(
                "axis must be 0 (-3), 1 (-2), 2 (-1) or 'x', 'y', 'z' for a TProfile3D"
            )

    def values(self, flow=False):
        raise NotImplementedError(repr(self))

    def values_errors(self, flow=False, error_mode=""):
        raise NotImplementedError(repr(self))

    def to_boost(self):
        raise NotImplementedError(repr(self))

    def to_hist(self):
        return uproot4.extras.hist().Hist(self.to_boost())

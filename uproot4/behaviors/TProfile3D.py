# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import uproot4.behaviors.TH1


class TProfile3D(object):
    def edges(self, axis):
        if axis == 0 or axis == "x":
            return uproot4.behaviors.TH1._edges(self.member("fXaxis"))
        elif axis == 1 or axis == "y":
            return uproot4.behaviors.TH1._edges(self.member("fYaxis"))
        elif axis == 2 or axis == "z":
            return uproot4.behaviors.TH1._edges(self.member("fZaxis"))
        else:
            raise ValueError("axis must be 0, 1, 2 or 'x', 'y', 'z' for a TProfile3D")

    def values(self):
        raise NotImplementedError(repr(self))

    def values_errors(self, error_mode=0):
        raise NotImplementedError(repr(self))

    @property
    def np(self):
        raise NotImplementedError(repr(self))

    @property
    def bh(self):
        raise NotImplementedError(repr(self))

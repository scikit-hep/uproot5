# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

"""
Defines the behavior of ``TProfile2D``.
"""

from __future__ import absolute_import

import numpy

import uproot4.behaviors.TH1
import uproot4.behaviors.TH2
import uproot4.behaviors.TProfile


class TProfile2D(uproot4.behaviors.TProfile.Profile):
    """
    Behaviors for two-dimensional profiles: ROOT's ``TProfile2D``.
    """

    no_inherit = (uproot4.behaviors.TH2.TH2,)

    @property
    def axes(self):
        return (self.member("fXaxis"), self.member("fYaxis"))

    def axis(self, axis):
        if axis == 0 or axis == -2 or axis == "x":
            return self.member("fXaxis")
        elif axis == 1 or axis == -1 or axis == "y":
            return self.member("fYaxis")
        else:
            raise ValueError("axis must be 0 (-2), 1 (-1) or 'x', 'y' for a TProfile2D")

    def effective_entries(self, flow=False):
        fBinEntries = numpy.asarray(self.member("fBinEntries"))
        out = uproot4.behaviors.TProfile._effective_entries_1d(
            fBinEntries.reshape(-1),
            numpy.asarray(self.member("fBinSumw2")).reshape(-1),
            self.member("fNcells"),
        )
        out = out.reshape(fBinEntries.shape)
        if flow:
            return out
        else:
            return out[1:-1, 1:-1]

    def values(self, flow=False):
        (root_cont,) = self.base(uproot4.models.TArray.Model_TArray)
        root_cont = numpy.asarray(root_cont, dtype=numpy.float64)
        out = uproot4.behaviors.TProfile._values_1d(
            numpy.asarray(self.member("fBinEntries")).reshape(-1),
            root_cont.reshape(-1),
        )
        out = out.reshape(root_cont.shape)
        if flow:
            return out
        else:
            return out[1:-1, 1:-1]

    def values_errors(self, flow=False, error_mode=""):
        (root_cont,) = self.base(uproot4.models.TArray.Model_TArray)
        root_cont = numpy.asarray(root_cont, dtype=numpy.float64)
        fSumw2 = self.member("fSumw2", none_if_missing=True)
        if fSumw2 is not None:
            fSumw2 = numpy.asarray(fSumw2).reshape(-1)
        out = uproot4.behaviors.TProfile._values_errors_1d(
            error_mode,
            numpy.asarray(self.member("fBinEntries")).reshape(-1),
            root_cont.reshape(-1),
            fSumw2,
            self.member("fNcells"),
            numpy.asarray(self.member("fBinSumw2")).reshape(-1),
        )
        out = out.reshape(root_cont.shape)
        if flow:
            return out
        else:
            return out[1:-1, 1:-1]

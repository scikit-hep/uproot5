# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import uproot4.behaviors.TBranch


class TTree(uproot4.behaviors.TBranch.HasBranches):
    @property
    def name(self):
        return self.member("fName")

    @property
    def title(self):
        return self.member("fTitle")

    def __repr__(self):
        return "<TTree {0} ({1} branches)>".format(repr(self.name), len(self))

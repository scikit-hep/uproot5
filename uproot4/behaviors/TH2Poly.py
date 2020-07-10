# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import


class TH2Poly(object):
    @property
    def np(self):
        raise NotImplementedError(repr(self))

    @property
    def bh(self):
        raise NotImplementedError(repr(self))

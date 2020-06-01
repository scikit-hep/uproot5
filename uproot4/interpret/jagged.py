# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import


class CompactJaggedArray(object):
    def __init__(self, offsets, content):
        self._offsets = offsets
        self._content = content

    @property
    def offsets(self):
        return self._offsets

    @property
    def content(self):
        return self._content

    @property
    def parents(self):
        raise NotImplementedError

    @property
    def localindex(self):
        raise NotImplementedError


class JaggedArray(object):
    def __init__(self, starts, stops, content):
        self._starts = starts
        self._stops = stops
        self._content = content

    @property
    def starts(self):
        return self._starts

    @property
    def stops(self):
        return self._stops

    @property
    def content(self):
        return self._content

    @property
    def compact(self):
        raise NotImplementedError

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import


def streamer_named(classname, version=None):
    import uproot4

    uproot4.streamers

    raise NotImplementedError


def class_named(classname, version=None):
    import uproot4

    uproot4.classes

    raise NotImplementedError

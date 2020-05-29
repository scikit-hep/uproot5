# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import sys
import json

try:
    from io import StringIO
except ImportError:
    from StringIO import StringIO

import numpy
import pytest
import skhep_testdata

import uproot4


def test():
    with uproot4.open(skhep_testdata.data_path("uproot-mc10events.root")) as f:
        assert f["Events"].name == "Events"
        assert f["Events/Info"].name == "Info"
        assert f["Events/Info/evtNum"].name == "evtNum"
        assert f["Events/evtNum"].name == "evtNum"
        assert f["Events"]["evtNum"].name == "evtNum"
        assert f["Events"]["/Info"].name == "Info"
        with pytest.raises(KeyError):
            f["Events"]["/evtNum"]

    # sample-6.20.04-uncompressed

    # raise Exception

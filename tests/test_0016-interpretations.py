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


def test_get_key():
    with uproot4.open(skhep_testdata.data_path("uproot-mc10events.root")) as f:
        assert f["Events"].name == "Events"
        assert f["Events/Info"].name == "Info"
        assert f["Events/Info/evtNum"].name == "evtNum"
        assert f["Events"]["Info/evtNum"].name == "evtNum"
        assert f["Events"]["/Info/evtNum"].name == "evtNum"
        assert f["Events/evtNum"].name == "evtNum"
        assert f["Events"]["evtNum"].name == "evtNum"
        assert f["Events"]["/Info"].name == "Info"
        assert f["Events"]["/Info/"].name == "Info"
        with pytest.raises(KeyError):
            f["Events"]["/evtNum"]


def test_basket_data():
    with uproot4.open(
        skhep_testdata.data_path("uproot-sample-6.20.04-uncompressed.root"),
    ) as f:
        assert f["sample/i4"].basket_key(3).fSeekKey == 35042
        assert f["sample/i4"].basket(3).data.view(">i4").tolist() == [
            6,
            7,
            8,
            9,
            10,
            11,
            12,
        ]

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import pytest
import skhep_testdata

import uproot4


def test_empty():
    with uproot4.open(skhep_testdata.data_path("uproot-empty.root")) as f:
        t = f["tree"]
        assert t["x"].array(library="np").tolist() == []
        assert t["y"].array(library="np").tolist() == []
        assert t["z"].array(library="np").tolist() == []

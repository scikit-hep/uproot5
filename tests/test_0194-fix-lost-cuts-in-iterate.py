# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import numpy
import pytest
import skhep_testdata

import uproot4


def test():
    with uproot4.open(skhep_testdata.data_path("uproot-Zmumu.root")) as f:
        for arrays in f["events"].iterate("px1", step_size=1000, cut="px1 > 0", library="np"):
            assert numpy.all(arrays["px1"] > 0)

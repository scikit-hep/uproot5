# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import numpy
import pytest
import skhep_testdata

import uproot4


def test_boost_1d():
    # {'hpx': 'TH1F', 'hpxpy': 'TH2F', 'hprof': 'TProfile'}

    with uproot4.open(skhep_testdata.data_path("uproot-hepdata-example.root")) as f:
        print(f["hpx"].hello())

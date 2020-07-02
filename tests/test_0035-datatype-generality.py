# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import numpy
import pytest
import skhep_testdata

import uproot4


def test_TH2_in_ttree():
    with uproot4.open(skhep_testdata.data_path("uproot-issue-tbranch-of-th2.root"))[
        "g4SimHits/tree"
    ] as tree:
        assert (
            tree["histogram"].array(library="np")[0].member("fXaxis").member("fName")
            == "xaxis"
        )

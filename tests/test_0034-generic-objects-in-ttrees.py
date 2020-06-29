# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import sys
import json

import numpy
import pytest
import skhep_testdata

import uproot4


def test_histograms_outside_of_ttrees():
    with uproot4.open(skhep_testdata.data_path("uproot-hepdata-example.root")) as f:
        contents = numpy.asarray(f["hpx"].bases[-1])
        assert (contents.min(), contents.max()) == (0.0, 2417.0)

        contents = numpy.asarray(f["hpxpy"].bases[-1])
        assert (contents.min(), contents.max()) == (0.0, 497.0)

        contents = numpy.asarray(f["hprof"].bases[-1].bases[-1])
        assert (contents.min(), contents.max()) == (0.0, 3054.7299575805664)

        numpy.asarray(f["ntuple"])


# def test_gohep_nosplit_file():
#     with uproot4.open(skhep_testdata.data_path("uproot-small-evnt-tree-nosplit.root"))["tree/evt"] as branch:
#         print(branch.array(library="np", entry_start=5))

#     raise Exception

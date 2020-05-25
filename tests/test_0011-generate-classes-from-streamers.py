# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import sys
import json

import numpy
import pytest
import skhep_testdata

import uproot4
import uproot4.reading


def test():
    with uproot4.open(skhep_testdata.data_path("uproot-histograms.root")) as f:
        assert f.file.class_named("TH1", 7).member_names == [
            "fNcells",
            "fXaxis",
            "fYaxis",
            "fZaxis",
            "fBarOffset",
            "fBarWidth",
            "fEntries",
            "fTsumw",
            "fTsumw2",
            "fTsumwx",
            "fTsumwx2",
            "fMaximum",
            "fMinimum",
            "fNormFactor",
            "fContour",
            "fSumw2",
            "fOption",
            "fFunctions",
            "fBufferSize",
            "fBuffer",
            "fBinStatErrOpt",
        ]

        print(f.file.class_named("TH1", 7).class_code)
        print(f.file.class_named("TH1F", 2).class_code)

        print(f.get("one"))

    # raise Exception

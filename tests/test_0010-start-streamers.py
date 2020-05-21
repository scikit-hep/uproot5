# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import sys
import json

import numpy
import pytest
import skhep_testdata

import uproot4
import uproot4.reading


# def test():
#     with uproot4.open(skhep_testdata.data_path("uproot-histograms.root")) as f:
#         for k, v in f.file.streamers.items():
#             print(k, json.dumps(v.tojson(), indent=4))

#     raise Exception

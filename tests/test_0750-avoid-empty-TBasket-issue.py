# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

import json
import sys

import numpy
import pytest
import skhep_testdata

import uproot


def test():
    with uproot.open(skhep_testdata.data_path("uproot-issue-750.root"))[
        "tout/Electron_eta"
    ] as branch:
        branch.array(library="np")

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

import io
import os

import numpy as np
import pytest
import skhep_testdata

import uproot


def test():
    with uproot.open(skhep_testdata.data_path("uproot-issue-359.root")) as file:
        matrix = file["covmatrixOCratio"]
        num_elements = matrix.member("fNrows") * (matrix.member("fNcols") + 1) // 2
        assert len(matrix.member("fElements")) == num_elements

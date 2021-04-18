# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

import numpy as np
import pytest
import skhep_testdata

import uproot


def test():
    for arrays in uproot.iterate(
        skhep_testdata.data_path("uproot-issue335.root") + ":empty_tree",
        ["var"],
        library="np",
    ):
        pass

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import pytest
import skhep_testdata

import uproot4


def test_common_offsets():
    # this file has terrible branch alignment
    with uproot4.open(skhep_testdata.data_path("uproot-hepdata-example.root")) as f:
        assert f["ntuple;1"].common_entry_offsets() == [0, 75000]

    # this file has just one branch
    with uproot4.open(skhep_testdata.data_path("uproot-foriter.root")) as f:
        assert f["foriter;1"].common_entry_offsets() == [0, 6, 12, 18, 24, 30, 36, 42, 46]

    # I could not find a testdata file with many well-aligned branches

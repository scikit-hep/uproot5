# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

from __future__ import absolute_import

import pytest
import skhep_testdata

import uproot


def test():
    file = uproot.open(skhep_testdata.data_path("uproot-Zmumu-uncompressed.root"))
    basket = file["events/px1"].basket(0)
    file.close()
    assert basket.data[0] == 192

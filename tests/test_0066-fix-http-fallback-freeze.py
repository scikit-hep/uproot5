# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import numpy
import pytest

import uproot4


@pytest.mark.network
def test():
    with uproot4.open("http://scikit-hep.org/uproot/examples/HZZ.root:events") as t:
        t["MET_px"].array()
        t["MET_py"].array()

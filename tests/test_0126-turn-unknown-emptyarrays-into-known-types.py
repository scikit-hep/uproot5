# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import numpy
import pytest
import skhep_testdata

import uproot4


def test():
    awkward1 = pytest.importorskip("awkward1")
    with uproot4.open(skhep_testdata.data_path("uproot-vectorVectorDouble.root")) as f:
        array = f["t/x"].array(entry_stop=2)
        assert str(awkward1.type(array)) == "2 * var * var * float64"

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

import pytest

import numpy as np
from numpy.testing import assert_array_equal
import skhep_testdata
import uproot


def test_numpy():
    with uproot.open(skhep_testdata.data_path("uproot-HZZ.root")) as f:
        a = f["events/Muon_Px"].array(entry_start=1, entry_stop=1, library="np")
        assert isinstance(a, np.ndarray)
        assert len(a) == 0


def test_awkward():
    ak = pytest.importorskip("awkward")

    with uproot.open(skhep_testdata.data_path("uproot-HZZ.root")) as f:
        a = f["events/Muon_Px"].array(entry_start=1, entry_stop=1)
        assert isinstance(a, ak.Array)
        assert len(a) == 0

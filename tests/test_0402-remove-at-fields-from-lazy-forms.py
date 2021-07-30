# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

from __future__ import absolute_import

import numpy
import pytest
import skhep_testdata

import uproot

awkward = pytest.importorskip("awkward")


def test():
    array = uproot.lazy(skhep_testdata.data_path("uproot-HZZ-objects.root") + ":events")
    assert array.jetp4.fP.fX[:5].tolist() == [
        [],
        [-38.87471389770508],
        [],
        [-71.6952133178711, 36.60636901855469, -28.866418838500977],
        [3.880161762237549, 4.979579925537109],
    ]

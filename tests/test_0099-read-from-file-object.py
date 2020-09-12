# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import pytest
import skhep_testdata

import uproot4


def test():
    with open(skhep_testdata.data_path("uproot-Zmumu.root"), "rb") as f:
        assert uproot4.open({f: "events"})["px1"].array()[:10].tolist() == [
            -41.1952876442,
            35.1180497674,
            35.1180497674,
            34.1444372454,
            22.7835819537,
            -19.8623073126,
            -19.8623073126,
            -20.1773731496,
            71.1437106445,
            51.0504859191,
        ]

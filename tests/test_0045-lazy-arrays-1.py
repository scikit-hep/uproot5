# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import numpy
import pytest
import skhep_testdata

import uproot4


def test_array_cast():
    with uproot4.open(skhep_testdata.data_path("uproot-Zmumu.root"))[
        "events"
    ] as events:
        assert numpy.array(events["px1"])[:5].tolist() == [
            -41.1952876442,
            35.1180497674,
            35.1180497674,
            34.1444372454,
            22.7835819537,
        ]


# def test_awkward():
#     awkward1 = pytest.importorskip("awkward1")
#     files = (
#         skhep_testdata.data_path("uproot-sample-6.20.04-uncompressed.root").replace(
#             "6.20.04", "*"
#         )
#         + ":sample"
#     )
#     array = uproot4.lazy(files)

#     print(array)

#     raise Exception

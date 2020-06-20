# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import numpy
import pytest
import skhep_testdata

import uproot4


def test_fallback_reading():
    # with uproot4.open(
    #     skhep_testdata.data_path("uproot-small-evnt-tree-fullsplit.root")
    # ) as f:
    #     f["tree:evt/P3/P3.Py"]
    #     assert f.file._streamers is None

    with uproot4.open(skhep_testdata.data_path("uproot-demo-double32.root")) as f:
        f["T/fD64"]
        assert f.file._streamers is not None

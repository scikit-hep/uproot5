# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

import numpy
import pytest
import skhep_testdata

import uproot


def test_fallback_reading():
    with uproot.open(
        skhep_testdata.data_path("uproot-small-evnt-tree-fullsplit.root")
    ) as f:
        f["tree:evt/P3/P3.Py"]
        assert f.file._streamers is None

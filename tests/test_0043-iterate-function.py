# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import numpy
import pytest
import skhep_testdata

import uproot4


def test_num_entries_for():
    with uproot4.open(skhep_testdata.data_path("uproot-HZZ.root"))["events"] as events:
        assert events.num_entries_for("1 kB") == 12
        assert events.num_entries_for("10 kB") == 118
        assert events.num_entries_for("0.1 MB") == 1213

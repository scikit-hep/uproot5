# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import numpy
import pytest
import skhep_testdata

import uproot4


def test():
    one = skhep_testdata.data_path("uproot-sample-6.16.00-uncompressed.root")
    two = skhep_testdata.data_path("uproot-sample-6.18.00-uncompressed.root")
    bad = one.replace(".root", "-DOES-NOT-EXIST.root")
    okay = one.replace(".root", "-DOES-NOT-EXIST-*.root")

    assert len(list(uproot4.iterate([one, two], step_size="1 TB", library="np"))) == 2

    with pytest.raises(uproot4._util._FileNotFoundError):
        list(uproot4.iterate([one, two, bad], library="np"))

    assert (
        len(list(uproot4.iterate([one, two, okay], step_size="1 TB", library="np")))
        == 2
    )

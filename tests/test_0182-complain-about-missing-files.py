# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

import numpy
import pytest
import skhep_testdata

import uproot


def test():
    one = skhep_testdata.data_path("uproot-sample-6.16.00-uncompressed.root")
    two = skhep_testdata.data_path("uproot-sample-6.18.00-uncompressed.root")
    bad = one.replace(".root", "-DOES-NOT-EXIST.root")
    okay = one.replace(".root", "-DOES-NOT-EXIST-*.root")

    assert len(list(uproot.iterate([one, two], step_size="1 TB", library="np"))) == 2

    with pytest.raises(uproot._util._FileNotFoundError):
        list(uproot.iterate([one, two, bad], library="np"))

    assert (
        len(list(uproot.iterate([one, two, okay], step_size="1 TB", library="np"))) == 2
    )

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

import json
import queue
import sys

import numpy
import pytest
import skhep_testdata

import uproot


def test():
    filename = skhep_testdata.data_path("test_ntuple_int_float.root")
    with uproot.open(filename) as f:
        R = f["ntuple"]
        assert R.keys() == ["one_integers", "two_floats"]
        assert [r.type_name for r in R.header.field_records] == [
            "std::int32_t",
            "float",
        ]
        assert R.header.crc32 == R.footer.header_crc32
        assert all(R.arrays(entry_stop=3)["one_integers"] == numpy.array([9,8,7]))
        assert all(R.arrays("one_integers", entry_stop=3)["one_integers"] == numpy.array([9,8,7]))

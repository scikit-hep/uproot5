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
        obj = f["ntuple"]
        assert obj.keys() == ["one_integers", "two_floats"]
        assert [r.type_name for r in obj.header.field_records] == [
            "std::int32_t",
            "float",
        ]
        assert obj.header.crc32 == obj.footer.header_crc32

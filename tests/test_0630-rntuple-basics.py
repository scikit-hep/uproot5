# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

import json
import queue
import sys

import numpy
import pytest
import skhep_testdata

import uproot


def test_flat():
    filename = skhep_testdata.data_path("test_ntuple_int_float.root")
    with uproot.open(filename) as f:
        R = f["ntuple"]
        assert R.keys() == ["one_integers", "two_floats"]
        assert [r.type_name for r in R.header.field_records] == [
            "std::int32_t",
            "float",
        ]
        assert R.header.crc32 == R.footer.header_crc32
        assert all(R.arrays(entry_stop=3)["one_integers"] == numpy.array([9, 8, 7]))
        assert all(
            R.arrays("one_integers", entry_stop=3)["one_integers"]
            == numpy.array([9, 8, 7])
        )
        assert all(
            R.arrays(entry_start=1, entry_stop=3)["one_integers"] == numpy.array([8, 7])
        )

    filename = skhep_testdata.data_path("test_ntuple_int_5e4.root")
    with uproot.open(filename) as f:
        R = f["ntuple"]
        assert all(
            R.arrays(entry_stop=3)["one_integers"] == numpy.array([50000, 49999, 49998])
        )
        assert all(R.arrays(entry_start=-3)["one_integers"] == numpy.array([3, 2, 1]))


def test_jagged():
    filename = skhep_testdata.data_path("test_ntuple_int_vfloat_tlv_vtlv.root")
    with uproot.open(filename) as f:
        R = f["ntuple"]
        assert R.keys() == ["one_integers", "two_v_floats", "three_LV", "four_v_LVs"]

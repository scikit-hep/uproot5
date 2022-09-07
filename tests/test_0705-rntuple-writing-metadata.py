# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

import json
import os
import queue
import sys

import awkward as ak
import numpy
import pytest
import skhep_testdata

import uproot


def test_flat(tmp_path):
    filepath = os.path.join(tmp_path, "test.root")

    with uproot.recreate(filepath) as file:
        akform = ak._v2.forms.RecordForm(
            [
                ak._v2.forms.NumpyForm("float64"),
                ak._v2.forms.NumpyForm("int32"),
                ak._v2.forms.NumpyForm("bool"),
            ],
            ["one", "two", "three"],
        )
        file.mkntuple("ntuple", akform)

    frs = uproot.open(filepath)["ntuple"].header.field_records
    assert frs[0].parent_field_id == 0
    assert frs[1].parent_field_id == 1
    assert frs[2].parent_field_id == 2
    assert frs[0].field_name == "one"
    assert frs[1].field_name == "two"
    assert frs[2].field_name == "three"
    assert frs[0].type_name == "double"
    assert frs[1].type_name == "std::int32_t"
    assert frs[2].type_name == "bit"

    crs = uproot.open(filepath)["ntuple"].header.column_records
    assert crs[0].type == 7
    assert crs[1].type == 11
    assert crs[2].type == 6
    assert crs[0].field_id == 0
    assert crs[1].field_id == 1
    assert crs[2].field_id == 2
    assert crs[0].nbits == 64
    assert crs[1].nbits == 32
    assert crs[2].nbits == 1

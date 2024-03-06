# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import json
import os
import queue
import sys

import numpy
import pytest
import skhep_testdata

import uproot

ak = pytest.importorskip("awkward")


@pytest.mark.skip(
    reason="RNTuple writing is pending until specification 1.0.0 is released."
)
def test_header(tmp_path):
    filepath = os.path.join(tmp_path, "test.root")

    with uproot.recreate(filepath) as file:
        akform = ak.forms.RecordForm(
            [
                ak.forms.NumpyForm("float64"),
                ak.forms.NumpyForm("int32"),
                ak.forms.NumpyForm("bool"),
            ],
            ["one", "two", "three"],
        )
        file.mkrntuple("ntuple", akform)

    file = uproot.open(filepath)["ntuple"]

    header = file.header
    assert header.crc32 == file.footer.header_crc32

    frs = header.field_records
    assert frs[0].parent_field_id == 0
    assert frs[1].parent_field_id == 1
    assert frs[2].parent_field_id == 2
    assert frs[0].field_name == "one"
    assert frs[1].field_name == "two"
    assert frs[2].field_name == "three"
    assert frs[0].type_name == "double"
    assert frs[1].type_name == "std::int32_t"
    assert frs[2].type_name == "bit"

    crs = header.column_records
    assert crs[0].type == 7
    assert crs[1].type == 11
    assert crs[2].type == 6
    assert crs[0].field_id == 0
    assert crs[1].field_id == 1
    assert crs[2].field_id == 2
    assert crs[0].nbits == 64
    assert crs[1].nbits == 32
    assert crs[2].nbits == 1


@pytest.mark.skip(
    reason="RNTuple writing is pending until specification 1.0.0 is released."
)
def test_writable(tmp_path):
    filepath = os.path.join(tmp_path, "test.root")

    with uproot.recreate(filepath) as file:
        akform = ak.forms.RecordForm(
            [
                ak.forms.NumpyForm("int32"),
            ],
            ["one"],
        )
        file.mkrntuple("ntuple", akform)
        assert type(file["ntuple"]).__name__ == "WritableNTuple"


# FIXME get ROOT to recognize it
# ROOT = pytest.importorskip("ROOT")


# def test_ROOT(tmp_path, capfd):
#     filepath = os.path.join(tmp_path, "test.root")

#     with uproot.recreate(filepath) as file:
#         akform = ak.forms.RecordForm(
#             [
#                 ak.forms.NumpyForm("float64"),
#                 ak.forms.NumpyForm("int32"),
#             ],
#             ["one", "two"],
#         )
#         file.mkrntuple("ntuple", akform)
#     RT = ROOT.Experimental.RNTupleReader.Open("ntuple", filepath)
#     RT.PrintInfo()
#     out = capfd.readouterr().out
#     assert "* Field 1   : one (double)" in out
#     assert "* Field 2   : two (std::int32_t)" in out

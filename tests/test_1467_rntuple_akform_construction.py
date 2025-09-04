# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import pytest
import skhep_testdata

import uproot

ak = pytest.importorskip("awkward")


def test_akform_logic():
    filepath = skhep_testdata.data_path(
        "cmsopendata2015_ttbar_19980_NANOAOD_RNTupleImporter_rntuple_v1-0-0-1.root"
    )

    with uproot.open(filepath) as file:
        obj = file["Events"]
        arrays = obj.arrays()

    # This is a very simple test, but if something was wrong it would have crashed before getting here
    assert len(arrays.fields) == 969

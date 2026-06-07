# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import os

import awkward as ak
import pytest
import skhep_testdata

import uproot

graphed_awkward = pytest.importorskip("graphed_awkward")


def test_graphed_write_roundtrip(tmp_path):
    test_path = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    ref = uproot.open(test_path).arrays()

    g_array = uproot.graphed(test_path, library="ak")
    out_file = os.path.join(tmp_path, "graphed_out.root")
    uproot.graphed_write(g_array, out_file, tree_name="events")

    back = uproot.open(out_file + ":events").arrays()
    assert set(back.fields) == set(ref.fields)
    assert ak.almost_equal(back.px1, ref.px1)
    assert ak.almost_equal(back.py2, ref.py2)


def test_graphed_write_compute_false_returns_array(tmp_path):
    test_path = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    g_array = uproot.graphed(test_path, library="ak")
    result = uproot.graphed_write(
        g_array, os.path.join(tmp_path, "unwritten.root"), compute=False
    )
    assert hasattr(result, "fields")  # a computed awkward array, not a path
    assert not os.path.exists(os.path.join(tmp_path, "unwritten.root"))

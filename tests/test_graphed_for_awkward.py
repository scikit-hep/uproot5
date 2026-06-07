# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import awkward as ak
import pytest
import skhep_testdata

import uproot

graphed_awkward = pytest.importorskip("graphed_awkward")


def test_single_graphed_array():
    test_path = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    ak_array = uproot.open(test_path).arrays()

    g_array = uproot.graphed(test_path, library="ak")
    out = uproot.graphed_compute(g_array)

    assert ak.almost_equal(out, ak_array)


def test_graphed_expression_matches_uproot():
    test_path = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    ak_array = uproot.open(test_path).arrays()

    g_array = uproot.graphed(test_path, library="ak")
    out = uproot.graphed_compute(g_array.px1 + g_array.px2)

    assert ak.almost_equal(out, ak_array.px1 + ak_array.px2)


def test_graphed_concatenation():
    test_path1 = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    test_path2 = skhep_testdata.data_path("uproot-Zmumu-uncompressed.root") + ":events"

    ak_array = uproot.concatenate([test_path1, test_path2])
    out = uproot.graphed_compute(uproot.graphed([test_path1, test_path2], library="ak"))

    assert ak.almost_equal(out, ak_array)


def test_metadata_only_construction():
    # construction reads ONLY metadata: the source form is a typetracer (no event data is read)
    test_path = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    g_array = uproot.graphed(test_path, library="ak")
    assert ak.backend(g_array.session.form(g_array).tt) == "typetracer"


def test_np_library_not_implemented():
    test_path = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    with pytest.raises(NotImplementedError):
        uproot.graphed(test_path, library="np")

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
"""Construction-time behaviour of ``uproot.graphed`` (metadata only).

Actually *running* a graphed analysis is covered in ``test_graphed_executor.py`` — through the real
``graphed-exec-local`` executors, not a ``materialize`` shortcut.
"""

import awkward as ak
import pytest
import skhep_testdata

import uproot

graphed_awkward = pytest.importorskip("graphed.awkward")


def test_metadata_only_construction():
    # construction reads ONLY metadata: the source form is a typetracer (no event data is read)
    test_path = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    g_array = uproot.graphed(test_path, library="ak")
    assert ak.backend(g_array.session.form(g_array).tt) == "typetracer"


def test_records_a_backend_agnostic_graph():
    # building an expression records nodes without reading data
    test_path = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    g_array = uproot.graphed(test_path, library="ak")
    before = g_array.session.node_count()
    _ = g_array.px1 + g_array.px2
    assert g_array.session.node_count() > before


def test_np_library_not_implemented():
    test_path = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    with pytest.raises(NotImplementedError):
        uproot.graphed(test_path, library="np")

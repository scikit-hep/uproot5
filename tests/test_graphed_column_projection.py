# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import awkward as ak
import pytest
import skhep_testdata

import uproot

graphed_awkward = pytest.importorskip("graphed_awkward")

from uproot._graphed import _GraphedTTreeSource


def test_column_projection_sanity_check():
    test_path = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    ak_array = uproot.open(test_path).arrays()

    g_array = uproot.graphed(test_path, library="ak")
    out = uproot.graphed_compute(g_array.px1 + g_array.py1)

    assert ak.almost_equal(out, ak_array.px1 + ak_array.py1)


def test_necessary_columns_are_minimal():
    test_path = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    g_array = uproot.graphed(test_path, library="ak")
    assert uproot.necessary_columns(g_array.px1 + g_array.px2) == {
        "events": frozenset({"px1", "px2"})
    }


def test_compute_reads_only_needed_branches():
    # the over-touching guard (cf. dask-awkward column projection): computing px1+px2 must read
    # ONLY those two TBranches from the file, never the others.
    test_path = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    g_array = uproot.graphed(test_path, library="ak")
    uproot.graphed_compute(g_array.px1 + g_array.px2)

    source = next(
        s for s in g_array.session._sources.values() if isinstance(s, _GraphedTTreeSource)
    )
    assert set(source.last_columns_read) == {"px1", "px2"}


def test_unprojected_compute_reads_everything():
    test_path = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    g_array = uproot.graphed(test_path, library="ak")
    uproot.graphed_compute(g_array.px1 + g_array.px2, project=False)

    source = next(
        s for s in g_array.session._sources.values() if isinstance(s, _GraphedTTreeSource)
    )
    # without projection every selected branch is read
    assert set(source.last_columns_read) == set(uproot.open(test_path).keys())

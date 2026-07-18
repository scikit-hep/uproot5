# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
"""Necessary-buffer (column) projection for ``uproot.graphed`` — the dask-awkward column-projection
analogue. Projection is metadata-only; the executor (``test_graphed_executor.py``) then reads only the
projected ``TBranches`` per partition.
"""

import os
import sys

import pytest
import skhep_testdata

import uproot

graphed_awkward = pytest.importorskip("graphed.awkward")

sys.path.insert(0, os.path.dirname(__file__))


def test_necessary_columns_are_minimal():
    test_path = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    g_array = uproot.graphed(test_path, library="ak")
    assert uproot.necessary_columns(g_array.px1 + g_array.px2) == {
        "events": frozenset({"px1", "px2"})
    }


def test_executor_reads_exactly_the_projected_columns():
    # the over-touching guard: the columns the executor reads per partition (graphed_uproot_analysis.
    # COLUMNS) are exactly the necessary buffers of that analysis — nothing more.
    import graphed_uproot_analysis as gu

    test_path = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    g_array = uproot.graphed(test_path, library="ak")
    projected = uproot.necessary_columns(gu.analysis(g_array))["events"]
    assert set(gu.COLUMNS) == set(projected)

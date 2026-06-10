# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
Regression tests for PR #1658: RNTuple/TBranch behavior edge cases.

Covers:
  - HasFields.__getitem__ raises KeyInFileError (not AttributeError) on RField
  - Module-level RNTuple.iterate tracks global offsets across files
  - ak_add_doc forwarded in leaf TBranch .arrays() and .iterate() delegates
"""

from __future__ import annotations

import numpy as np
import pytest

import skhep_testdata

import uproot
from uproot.behaviors.RNTuple import iterate as rn_iterate


# ---------------------------------------------------------------------------
# Fix 1: RField.__getitem__ with non-recursive miss raises KeyInFileError
# ---------------------------------------------------------------------------


def test_rfield_getitem_nonexistent_raises_key_in_file_error():
    """field['.nonexistent'] must raise KeyInFileError, not AttributeError."""
    filename = skhep_testdata.data_path("test_nested_structs_rntuple_v1-0-0-0.root")
    with uproot.open(filename) as f:
        ntuple = f["ntuple"]
        field = next(iter(ntuple.values()))
        with pytest.raises(uproot.KeyInFileError):
            field[".nonexistent"]


def test_rfield_get_returns_default_for_missing_key():
    """.get() on an RField must return the default for a missing key."""
    filename = skhep_testdata.data_path("test_nested_structs_rntuple_v1-0-0-0.root")
    with uproot.open(filename) as f:
        ntuple = f["ntuple"]
        field = next(iter(ntuple.values()))
        assert field.get(".nonexistent", "default") == "default"


def test_rfield_in_operator_missing_key():
    """'in' operator on an RField must return False for a missing key."""
    filename = skhep_testdata.data_path("test_nested_structs_rntuple_v1-0-0-0.root")
    with uproot.open(filename) as f:
        ntuple = f["ntuple"]
        field = next(iter(ntuple.values()))
        assert ".nonexistent" not in field


# ---------------------------------------------------------------------------
# Fix 2: module-level RNTuple.iterate tracks global offsets across files
# ---------------------------------------------------------------------------


def test_rntuple_iterate_multifile_global_offsets():
    """report.global_entry_start/stop must be continuous across files."""
    filename = skhep_testdata.data_path("test_int_float_rntuple_v1-0-0-0.root")
    results = list(
        rn_iterate([f"{filename}:ntuple", f"{filename}:ntuple"], report=True)
    )
    assert len(results) == 2, "expected one batch per file"
    _, rep0 = results[0]
    _, rep1 = results[1]
    # Second file's global start must equal first file's global stop
    assert rep0.global_entry_start == 0
    assert rep1.global_entry_start == rep0.global_entry_stop
    # Per-file (tree) entries restart at 0 for each file
    assert rep0.tree_entry_start == 0
    assert rep1.tree_entry_start == 0


def test_rntuple_iterate_multifile_no_report_array_lengths():
    """Arrays yielded without report must have correct total entry count."""
    filename = skhep_testdata.data_path("test_int_float_rntuple_v1-0-0-0.root")
    with uproot.open(f"{filename}:ntuple") as f:
        single_count = f.num_entries

    total = 0
    for arrays in rn_iterate([f"{filename}:ntuple", f"{filename}:ntuple"]):
        total += len(arrays[arrays.fields[0]])
    assert total == 2 * single_count


# ---------------------------------------------------------------------------
# Fix 3: ak_add_doc forwarded in leaf TBranch .arrays() and .iterate()
# ---------------------------------------------------------------------------


def test_leaf_tbranch_arrays_forwards_ak_add_doc():
    """TBranch.arrays(ak_add_doc=True) must annotate inner arrays with __doc__."""
    filename = skhep_testdata.data_path("uproot-HZZ.root")
    with uproot.open(filename) as f:
        branch = f["events"]["NJet"]
        result = branch.arrays(
            entry_start=0, entry_stop=5, ak_add_doc=True, array_cache=None
        )
    doc = result["NJet"].layout.parameters.get("__doc__")
    assert doc is not None, "ak_add_doc=True must set __doc__ on the inner array"
    assert doc == branch.title


def test_leaf_tbranch_iterate_forwards_ak_add_doc():
    """TBranch.iterate(ak_add_doc=True) must annotate inner arrays with __doc__."""
    filename = skhep_testdata.data_path("uproot-HZZ.root")
    with uproot.open(filename) as f:
        branch = f["events"]["NJet"]
        chunks = list(
            branch.iterate(entry_stop=10, ak_add_doc=True, step_size=10)
        )
    assert len(chunks) == 1
    doc = chunks[0]["NJet"].layout.parameters.get("__doc__")
    assert doc is not None, "ak_add_doc=True must set __doc__ on the inner array"
    assert doc == branch.title

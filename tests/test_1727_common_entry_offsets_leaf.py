# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
"""``TBranch.common_entry_offsets`` of a branch without sub-branches is its own
``entry_offsets``."""

from __future__ import annotations

import pytest
import skhep_testdata

import uproot


@pytest.mark.parametrize(
    "file, tree, branch",
    [
        ("uproot-HZZ.root", "events", "Muon_Px"),
        ("uproot-foriter.root", "foriter", "data"),
        ("uproot-hepdata-example.root", "ntuple", "px"),
    ],
)
def test_a_leaf_branch_reports_its_own_entry_offsets(file, tree, branch):
    leaf = uproot.open(skhep_testdata.data_path(file) + ":" + tree)[branch]
    assert not leaf.branches
    assert len(leaf.entry_offsets) > 2
    assert leaf.common_entry_offsets() == leaf.entry_offsets

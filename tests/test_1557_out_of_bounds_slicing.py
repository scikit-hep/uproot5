# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import pytest
import skhep_testdata
import numpy
import awkward

import uproot

@pytest.fixture(scope="module")
def tree():
    """
    Fixture to open the standard HZZ test file once for this module.
    """
    with uproot.open(skhep_testdata.data_path("uproot-HZZ.root")) as file:
        yield file["events"]

def test_hard_out_of_bounds_returns_empty(tree):
    """
    Regression test: Reading far beyond the end of the file should return an empty array.
    """
    total = tree.num_entries
    
    # Test on a Jagged array (Muon_Px)
    data = tree["Muon_Px"].array(
        entry_start=total + 100, 
        entry_stop=total + 200,
        library="np"
    )
    assert len(data) == 0
    assert isinstance(data, numpy.ndarray)

    # Test on a Flat array (NJet)
    data_flat = tree["NJet"].array(
        entry_start=total + 100, 
        entry_stop=total + 200,
        library="np"
    )
    assert len(data_flat) == 0

def test_clamped_out_of_bounds_truncates(tree):
    """
    Regression test: Reading a slice that overlaps the end of the file should 
    be silently truncated (clamped) to the valid range.
    """
    total = tree.num_entries
    
    # Range: [total-5, total+100] -> Should be clamped to [total-5, total]
    data = tree["Muon_Px"].array(
        entry_start=total - 5, 
        entry_stop=total + 100,
        library="ak"
    )
    assert len(data) == 5
    assert isinstance(data, awkward.highlevel.Array)

def test_exact_boundary_start(tree):
    """
    Regression test: Starting a read exactly at the end of the file 
    should return an empty array.
    """
    total = tree.num_entries
    
    data = tree["NJet"].array(
        entry_start=total, 
        entry_stop=total + 10,
        library="np"
    )
    assert len(data) == 0

def test_massive_negative_index_is_pythonic(tree):
    """
    Regression test: A negative start index larger than the file size 
    should return the entire array.
    """
    total = tree.num_entries
    data = tree["NJet"].array(entry_start=-1_000_000, library="np")
    assert len(data) == total

def test_iterate_out_of_bounds_yields_nothing(tree):
    """
    Regression test: Iterating over a range entirely outside the file 
    should yield zero batches.
    """
    total = tree.num_entries
    iterator = tree.iterate(
        ["NJet"], 
        step_size=100, 
        entry_start=total + 100, 
        entry_stop=total + 200,
        library="np"
    )
    batches = list(iterator)
    assert len(batches) == 0

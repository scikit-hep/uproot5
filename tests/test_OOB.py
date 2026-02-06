# ==============================================================================
# QA VALIDATION SCRIPT FOR UPROOT OUT-OF-BOUNDS (OOB) SLICING BEHAVIOR
#
# PURPOSE:
# 1. To explicitly test the behavior of Uproot when reading data ranges
#    that are outside the valid number of entries in a TTree.
# 2. To provide verbose logging output to demonstrate the "What, How, and Why"
#    of each test case for debugging and educational purposes.
# 3. To include "negative tests" (tests expected to fail) to prove that
#    our testing framework is capable of catching regressions.
#
# HOW TO RUN FOR MAX VERBOSITY:
# pytest -v -s tests/test_OOB_verbose.py
#   -v: Verbose mode
#   -s: Show print/logging statements
# ==============================================================================

import pytest
import uproot
import skhep_testdata
import numpy as np
import awkward as ak
import logging

# --- Setup a logger for detailed, professional output ---
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Use local test data to ensure speed and reliability ---
FILE_PATH = skhep_testdata.data_path("uproot-HZZ.root")


@pytest.fixture(scope="module")
def tree():
    """Fixture to open the file once for all tests, an optimal practice."""
    logger.info(f"\nSetting up test fixture: Opening file {FILE_PATH}...")
    with uproot.open(FILE_PATH) as file:
        yield file["events"]
    logger.info("Test fixture teardown complete.")


# --- Define the archetypes of data we will test against ---
BRANCH_ARCHETYPES = [
    pytest.param(
        "NJet", id="Flat-Integer-Branch"
    ),  # Simple, fixed-size numbers per event
    pytest.param(
        "Jet_Px", id="Jagged-Float-Branch"
    ),  # Variable number of floats per event
]

# ==============================================================================
# SECTION 1: POSITIVE TESTS (BEHAVIORS WE EXPECT TO PASS)
# These tests confirm that Uproot behaves correctly and predictably.
# ==============================================================================


@pytest.mark.parametrize("branch_name", BRANCH_ARCHETYPES)
def test_hard_out_of_bounds_returns_empty(tree, branch_name):
    """
    Test Case 1: The 'completely outside' scenario.
    Hypothesis: Reading a slice far beyond the file's end should return an empty array.
    """
    logger.info(f"\n--- Testing: Hard Out-of-Bounds on branch '{branch_name}' ---")
    total = tree.num_entries
    branch = tree[branch_name]
    start_idx, stop_idx = total + 100, total + 200

    logger.info(f"Total entries in tree: {total}")
    logger.info(f"Requesting slice from index {start_idx} to {stop_idx}...")

    data = branch.array(entry_start=start_idx, entry_stop=stop_idx)

    logger.info(
        f"RESULT: Received an array of type {type(data).__name__} with length {len(data)}."
    )
    assert len(data) == 0
    logger.info("SUCCESS: Assertion `len(data) == 0` passed as expected.")


@pytest.mark.parametrize("branch_name", BRANCH_ARCHETYPES)
def test_clamped_out_of_bounds_truncates(tree, branch_name):
    """
    Test Case 2: The 'overlapping' scenario.
    Hypothesis: Reading a slice that starts inside but ends outside should be silently truncated.
    """
    logger.info(f"\n--- Testing: Clamped Out-of-Bounds on branch '{branch_name}' ---")
    total = tree.num_entries
    branch = tree[branch_name]
    start_idx, stop_idx = total - 5, total + 100

    logger.info(f"Total entries in tree: {total}")
    logger.info(f"Requesting slice from index {start_idx} to {stop_idx}...")

    data = branch.array(entry_start=start_idx, entry_stop=stop_idx)

    logger.info(
        f"RESULT: Received an array of type {type(data).__name__} with length {len(data)}."
    )
    assert len(data) == 5
    logger.info("SUCCESS: Assertion `len(data) == 5` passed as expected.")


def test_massive_negative_index_is_pythonic(tree):
    """
    Test Case 3: The 'large negative index' scenario.
    Hypothesis: A negative start index larger than the file size should act like Python's
                list slicing and return the entire array.
    """
    logger.info("\n--- Testing: Massive Negative Index Slicing ---")
    branch = tree["NJet"]  # Use a simple branch for speed
    total = tree.num_entries
    start_idx = -1_000_000

    logger.info(f"Total entries in tree: {total}")
    logger.info(f"Requesting slice with start_idx = {start_idx}...")

    data = branch.array(entry_start=start_idx)

    logger.info(
        f"RESULT: Received an array of type {type(data).__name__} with length {len(data)}."
    )
    assert len(data) == total
    logger.info(f"SUCCESS: Assertion `len(data) == {total}` passed as expected.")


def test_iterate_out_of_bounds_yields_nothing(tree):
    """
    Test Case 4: The 'iterator' scenario.
    Hypothesis: Iterating over a range entirely outside the file should yield zero batches.
    """
    logger.info("\n--- Testing: Out-of-Bounds with an Iterator ---")
    total = tree.num_entries
    start_idx, stop_idx = total + 100, total + 200

    logger.info(f"Total entries in tree: {total}")
    logger.info(f"Iterating from index {start_idx} to {stop_idx}...")

    iterator = tree.iterate(
        ["NJet"], step_size=10, entry_start=start_idx, entry_stop=stop_idx
    )

    batches = list(iterator)

    logger.info(f"RESULT: Iterator yielded {len(batches)} batches.")
    assert len(batches) == 0
    logger.info("SUCCESS: Assertion `len(batches) == 0` passed as expected.")


# ==============================================================================
# SECTION 2: NEGATIVE TESTS (BEHAVIORS WE EXPECT TO FAIL)
# These tests verify that our validation logic is sound. A test that can't fail
# is not a real test. We use `pytest.raises` to confirm that an intentional
# mistake is correctly caught as an `AssertionError`.
# ==============================================================================


def test_can_detect_failure_in_clamping_logic(tree):
    """
    META-TEST: This test PASSES if our assertion logic correctly FAILS.
    Hypothesis: If the clamping logic were broken and returned the wrong number of
                entries, our test suite should be able to detect it.
    """
    logger.info("\n--- META-TEST: Verifying failure detection in clamping test ---")
    total = tree.num_entries

    # This is the correct action that should return 5 entries
    data = tree["Jet_Px"].array(entry_start=total - 5, entry_stop=total + 100)
    logger.info(
        f"Action: Performed a clamped read. Correct length should be 5, actual is {len(data)}."
    )

    # This is the test for the test itself
    with pytest.raises(AssertionError):
        logger.info(
            "Intentionally making a WRONG assertion to ensure pytest catches it."
        )
        # This should return 5, so asserting it is 0 should raise an AssertionError.
        # pytest.raises will catch this error, and therefore this test will PASS.
        assert len(data) == 0

    logger.info(
        "SUCCESS: `pytest.raises(AssertionError)` correctly caught the intentional mistake."
    )

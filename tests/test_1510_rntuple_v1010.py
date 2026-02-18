# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
Tests for RNTuple version 1.0.1.0 support (issue #1510).

RNTuple v1.0.1.0 added linked attribute sets to the footer structure.
This test ensures that both v1.0.1.0 files (with attributes) and v1.0.0.0
files (without attributes) can be read correctly.
"""

import skhep_testdata

import uproot


def test_rntuple_v1010_footer_parsing():
    """
    Test that RNTuple v1.0.1.0 files can be read.

    The v1.0.1.0 format added a "linked attribute sets" list frame to the
    footer, appearing between cluster_group_records and the checksum.
    Without proper support, this causes an AssertionError due to checksum
    mismatch.
    """
    filename = skhep_testdata.data_path("ntpl001_staff_rntuple_v1-0-1-0.root")

    with uproot.open(filename) as f:
        obj = f["Staff"]

        # Verify version
        assert obj.member("fVersionEpoch") == 1
        assert obj.member("fVersionMajor") == 0
        assert obj.member("fVersionMinor") == 1
        assert obj.member("fVersionPatch") == 0

        # Access footer - should not raise AssertionError
        footer = obj.footer

        # Verify footer structure includes linked_attribute_sets
        assert hasattr(footer, "linked_attribute_sets")
        assert isinstance(footer.linked_attribute_sets, list)

        # Verify checksum validation passes
        import xxhash

        chunk = obj._footer_chunk
        computed = xxhash.xxh3_64_intdigest(chunk.raw_data[:-8])
        stored = footer.checksum
        assert (
            computed == stored
        ), f"Footer checksum mismatch: {computed:016x} != {stored:016x}"

        # Verify we can retrieve keys (original failing operation)
        keys = obj.keys()
        assert len(keys) > 0


def test_rntuple_v1000_backward_compatibility():
    """
    Verify backward compatibility: v1.0.0.0 files still work with the fix.

    The fix adds a linked_attribute_sets reader that should gracefully handle
    older files by reading an empty list frame (num_items=0).
    """
    filename = skhep_testdata.data_path("ntpl001_staff_rntuple_v1-0-0-0.root")

    with uproot.open(filename) as f:
        obj = f["Staff"]

        # Verify version
        assert obj.member("fVersionEpoch") == 1
        assert obj.member("fVersionMajor") == 0
        assert obj.member("fVersionMinor") == 0
        assert obj.member("fVersionPatch") == 0

        # Access footer - should still work
        footer = obj.footer

        # Footer should have linked_attribute_sets field (but empty)
        assert hasattr(footer, "linked_attribute_sets")
        assert isinstance(footer.linked_attribute_sets, list)
        assert len(footer.linked_attribute_sets) == 0

        # Verify checksum validation still passes
        import xxhash

        chunk = obj._footer_chunk
        computed = xxhash.xxh3_64_intdigest(chunk.raw_data[:-8])
        stored = footer.checksum
        assert (
            computed == stored
        ), f"Footer checksum mismatch: {computed:016x} != {stored:016x}"

        # Verify normal operations still work
        keys = obj.keys()
        assert len(keys) > 0

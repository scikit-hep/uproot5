# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
Regression tests for PR #1650: RNTuple footer leak on extend, broken
WritableNTuple.compression, directory __delitem__ KeyError.
"""

from __future__ import annotations

import numpy as np
import pytest

import uproot
import uproot.exceptions


def test_rntuple_extend_no_freesegments_leak(tmp_path):
    """
    Every call to WritableNTuple.extend() must fully release the old footer
    region (key header + payload), not just the key header.

    Before the fix, NTuple.extend released only old_footer_key.allocation
    (= num_bytes, ~70 bytes) instead of num_bytes + compressed_bytes.
    Repeated extends left the footer payload stranded in FreeSegments,
    causing permanent file bloat.

    We verify the fix by checking that the file size after N extends is
    within a reasonable bound and that the file is still readable.
    """
    fname = tmp_path / "rntuple_extend_leak.root"
    f = uproot.recreate(fname)
    f["t"] = {"x": np.array([1.0, 2.0, 3.0], dtype="f4")}
    ntuple = f["t"]
    n_extends = 10
    for i in range(n_extends):
        ntuple.extend({"x": np.array([float(i), float(i + 1)], dtype="f4")})
    f.close()

    # File must be readable and contain all entries
    with uproot.open(fname) as rf:
        arr = rf["t"]["x"].array()
        assert len(arr) == 3 + n_extends * 2

    # Sanity check: file should not grow unboundedly.
    # Without the fix, each extend leaked ~footer_size bytes that could never
    # be reused.  A simple bound: file should be under 20 kB for this tiny dataset.
    size = fname.stat().st_size
    assert size < 20_000, f"File unexpectedly large ({size} bytes), possible leak"


def test_writable_ntuple_compression_getter(tmp_path):
    """
    WritableNTuple.compression must return the file-level compression object
    without raising AttributeError (which it did before the fix because it
    referenced self._cascading._branch_data which does not exist on NTuple).
    """
    fname = tmp_path / "ntuple_compression.root"
    f = uproot.recreate(fname, compression=uproot.ZLIB(1))
    f["t"] = {"x": np.array([1.0], dtype="f4")}
    ntuple = f["t"]
    comp = ntuple.compression
    assert comp is not None
    assert isinstance(comp, uproot.compression.Compression)
    f.close()


def test_writable_ntuple_compression_setter_raises(tmp_path):
    """
    WritableNTuple.compression setter must raise NotImplementedError (not
    AttributeError) since per-RNTuple compression is not yet supported.
    """
    fname = tmp_path / "ntuple_compression_set.root"
    f = uproot.recreate(fname)
    f["t"] = {"x": np.array([1.0], dtype="f4")}
    ntuple = f["t"]
    with pytest.raises(NotImplementedError):
        ntuple.compression = uproot.ZLIB(1)
    f.close()


def test_writable_directory_delitem_nonexistent_raises_key_error(tmp_path):
    """
    Deleting a nonexistent key from a WritableDirectory must raise a KeyError
    subclass (uproot.exceptions.KeyInFileError), not AttributeError.

    Before the fix, _del() called key.seek_location without checking whether
    get_key() returned None, resulting in:
        AttributeError: 'NoneType' object has no attribute 'seek_location'
    """
    fname = tmp_path / "del_nonexistent.root"
    f = uproot.recreate(fname)
    f["t"] = {"x": np.array([1, 2, 3], dtype="i4")}
    with pytest.raises(KeyError):
        del f["nonexistent_key"]
    f.close()


def test_writable_directory_delitem_nonexistent_is_key_in_file_error(tmp_path):
    """
    The error raised for deleting a nonexistent key should specifically be
    uproot.exceptions.KeyInFileError (a KeyError subclass), matching the
    behaviour of __getitem__ (_get method).
    """
    fname = tmp_path / "del_nonexistent2.root"
    f = uproot.recreate(fname)
    f["t"] = {"x": np.array([1], dtype="i4")}
    with pytest.raises(uproot.exceptions.KeyInFileError):
        del f["does_not_exist"]
    f.close()

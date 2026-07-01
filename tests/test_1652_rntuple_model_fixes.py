# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import numpy as np
import pytest

import uproot
from uproot.models.RNTuple import (
    FieldClusterMetadata,
    Model_ROOT_3a3a_RNTuple,
    _extract_bits,
)

skhep_testdata = pytest.importorskip("skhep_testdata")


def _byte_split(values):
    """Byte-split an array the way RNTuple stores split-encoded columns."""
    num = len(values)
    raw = values.view(np.uint8).reshape(num, values.dtype.itemsize)
    out = np.empty(num * values.dtype.itemsize, np.uint8)
    for b in range(values.dtype.itemsize):
        out[b * num : (b + 1) * num] = raw[:, b]
    return out


def _make_metadata(**kwargs):
    defaults = dict(
        ncol=0,
        dtype_byte=0x18,
        dtype_str="float32",
        dtype=np.dtype("float32"),
        dtype_toread=np.dtype("float32"),
        dtype_result=np.dtype("float32"),
        split=False,
        zigzag=False,
        delta=False,
        isbit=False,
        nbits=32,
    )
    defaults.update(kwargs)
    return FieldClusterMetadata(**defaults)


def _deserialize(destination, field_metadata):
    obj = Model_ROOT_3a3a_RNTuple.__new__(Model_ROOT_3a3a_RNTuple)
    Model_ROOT_3a3a_RNTuple.deserialize_page_decompressed_buffer(
        obj, destination, field_metadata
    )


def test_split_column_into_wider_destination():
    # Regression for split-column sizing when the raw dtype (float32) is
    # narrower than dtype_result (float64), as happens with multiple column
    # representations. The raw data only occupies the first num*4 bytes of the
    # float64-wide destination; the split unshuffle must not pull in the
    # garbage tail bytes (which previously caused a broadcast failure).
    values = np.array([1.5, -2.25, 3.0, 42.0], dtype=np.float32)
    num = len(values)
    destination = np.zeros(num, dtype=np.float64)
    destination.view(np.uint8)[: num * 4] = _byte_split(values)

    field_metadata = _make_metadata(
        split=True,
        dtype=np.dtype("float32"),
        dtype_result=np.dtype("float64"),
    )
    _deserialize(destination, field_metadata)

    np.testing.assert_allclose(destination.view(np.float32)[:num], values)


def test_split_column_same_dtype_roundtrip():
    # The common split case where dtype == dtype_result must be unchanged.
    values = np.array([1.5, -2.25, 3.0, 42.0, -7.5], dtype=np.float32)
    num = len(values)
    destination = np.zeros(num, dtype=np.float32)
    destination.view(np.uint8)[:] = _byte_split(values)

    field_metadata = _make_metadata(split=True)
    _deserialize(destination, field_metadata)

    np.testing.assert_allclose(destination, values)


def test_plain_column_is_left_untouched():
    # Plain columns need no transform; the data is already in place and should
    # be returned without an extra round-trip copy.
    values = np.array([1, 2, 3, 4], dtype=np.int64)
    destination = values.copy()

    field_metadata = _make_metadata(
        dtype_byte=0x09,
        dtype_str="int64",
        dtype=np.dtype("int64"),
        dtype_toread=np.dtype("int64"),
        dtype_result=np.dtype("int64"),
        nbits=64,
    )
    _deserialize(destination, field_metadata)

    np.testing.assert_array_equal(destination, values)


def test_extract_bits_roundtrip():
    # _extract_bits should faithfully unpack nbits-wide values; this also guards
    # against the removed dead allocation reintroducing a bug.
    nbits = 12
    original = np.array([0, 1, 4095, 1234, 2048], dtype=np.uint32)
    n = len(original)
    total_bits = n * nbits
    packed_words = int(np.ceil(total_bits / 32))
    bitstream = np.zeros(packed_words * 32, dtype=np.uint8)
    for i, v in enumerate(original):
        for b in range(nbits):
            if (v >> b) & 1:
                bit_index = i * nbits + b
                bitstream[bit_index] = 1
    packed = np.packbits(bitstream, bitorder="little").view(np.uint32)

    result = _extract_bits(packed, nbits)
    np.testing.assert_array_equal(result[:n], original)


def test_custom_floats_roundtrip_still_passes():
    # Safety net for the real32trunc / real32quant deserialization paths.
    filename = skhep_testdata.data_path("test_float_types_rntuple_v1-0-0-0.root")
    with uproot.open(filename) as f:
        arrays = f["ntuple"].arrays()
        assert len(arrays) > 0


def test_split_zigzag_ints_roundtrip_still_passes():
    # Safety net for the split + zigzag integer deserialization path.
    filename = skhep_testdata.data_path("test_int_5e4_rntuple_v1-0-0-0.root")
    with uproot.open(filename) as f:
        arrays = f["ntuple"].arrays()
        assert len(arrays) > 0


def test_multiple_representations_still_passes():
    # Safety net for the suppressed-columns / multiple-representations path
    # which involves dtype != dtype_result handling.
    filename = skhep_testdata.data_path(
        "test_multiple_representations_rntuple_v1-0-0-0.root"
    )
    with uproot.open(filename) as f:
        arrays = f["ntuple"].arrays()
        assert np.allclose(arrays.real, [1, 2, 3])

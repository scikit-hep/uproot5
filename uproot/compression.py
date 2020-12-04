# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

"""
This module defines an interface to compression algorithms used by ROOT, as well
as functions for compressing and decompressing a :doc:`uproot.source.chunk.Chunk`.
"""

from __future__ import absolute_import

import struct

import numpy

import uproot
import uproot.const


class Compression(object):
    """
    Abstract class for objects that describe compression algorithms and levels.
    """

    def __init__(self, level):
        self.level = level

    def __repr__(self):
        return "{0}({1})".format(type(self).__name__, self._level)

    @classmethod
    def from_code(cls, code):
        """
        Constructs a :doc:`uproot.compression.Compression` from a raw
        ``fCompress`` integer.
        """
        return cls.from_code_pair(code // 100, code % 100)

    @classmethod
    def from_code_pair(cls, algorithm, level):
        """
        Constructs a :doc:`uproot.compression.Compression` from a pair of
        integers representing ``algorithm`` and ``level``.
        """
        if algorithm == 0 or level == 0:
            return None
        elif algorithm in algorithm_codes:
            return algorithm_codes[algorithm](level)
        else:
            raise ValueError(
                "unrecognized compression algorithm code: {0}".format(algorithm)
            )

    @property
    def code(self):
        """
        This :doc:`uproot.compression.Compression` as a raw ``fCompress``
        integer.
        """
        algorithm, level = self.code_pair
        return algorithm * 100 + level

    @property
    def code_pair(self):
        """
        This :doc:`uproot.compression.Compression` as a 2-tuple of integers
        representing algorithm and level.
        """
        for const, cls in algorithm_codes.items():
            if type(self) is cls:
                return const, self._level
        else:
            raise ValueError("unrecognized compression type: {0}".format(type(self)))

    @property
    def level(self):
        """
        The compression level: 0 is uncompressed, 1 is minimally compressed, and
        9 is maximally compressed.
        """
        return self._level

    @level.setter
    def level(self, value):
        if not uproot._util.isint(value):
            raise TypeError("Compression level must be an integer")
        if not 0 <= value <= 9:
            raise ValueError("Compression level must be between 0 and 9 (inclusive)")
        self._level = int(value)


class ZLIB(Compression):
    """
    Args:
        level (int, 0-9): Compression level: 0 is uncompressed, 1 is minimally
            compressed, and 9 is maximally compressed.

    Represents the ZLIB compression algorithm.

    Uproot uses ``zlib`` from the Python standard library.
    """

    @classmethod
    def decompress(cls, data, uncompressed_bytes=None):
        import zlib

        return zlib.decompress(data)


class LZMA(Compression):
    """
    Args:
        level (int, 0-9): Compression level: 0 is uncompressed, 1 is minimally
            compressed, and 9 is maximally compressed.

    Represents the LZMA compression algorithm.

    Uproot uses ``lzma`` from the Python 3 standard library.

    In Python 2, ``backports.lzma`` must be installed.
    """

    @classmethod
    def decompress(cls, data, uncompressed_bytes=None):
        lzma = uproot.extras.lzma()
        return lzma.decompress(data)


class LZ4(Compression):
    """
    Args:
        level (int, 0-9): Compression level: 0 is uncompressed, 1 is minimally
            compressed, and 9 is maximally compressed.

    Represents the LZ4 compression algorithm.

    The ``zl4`` and ``xxhash`` libraries must be installed.
    """

    @classmethod
    def decompress(cls, data, uncompressed_bytes=None):
        lz4_block = uproot.extras.lz4_block()
        if uncompressed_bytes is None:
            raise ValueError(
                "lz4 block decompression requires the number of uncompressed bytes"
            )
        return lz4_block.decompress(data, uncompressed_size=uncompressed_bytes)


class ZSTD(Compression):
    """
    Args:
        level (int, 0-9): Compression level: 0 is uncompressed, 1 is minimally
            compressed, and 9 is maximally compressed.

    Represents the ZSTD compression algorithm.

    The ``zstandard`` library must be installed.
    """

    @classmethod
    def decompress(cls, data, uncompressed_bytes=None):
        zstandard = uproot.extras.zstandard()
        dctx = zstandard.ZstdDecompressor()
        return dctx.decompress(data)


algorithm_codes = {
    uproot.const.kZLIB: ZLIB,
    uproot.const.kLZMA: LZMA,
    uproot.const.kLZ4: LZ4,
    uproot.const.kZSTD: ZSTD,
}


_decompress_header_format = struct.Struct("2sBBBBBBB")
_decompress_checksum_format = struct.Struct(">Q")


def decompress(chunk, cursor, context, compressed_bytes, uncompressed_bytes):
    """
    Args:
        chunk (:doc:`uproot.source.chunk.Chunk`): Buffer of contiguous data
            from the file :doc:`uproot.source.chunk.Source`.
        cursor (:doc:`uproot.source.cursor.Cursor`): Current position in
            that ``chunk``.
        context (dict): Auxiliary data used in deserialization.
        compressed_bytes (int): Number of compressed bytes to decompress.
        uncompressed_bytes (int): Number of uncompressed bytes to expect after
            decompression.

    Decompresses ``compressed_bytes`` of a :doc:`uproot.source.chunk.Chunk`
    of data, starting at the ``cursor``.

    This function parses ROOT's 9-byte compression headers (17 bytes for LZ4
    because it includes a checksum), combining blocks if there are more than
    one, returning the result as a new :doc:`uproot.source.chunk.Chunk`.
    """
    assert compressed_bytes >= 0
    assert uncompressed_bytes >= 0

    start = cursor.copy()
    filled = 0
    num_blocks = 0

    while cursor.displacement(start) < compressed_bytes:
        # https://github.com/root-project/root/blob/master/core/zip/src/RZip.cxx#L217
        # https://github.com/root-project/root/blob/master/core/lzma/src/ZipLZMA.c#L81
        # https://github.com/root-project/root/blob/master/core/lz4/src/ZipLZ4.cxx#L38
        algo, method, c1, c2, c3, u1, u2, u3 = cursor.fields(
            chunk, _decompress_header_format, context
        )
        block_compressed_bytes = c1 + (c2 << 8) + (c3 << 16)
        block_uncompressed_bytes = u1 + (u2 << 8) + (u3 << 16)

        if algo == b"ZL":
            cls = ZLIB
            data = cursor.bytes(chunk, block_compressed_bytes, context)

        elif algo == b"XZ":
            cls = LZMA
            data = cursor.bytes(chunk, block_compressed_bytes, context)

        elif algo == b"L4":
            cls = LZ4
            block_compressed_bytes -= 8
            expected_checksum = cursor.field(
                chunk, _decompress_checksum_format, context
            )
            data = cursor.bytes(chunk, block_compressed_bytes, context)

            xxhash = uproot.extras.xxhash()
            computed_checksum = xxhash.xxh64(data).intdigest()
            if computed_checksum != expected_checksum:
                raise ValueError(
                    """computed checksum {0} didn't match expected checksum {1}
in file {2}""".format(
                        computed_checksum, expected_checksum, chunk.source.file_path
                    )
                )

        elif algo == b"ZS":
            cls = ZSTD
            data = cursor.bytes(chunk, block_compressed_bytes, context)

        elif algo == b"CS":
            raise ValueError(
                """unsupported compression algorithm: {0} (according to """
                """ROOT comments, it hasn't been used in 20 years!
in file {1}""".format(
                    algo, chunk.source.file_path
                )
            )

        else:
            raise ValueError(
                """unrecognized compression algorithm: {0}
in file {1}""".format(
                    algo, chunk.source.file_path
                )
            )

        uncompressed_bytestring = cls.decompress(data, block_uncompressed_bytes)

        if len(uncompressed_bytestring) != block_uncompressed_bytes:
            raise ValueError(
                """after successfully decompressing {0} blocks, a block of """
                """compressed size {1} decompressed to {2} bytes, but the """
                """block header expects {3} bytes.
in file {4}""".format(
                    num_blocks,
                    block_compressed_bytes,
                    len(uncompressed_bytestring),
                    block_uncompressed_bytes,
                    chunk.source.file_path,
                )
            )

        uncompressed_array = numpy.frombuffer(
            uncompressed_bytestring, dtype=uproot.source.chunk.Chunk._dtype
        )

        if num_blocks == 0:
            if uncompressed_bytes == block_uncompressed_bytes:
                # the usual case: only one block
                output = uncompressed_array
                break

            else:
                output = numpy.empty(
                    uncompressed_bytes, dtype=uproot.source.chunk.Chunk._dtype
                )

        output[filled : filled + block_uncompressed_bytes] = uncompressed_array
        filled += block_uncompressed_bytes
        num_blocks += 1

    return uproot.source.chunk.Chunk.wrap(chunk.source, output)

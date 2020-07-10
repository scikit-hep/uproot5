# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import struct

import numpy

import uproot4.source.chunk
import uproot4.const
import uproot4._util
import uproot4.extras


class Compression(object):
    @classmethod
    def from_code_pair(cls, algorithm, level):
        if algorithm == 0 or level == 0:
            return None
        elif algorithm in algorithm_codes:
            return algorithm_codes[algorithm](level)
        else:
            raise ValueError(
                "unrecognized compression algorithm code: {0}".format(algorithm)
            )

    @classmethod
    def from_code(cls, code):
        return cls.from_code_pair(code // 100, code % 100)

    def __init__(self, level):
        self.level = level

    @property
    def level(self):
        return self._level

    @level.setter
    def level(self, value):
        if not uproot4._util.isint(value):
            raise TypeError("Compression level must be an integer")
        if not 0 <= value <= 9:
            raise ValueError("Compression level must be between 0 and 9 (inclusive)")
        self._level = int(value)

    def __repr__(self):
        return "{0}({1})".format(type(self).__name__, self._level)

    @property
    def code_pair(self):
        for const, cls in algorithm_codes.items():
            if type(self) is cls:
                return const, self._level
        else:
            raise ValueError("unrecognized compression type: {0}".format(type(self)))

    @property
    def code(self):
        algorithm, level = self.code_pair
        return algorithm * 100 + level


class ZLIB(Compression):
    @classmethod
    def decompress(cls, data, uncompressed_bytes=None):
        import zlib

        return zlib.decompress(data)


class LZMA(Compression):
    @classmethod
    def decompress(cls, data, uncompressed_bytes=None):
        lzma = uproot4.extras.lzma()
        return lzma.decompress(data)


class LZ4(Compression):
    @classmethod
    def decompress(cls, data, uncompressed_bytes=None):
        lz4_block = uproot4.extras.lz4_block()
        if uncompressed_bytes is None:
            raise ValueError(
                "lz4 block decompression requires the number of uncompressed bytes"
            )
        return lz4_block.decompress(data, uncompressed_size=uncompressed_bytes)


class ZSTD(Compression):
    @classmethod
    def decompress(cls, data, uncompressed_bytes=None):
        zstandard = uproot4.extras.zstandard()
        dctx = zstandard.ZstdDecompressor()
        return dctx.decompress(data)


algorithm_codes = {
    uproot4.const.kZLIB: ZLIB,
    uproot4.const.kLZMA: LZMA,
    uproot4.const.kLZ4: LZ4,
    uproot4.const.kZSTD: ZSTD,
}


_decompress_header_format = struct.Struct("2sBBBBBBB")
_decompress_checksum_format = struct.Struct(">Q")


def decompress(chunk, cursor, context, compressed_bytes, uncompressed_bytes):
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

            xxhash = uproot4.extras.xxhash()
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
            uncompressed_bytestring, dtype=uproot4.source.chunk.Chunk._dtype
        )

        if num_blocks == 0:
            if uncompressed_bytes == block_uncompressed_bytes:
                # the usual case: only one block
                output = uncompressed_array
                break

            else:
                output = numpy.empty(
                    uncompressed_bytes, dtype=uproot4.source.chunk.Chunk._dtype
                )

        output[filled : filled + block_uncompressed_bytes] = uncompressed_array
        filled += block_uncompressed_bytes
        num_blocks += 1

    return uproot4.source.chunk.Chunk.wrap(chunk.source, output)

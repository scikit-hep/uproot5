# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
This module defines an interface to compression algorithms used by ROOT, as well
as functions for compressing and decompressing a :doc:`uproot.source.chunk.Chunk`.
"""


import struct
import threading

import numpy

import uproot
import uproot.const


class Compression:
    """
    Abstract class for objects that describe compression algorithms and levels.
    """

    def __init__(self, level):
        self.level = level

    def __repr__(self):
        return f"{type(self).__name__}({self._level})"

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
            raise ValueError(f"unrecognized compression algorithm code: {algorithm}")

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
            raise ValueError(f"unrecognized compression type: {type(self)}")

    @property
    def level(self):
        """
        The compression level: 0 is uncompressed, 1 is minimally compressed, and
        higher levels increase compression. (See specific algorithms for maximum values.)
        """
        return self._level

    def __eq__(self, other):
        if isinstance(other, Compression):
            return self.name == other.name and self.level == other.level
        else:
            return False


class _DecompressZLIB:
    name = "ZLIB"
    _2byte = b"ZL"
    _method = b"\x08"

    def decompress(self, data, uncompressed_bytes=None):
        import zlib

        return zlib.decompress(data)


class ZLIB(Compression, _DecompressZLIB):
    """
    Args:
        level (int, 0-9): Compression level: 0 is uncompressed, 1 is minimally
            compressed, and 9 is maximally compressed.

    Represents the ZLIB compression algorithm.

    Uproot uses ``zlib`` from the Python standard library.
    """

    def __init__(self, level):
        _DecompressZLIB.__init__(self)
        Compression.__init__(self, level)

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

    def compress(self, data):
        import zlib

        return zlib.compress(data, self._level)


class _DecompressLZMA:
    name = "LZMA"
    _2byte = b"XZ"
    _method = b"\x00"

    def decompress(self, data, uncompressed_bytes=None):
        lzma = uproot.extras.lzma()
        return lzma.decompress(data)


class LZMA(Compression, _DecompressLZMA):
    """
    Args:
        level (int, 0-9): Compression level: 0 is uncompressed, 1 is minimally
            compressed, and 9 is maximally compressed.

    Represents the LZMA compression algorithm.

    Uproot uses ``lzma`` from the Python 3 standard library.

    In Python 2, ``backports.lzma`` must be installed.
    """

    def __init__(self, level):
        _DecompressLZMA.__init__(self)
        Compression.__init__(self, level)

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

    def compress(self, data):
        lzma = uproot.extras.lzma()
        return lzma.compress(data, preset=self._level)


class _DecompressLZ4:
    name = "LZ4"
    _2byte = b"L4"
    _method = b"\x01"

    def decompress(self, data, uncompressed_bytes=None):
        lz4_block = uproot.extras.lz4_block()
        if uncompressed_bytes is None:
            raise ValueError(
                "lz4 block decompression requires the number of uncompressed bytes"
            )
        return lz4_block.decompress(data, uncompressed_size=uncompressed_bytes)


class LZ4(Compression, _DecompressLZ4):
    """
    Args:
        level (int, 0-9): Compression level: 0 is uncompressed, 1 is minimally
            compressed, and 9 is maximally compressed.

    Represents the LZ4 compression algorithm.

    The ``zl4`` and ``xxhash`` libraries must be installed.
    """

    def __init__(self, level):
        _DecompressLZ4.__init__(self)
        Compression.__init__(self, level)

    @property
    def level(self):
        """
        The compression level: 0 is uncompressed, 1 is minimally compressed, and
        12 is maximally compressed.
        """
        return self._level

    @level.setter
    def level(self, value):
        if not uproot._util.isint(value):
            raise TypeError("Compression level must be an integer")
        if not 0 <= value <= 12:
            raise ValueError("Compression level must be between 0 and 12 (inclusive)")
        self._level = int(value)

    def compress(self, data):
        lz4_block = uproot.extras.lz4_block()
        return lz4_block.compress(data, compression=self._level, store_size=False)


class _DecompressZSTD:
    name = "ZSTD"
    _2byte = b"ZS"
    _method = b"\x01"

    def __init__(self):
        # ZstdDecompressor resource is not thread-safe
        self._decompressor = threading.local()

    @property
    def decompressor(self):
        if not hasattr(self._decompressor, "obj"):
            zstandard = uproot.extras.zstandard()
            self._decompressor.obj = zstandard.ZstdDecompressor()
        return self._decompressor.obj

    def decompress(self, data, uncompressed_bytes=None):
        return self.decompressor.decompress(data)


class ZSTD(Compression, _DecompressZSTD):
    """
    Args:
        level (int, 0-9): Compression level: 0 is uncompressed, 1 is minimally
            compressed, and 9 is maximally compressed.

    Represents the ZSTD compression algorithm.

    The ``zstandard`` library must be installed.
    """

    def __init__(self, level):
        _DecompressZSTD.__init__(self)
        Compression.__init__(self, level)
        self._compressor = None

    @property
    def level(self):
        """
        The compression level: 0 is uncompressed, 1 is minimally compressed, and
        22 is maximally compressed.
        """
        return self._level

    @level.setter
    def level(self, value):
        if not uproot._util.isint(value):
            raise TypeError("Compression level must be an integer")
        if not 0 <= value <= 22:
            raise ValueError("Compression level must be between 0 and 22 (inclusive)")
        self._level = int(value)

    @property
    def compressor(self):
        if self._compressor is None:
            zstandard = uproot.extras.zstandard()
            self._compressor = zstandard.ZstdCompressor(level=self._level)
        return self._compressor

    def compress(self, data):
        return self.compressor.compress(data)


algorithm_codes = {
    uproot.const.kZLIB: ZLIB,
    uproot.const.kLZMA: LZMA,
    uproot.const.kLZ4: LZ4,
    uproot.const.kZSTD: ZSTD,
}

_decompress_ZLIB = _DecompressZLIB()
_decompress_LZMA = _DecompressLZMA()
_decompress_LZ4 = _DecompressLZ4()
_decompress_ZSTD = _DecompressZSTD()

_decompress_header_format = struct.Struct("2sBBBBBBB")
_decompress_checksum_format = struct.Struct(">Q")


def decompress(
    chunk, cursor, context, compressed_bytes, uncompressed_bytes, block_info=None
):
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
        block_info (None or empty list): List to fill with
            ``(compression type class, num compressed bytes, num uncompressed bytes)``
            observed in each compressed block.

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
        decompress.hook_before_block(
            chunk=chunk,
            cursor=cursor,
            context=context,
            compressed_bytes=compressed_bytes,
            uncompressed_bytes=uncompressed_bytes,
            start=start,
            filled=filled,
            num_blocks=num_blocks,
        )

        # https://github.com/root-project/root/blob/master/core/zip/src/RZip.cxx#L217
        # https://github.com/root-project/root/blob/master/core/lzma/src/ZipLZMA.c#L81
        # https://github.com/root-project/root/blob/master/core/lz4/src/ZipLZ4.cxx#L38
        algo, method, c1, c2, c3, u1, u2, u3 = cursor.fields(
            chunk, _decompress_header_format, context
        )
        block_compressed_bytes = c1 + (c2 << 8) + (c3 << 16)
        block_uncompressed_bytes = u1 + (u2 << 8) + (u3 << 16)

        if algo == _decompress_ZLIB._2byte:
            decompressor = _decompress_ZLIB
            data = cursor.bytes(chunk, block_compressed_bytes, context)

        elif algo == _decompress_LZMA._2byte:
            decompressor = _decompress_LZMA
            data = cursor.bytes(chunk, block_compressed_bytes, context)

        elif algo == _decompress_LZ4._2byte:
            decompressor = _decompress_LZ4
            block_compressed_bytes -= 8
            expected_checksum = cursor.field(
                chunk, _decompress_checksum_format, context
            )
            data = cursor.bytes(chunk, block_compressed_bytes, context)

            xxhash = uproot.extras.xxhash()
            computed_checksum = xxhash.xxh64(data).intdigest()
            if computed_checksum != expected_checksum:
                raise ValueError(
                    """computed checksum {} didn't match expected checksum {}
in file {}""".format(
                        computed_checksum, expected_checksum, chunk.source.file_path
                    )
                )

        elif algo == _decompress_ZSTD._2byte:
            decompressor = _decompress_ZSTD
            data = cursor.bytes(chunk, block_compressed_bytes, context)

        elif algo == b"CS":
            raise ValueError(
                f"""unsupported compression algorithm: {algo} (according to """
                f"""ROOT comments, it hasn't been used in 20 years!
in file {chunk.source.file_path}"""
            )

        else:
            raise ValueError(
                f"""unrecognized compression algorithm: {algo}
in file {chunk.source.file_path}"""
            )

        if block_info is not None:
            block_info.append(
                (decompressor.name, block_compressed_bytes, block_uncompressed_bytes)
            )

        uncompressed_bytestring = decompressor.decompress(
            data, block_uncompressed_bytes
        )

        if len(uncompressed_bytestring) != block_uncompressed_bytes:
            raise ValueError(
                """after successfully decompressing {} blocks, a block of """
                """compressed size {} decompressed to {} bytes, but the """
                """block header expects {} bytes.
in file {}""".format(
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

        decompress.hook_after_block(
            chunk=chunk,
            cursor=cursor,
            context=context,
            compressed_bytes=compressed_bytes,
            uncompressed_bytes=uncompressed_bytes,
            start=start,
            filled=filled,
            num_blocks=num_blocks,
            output=output,
        )

    return uproot.source.chunk.Chunk.wrap(chunk.source, output)


def hook_before_block(**kwargs):
    pass


def hook_after_block(**kwargs):
    pass


decompress.hook_before_block = hook_before_block
decompress.hook_after_block = hook_after_block

_3BYTE_MAX = 2**24 - 1
_4byte = struct.Struct("<I")  # compressed sizes are 3-byte little endian!


def compress(data, compression):
    """
    Args:
        data (bytes, memoryview, or NumPy array): Data to compress.
        compression (:doc:`uproot.compression.Compression`): Algorithm and level
            to use in compressing the data.

    Compress ``data`` using ``compression`` and return a bytes object (or the
    original object in some circumstances).

    This function generates ROOT's 9-byte compression headers (17 bytes for LZ4
    because it includes a checksum), with multiple blocks for each 2**24 - 1 bytes
    of input. The return value has all blocks concatenated into a single bytes
    object, and the splitting into multiple blocks is necessary because the compression
    headers can only specify uncompressed and compressed sizes up to 2**24 - 1.

    If the compression level (``compression.level``) is zero or the compressed
    output would be larger than the input, the input is returned instead, in whatever
    format (bytes, memoryview, or NumPy array) it was provided.
    """
    if compression is None or compression.level == 0:
        return data

    out = []
    next = data

    while len(next) > 0:
        block, next = next[:_3BYTE_MAX], next[_3BYTE_MAX:]

        compressed = compression.compress(block)
        len_compressed = len(compressed)

        if isinstance(compression, LZ4):
            xxhash = uproot.extras.xxhash()
            computed_checksum = xxhash.xxh64(compressed).intdigest()
            checksum = _decompress_checksum_format.pack(computed_checksum)
            len_compressed += 8
        else:
            checksum = b""

        uncompressed_size = _4byte.pack(len(block))[:-1]
        compressed_size = _4byte.pack(len_compressed)[:-1]

        out.append(
            compression._2byte
            + compression._method
            + compressed_size
            + uncompressed_size
            + checksum
        )

        out.append(compressed)

    out = b"".join(out)

    if len(out) < len(data):
        return out
    else:
        return data

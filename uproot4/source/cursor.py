# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

"""
Defines the Cursor, which is a universal pointer/interpreter at point of data
in a ROOT file.
"""

from __future__ import absolute_import

import sys
import struct

import numpy

import uproot4
import uproot4.deserialization


_printable_characters = (
    "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLM"
    "NOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
)
_raw_double32 = struct.Struct(">f")
_raw_float16 = struct.Struct(">BH")


class Cursor(object):
    """
    Represents a position in a ROOT file, which may be held for later reference
    or advanced while interpreting bytes from Chunks.
    """

    __slots__ = ["_index", "_origin", "_refs"]

    def __init__(self, index, origin=0, refs=None):
        """
        Args:
            index (int): Global position in the ROOT file.
            origin (int): Placeholder that is sometimes useful for arithmetic.
            refs (None or dict): References to data already read, for
                `read_object_any`.
        """
        self._index = index
        self._origin = origin
        self._refs = refs

    def __repr__(self):
        if self._origin == 0:
            o = ""
        else:
            o = ", origin={0}".format(self._origin)

        if self._refs is None or len(self._refs) == 0:
            r = ""
        elif self._refs is None or len(self._refs) < 3:
            r = ", {0} refs: {1}".format(
                len(self._refs), ", ".join(str(x) for x in self._refs)
            )
        else:
            r = ", {0} refs: {1}...".format(
                len(self._refs), ", ".join(str(x) for x in list(self._refs)[:3])
            )

        return "Cursor({0}{1}{2})".format(self._index, o, r)

    @property
    def index(self):
        """
        Global position in the ROOT file.
        """
        return self._index

    @property
    def origin(self):
        """
        Placeholder that is sometimes useful for arithmetic.
        """
        return self._origin

    @property
    def refs(self):
        """
        References to data already read, for `read_object_any`.
        """
        if self._refs is None:
            self._refs = {}
        return self._refs

    def displacement(self, other=None):
        """
        The number of bytes between this Cursor and its origin (if None)
        or the other Cursor (if provided).

        If the displacement is positive, this Cursor is later in the file
        than the origin/other; if negative, it is earlier.
        """
        if other is None:
            return self._index - self._origin
        else:
            return self._index - other._index

    def copy(self, link_refs=True):
        """
        Returns a copy of this Cursor. If `link_refs` is True, any `refs` will
        be *referenced*, rather than *copied*.
        """
        if link_refs or self._refs is None:
            return Cursor(self._index, origin=self._origin, refs=self._refs)
        else:
            return Cursor(self._index, origin=self._origin, refs=dict(self._refs))

    def move_to(self, index):
        """
        Move the cursor to a specified byte position.
        """
        self._index = index

    def skip(self, num_bytes):
        """
        Move the index forward `num_bytes`.
        """
        self._index += num_bytes

    def skip_after(self, obj):
        """
        Move the index after an object with a starting `obj.cursor` and an
        `obj.num_bytes`.
        """
        start_cursor = getattr(obj, "cursor", None)
        num_bytes = getattr(obj, "num_bytes", None)
        if (
            start_cursor is None
            or not isinstance(start_cursor, Cursor)
            or num_bytes is None
        ):
            raise TypeError(
                "Cursor.skip_after can only be used on an object with a "
                "`cursor` and `num_bytes`, not {0}".format(type(obj))
            )
        self._index = start_cursor.index + num_bytes

    def skip_over(self, chunk, context):
        """
        Move the index after serialized data for an object with
        numbytes_version.
        """
        num_bytes, version = uproot4.deserialization.numbytes_version(
            chunk, self, context, move=False
        )
        if num_bytes is None:
            raise TypeError(
                "Cursor.skip_over can only be used on an object with non-null "
                "`num_bytes`"
            )
        self._index += num_bytes

    def fields(self, chunk, format, context, move=True):
        """
        Interpret data at this index of the Chunk with a `struct.Struct`
        format. Returns a tuple (length determined by `format`).

        If `move` is False, only peek: don't update the index.
        """
        start = self._index
        stop = start + format.size
        if move:
            self._index = stop
        return format.unpack(chunk.get(start, stop, self, context))

    def field(self, chunk, format, context, move=True):
        """
        Interpret data at this index of the Chunk with a `struct.Struct`
        format, returning a single item instead of a tuple (the first).

        If `move` is False, only peek: don't update the index.
        """
        start = self._index
        stop = start + format.size
        if move:
            self._index = stop
        return format.unpack(chunk.get(start, stop, self, context))[0]

    def double32(self, chunk, context, move=True):
        # https://github.com/root-project/root/blob/e87a6311278f859ca749b491af4e9a2caed39161/io/io/src/TBufferFile.cxx#L448-L464
        start = self._index
        stop = start + _raw_double32.size
        if move:
            self._index = stop
        return _raw_double32.unpack(chunk.get(start, stop, self, context))[0]

    def float16(self, chunk, num_bits, context, move=True):
        # https://github.com/root-project/root/blob/e87a6311278f859ca749b491af4e9a2caed39161/io/io/src/TBufferFile.cxx#L432-L442
        # https://github.com/root-project/root/blob/e87a6311278f859ca749b491af4e9a2caed39161/io/io/src/TBufferFile.cxx#L482-L499

        start = self._index
        stop = start + _raw_float16.size
        if move:
            self._index = stop

        exponent, mantissa = _raw_float16.unpack(chunk.get(start, stop, self, context))
        out = numpy.array([exponent], numpy.int32)
        out <<= 23
        out |= (mantissa & ((1 << (num_bits + 1)) - 1)) << (23 - num_bits)
        out = out.view(numpy.float32)

        if (1 << (num_bits + 1) & mantissa) != 0:
            out = -out

        return out.item()

    def bytes(self, chunk, length, context, move=True, copy_if_memmap=False):
        """
        Interpret data at this index of the Chunk as raw bytes with a
        given `length`.

        If `move` is False, only peek: don't update the index.

        If `copy_if_memmap` is True and the chunk is a np.memmap, it is copied.
        """
        start = self._index
        stop = start + length
        if move:
            self._index = stop
        out = chunk.get(start, stop, self, context)
        if copy_if_memmap:
            step = out
            while getattr(step, "base", None) is not None:
                if isinstance(step, numpy.memmap):
                    return numpy.array(out, copy=True)
                step = step.base
        return out

    def array(self, chunk, length, dtype, context, move=True):
        """
        Interpret data at this index of the Chunk as an array with a
        given `length` and `dtype`.

        If `move` is False, only peek: don't update the index.
        """
        start = self._index
        stop = start + length * dtype.itemsize
        if move:
            self._index = stop
        return numpy.frombuffer(chunk.get(start, stop, self, context), dtype=dtype)

    _u1 = numpy.dtype("u1")
    _i4 = numpy.dtype(">i4")

    def bytestring(self, chunk, context, move=True):
        """
        Interpret data at this index of the Chunk as a ROOT bytestring
        (first 1 or 5 bytes indicate size).

        If `move` is False, only peek: don't update the index.
        """
        start = self._index
        stop = start + 1
        length = chunk.get(start, stop, self, context)[0]
        if length == 255:
            start = stop
            stop = start + 4
            length_data = chunk.get(start, stop, self, context)
            length = numpy.frombuffer(length_data, dtype=self._u1).view(self._i4)[0]
        start = stop
        stop = start + length
        if move:
            self._index = stop
        data = chunk.get(start, stop, self, context)
        if hasattr(data, "tobytes"):
            return data.tobytes()
        else:
            return data.tostring()

    def string(self, chunk, context, move=True):
        """
        Interpret data at this index of the Chunk as a Python str
        (first 1 or 5 bytes indicate size).

        The encoding is assumed to be UTF-8, but errors are surrogate-escaped.

        If `move` is False, only peek: don't update the index.
        """
        out = self.bytestring(chunk, context, move=move)
        if uproot4._util.py2:
            return out
        else:
            return out.decode(errors="surrogateescape")

    def bytestring_with_length(self, chunk, context, length, move=True):
        """
        Interpret data at this index of the Chunk as an unprefixed, unsuffixed
        bytestring with a given length.

        If `move` is False, only peek: don't update the index.
        """
        start = self._index
        stop = start + length
        if move:
            self._index = stop
        data = chunk.get(start, stop, self, context)
        if hasattr(data, "tobytes"):
            return data.tobytes()
        else:
            return data.tostring()

    def string_with_length(self, chunk, context, length, move=True):
        """
        Interpret data at this index of the Chunk as an unprefixed, unsuffixed
        Python str with a given length.

        If `move` is False, only peek: don't update the index.
        """
        out = self.bytestring_with_length(chunk, context, length, move=move)
        if uproot4._util.py2:
            return out
        else:
            return out.decode(errors="surrogateescape")

    def classname(self, chunk, context, move=True):
        """
        Interpret data at this index of the Chunk as a ROOT class
        name, which is the only usage of null-terminated strings (rather than
        1-to-5 byte size prefix, as in the `string` method).

        The encoding is assumed to be UTF-8, but errors are surrogate-escaped.

        If `move` is False, only peek: don't update the index.
        """
        remainder = chunk.remainder(self._index, self, context)
        local_stop = 0
        char = None
        while char != 0:
            if local_stop > len(remainder):
                raise OSError(
                    """C-style string has no terminator (null byte) in Chunk {0}:{1}
of file path {2}""".format(
                        self._start, self._stop, self._source.file_path
                    )
                )
            char = remainder[local_stop]
            local_stop += 1

        if move:
            self._index += local_stop

        out = remainder[: local_stop - 1]
        if hasattr(out, "tobytes"):
            out = out.tobytes()
        else:
            out = out.tostring()

        if uproot4._util.py2:
            return out
        else:
            return out.decode(errors="surrogateescape")

    def debug(
        self,
        chunk,
        context={},
        limit_bytes=None,
        dtype=None,
        offset=0,
        stream=sys.stdout,
    ):
        """
        Args:
            chunk (Chunk): Data to interpret.
            limit_bytes (None or int): Maximum number of bytes to view or None
                to see all bytes to the end of the Chunk.
            dtype (None or dtype): If a dtype, additionally interpret the data
                as numbers.
            offset (int): Number of bytes offset for interpreting as a dtype.
            stream: Object with a `write` method for writing the output.

        Peek at data by printing it to the `stream` (usually stdout). The data
        are always presented as decimal bytes and ASCII characters, but may
        also be interpreted as numbers.

        Example output with dtype=">f4" and offset=3.

            --+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+-
            123 123 123  63 140 204 205  64  12 204 205  64  83  51  51  64 140 204 205  64
              {   {   {   ? --- --- ---   @ --- --- ---   @   S   3   3   @ --- --- ---   @
                                    1.1             2.2             3.3             4.4
                --+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+-
                176   0   0  64 211  51  51  64 246 102 102  65  12 204 205  65  30 102 102  66
                --- --- ---   @ ---   3   3   @ ---   f   f   A --- --- ---   A ---   f   f   B
                        5.5             6.6             7.7             8.8             9.9
                --+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+-
                202   0   0  67  74   0   0  67 151 128   0 123 123
                --- --- ---   C   J --- ---   C --- --- ---   {   {
                      101.0           202.0           303.0
        """
        data = chunk.remainder(self._index, self, context)
        if limit_bytes is not None:
            data = data[:limit_bytes]

        if dtype is not None:
            if not isinstance(dtype, numpy.dtype):
                dtype = numpy.dtype(dtype)

            interpreted = [None] * len(data)
            i = offset
            interpreted_length = (
                (len(data) - offset) // dtype.itemsize
            ) * dtype.itemsize
            for x in data[offset : offset + interpreted_length].view(dtype):
                i += dtype.itemsize
                interpreted[i - 1] = x

            formatter = u"{{0:>{0}.{0}s}}".format(dtype.itemsize * 4 - 1)

        for line_start in range(0, int(numpy.ceil(len(data) / 20.0)) * 20, 20):
            line_data = data[line_start : line_start + 20]

            prefix = u""
            if dtype is not None:
                nones = 0
                for x in interpreted[line_start:]:
                    if x is None:
                        nones += 1
                    else:
                        break
                fill = max(0, dtype.itemsize - 1 - nones)
                line_interpreted = [None] * fill + interpreted[
                    line_start : line_start + 20
                ]
                prefix = u"    " * fill
                interpreted_prefix = u"    " * (fill + nones + 1 - dtype.itemsize)

            stream.write(prefix + (u"--+-" * 20) + u"\n")
            stream.write(
                prefix + u" ".join(u"{0:3d}".format(x) for x in line_data) + u"\n"
            )
            stream.write(
                prefix
                + u" ".join(
                    u"{0:>3s}".format(chr(x))
                    if chr(x) in _printable_characters
                    else u"---"
                    for x in line_data
                )
                + u"\n"
            )

            if dtype is not None:
                stream.write(
                    interpreted_prefix
                    + u" ".join(
                        formatter.format(str(x))
                        for x in line_interpreted
                        if x is not None
                    )
                    + u"\n"
                )

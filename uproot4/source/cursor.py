# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import numpy

import uproot4


class Cursor(object):
    __slots__ = ["_index", "_origin", "_refs"]

    def __init__(self, index, origin=0, refs=None):
        self._index = index
        self._origin = origin
        self._refs = refs

    @property
    def index(self):
        return self._index

    @property
    def origin(self):
        return self._origin

    @property
    def refs(self):
        if self._refs is None:
            self._refs = {}
        return self._refs

    def copy(self, link_refs=False):
        if link_refs or self._refs is None:
            return Cursor(self._index, origin=self._origin, refs=self._refs)
        else:
            return Cursor(self._index, origin=self._origin, refs=dict(self._refs))

    def skip(self, num_bytes):
        self._index += num_bytes

    def fields(self, chunk, format, move=True):
        start = self._index
        stop = start + format.size
        if move:
            self._index = stop
        return format.unpack(chunk.get(start, stop))

    def field(self, chunk, format, move=True):
        start = self._index
        stop = start + format.size
        if move:
            self._index = stop
        return format.unpack(chunk.get(start, stop))[0]

    def bytes(self, chunk, length, move=True):
        start = self._index
        stop = start + length
        if move:
            self._index = stop
        return chunk.get(start, stop)

    def array(self, chunk, length, dtype, move=True):
        start = self._index
        stop = start + length * dtype.itemsize
        if move:
            self._index = stop
        return numpy.frombuffer(chunk.get(start, stop), dtype=dtype)

    _u1 = numpy.dtype("u1")
    _i4 = numpy.dtype(">i4")

    def bytestring(self, chunk, move=True):
        start = self._index
        stop = start + 1
        length = chunk.get(start, stop)[0]
        if length == 255:
            start = stop
            stop = start + 4
            length = numpy.frombuffer(chunk.get(start, stop), dtype=self._u1).view(
                self._i4
            )[0]
        start = stop
        stop = start + length
        if move:
            self._index = stop
        return chunk.get(start, stop).tostring()

    def string(self, chunk, move=True):
        out = self.bytestring(chunk, move=move)
        if uproot4._util.py2:
            return out
        else:
            return out.decode(errors="surrogateescape")

    def class_string(self, chunk, move=True):
        remainder = chunk.remainder(self._index)
        local_stop = 0
        char = None
        while char != 0:
            if local_stop > len(remainder):
                raise IOError(
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
        if uproot4._util.py2:
            return out
        else:
            return out.decode(errors="surrogateescape")

    _printable = (
        "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLM"
        "NOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
    )

    def debug(
        self, chunk, limit_bytes=None, dtype=None, offset=0, stream=sys.stdout,
    ):
        data = chunk.remainder(self._index)
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
                    u"{0:>3s}".format(chr(x)) if chr(x) in self._printable else u"---"
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

# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import os
import sys

import numpy

import uproot4
import uproot4.futures
import uproot4._util


class Resource(object):
    pass


class Source(object):
    pass


class FileResource(Resource):
    __slots__ = ["_file_path", "_file"]

    def __init__(self, file_path):
        self._file_path = file_path
        self._file = None

    @property
    def file_path(self):
        return self._file_path

    @property
    def file(self):
        return self._file

    @property
    def ready(self):
        return self._file is not None and not self._file.closed

    def __enter__(self):
        self._file = open(self._file_path, "rb")

    def __exit__(self, exception_type, exception_value, traceback):
        self._file.__exit__(exception_type, exception_value, traceback)

    def get(self, start, stop):
        self._file.seek(start)
        return self._file.read(stop - start)


class FileSource(Source):
    __slots__ = ["_file_path", "_executor"]

    def __init__(self, file_path, num_workers=1):
        if not os.path.exists(file_path):
            raise IOError("file not found: {0}".format(file_path))

        self._file_path = file_path
        if num_workers == 1:
            self._executor = uproot4.futures.ResourceExecutor(FileResource(file_path))
        elif num_workers > 1:
            self._executor = uproot4.futures.ThreadResourceExecutor(
                [FileResource(file_path) for x in range(num_workers)]
            )
        else:
            raise ValueError("num_workers must be at least 1")

    @property
    def file_path(self):
        return self._file_path

    @property
    def executor(self):
        return self._executor

    @property
    def num_workers(self):
        return self._executor.num_workers

    @property
    def ready(self):
        return self._executor.ready

    def __enter__(self):
        self._executor.__enter__()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self._executor.__exit__(exception_type, exception_value, traceback)

    @staticmethod
    def getter(start, stop):
        return lambda resource: resource.get(start, stop)

    def chunks(self, ranges):
        out = []
        for start, stop in ranges:
            out.append(
                Chunk(
                    self, start, stop, self._executor.submit(self.getter(start, stop))
                )
            )
        return out


class Chunk(object):
    __slots__ = ["_source", "_start", "_stop", "_future", "_raw_data"]

    def __init__(self, source, start, stop, future):
        self._source = source
        self._start = start
        self._stop = stop
        self._future = future
        self._raw_data = None

    @property
    def source(self):
        return self._source

    @property
    def start(self):
        return self._start

    @property
    def stop(self):
        return self._stop

    @property
    def future(self):
        return self._future

    def wait(self):
        if self._raw_data is None:
            self._raw_data = self._future.result()
            if len(self._raw_data) != self._stop - self._start:
                raise IOError(
                    """expected Chunk of length {0},
received Chunk of length {1}
for file path {2}""".format(
                        len(self._raw_data),
                        self._stop - self._start,
                        self._source.file_path,
                    )
                )

    @property
    def raw_data(self):
        self.wait()
        return self._raw_data

    def get(self, start, stop):
        local_start = start - self._start
        local_stop = stop - self._start
        if 0 <= local_start and stop <= self._stop:
            self.wait()
            return self.raw_data[local_start:local_stop]
        else:
            raise IOError(
                """attempting to get bytes {0}:{1}
 outside expected range {2}:{3} for this Chunk
of file path {4}""".format(
                    start, stop, self._start, self._stop, self._source.file_path,
                )
            )

    def remainder(self, start):
        local_start = start - self._start
        if 0 <= local_start:
            self.wait()
            return self.raw_data[local_start:]
        else:
            raise IOError(
                """attempting to get byte {0}
 outside expected range {1}:{2} for this Chunk
of file path {3}""".format(
                    start, self._start, self._stop, self._source.file_path,
                )
            )


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

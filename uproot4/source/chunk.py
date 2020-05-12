# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import numpy


class Resource(object):
    @staticmethod
    def getter(start, stop):
        return lambda resource: resource.get(start, stop)


class Source(object):
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

    def chunks(self, ranges):
        chunks = []
        for start, stop in ranges:
            future = self._executor.submit(Resource.getter(start, stop))
            chunks.append(Chunk(self, start, stop, future,))
        return chunks


class Chunk(object):
    __slots__ = ["_source", "_start", "_stop", "_future", "_raw_data"]

    _dtype = numpy.dtype(numpy.uint8)

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
            self._raw_data = numpy.frombuffer(self._future.result(), dtype=self._dtype)
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

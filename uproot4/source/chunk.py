# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import


class Resource(object):
    pass


class Source(object):
    pass


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

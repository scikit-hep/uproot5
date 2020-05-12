# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import os

import numpy

import uproot4
import uproot4.futures


class Resource(object):
    pass


class Source(object):
    pass


class FileResource(Resource):
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

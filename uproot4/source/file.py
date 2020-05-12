# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import os

import uproot4.source.chunk


class FileResource(uproot4.source.chunk.Resource):
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


class FileSource(uproot4.source.chunk.Source):
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
                uproot4.source.chunk.Chunk(
                    self, start, stop, self._executor.submit(self.getter(start, stop))
                )
            )
        return out

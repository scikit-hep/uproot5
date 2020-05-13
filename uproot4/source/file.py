# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import uproot4.source.chunk
import uproot4.source.futures


class FileResource(uproot4.source.chunk.Resource):
    __slots__ = ["_file_path", "_file"]

    def __init__(self, file_path):
        self._file_path = file_path
        self._file = open(self._file_path, "rb")

    @property
    def file_path(self):
        return self._file_path

    @property
    def file(self):
        return self._file

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self._file.__exit__(exception_type, exception_value, traceback)

    def get(self, start, stop):
        self._file.seek(start)
        return self._file.read(stop - start)


class FileSource(uproot4.source.chunk.MultiThreadedSource):
    __slots__ = ["_file_path", "_executor"]

    def __init__(self, file_path, num_workers=1):
        self._file_path = file_path
        if num_workers == 1:
            self._executor = uproot4.source.futures.ResourceExecutor(
                FileResource(file_path)
            )
        elif num_workers > 1:
            self._executor = uproot4.source.futures.ThreadResourceExecutor(
                [FileResource(file_path) for x in range(num_workers)]
            )
        else:
            raise ValueError("num_workers must be at least 1")

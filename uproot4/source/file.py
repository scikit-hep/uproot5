# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

"""
Source and Resource for plain file handle physical I/O.
"""

from __future__ import absolute_import

import os
import os.path

import uproot4.source.chunk
import uproot4.source.futures


class FileResource(uproot4.source.chunk.Resource):
    """
    Resource wrapping a plain file handle.
    """

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

    @property
    def closed(self):
        return self._file.closed

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self._file.__exit__(exception_type, exception_value, traceback)

    def get(self, start, stop):
        self._file.seek(start)
        return self._file.read(stop - start)


class FileSource(uproot4.source.chunk.MultithreadedSource):
    """
    Source managing one synchronous or multiple asynchronous file handles as a
    context manager.
    """

    __slots__ = ["_file_path", "_executor"]

    def __init__(self, file_path, **options):
        """
        Args:
            file_path (str): Path to the file.
            num_workers (int): If 0, one synchronous ResourceExecutor is
                created; if 1 or more, a collection of asynchronous
                ThreadResourceExecutors are created.
        """
        num_workers = options["num_workers"]
        self._num_requests = 0
        self._num_requested_chunks = 0
        self._num_requested_bytes = 0

        self._file_path = file_path
        self._resource = FileResource(file_path)
        self._num_bytes = os.path.getsize(self._file_path)

        if num_workers == 0:
            self._executor = uproot4.source.futures.ResourceExecutor(self._resource)
        else:
            self._executor = uproot4.source.futures.ThreadResourceExecutor(
                [FileResource(file_path) for x in range(num_workers)]
            )

    @property
    def num_bytes(self):
        """
        The number of bytes in the file.
        """
        return self._num_bytes

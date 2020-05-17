# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

"""
Source and Resource for a memory mapped file, which is never multithreaded.
"""

from __future__ import absolute_import

import numpy

import uproot4.source.chunk
import uproot4.source.futures
import uproot4._util


class MemmapSource(uproot4.source.chunk.Source):
    """
    Source for a memory-mapped file.

    Threading is unnecessary because a memory-map is stateless.
    """

    __slots__ = ["_file_path", "_file"]

    _dtype = uproot4.source.chunk.Chunk._dtype

    def __init__(self, file_path):
        """
        Args:
            file_path (str): Path to the file.
        """
        self._file_path = file_path
        self._file = numpy.memmap(self._file_path, dtype=self._dtype, mode="r")

    @property
    def file(self):
        """
        Path to the file.
        """
        return self._file

    def __enter__(self):
        """
        Passes `__enter__` to the memory-map.

        Returns self.
        """
        if hasattr(self._file._mmap, "__enter__"):
            self._file._mmap.__enter__()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """
        Passes `__exit__` to the memory-map or otherwise closes the file.
        """
        if hasattr(self._file._mmap, "__exit__"):
            self._file._mmap.__exit__(exception_type, exception_value, traceback)
        else:
            self._file._mmap.close()

    def chunk(self, start, stop, exact=True):
        """
        Args:
            start (int): The start (inclusive) byte position for the desired
                chunk.
            stop (int): The stop (exclusive) byte position for the desired
                chunk.
            exact (bool): If False, attempts to access bytes beyond the
                end of the Chunk raises a RefineChunk; if True, it raises
                an OSError with an informative message.

        Returns a single Chunk that has already been filled synchronously.
        """
        future = uproot4.source.futures.TrivialFuture(self._file[start:stop])
        return uproot4.source.chunk.Chunk(self, start, stop, future, exact)

    def chunks(self, ranges, exact=True, notifications=None):
        """
        Args:
            ranges (iterable of (int, int)): The start (inclusive) and stop
                (exclusive) byte ranges for each desired chunk.
            exact (bool): If False, attempts to access bytes beyond the
                end of the Chunk raises a RefineChunk; if True, it raises
                an OSError with an informative message.
            notifications (None or Queue): If not None, Chunks will be put
                on this Queue immediately after they are ready.

        Returns a list of Chunks that are already filled with data.
        """
        if uproot4._util.py2:
            try:
                self._file._mmap.tell()
            except ValueError:
                raise OSError("memmap is closed for file {0}".format(self._file_path))

        elif self._file._mmap.closed:
            raise OSError("memmap is closed for file {0}".format(self._file_path))

        chunks = []
        for start, stop in ranges:
            future = uproot4.source.futures.TrivialFuture(self._file[start:stop])
            chunk = uproot4.source.chunk.Chunk(self, start, stop, future, exact)
            if notifications is not None:
                future.add_done_callback(
                    uproot4.source.chunk.Resource.notifier(chunk, notifications)
                )
            chunks.append(chunk)
        return chunks

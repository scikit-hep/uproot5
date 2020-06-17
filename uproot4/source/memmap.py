# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

"""
Source and Resource for a memory mapped file, which is never multithreaded.
"""

from __future__ import absolute_import

import numpy

import uproot4.source.chunk
import uproot4.source.futures
import uproot4.source.file
import uproot4._util


class MemmapSource(uproot4.source.chunk.Source):
    """
    Source for a memory-mapped file.

    Threading is unnecessary because a memory-map is stateless.
    """

    __slots__ = ["_file_path", "_file", "_fallback"]

    _dtype = uproot4.source.chunk.Chunk._dtype

    def __init__(self, file_path, **options):
        """
        Args:
            file_path (str): Path to the file.
        """
        num_fallback_workers = options["num_fallback_workers"]
        self._num_requests = 0
        self._num_requested_chunks = 0
        self._num_requested_bytes = 0

        self._file_path = file_path
        try:
            self._file = numpy.memmap(self._file_path, dtype=self._dtype, mode="r")
            self._fallback = None
        except (OSError, IOError):
            self._file = None
            self._fallback = uproot4.source.file.FileSource(
                file_path, num_workers=num_fallback_workers
            )

    @property
    def file(self):
        """
        Path to the file.
        """
        return self._file

    @property
    def fallback(self):
        """
        Fallback FileSource or None; only created if opening a memory map
        raised OSError or IOError.
        """
        return self._fallback

    def __enter__(self):
        """
        Passes `__enter__` to the memory-map.

        Returns self.
        """
        if self._fallback is None:
            if hasattr(self._file._mmap, "__enter__"):
                self._file._mmap.__enter__()
        else:
            self._fallback.__enter__()

        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """
        Passes `__exit__` to the memory-map or otherwise closes the file.
        """
        if self._fallback is None:
            if hasattr(self._file._mmap, "__exit__"):
                self._file._mmap.__exit__(exception_type, exception_value, traceback)
            else:
                self._file._mmap.close()
        else:
            self._fallback.__exit__(exception_type, exception_value, traceback)

    @property
    def closed(self):
        """
        True if the associated file/connection/thread pool is closed; False
        otherwise.
        """
        if uproot4._util.py2:
            try:
                self._file._mmap.tell()
            except ValueError:
                return True
        elif self._file._mmap.closed:
            return True
        else:
            return False

    @property
    def num_bytes(self):
        """
        The number of bytes in the file.
        """
        if self._fallback is None:
            return self._file._mmap.size()
        else:
            return self._fallback.num_bytes

    def chunk(self, start, stop, exact=True):
        """
        Args:
            start (int): The start (inclusive) byte position for the desired
                chunk.
            stop (int or None): If an int, the stop (exclusive) byte position
                for the desired chunk; if None, stop at the end of the file.
            exact (bool): If False, attempts to access bytes beyond the
                end of the Chunk raises a RefineChunk; if True, it raises
                an OSError with an informative message.

        Returns a single Chunk that has already been filled synchronously.
        """
        self._num_requests += 1
        self._num_requested_chunks += 1
        self._num_requested_bytes += stop - start

        if self._fallback is None:
            if self.closed:
                raise OSError("memmap is closed for file {0}".format(self._file_path))

            data = numpy.array(self._file[start:stop], copy=True)
            future = uproot4.source.futures.TrivialFuture(data)
            return uproot4.source.chunk.Chunk(self, start, stop, future, exact)

        else:
            return self._fallback(start, stop, exact=exact)

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
        self._num_requests += 1
        self._num_requested_chunks += len(ranges)
        self._num_requested_bytes += sum(stop - start for start, stop in ranges)

        if self._fallback is None:
            if self.closed:
                raise OSError("memmap is closed for file {0}".format(self._file_path))

            chunks = []
            for start, stop in ranges:
                data = numpy.array(self._file[start:stop], copy=True)
                future = uproot4.source.futures.TrivialFuture(data)
                chunk = uproot4.source.chunk.Chunk(self, start, stop, future, exact)
                if notifications is not None:
                    future.add_done_callback(
                        uproot4.source.chunk.Resource.notifier(chunk, notifications)
                    )
                chunks.append(chunk)
            return chunks

        else:
            return self._fallback(ranges, exact=exact, notifications=notifications)

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

"""
Defines the basic types of Uproot physical I/O.

A Source represents one ROOT file (local or remote) as a source of Chunks.

Chunks are byte ranges within the file that can fill asynchronously.
"""

from __future__ import absolute_import

import numpy

import uproot4.deserialization
import uproot4.source.futures
import uproot4.source.cursor


class Resource(object):
    """
    Abstract base class for a file handle whose lifetime is linked to Threads
    and/or a thread pool Executor.

    A Resource instance is passed as the first argument of a TaskFuture's task
    function.
    """

    @staticmethod
    def getter(start, stop):
        """
        Creates a function to submit to an Executor, which fetches bytes.
        """
        return lambda resource: resource.get(start, stop)

    @staticmethod
    def notifier(chunk, notifications):
        return lambda future: notifications.put(chunk)


class Source(object):
    """
    Abstract base class for physical I/O in Uproot.

    These are all context managers that shut down any thread pools (Executors)
    and close any file handles (Resources) when finished.
    """

    @property
    def file_path(self):
        """
        The original path to the file (or URL, etc).
        """
        return self._file_path

    @property
    def num_requests(self):
        """
        The number of requests that have been made.
        """
        return self._num_requests

    @property
    def num_requested_chunks(self):
        """
        The number of chunks that have been requested.
        """
        return self._num_requested_chunks

    @property
    def num_requested_bytes(self):
        """
        The number of bytes that have been requested.
        """
        return self._num_requested_bytes

    def close(self):
        """
        Manually calls `__exit__`.
        """
        self.__exit__(None, None, None)

    @property
    def closed(self):
        """
        True if the associated file/connection/thread pool is closed; False
        otherwise.
        """
        return self._executor.closed

    def begin_end_chunks(self, begin_guess_bytes, end_guess_bytes):
        """
        Args:
            begin_guess_bytes (int): Number of bytes to try to take from the
                beginning of the file.
            end_guess_bytes (int): Number of bytes to try to take from the
                end of the file.

        Returns two Chunks, one from the beginning of the file and the other
        from the end of the file, which may be the same Chunk if these regions
        overlap. The Chunks are filled asynchronously.
        """
        num_bytes = self.num_bytes
        if begin_guess_bytes + end_guess_bytes > num_bytes:
            chunk = self.chunk(0, num_bytes, exact=False)
            return chunk, chunk
        else:
            return self.chunks(
                [(0, begin_guess_bytes), (num_bytes - end_guess_bytes, num_bytes)],
                exact=False,
            )


class MultithreadedSource(Source):
    """
    Base class for Sources that maintain an Executor.
    """

    @property
    def executor(self):
        """
        The Executor, which may be manage Threads and other Resources.
        """
        return self._executor

    @property
    def num_workers(self):
        """
        The number of workers (int).

        If 0, this Source is synchronous: calling `chunks` blocks until all
        Chunks are full.

        If 1 or more, this Source is asynchronous: the thread or threads fill
        Chunks in the background.
        """
        return self._executor.num_workers

    def __enter__(self):
        """
        Passes down __enter__, but most Sources do nothing when entering a
        context block. (Resources are provisioned in the constructor.)

        Returns self.
        """
        self._executor.__enter__()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """
        Passes down __exit__. Most Sources shut down thread pools and close
        file handles when exiting a context block.
        """
        self._executor.__exit__(exception_type, exception_value, traceback)
        self._resource.__exit__(exception_type, exception_value, traceback)

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
        self._num_requests += 1
        self._num_requested_chunks += 1
        self._num_requested_bytes += stop - start

        future = uproot4.source.futures.TrivialFuture(self._resource.get(start, stop))
        return Chunk(self, start, stop, future, exact)

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

        Returns a list of Chunks that may already be filled with data or are
        filling in another thread and only block when their bytes are needed.
        """
        self._num_requests += 1
        self._num_requested_chunks += len(ranges)
        self._num_requested_bytes += sum(stop - start for start, stop in ranges)

        chunks = []
        for start, stop in ranges:
            future = self._executor._prepare(Resource.getter(start, stop))
            chunk = Chunk(self, start, stop, future, exact)
            if notifications is not None:
                future.add_done_callback(Resource.notifier(chunk, notifications))
            self._executor.submit(future)
            chunks.append(chunk)
        return chunks


class RefineChunk(Exception):
    __slots__ = ["start", "stop", "chunk_start", "chunk_stop"]

    def __init__(self, start, stop, chunk_start, chunk_stop):
        self.start = start
        self.stop = stop
        self.chunk_start = chunk_start
        self.chunk_stop = chunk_stop

    def __repr__(self):
        return "RefineChunk({0}, {1}, {2}, {3})".format(
            self.start, self.stop, self.chunk_start, self.chunk_stop
        )


class Chunk(object):
    """
    A range of bytes from a Source, which may be synchronously filled by
    Source.chunks or asynchronously filled.

    It only blocks when `raw_data`, `get`, or `remainder` is called.
    """

    __slots__ = ["_source", "_start", "_stop", "_future", "_raw_data", "_exact"]

    _dtype = numpy.dtype(numpy.uint8)

    @classmethod
    def wrap(cls, source, data):
        """
        Wrap a `data` buffer with a Chunk interface, linking it to a given
        Source. Used for presenting uncompressed data as Chunks.
        """
        future = uproot4.source.futures.TrivialFuture(data)
        return Chunk(source, 0, len(data), future, True)

    def __init__(self, source, start, stop, future, exact):
        """
        Args:
            source (Source): Parent from which this Chunk is derived.
            start (int): Starting byte position (inclusive, global in Source).
            stop (int): Stopping byte position (exclusive, global in Source).
            future (Future): Fills `raw_data` on demand.
            exact (bool): If False, attempts to access bytes beyond the
                end of this Chunk raises a RefineChunk; if True, it raises
                an OSError with an informative message.
        """
        self._source = source
        self._start = start
        self._stop = stop
        self._future = future
        self._exact = exact
        self._raw_data = None

    def __repr__(self):
        if self._exact:
            e = ""
        else:
            e = " (inexact)"

        return "<Chunk {0}-{1}{2}>".format(self._start, self._stop, e)

    @property
    def source(self):
        """
        Source from which this Chunk is derived.
        """
        return self._source

    @property
    def start(self):
        """
        Starting byte position (inclusive, global in Source).
        """
        return self._start

    @property
    def stop(self):
        """
        Stopping byte position (exclusive, global in Source).
        """
        return self._stop

    @property
    def future(self):
        """
        Fills `raw_data` on demand.
        """
        return self._future

    @property
    def exact(self):
        """
        If False, attempts to access bytes beyond the end of this Chunk raises
        a RefineChunk; if True, it raises an OSError with an informative message.
        """
        return self._exact

    def __contains__(self, range):
        """
        True if the range (start, stop) is fully contained within the Chunk;
        False otherwise.
        """
        start, stop = range
        if isinstance(start, uproot4.source.cursor.Cursor):
            start = start.index
        if isinstance(stop, uproot4.source.cursor.Cursor):
            stop = stop.index
        return self._start <= start and stop <= self._stop

    def wait(self):
        """
        Explicitly block until `raw_data` is filled.
        """
        if self._raw_data is None:
            self._raw_data = numpy.frombuffer(self._future.result(), dtype=self._dtype)
            if self._exact and len(self._raw_data) != self._stop - self._start:
                raise OSError(
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
        """
        Data from the Source as a NumPy array with dtype uint8.

        Accessing this member blocks until it is filled.
        """
        self.wait()
        return self._raw_data

    def get(self, start, stop, cursor, context):
        """
        Args:
            start (int): Starting byte position to extract (inclusive, global
                in Source).
            stop (int): Stopping byte position to extract (exclusive, global
                in Source).
            cursor (Cursor): The Cursor that is currently reading this Chunk.
            context (dict): Information about the current state of deserialization.

        Returns a subinterval of the `raw_data` using global coordinates as a
        NumPy array with dtype uint8.

        The start and stop must be `Chunk.start <= start <= stop <= Chunk.stop`.

        Calling this function blocks until `raw_data` is filled.
        """
        self.wait()

        if (start, stop) in self:
            local_start = start - self._start
            local_stop = stop - self._start
            return self._raw_data[local_start:local_stop]

        elif self._exact:
            raise uproot4.deserialization.DeserializationError(
                """attempting to get bytes {0}:{1}
outside expected range {2}:{3} for this Chunk""".format(
                    start, stop, self._start, self._stop
                ),
                self,
                cursor.copy(),
                context,
                self._source.file_path,
            )

        else:
            raise RefineChunk(start, stop, self._start, self._stop)

    def remainder(self, start, cursor, context):
        """
        Args:
            start (int): Starting byte position to extract (inclusive, global
                in Source).
            context (dict): Information about the current state of deserialization.

        Returns a subinterval of the `raw_data` from `start` to the end of the
        Chunk as a NumPy array with dtype uint8.

        The start must be `Chunk.start <= start < Chunk.stop`.

        Calling this function blocks until `raw_data` is filled.
        """
        self.wait()

        if self._start <= start:
            local_start = start - self._start
            return self._raw_data[local_start:]

        else:
            raise uproot4.deserialization.DeserializationError(
                """attempting to get bytes after {0}
outside expected range {1}:{2} for this Chunk""".format(
                    start, self._start, self._stop
                ),
                self,
                cursor.copy(),
                context,
                self._source.file_path,
            )

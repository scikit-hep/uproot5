# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

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

    def file_path(self):
        """
        The original path to the file (or URL, etc).
        """
        return self._file_path


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
    def num_bytes(self):
        """
        The number of bytes in the file.
        """
        return self._num_bytes

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


class MultithreadedSource(Source):
    """
    Base class for Sources that maintain an ResourceThreadPoolExecutor.
    """

    def __repr__(self):
        path = repr(self._file_path)
        if len(self._file_path) > 10:
            path = repr("..." + self._file_path[-10:])
        return "<{0} {1} ({2} workers) at 0x{3:012x}>".format(
            type(self).__name__, path, self.num_workers, id(self)
        )

    @property
    def executor(self):
        return self._executor

    @property
    def num_workers(self):
        return self._executor.num_workers

    @property
    def closed(self):
        return self._executor.closed

    def __enter__(self):
        self._executor.__enter__()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self._executor.__exit__(exception_type, exception_value, traceback)

    def chunk(self, start, stop):
        self._num_requests += 1
        self._num_requested_chunks += 1
        self._num_requested_bytes += stop - start

        future = self.ResourceClass.future(self, start, stop)
        chunk = Chunk(self, start, stop, future)
        self._executor.submit(future)
        return chunk

    def chunks(self, ranges, notifications):
        self._num_requests += 1
        self._num_requested_chunks += len(ranges)
        self._num_requested_bytes += sum(stop - start for start, stop in ranges)

        chunks = []
        for start, stop in ranges:
            future = self.ResourceClass.future(self, start, stop)
            chunk = Chunk(self, start, stop, future)
            future._set_notify(notifier(chunk, notifications))
            self._executor.submit(future)
            chunks.append(chunk)
        return chunks


def notifier(chunk, notifications):
    def notify():
        notifications.put(chunk)

    return notify


class Chunk(object):
    """
    A range of bytes from a Source, which may be synchronously filled by
    Source.chunks or asynchronously filled.

    It only blocks when `raw_data`, `get`, or `remainder` is called.
    """

    _dtype = numpy.dtype(numpy.uint8)

    @classmethod
    def wrap(cls, source, data):
        """
        Wrap a `data` buffer with a Chunk interface, linking it to a given
        Source. Used for presenting uncompressed data as Chunks.
        """
        future = uproot4.source.futures.NoFuture(data)
        return Chunk(source, 0, len(data), future)

    def __init__(self, source, start, stop, future):
        """
        Args:
            source (Source): Parent from which this Chunk is derived.
            start (int): Starting byte position (inclusive, global in Source).
            stop (int): Stopping byte position (exclusive, global in Source).
            future (Future): Fills `raw_data` on demand.
        """
        self._source = source
        self._start = start
        self._stop = stop
        self._future = future
        self._raw_data = None

    def __repr__(self):
        return "<Chunk {0}-{1}>".format(self._start, self._stop)

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
            if len(self._raw_data) != self._stop - self._start:
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

        else:
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

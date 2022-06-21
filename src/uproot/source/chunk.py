# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

"""
This module defines a :doc:`uproot.source.chunk.Chunk`, which is a range of bytes
requested from a file. All interaction between the "physical layer" and the
"interpretation layer" is through a :doc:`uproot.source.cursor.Cursor`'s
interpretation of a :doc:`uproot.source.chunk.Chunk`.

Also defines abstract classes for :doc:`uproot.source.chunk.Resource` and
:doc:`uproot.source.chunk.Source`, the primary types of the "physical layer."
"""


import numbers

import numpy

import uproot


class Resource:
    """
    Abstract class for a file handle whose lifetime may be linked to threads
    in a thread pool executor.

    A :doc:`uproot.source.chunk.Resource` instance is always the first
    argument of functions evaluated by a
    :doc:`uproot.source.futures.ResourceFuture`.
    """

    @property
    def file_path(self):
        """
        A path to the file (or URL).
        """
        return self._file_path


class Source:
    """
    Abstract class for physically reading and writing data from a file, which
    might be remote.

    In addition to the file handle, a :doc:`uproot.source.chunk.Source` might
    manage a :doc:`uproot.source.futures.ResourceThreadPoolExecutor` to read
    the file in parallel. Stopping these threads is part of the act of closing
    the file.
    """

    def chunk(self, start, stop):
        """
        Args:
            start (int): Seek position of the first byte to include.
            stop (int): Seek position of the first byte to exclude
                (one greater than the last byte to include).

        Request a byte range of data from the file as a
        :doc:`uproot.source.chunk.Chunk`.
        """
        pass

    def chunks(self, ranges, notifications):
        """
        Args:
            ranges (list of (int, int) 2-tuples): Intervals to fetch
                as (start, stop) pairs in a single request, if possible.
            notifications (``queue.Queue``): Indicator of completed
                chunks. After each gets filled, it is ``put`` on the
                queue; a listener should ``get`` from this queue
                ``len(ranges)`` times.

        Request a set of byte ranges from the file.

        This method has two outputs:

        * The method returns a list of unfilled
          :doc:`uproot.source.chunk.Chunk` objects, which get filled
          in a background thread. If you try to read data from an
          unfilled chunk, it will wait until it is filled.
        * The method also puts the same :doc:`uproot.source.chunk.Chunk`
          objects onto the ``notifications`` queue as soon as they are
          filled.

        Reading data from chunks on the queue can be more efficient than
        reading them from the returned list. The total reading time is the
        same, but work on the filled chunks can be better parallelized if
        it is triggered by already-filled chunks, rather than waiting for
        chunks to be filled.
        """
        pass

    @property
    def file_path(self):
        """
        A path to the file (or URL).
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
        The number of requests that have been made (performance counter).
        """
        return self._num_requests

    @property
    def num_requested_chunks(self):
        """
        The number of :doc:`uproot.source.chunk.Chunk` objects that have been
        requested (performance counter).
        """
        return self._num_requested_chunks

    @property
    def num_requested_bytes(self):
        """
        The number of bytes that have been requested (performance counter).
        """
        return self._num_requested_bytes

    def close(self):
        """
        Manually closes the file(s) and stops any running threads.
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
    Abstract class for a :doc:`uproot.source.chunk.Source` that maintains a
    :doc:`uproot.source.futures.ResourceThreadPoolExecutor`.
    """

    def __repr__(self):
        path = repr(self._file_path)
        if len(self._file_path) > 10:
            path = repr("..." + self._file_path[-10:])
        return "<{} {} ({} workers) at 0x{:012x}>".format(
            type(self).__name__, path, self.num_workers, id(self)
        )

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

    @property
    def executor(self):
        """
        The :doc:`uproot.source.futures.ResourceThreadPoolExecutor`
        """
        return self._executor

    @property
    def num_workers(self):
        """
        The number of :doc:`uproot.source.futures.ResourceWorker` threads in
        the :doc:`uproot.source.futures.ResourceThreadPoolExecutor`.
        """
        return self._executor.num_workers

    @property
    def closed(self):
        """
        True if the :doc:`uproot.source.futures.ResourceThreadPoolExecutor` has
        been shut down and the file handles have been closed.
        """
        return self._executor.closed

    def __enter__(self):
        self._executor.__enter__()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self._executor.__exit__(exception_type, exception_value, traceback)


def notifier(chunk, notifications):  # noqa: D103
    def notify():
        notifications.put(chunk)

    return notify


class Chunk:
    """
    Args:
        source (:doc:`uproot.source.chunk.Source`): Source from which the
            data were derived.
        start (int): Seek position of the first byte to include.
        stop (int): Seek position of the first byte to exclude
            (one greater than the last byte to include).
        future (:doc:`uproot.source.futures.TrivialFuture` or :doc:`uproot.source.futures.Future`): Handle
            to the synchronous or asynchronous data. A chunk is "filled"
            when the ``future`` completes.

    A range of bytes from a :doc:`uproot.source.chunk.Source`, which may be
    synchronously or asynchronously filled.

    The following methods must wait for the
    :ref:`uproot.source.chunk.Chunk.future` to complete (to be filled):

    * :ref:`uproot.source.chunk.Chunk.wait`: Waits and nothing else.
    * :ref:`uproot.source.chunk.Chunk.raw_data`: The data as a
      ``numpy.ndarray`` of ``numpy.uint8``.
    * :ref:`uproot.source.chunk.Chunk.get`: A subinterval of the data as
      a ``numpy.ndarray`` of ``numpy.uint8``.
    * :ref:`uproot.source.chunk.Chunk.remainder`: A subinterval from the
      :doc:`uproot.source.cursor.Cursor` to the end of the
      :doc:`uproot.source.chunk.Chunk`.
    """

    _dtype = numpy.dtype(numpy.uint8)

    @classmethod
    def wrap(cls, source, data, start=0):
        """
        Args:
            source (:doc:`uproot.source.chunk.Source`): Source to attach to
                the new chunk.
            data (``numpy.ndarray`` of ``numpy.uint8``): Data for the new chunk.
            start (int): Virtual starting position for this chunk; if ``X``,
                then a :doc:`uproot.source.cursor.Cursor` is valid from ``X``
                to ``X + len(data)``.

        Manually creates a synchronous :doc:`uproot.source.chunk.Chunk`.
        """
        future = uproot.source.futures.TrivialFuture(data)
        return Chunk(source, start, start + len(data), future)

    def __init__(self, source, start, stop, future, is_memmap=False):
        self._source = source
        self._start = start
        self._stop = stop
        self._future = future
        self._raw_data = None
        self._is_memmap = is_memmap

    def __repr__(self):
        return f"<Chunk {self._start}-{self._stop}>"

    @property
    def source(self):
        """
        Source from which this Chunk is derived.
        """
        return self._source

    @property
    def start(self):
        """
        Seek position of the first byte to include.
        """
        return self._start

    @property
    def stop(self):
        """
        Seek position of the first byte to exclude (one greater than the last
        byte to include).
        """
        return self._stop

    @property
    def future(self):
        """
        Handle to the synchronous or asynchronous data. A chunk is "filled"
        when the ``future`` completes.
        """
        return self._future

    @property
    def is_memmap(self):
        """
        If True, the `raw_data` is or will be a view into a memmap file, which
        must be handled carefully. Accessing that data after the file is closed
        causes a segfault.
        """
        return self._is_memmap

    def detach_memmap(self):
        """
        Returns a Chunk (possibly this one) that is not tied to live memmap data.
        Such a Chunk can be accessed after the file is closed without segfaults.
        """
        if self._is_memmap:
            if self._future is None:
                assert self._raw_data is not None
                future = uproot.source.futures.TrivialFuture(
                    numpy.array(self._raw_data, copy=True)
                )
            else:
                assert isinstance(self._future, uproot.source.futures.TrivialFuture)
                future = uproot.source.futures.TrivialFuture(
                    numpy.array(self._future._result, copy=True)
                )
            return Chunk(self._source, self._start, self._stop, future)

        else:
            return self

    def __contains__(self, range):
        start, stop = range
        if isinstance(start, uproot.source.cursor.Cursor):
            start = start.index
        if isinstance(stop, uproot.source.cursor.Cursor):
            stop = stop.index
        return self._start <= start and stop <= self._stop

    def wait(self, insist=True):
        """
        Args:
            insist (bool or int): If True, raise an OSError if ``raw_data`` does
                does not have exactly ``stop - start`` bytes. If False, do not check.
                If an integer, only raise an OSError if data up to that index can't
                be supplied (i.e. require ``len(raw_data) >= insist - start``).

        Explicitly wait until the chunk is filled (the
        :ref:`uproot.source.chunk.Chunk.future` completes).
        """
        if self._raw_data is None:
            self._raw_data = numpy.frombuffer(self._future.result(), dtype=self._dtype)
            if insist is True:
                requirement = len(self._raw_data) == self._stop - self._start
            elif isinstance(insist, numbers.Integral):
                requirement = len(self._raw_data) >= insist - self._start
            elif insist is False:
                requirement = True
            else:
                raise TypeError(
                    """insist must be a bool or an int, not {}
for file path {}""".format(
                        repr(insist), self._source.file_path
                    )
                )

            if not requirement:
                raise OSError(
                    """expected Chunk of length {},
received {} bytes from {}
for file path {}""".format(
                        self._stop - self._start,
                        len(self._raw_data),
                        type(self._source).__name__,
                        self._source.file_path,
                    )
                )
            self._future = None

    @property
    def raw_data(self):
        """
        Data from the Source as a ``numpy.ndarray`` of ``numpy.uint8``.

        This method will wait until the chunk is filled (the
        :ref:`uproot.source.chunk.Chunk.future` completes), if it isn't
        already.
        """
        self.wait()
        return self._raw_data

    def get(self, start, stop, cursor, context):
        """
        Args:
            start (int): Seek position of the first byte to include.
            stop (int): Seek position of the first byte to exclude
                (one greater than the last byte to include).
            cursor (:doc:`uproot.source.cursor.Cursor`): A pointer to the
                current position in this chunk.
            context (dict): Auxiliary data used in deserialization.

        Returns a subinterval of the :ref:`uproot.source.chunk.Chunk.raw_data`
        as a ``numpy.ndarray`` of ``numpy.uint8``.

        Note that this ``start`` and ``stop`` are in the same coordinate
        system as the :ref:`uproot.source.chunk.Chunk.start` and
        :ref:`uproot.source.chunk.Chunk.stop`. That is, to get the whole
        chunk, use ``start=chunk.start`` and ``stop=chunk.stop``.

        This method will wait until the chunk is filled (the
        :ref:`uproot.source.chunk.Chunk.future` completes), if it isn't
        already.
        """
        if (start, stop) in self:
            self.wait(insist=stop)
            local_start = start - self._start
            local_stop = stop - self._start
            return self._raw_data[local_start:local_stop]

        else:
            raise uproot.deserialization.DeserializationError(
                """attempting to get bytes {}:{}
outside expected range {}:{} for this Chunk""".format(
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
            start (int): Seek position of the first byte to include.
            cursor (:doc:`uproot.source.cursor.Cursor`): A pointer to the
                current position in this chunk.
            context (dict): Auxiliary data used in deserialization.

        Returns a subinterval of the :ref:`uproot.source.chunk.Chunk.raw_data`
        as a ``numpy.ndarray`` of ``numpy.uint8`` from ``start`` to the end
        of the chunk.

        Note that this ``start`` is in the same coordinate system as the
        :ref:`uproot.source.chunk.Chunk.start`. That is, to get the whole
        chunk, use ``start=chunk.start``.

        This method will wait until the chunk is filled (the
        :ref:`uproot.source.chunk.Chunk.future` completes), if it isn't
        already.
        """
        self.wait()

        if self._start <= start:
            local_start = start - self._start
            return self._raw_data[local_start:]

        else:
            raise uproot.deserialization.DeserializationError(
                """attempting to get bytes after {}
outside expected range {}:{} for this Chunk""".format(
                    start, self._start, self._stop
                ),
                self,
                cursor.copy(),
                context,
                self._source.file_path,
            )

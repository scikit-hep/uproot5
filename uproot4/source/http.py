# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

"""
Source and Resource for multi-part HTTP(S) or single-part per Thread HTTP(S).
"""

from __future__ import absolute_import

import sys
import re
import threading
import time

try:
    import queue
    from http.client import HTTPConnection
    from http.client import HTTPSConnection
    from urllib.parse import urlparse
except ImportError:
    import Queue as queue
    from httplib import HTTPConnection
    from httplib import HTTPSConnection
    from urlparse import urlparse

import uproot4.source.futures
import uproot4.source.chunk
import uproot4._util


def make_connection(parsed_url, timeout):
    """
    Helper function to create a HTTPConnection or HTTPSConnection.
    """

    if parsed_url.scheme == "https":
        if uproot4._util.py2:
            return HTTPSConnection(
                parsed_url.netloc, parsed_url.port, None, None, False, timeout
            )
        else:
            return HTTPSConnection(
                parsed_url.netloc, parsed_url.port, None, None, timeout
            )

    elif parsed_url.scheme == "http":
        if uproot4._util.py2:
            return HTTPConnection(parsed_url.netloc, parsed_url.port, False, timeout)
        else:
            return HTTPConnection(parsed_url.netloc, parsed_url.port, timeout)

    else:
        raise ValueError(
            "unrecognized URL scheme for HTTP MultipartSource: {0}".format(
                parsed_url.scheme
            )
        )


class HTTPBackgroundThread(threading.Thread):
    class GetSize(object):
        __slots__ = ["done", "excinfo"]

        def __init__(self):
            self.done = threading.Event()
            self.excinfo = None

    class SinglepartWork(object):
        __slots__ = ["start", "stop", "future"]

        def __init__(self, start, stop, future):
            self.start = start
            self.stop = stop
            self.future = future

    class MultipartWork(object):
        __slots__ = ["ranges", "range_string", "futures"]

        def __init__(self, ranges, range_string, futures):
            self.ranges = ranges
            self.range_string = range_string
            self.futures = futures

    def __init__(self, file_path, timeout, work_queue, num_fallback_workers):
        """
        Args:
            file_path (str): URL starting with "http://" or "https://".
            timeout (None or float): Number of seconds before giving up on a
                remote file.
            work_queue (queue.Queue): Incoming communication from the main
                thread.
            num_fallback_workers (int): Number of workers to pass to a fallback
                MultithreadedHTTPSource, if necessary.
        """
        super(HTTPBackgroundThread, self).__init__()
        self.daemon = True
        self._file_path = file_path
        self._parsed_url = urlparse(file_path)
        self._connection = make_connection(self._parsed_url, timeout)
        self._timeout = timeout
        self._work_queue = work_queue
        self._num_fallback_workers = num_fallback_workers
        self._fallback = None
        self._num_bytes = None

    @property
    def file_path(self):
        """
        URL starting with "http://" or "https://".
        """
        return self._file_path

    @property
    def parsed_url(self):
        """
        URL parsed by urlparse.
        """
        return self._parsed_url

    @property
    def connection(self):
        """
        Used only for multi-part GET.
        """
        return self._connection

    @property
    def timeout(self):
        """
        Number of seconds before giving up on a remote file.
        """
        return self._timeout

    @property
    def work_queue(self):
        """
        Incoming communication from the main thread.
        """
        return self._work_queue

    @property
    def fallback(self):
        """
        Fallback MultithreadedHTTPSource or None; only created if the server
        returned a code other than 206 in response to a multi-part GET.
        """
        return self._fallback

    @property
    def num_fallback_workers(self):
        return self._num_fallback_workers

    def fill_with_singlepart(self, start, stop, future):
        """
        Fills the `future` with data from a single-part range request.
        """
        try:
            self._connection.request(
                "GET",
                self._parsed_url.path,
                headers={"Range": "bytes={0}-{1}".format(start, stop - 1)},
            )

            response = self._connection.getresponse()

            if response.status == 200:
                future._result = response.read()[start:stop]

            elif response.status == 206:
                future._result = response.read()

            else:
                raise OSError(
                    """remote server responded with status {0}
for URL {1}""".format(
                        response.status, self._file_path
                    )
                )

        except Exception:
            future._excinfo = sys.exc_info()

        future._set_finished()

    def make_fallback(self):
        """
        Creates a fallback MultithreadedHTTPSource because the server didn't
        respond with 206.
        """
        self._fallback = MultithreadedHTTPSource(
            self._file_path,
            num_workers=self._num_fallback_workers,
            timeout=self._timeout,
        )

    def fill_with_fallback(self, ranges, futures):
        """
        Fills the existing `futures` with data from Chunks made by the
        `fallback` MultithreadedHTTPSource.

        This will only be called before the HTTPSource notices that there is a
        `fallback` and sends requests for `chunks` to it directly.
        """
        if any(x is None or y is None for x, y in ranges):
            (_, begin_guess_bytes), (end_guess_bytes, _) = ranges
            num_bytes = self._num_bytes

            if begin_guess_bytes + end_guess_bytes > num_bytes:
                ranges = [
                    (0, num_bytes),
                    (0, num_bytes),
                ]
            else:
                ranges = [
                    (0, begin_guess_bytes),
                    (num_bytes - end_guess_bytes, num_bytes),
                ]

            (
                (futures[b"begin"]._start, futures[b"begin"]._stop),
                (futures[b"end"]._start, futures[b"end"]._stop),
            ) = ranges

            mapping = {
                "{0}-{1}".format(ranges[0][0], ranges[0][1] - 1).encode(): b"begin",
                "{0}-{1}".format(ranges[1][0], ranges[1][1] - 1).encode(): b"end",
            }

        try:
            chunks = self._fallback.chunks(ranges)

        except Exception:
            for future in futures.values():
                future._excinfo = sys.exc_info()
                future._set_finished()

        else:
            for (start, stop), chunk in zip(ranges, chunks):
                r = "{0}-{1}".format(start, stop - 1).encode()
                future = futures.get(r)
                if future is None:
                    future = futures[mapping[r]]

                if hasattr(chunk.future, "_finished"):
                    try:
                        chunk.future._finished.wait()
                    except Exception:
                        future._excinfo = chunk.future._excinfo
                        future._set_finished()
                        continue

                future._result = chunk.future._result
                future._excinfo = getattr(chunk.future, "_excinfo", None)
                future._set_finished()

    _content_range_size = re.compile(b"Content-Range: bytes ([0-9]+-[0-9]+)/([0-9]+)")
    _content_range = re.compile(b"Content-Range: bytes ([0-9]+-[0-9]+)")

    @staticmethod
    def next_header(response):
        """
        Helper function to get the next header of multi-part content.
        """
        line = response.fp.readline()
        r, size = None, None
        while r is None:
            m = HTTPBackgroundThread._content_range_size.match(line)
            if m is not None:
                r = m.group(1)
                size = int(m.group(2))
            else:
                m = HTTPBackgroundThread._content_range.match(line)
                if m is not None:
                    r = m.group(1)
                    size = None
            line = response.fp.readline()
            if len(line.strip()) == 0:
                break
        return r, size

    @staticmethod
    def raise_missing(i, futures, file_path):
        """
        Helper function to put a "not enough byte ranges" exception on all
        unfinished Futures.
        """
        for future in futures.values():
            if not future._finished.is_set():
                try:
                    raise OSError(
                        """found {0} of {1} expected headers in HTTP multipart
for URL {2}""".format(
                            i, len(futures), file_path
                        )
                    )
                except Exception:
                    future._excinfo = sys.exc_info()
                future._set_finished()

    @staticmethod
    def raise_unrecognized(r, futures, file_path):
        """
        Helper function to put a "unrecognized byte range" exception on all
        unfinished Futures.
        """
        for future in futures.values():
            if not future._finished.is_set():
                try:
                    raise OSError(
                        """unrecognized byte range in headers of HTTP multipart: {0}
for URL {1}""".format(
                            repr(r), file_path
                        )
                    )
                except Exception:
                    future._excinfo = sys.exc_info()
                future._set_finished()

    @staticmethod
    def raise_wrong_length(actual, expected, r, future, file_path):
        """
        Helper function to put a "wrong length" exception on the relevant Future.
        """
        try:
            raise OSError(
                """wrong chunk length {0} (expected {1}) for byte range {2} in HTTP multipart:
for URL {3}""".format(
                    actual, expected, repr(r), file_path
                )
            )
        except Exception:
            future._excinfo = sys.exc_info()
        future._set_finished()

    def fill_with_multipart(self, ranges, range_string, futures):
        is_begin_end = any(x is None or y is None for x, y in ranges)

        try:
            self._connection.request(
                "GET", self._parsed_url.path, headers={"Range": range_string}
            )
            response = self._connection.getresponse()
        except Exception:
            for future in futures.values():
                future._excinfo = sys.exc_info()
                future._set_finished()
            return

        multipart_supported = response.status == 206

        if multipart_supported and not is_begin_end:
            for k, x in response.getheaders():
                if k.lower() == "content-length":
                    content_length = int(x)
                    for start, stop in ranges:
                        if content_length == stop - start:
                            multipart_supported = False
            else:
                multipart_supported = False

        if not multipart_supported:
            for k, x in response.getheaders():
                if k.lower() == "content-length":
                    self._num_bytes = int(x)
            response.close()
            self.make_fallback()
            self.fill_with_fallback(ranges, futures)
            return

        for i in range(len(futures)):
            r, size = self.next_header(response)
            if r is None:
                self.raise_missing(i, futures, self._file_path)
                break

            first, last = r.split(b"-")
            first, last = int(first), int(last)
            length = last + 1 - first

            future = futures.get(r)
            if future is None:
                if first == 0:
                    future = futures.get(b"begin")
                    future._start = first
                    future._stop = last + 1

                elif last + 1 == size:
                    future = futures.get(b"end")
                    future._start = first
                    future._stop = last + 1

            if future is None:
                self.raise_unrecognized(r, futures, self._file_path)
                break

            future._result = response.read(length)
            if len(future._result) != length:
                self.raise_wrong_length(
                    len(future._result), length, r, future, self._file_path
                )
                break

            future._set_finished()

        response.close()

    @property
    def num_bytes(self):
        """
        The number of bytes in the file.
        """
        if self._num_bytes is None:
            work = HTTPBackgroundThread.GetSize()
            self._work_queue.put(work)
            work.done.wait()
            if work.excinfo is not None:
                uproot4.source.futures.delayed_raise(*work.excinfo)

        return self._num_bytes

    def fill_size(self, work):
        try:
            self._connection.request("HEAD", self._parsed_url.path)
            response = self._connection.getresponse()

        except Exception:
            work.excinfo = sys.exc_info()

        else:
            if response.status != 200:
                try:
                    raise OSError(
                        """response status was {0}, rather than 200
in file {1}""".format(
                            response.status, self._file_path
                        )
                    )
                except Exception:
                    work.excinfo = sys.exc_info()

            else:
                for k, x in response.getheaders():
                    if k.lower() == "content-length":
                        self._num_bytes = int(x)
                        break

                else:
                    try:
                        raise OSError(
                            """response headers did not include content-length: {0}
in file {1}""".format(
                                repr(dict(response.getheaders())), self._file_path
                            )
                        )
                    except Exception:
                        work.excinfo = sys.exc_info()

        work.done.set()

    def run(self):
        """
        Listens to the `work_queue`, processing each (ranges, futures) it
        recieves.

        If it finds a None on the `work_queue`, the Thread shuts down.
        """
        while True:
            work = self._work_queue.get()
            if work is None:
                break

            if isinstance(work, self.MultipartWork):
                if self._fallback is not None:
                    self.fill_with_fallback(work.ranges, work.futures)
                else:
                    self.fill_with_multipart(
                        work.ranges, work.range_string, work.futures
                    )

            elif isinstance(work, self.SinglepartWork):
                self.fill_with_singlepart(work.start, work.stop, work.future)

            elif isinstance(work, self.GetSize):
                self.fill_size(work)

            else:
                raise AssertionError(
                    "unrecognized message for HTTPBackgroundThread: {0}".format(
                        type(work)
                    )
                )

        self._connection.close()


class HTTPSource(uproot4.source.chunk.Source):
    """
    Source managing one asynchronous HTTP(S) capable of multi-part GET requests.

    This Source always has exactly 1 background Thread, since the result of a
    multi-part request has to be parsed by a single Thread.

    See MultithreadedHTTPSource for one Thread per part (one part per request).

    If a remote server fails to respond appropriately to a multi-part request
    (206), this falls back to a nested MultithreadedHTTPSource.
    """

    __slots__ = ["_file_path", "_parsed_url", "_connection", "_work_queue", "_worker"]

    def __init__(self, file_path, **options):
        """
        Args:
            file_path (str): URL starting with "http://" or "https://".
            num_fallback_workers (int): Number of workers to pass to a fallback
                MultithreadedHTTPSource, if necessary.
            timeout (None or float): Number of seconds before giving up on a
                remote file.
        """
        self._timeout = options["timeout"]
        num_fallback_workers = options["num_fallback_workers"]
        self._num_requests = 0
        self._num_requested_chunks = 0
        self._num_requested_bytes = 0

        self._file_path = file_path
        self._work_queue = queue.Queue()
        self._worker = HTTPBackgroundThread(
            file_path, self._timeout, self._work_queue, num_fallback_workers
        )
        self._worker.start()

    @property
    def file_path(self):
        """
        URL starting with "http://" or "https://".
        """
        return self._file_path

    @property
    def timeout(self):
        """
        Number of seconds before giving up on a remote file.
        """
        return self._timeout

    @property
    def work_queue(self):
        """
        FIFO for the single background Thread.
        """
        return self._work_queue

    @property
    def worker(self):
        """
        The single background Thread.
        """
        return self._worker

    @property
    def has_fallback(self):
        """
        If True, the server failed to respond appropriately to a multi-part GET
        (206) and will henceforth be accessed through a nested
        MultithreadedHTTPSource.
        """
        return self._worker.fallback is not None

    @property
    def fallback(self):
        """
        A nested MultithreadedHTTPSource or None; only created if the remote
        server fails to respond appriately to a multi-part GET (206).
        """
        return self._worker.fallback

    @property
    def num_fallback_workers(self):
        """
        Number of workers to use with the `fallback` MultithreadedHTTPSource.
        """
        return self._worker.num_fallback_workers

    def __enter__(self):
        """
        Does nothing and returns self.
        """
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """
        Closes the HTTP(S) connection and passes `__exit__` to the worker
        Thread.
        """
        while self._worker.is_alive():
            self._work_queue.put(None)
            time.sleep(0.001)

    @property
    def closed(self):
        """
        True if the associated file/connection/thread pool is closed; False
        otherwise.
        """
        return not self._worker.is_alive()

    @property
    def num_bytes(self):
        """
        The number of bytes in the file.
        """
        return self._worker.num_bytes

    def chunk(self, start, stop, exact=True):
        """
        Args:
            start (int): The start (inclusive) byte position for the desired
                chunk.
            stop (int None): The stop (exclusive) byte position for the desired
                chunk.
            exact (bool): If False, attempts to access bytes beyond the
                end of the Chunk raises a RefineChunk; if True, it raises
                an OSError with an informative message.

        Returns a single Chunk that will be filled by the background thread.
        """
        self._num_requests += 1
        self._num_requested_chunks += 1
        self._num_requested_bytes += stop - start

        future = uproot4.source.futures.TaskFuture(None)
        chunk = uproot4.source.chunk.Chunk(self, start, stop, future, exact)
        self._work_queue.put(HTTPBackgroundThread.SinglepartWork(start, stop, future))
        return chunk

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

        Returns a list of Chunks that will be filled asynchronously by the
        multi-part GET or the `fallback` threads (asynchronously in either
        case).
        """
        self._num_requests += 1
        self._num_requested_chunks += len(ranges)
        self._num_requested_bytes += sum(stop - start for start, stop in ranges)

        if self._worker.fallback is not None:
            return self._worker.fallback.chunks(ranges)

        else:
            range_strings = []
            futures = {}
            chunks = []
            for start, stop in ranges:
                r = "{0}-{1}".format(start, stop - 1)
                range_strings.append(r)
                future = uproot4.source.futures.TaskFuture(None)
                futures[r.encode()] = future
                chunk = uproot4.source.chunk.Chunk(self, start, stop, future, exact)
                if notifications is not None:
                    future.add_done_callback(
                        uproot4.source.chunk.Resource.notifier(chunk, notifications)
                    )
                chunks.append(chunk)

            range_string = "bytes=" + ", ".join(range_strings)
            self._work_queue.put(
                HTTPBackgroundThread.MultipartWork(ranges, range_string, futures)
            )
            return chunks

    @staticmethod
    def _fix_start_stop(chunk):
        def fix(future):
            if future._excinfo is None:
                chunk._start = future._start
                chunk._stop = future._stop

        return fix

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
        self._num_requests += 1
        self._num_requested_chunks += 2
        self._num_requested_bytes += begin_guess_bytes + end_guess_bytes

        if self._worker.fallback is not None:
            return self._worker.fallback.begin_end_chunks(
                begin_guess_bytes, end_guess_bytes
            )

        else:
            range_strings = [
                "0-{0}".format(begin_guess_bytes - 1),
                "-{0}".format(end_guess_bytes),
            ]
            futures = {}
            chunks = []

            future = uproot4.source.futures.TaskFuture(None)
            chunk = uproot4.source.chunk.Chunk(self, 0, 0, future, exact=False)
            future.add_done_callback(self._fix_start_stop(chunk))
            futures[b"begin"] = future
            chunks.append(chunk)

            future = uproot4.source.futures.TaskFuture(None)
            chunk = uproot4.source.chunk.Chunk(self, 0, 0, future, exact=False)
            future.add_done_callback(self._fix_start_stop(chunk))
            futures[b"end"] = future
            chunks.append(chunk)

            range_string = "bytes=" + ", ".join(range_strings)
            self._work_queue.put(
                HTTPBackgroundThread.MultipartWork(
                    [(None, begin_guess_bytes), (end_guess_bytes, None)],
                    range_string,
                    futures,
                )
            )
            return chunks


class HTTPResource(uproot4.source.chunk.Resource):
    """
    Resource wrapping a HTTPConnection or HTTPSConnection.
    """

    __slots__ = ["_file_path", "_timeout", "_parsed_url", "_connection"]

    def __init__(self, file_path, timeout):
        """
        Args:
            file_path (str): URL starting with "http://" or "https://".
            timeout (float): Number of seconds before giving up on a remote
                file.
        """
        self._file_path = file_path
        self._timeout = timeout
        self._parsed_url = urlparse(file_path)
        self._connection = make_connection(self._parsed_url, timeout)

    @property
    def file_path(self):
        """
        URL starting with "http://" or "https://".
        """
        return self._file_path

    @property
    def timeout(self):
        """
        Number of seconds before giving up on a remote file.
        """
        return self._timeout

    @property
    def parsed_url(self):
        """
        URL parsed by urlparse.
        """
        return self._parsed_url

    @property
    def connection(self):
        """
        HTTPConnection or HTTPSConnection.
        """
        return self._connection

    def __enter__(self):
        """
        Does nothing and returns self.
        """
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """
        Closes the HTTP(S) connection.
        """
        self._connection.close()

    def get(self, start, stop):
        """
        Args:
            start (int): Starting byte position to extract (inclusive, global
                in Source).
            stop (int): Stopping byte position to extract (exclusive, global
                in Source).

        Returns a subinterval of the `raw_data` using global coordinates as a
        NumPy array with dtype uint8.

        The start and stop must be `Chunk.start <= start <= stop <= Chunk.stop`.

        Calling this function blocks until `raw_data` is filled.
        """
        self._connection.request(
            "GET",
            self._parsed_url.path,
            headers={"Range": "bytes={0}-{1}".format(start, stop - 1)},
        )

        response = self._connection.getresponse()
        if response.status != 206:
            raise OSError(
                """remote server does not support HTTP range requests
for URL {0}""".format(
                    self._file_path
                )
            )

        return response.read()


class MultithreadedHTTPSource(uproot4.source.chunk.MultithreadedSource):
    """
    Source managing one synchronous or multiple asynchronous HTTP(S) handles as
    a context manager.

    This Source always makes one request per Chunk (though they may come from
    many concurrent Threads). See HTTPSource for multi-part HTTP(S).
    """

    __slots__ = ["_file_path", "_executor", "_timeout"]

    def __init__(self, file_path, **options):
        """
        Args:
            file_path (str): URL starting with "http://" or "https://".
            num_workers (int): If 0, one synchronous ResourceExecutor is
                created; if 1 or more, a collection of asynchronous
                ThreadResourceExecutors are created.
            timeout (float): Number of seconds before giving up on a remote
                file.
        """
        timeout = options["timeout"]
        num_workers = options["num_workers"]
        self._num_requests = 0
        self._num_requested_chunks = 0
        self._num_requested_bytes = 0

        self._file_path = file_path
        self._resource = HTTPResource(file_path, timeout)
        self._num_bytes = None

        if num_workers == 0:
            self._executor = uproot4.source.futures.ResourceExecutor(self._resource)
        else:
            self._executor = uproot4.source.futures.ThreadResourceExecutor(
                [HTTPResource(file_path, timeout) for x in range(num_workers)]
            )

        self._timeout = timeout

    @property
    def timeout(self):
        """
        Number of seconds before giving up on a remote file.
        """
        return self._timeout

    @property
    def num_bytes(self):
        """
        The number of bytes in the file.
        """
        if self._num_bytes is None:
            self._resource._connection.request("HEAD", self._resource._parsed_url.path)
            response = self._resource._connection.getresponse()

            if response.status != 200:
                raise OSError(
                    """response status was {0}, rather than 200
in file {1}""".format(
                        response.status, self._file_path
                    )
                )

            for k, x in response.getheaders():
                if k.lower() == "content-length":
                    self._num_bytes = int(x)
                    break
            else:
                raise OSError(
                    """response headers did not include content-length: {0}
in file {1}""".format(
                        repr(dict(response.getheaders())), self._file_path
                    )
                )

        return self._num_bytes

    @property
    def closed(self):
        """
        True if the associated file/connection/thread pool is closed; False
        otherwise.
        """
        return self._executor.closed

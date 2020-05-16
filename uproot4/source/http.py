# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

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


class HTTPMultipartThread(threading.Thread):
    def __init__(self, file_path, connection, work_queue, num_fallback_workers):
        """
        Args:
            file_path (str): URL starting with "http://" or "https://".
            connection (HTTPConnection or HTTPSConnection): Used only for
                multi-part GET.
            work_queue (queue.Queue): Incoming communication from the main
                thread.
            num_fallback_workers (int): Number of workers to pass to a fallback
                HTTPSource, if necessary.
        """
        super(HTTPMultipartThread, self).__init__()
        self.daemon = True
        self._file_path = file_path
        self._connection = connection
        self._work_queue = work_queue
        self._num_fallback_workers = num_fallback_workers
        self._fallback = None

    @property
    def file_path(self):
        """
        URL starting with "http://" or "https://".
        """
        return self._file_path

    @property
    def connection(self):
        """
        Used only for multi-part GET.
        """
        return self._connection

    @property
    def work_queue(self):
        """
        Incoming communication from the main thread.
        """
        return self._work_queue

    @property
    def fallback(self):
        """
        Fallback HTTPSource or None; only created if the server returned a code
        other than 206 in response to a multi-part GET.
        """
        return self._fallback

    @property
    def num_fallback_workers(self):
        return self._num_fallback_workers

    def make_fallback(self):
        """
        Creates a fallback HTTPSource because the server didn't respond with
        206.
        """
        self._fallback = HTTPSource(self._file_path, self._num_fallback_workers)

    def fill_with_fallback(self, ranges, futures):
        """
        Fills the existing `futures` with data from Chunks made by the
        `fallback` HTTPSource.

        This will only be called before the HTTPMultipartSource notices
        that there is a `fallback` and sends requests for `chunks` to it
        directly.
        """
        chunks = self._fallback.chunks(ranges)

        for (start, stop), chunk in zip(ranges, chunks):
            r = "{0}-{1}".format(start, stop - 1).encode()
            assert r in futures
            future = futures[r]

            if hasattr(chunk.future, "_finished"):
                chunk.future._finished.wait()

            future._result = chunk.future._result
            future._excinfo = getattr(chunk.future, "_excinfo", None)
            future._set_finished()

    _content_range = re.compile(b"Content-Range: bytes ([0-9]+-[0-9]+)")

    @staticmethod
    def next_header(response):
        """
        Helper function to get the next header of multi-part content.
        """
        line = response.fp.readline()
        r = None
        while r is None:
            m = HTTPMultipartThread._content_range.match(line)
            if m is not None:
                r = m.group(1)
            line = response.fp.readline()
            if len(line.strip()) == 0:
                break
        return r

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

    def run(self):
        """
        Listens to the `work_queue`, processing each (ranges, futures) it
        recieves.

        If it finds a None on the `work_queue`, the Thread shuts down.
        """
        while True:
            pair = self._work_queue.get()
            if pair is None:
                break

            assert isinstance(pair, tuple) and len(pair) == 2
            ranges, futures = pair
            assert isinstance(futures, dict)

            if self._fallback is not None:
                self.fill_with_fallback(ranges, futures)
                break

            response = self._connection.getresponse()

            multipart_supported = response.status == 206

            if multipart_supported:
                content_length = int(response.headers["Content-Length"])
                for start, stop in ranges:
                    if content_length == stop - start:
                        multipart_supported = False

            if not multipart_supported:
                response.close()
                self.make_fallback()
                self.fill_with_fallback(ranges, futures)
                break

            for i in range(len(futures)):
                r = self.next_header(response)
                if r is None:
                    self.raise_missing(i, futures, self._file_path)
                    break
                if r not in futures:
                    self.raise_unrecognized(r, futures, self._file_path)
                    break

                future = futures[r]

                first, last = r.split(b"-")
                length = int(last) + 1 - int(first)

                future._result = response.read(length)
                if len(future._result) != length:
                    self.raise_wrong_length(
                        len(future._result), length, r, future, self._file_path
                    )
                    break

                future._set_finished()

            response.close()


class HTTPMultipartSource(uproot4.source.chunk.Source):
    """
    Source managing one asynchronous HTTP(S) capable of multi-part GET requests.

    This Source always has exactly 1 background Thread, since the result of a
    multi-part request has to be parsed by a single Thread.

    See HTTPSource for one Thread per part (one part per request).

    If a remote server fails to respond appropriately to a multi-part request
    (206), this falls back to a nested HTTPSource.
    """

    __slots__ = ["_file_path", "_parsed_url", "_connection", "_work_queue", "_worker"]

    def __init__(self, file_path, num_fallback_workers=1, timeout=None):
        """
        Args:
            file_path (str): URL starting with "http://" or "https://".
            num_fallback_workers (int): Number of workers to pass to a fallback
                HTTPSource, if necessary.
            timeout (float): Number of seconds before giving up on a remote
                file.
        """
        self._file_path = file_path
        self._timeout = timeout
        self._parsed_url = urlparse(file_path)
        self._connection = make_connection(self._parsed_url, timeout)
        self._work_queue = queue.Queue()
        self._worker = HTTPMultipartThread(
            file_path, self._connection, self._work_queue, num_fallback_workers
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
        (206) and will henceforth be accessed through a nested HTTPSource.
        """
        return self._worker.fallback is not None

    @property
    def fallback(self):
        """
        A nested HTTPSource or None; only created if the remote server fails to
        respond appriately to a multi-part GET (206).
        """
        return self._worker.fallback

    @property
    def num_fallback_workers(self):
        """
        Number of workers to use with the `fallback` HTTPSource.
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
        self._connection.close()
        while self._worker.is_alive():
            self._work_queue.put(None)
            time.sleep(0.001)

    def chunks(self, ranges):
        """
        Args:
            ranges (iterable of (int, int)): The start (inclusive) and stop
                (exclusive) byte ranges for each desired chunk.

        Returns a list of Chunks that will be filled asynchronously by the
        multi-part GET or the `fallback` threads (asynchronously in either
        case).
        """
        if self._worker.fallback is not None:
            return self._worker.fallback.chunks(ranges)

        range_strings = []
        futures = {}
        chunks = []
        for start, stop in ranges:
            r = "{0}-{1}".format(start, stop - 1)
            range_strings.append(r)
            future = uproot4.source.futures.TaskFuture(None)
            futures[r.encode()] = future
            chunk = uproot4.source.chunk.Chunk(self, start, stop, future)
            chunks.append(chunk)

        range_string = "bytes=" + ", ".join(range_strings)
        self._connection.request(
            "GET", self._parsed_url.path, headers={"Range": range_string}
        )

        self._work_queue.put((ranges, futures))
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


class HTTPSource(uproot4.source.chunk.MultiThreadedSource):
    """
    Source managing one synchronous or multiple asynchronous HTTP(S) handles as
    a context manager.

    This Source always makes one request per Chunk (though they may come from
    many concurrent Threads). See HTTPMultipartSource for multi-part HTTP(S).
    """

    __slots__ = ["_file_path", "_executor", "_timeout"]

    def __init__(self, file_path, num_workers=0, timeout=None):
        """
        Args:
            file_path (str): URL starting with "http://" or "https://".
            num_workers (int): If 0, one synchronous ResourceExecutor is
                created; if 1 or more, a collection of asynchronous
                ThreadResourceExecutors are created.
            timeout (float): Number of seconds before giving up on a remote
                file.
        """
        self._file_path = file_path

        if num_workers == 0:
            self._executor = uproot4.source.futures.ResourceExecutor(
                HTTPResource(file_path, timeout)
            )
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

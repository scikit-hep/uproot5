# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

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


def connection(parsed_url):
    if parsed_url.scheme == "https":
        return HTTPSConnection(parsed_url.netloc)
    elif parsed_url.scheme == "http":
        return HTTPConnection(parsed_url.netloc)
    else:
        raise ValueError(
            "unrecognized URL scheme for HTTP MultipartSource: {0}".format(
                parsed_url.scheme
            )
        )


class HTTPMultipartThread(threading.Thread):
    def __init__(self, file_path, connection, work_queue, num_fallback_workers):
        super(HTTPMultipartThread, self).__init__()
        self.daemon = True
        self._file_path = file_path
        self._connection = connection
        self._work_queue = work_queue
        self._num_fallback_workers = num_fallback_workers
        self._fallback = None

    @property
    def file_path(self):
        return self._file_path

    @property
    def connection(self):
        return self._connection

    @property
    def work_queue(self):
        return self._work_queue

    @property
    def fallback(self):
        return self._fallback

    @property
    def num_fallback_workers(self):
        return self._num_fallback_workers

    def make_fallback(self):
        self._fallback = HTTPSource(self._file_path, self._num_fallback_workers)

    def fill_with_fallback(self, ranges, futures):
        chunks = self._fallback.chunks(ranges)

        for (start, stop), chunk in zip(ranges, chunks):
            r = "{0}-{1}".format(start, stop - 1).encode()
            assert r in futures
            future = futures[r]

            if hasattr(chunk.future, "_finished"):
                chunk.future._finished.wait()

            future._result = chunk.future._result
            future._excinfo = getattr(chunk.future, "_excinfo", None)
            future._finished.set()

    _content_range = re.compile(b"Content-Range: bytes ([0-9]+-[0-9]+)")

    @staticmethod
    def next_header(response):
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
        for future in futures.values():
            if not future._finished.is_set():
                try:
                    raise IOError(
                        """found {0} of {1} expected headers in HTTP multipart
for URL {2}""".format(
                            i, len(futures), file_path
                        )
                    )
                except Exception:
                    future._excinfo = sys.exc_info()
                future._finished.set()

    @staticmethod
    def raise_unrecognized(r, futures, file_path):
        for future in futures.values():
            if not future._finished.is_set():
                try:
                    raise IOError(
                        """unrecognized byte range in headers of HTTP multipart: {0}
for URL {1}""".format(
                            repr(r), file_path
                        )
                    )
                except Exception:
                    future._excinfo = sys.exc_info()
                future._finished.set()

    @staticmethod
    def raise_wrong_length(actual, expected, r, future, file_path):
        try:
            raise IOError(
                """wrong chunk length {0} (expected {1}) for byte range {2} in HTTP multipart:
for URL {3}""".format(
                    actual, expected, repr(r), file_path
                )
            )
        except Exception:
            future._excinfo = sys.exc_info()
        future._finished.set()

    def run(self):
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
            if response.status != 206:
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

                future._finished.set()

            response.close()


class HTTPMultipartSource(uproot4.source.chunk.Source):
    __slots__ = ["_file_path", "_parsed_url", "_connection", "_work_queue", "_worker"]

    def __init__(self, file_path, num_fallback_workers=1):
        self._file_path = file_path
        self._parsed_url = urlparse(file_path)
        self._connection = connection(self._parsed_url)
        self._work_queue = queue.Queue()
        self._worker = HTTPMultipartThread(
            file_path, self._connection, self._work_queue, num_fallback_workers
        )
        self._worker.start()

    @property
    def file_path(self):
        return self._file_path

    @property
    def parsed_url(self):
        return self._parsed_url

    @property
    def connection(self):
        return self._connection

    @property
    def work_queue(self):
        return self._work_queue

    @property
    def worker(self):
        return self._worker

    @property
    def has_fallback(self):
        return self._worker.fallback is not None

    @property
    def num_fallback_workers(self):
        return self._worker.num_fallback_workers

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self._connection.close()
        self._connection = None
        while self._worker.is_alive():
            self._work_queue.put(None)
            time.sleep(0.001)

    def chunks(self, ranges):
        if self._worker.fallback is not None:
            return self._worker.fallback.chunks(ranges)

        range_strings = []
        futures = {}
        chunks = []
        for start, stop in ranges:
            r = "{0}-{1}".format(start, stop - 1)
            range_strings.append(r)
            futures[r.encode()] = future = uproot4.source.futures.TaskFuture(None)
            chunks.append(uproot4.source.chunk.Chunk(self, start, stop, future))

        range_string = "bytes=" + ", ".join(range_strings)
        self._connection.request(
            "GET", self._parsed_url.path, headers={"Range": range_string}
        )

        self._work_queue.put((ranges, futures))
        return chunks


class HTTPResource(uproot4.source.chunk.Resource):
    def __init__(self, file_path):
        self._file_path = file_path
        self._parsed_url = urlparse(file_path)
        self._connection = connection(self._parsed_url)

    @property
    def file_path(self):
        return self._file_path

    @property
    def parsed_url(self):
        return self._parsed_url

    @property
    def connection(self):
        return self._connection

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self._connection.close()

    def get(self, start, stop):
        self._connection.request(
            "GET",
            self._parsed_url.path,
            headers={"Range": "bytes={0}-{1}".format(start, stop - 1)},
        )

        response = self._connection.getresponse()
        if response.status != 206:
            raise IOError(
                """remote server does not support HTTP range requests
for URL {0}""".format(
                    self._file_path
                )
            )

        return response.read()


class HTTPSource(uproot4.source.chunk.Source):
    def __init__(self, file_path, num_workers=1):
        self._file_path = file_path
        if num_workers == 1:
            self._executor = uproot4.source.futures.ResourceExecutor(
                HTTPResource(file_path)
            )
        elif num_workers > 1:
            self._executor = uproot4.source.futures.ThreadResourceExecutor(
                [HTTPResource(file_path) for x in range(num_workers)]
            )
        else:
            raise ValueError("num_workers must be at least 1")

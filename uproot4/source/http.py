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

import numpy

import uproot4.futures
import uproot4.source.chunk


class MultipartThread(threading.Thread):
    def __init__(self, file_path, connection, work_queue):
        super(MultipartThread, self).__init__()
        self.daemon = True
        self._file_path = file_path
        self._connection = connection
        self._work_queue = work_queue

    @property
    def file_path(self):
        return self._file_path

    @property
    def connection(self):
        return self._connection

    @property
    def work_queue(self):
        return self._work_queue

    _content_range = re.compile(b"Content-Range: bytes ([0-9]+-[0-9]+)")

    @staticmethod
    def next_header(response):
        line = response.fp.readline()
        r = None
        while r is None:
            m = MultipartThread._content_range.match(line)
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
        print("start thread")

        while True:
            print("waiting for work")

            futures = self._work_queue.get()
            if futures is None:
                break

            print("got work")

            assert isinstance(futures, dict)
            response = self._connection.getresponse()

            print("got response", response.status)

            if response.status != 206:
                raise NotImplementedError("FIXME: switch to single-part")

            for i in range(len(futures)):
                print("for", i, "in futures")

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

                future._result = numpy.frombuffer(
                    response.read(length), dtype=numpy.uint8
                )

                if len(future._result) != length:
                    self.raise_wrong_length(
                        len(future._result), length, r, future, self._file_path
                    )
                    break

                print("done with future")
                future._finished.set()

            print("done with response")
            response.close()


class MultipartSource(uproot4.source.chunk.Source):
    def __init__(self, file_path):
        self._file_path = file_path
        self._parsed_url = urlparse(file_path)
        if self._parsed_url.scheme == "https":
            self._connection = HTTPSConnection(self._parsed_url.netloc)
        elif self._parsed_url.scheme == "http":
            self._connection = HTTPConnection(self._parsed_url.netloc)
        else:
            raise ValueError(
                "unrecognized URL scheme for HTTP MultipartSource: {0}".format(
                    self._parsed_url.scheme
                )
            )
        self._work_queue = queue.Queue()
        self._worker = MultipartThread(file_path, self._connection, self._work_queue)

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
    def ready(self):
        return self._worker.is_alive()

    def __enter__(self):
        self._worker.start()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self._connection.close()
        self._connection = None
        while self._worker.is_alive():
            self._work_queue.put(None)
            time.sleep(0.001)

    def chunks(self, ranges):
        range_strings = []
        futures = {}
        chunks = []
        for start, stop in ranges:
            r = "{0}-{1}".format(start, stop - 1)
            range_strings.append(r)
            futures[r.encode()] = future = uproot4.futures.TaskFuture(None)
            chunks.append(uproot4.source.chunk.Chunk(self, start, stop, future))

        range_string = "bytes=" + ", ".join(range_strings)
        self._connection.request(
            "GET", self._parsed_url.path, headers={"Range": range_string}
        )
        self._work_queue.put(futures)

        return chunks


# connection = http.client.HTTPConnection("example.com")
# connection.request("GET", "", headers={"Range": "bytes=0-99, 150-159, 200-399"})
# response = connection.getresponse()
# print(response.status)
# for k, v in response.headers.items():
#     print(repr(k), repr(v))

# print(next(response))


# print(response.read())

# connection = http.client.HTTPSConnection("scikit-hep.org")
# # connection.request("HEAD", "uproot/examples/Zmumu.root",
# headers={"Range": "bytes=0-1, 2-3"})
# # connection.request("HEAD", "uproot/examples/Zmumu.root",
# headers={"Range": "bytes=90000-170000"})
# connection.request("GET", "uproot/examples/Zmumu.root",
# headers={"Range": "bytes=0-99, 90000-170000"})
# response = connection.getresponse()
# print(response.status)
# for k, v in response.headers.items():
#     print(repr(k), repr(v))
# print(len(response.read()))

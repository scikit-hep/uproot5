# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import sys
import re

try:
    from http.client import HTTPConnection
    from http.client import HTTPSConnection
    from urllib.parse import urlparse
except ImportError:
    from httplib import HTTPConnection
    from httplib import HTTPSConnection
    from urlparse import urlparse
try:
    import queue
except ImportError:
    import Queue as queue

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
                parsed_url.hostname, parsed_url.port, None, None, False, timeout
            )
        else:
            return HTTPSConnection(
                parsed_url.hostname, parsed_url.port, None, None, timeout
            )

    elif parsed_url.scheme == "http":
        if uproot4._util.py2:
            return HTTPConnection(parsed_url.hostname, parsed_url.port, False, timeout)
        else:
            return HTTPConnection(parsed_url.hostname, parsed_url.port, timeout)

    else:
        raise ValueError(
            "unrecognized URL scheme for HTTP MultipartSource: {0}".format(
                parsed_url.scheme
            )
        )


def get_num_bytes(file_path, parsed_url, timeout):
    connection = make_connection(parsed_url, timeout)
    connection.request("HEAD", parsed_url.path)
    response = connection.getresponse()

    if response.status == 404:
        connection.close()
        raise uproot4._util._file_not_found(file_path, "HTTP(S) returned 404")

    if response.status != 200:
        connection.close()
        raise OSError(
            """HTTP response was {0}, rather than 200, in attempt to get file size
in file {1}""".format(
                response.status, file_path
            )
        )

    for k, x in response.getheaders():
        if k.lower() == "content-length" and x.strip() != "0":
            connection.close()
            return int(x)
    else:
        connection.close()
        raise OSError(
            """response headers did not include content-length: {0}
in file {1}""".format(
                dict(response.getheaders()), file_path
            )
        )


class HTTPResource(uproot4.source.chunk.Resource):
    def __init__(self, file_path, timeout):
        self._file_path = file_path
        self._timeout = timeout
        self._parsed_url = urlparse(file_path)

    @property
    def timeout(self):
        return self._timeout

    @property
    def parsed_url(self):
        return self._parsed_url

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        pass

    @staticmethod
    def future(source, start, stop):
        connection = make_connection(source.parsed_url, source.timeout)
        connection.request(
            "GET",
            source.parsed_url.path,
            headers={"Range": "bytes={0}-{1}".format(start, stop - 1)},
        )

        def task(resource):
            return resource.get(connection, start, stop)

        return uproot4.source.futures.ResourceFuture(task)

    def get(self, connection, start, stop):
        response = connection.getresponse()

        if response.status == 404:
            connection.close()
            raise uproot4._util._file_not_found(self.file_path, "HTTP(S) returned 404")

        if response.status != 206:
            connection.close()
            raise OSError(
                """remote server does not support HTTP range requests
for URL {0}""".format(
                    self._file_path
                )
            )
        try:
            return response.read()
        finally:
            connection.close()

    @staticmethod
    def multifuture(source, ranges, futures, results):
        connection = make_connection(source.parsed_url, source.timeout)

        range_strings = []
        for start, stop in ranges:
            range_strings.append("{0}-{1}".format(start, stop - 1))

        connection.request(
            "GET",
            source.parsed_url.path,
            headers={"Range": "bytes=" + ", ".join(range_strings)},
        )

        def task(resource):
            try:
                response = connection.getresponse()
                multipart_supported = resource.is_multipart_supported(ranges, response)

                if not multipart_supported:
                    resource.handle_no_multipart(source, ranges, futures, results)
                else:
                    resource.handle_multipart(source, futures, results, response)

            except Exception:
                excinfo = sys.exc_info()
                for future in futures.values():
                    future._set_excinfo(excinfo)

            finally:
                connection.close()

        return uproot4.source.futures.ResourceFuture(task)

    _content_range_size = re.compile(b"Content-Range: bytes ([0-9]+-[0-9]+)/([0-9]+)")
    _content_range = re.compile(b"Content-Range: bytes ([0-9]+-[0-9]+)")

    def is_multipart_supported(self, ranges, response):
        if response.status != 206:
            return False

        for k, x in response.getheaders():
            if k.lower() == "content-length":
                content_length = int(x)
                for start, stop in ranges:
                    if content_length == stop - start:
                        return False
        else:
            return True

    def handle_no_multipart(self, source, ranges, futures, results):
        source._set_fallback()

        notifications = queue.Queue()
        source.fallback.chunks(ranges, notifications)

        for x in uproot4._util.range(len(ranges)):
            chunk = notifications.get()
            results[chunk.start, chunk.stop] = chunk.raw_data
            futures[chunk.start, chunk.stop]._run(self)

    def handle_multipart(self, source, futures, results, response):
        for i in uproot4._util.range(len(futures)):
            range_string, size = self.next_header(response)
            if range_string is None:
                raise OSError(
                    """found {0} of {1} expected headers in HTTP multipart
for URL {2}""".format(
                        i, len(futures), self._file_path
                    )
                )

            start, last = range_string.split(b"-")
            start, last = int(start), int(last)
            stop = last + 1

            future = futures.get((start, stop))

            if future is None:
                raise OSError(
                    """unrecognized byte range in headers of HTTP multipart: {0}
for URL {1}""".format(
                        repr(range_string.decode()), self._file_path
                    )
                )

            length = stop - start
            results[start, stop] = response.read(length)

            if len(results[start, stop]) != length:
                raise OSError(
                    """wrong chunk length {0} (expected {1}) for byte range {2} "
                    "in HTTP multipart
for URL {3}""".format(
                        len(results[start, stop]),
                        length,
                        repr(range_string.decode()),
                        self._file_path,
                    )
                )

            future._run(self)

    def next_header(self, response):
        line = response.fp.readline()
        range_string, size = None, None
        while range_string is None:
            m = self._content_range_size.match(line)
            if m is not None:
                range_string = m.group(1)
                size = int(m.group(2))
            else:
                m = self._content_range.match(line)
                if m is not None:
                    range_string = m.group(1)
                    size = None
            line = response.fp.readline()
            if len(line.strip()) == 0:
                break
        return range_string, size

    @staticmethod
    def partfuture(results, start, stop):
        def task(resource):
            return results[start, stop]

        return uproot4.source.futures.ResourceFuture(task)


class MultithreadedHTTPSource(uproot4.source.chunk.MultithreadedSource):
    ResourceClass = HTTPResource

    def __init__(self, file_path, **options):
        num_workers = options["num_workers"]
        timeout = options["timeout"]
        self._num_requests = 0
        self._num_requested_chunks = 0
        self._num_requested_bytes = 0

        self._file_path = file_path
        self._num_bytes = None
        self._timeout = timeout

        self._executor = uproot4.source.futures.ResourceThreadPoolExecutor(
            [HTTPResource(file_path, timeout) for x in uproot4._util.range(num_workers)]
        )

    @property
    def timeout(self):
        return self._timeout

    @property
    def num_bytes(self):
        if self._num_bytes is None:
            self._num_bytes = get_num_bytes(
                self._file_path, self.parsed_url, self._timeout
            )
        return self._num_bytes

    @property
    def parsed_url(self):
        return self._executor.workers[0].resource.parsed_url


class HTTPSource(uproot4.source.chunk.Source):
    ResourceClass = HTTPResource

    def __init__(self, file_path, **options):
        num_fallback_workers = options["num_fallback_workers"]
        timeout = options["timeout"]
        self._num_requests = 0
        self._num_requested_chunks = 0
        self._num_requested_bytes = 0

        self._file_path = file_path
        self._timeout = timeout
        self._num_bytes = None

        self._executor = uproot4.source.futures.ResourceThreadPoolExecutor(
            [HTTPResource(file_path, timeout)]
        )
        self._fallback = None
        self._fallback_options = dict(options)
        self._fallback_options["num_workers"] = num_fallback_workers

    def __repr__(self):
        path = repr(self._file_path)
        if len(self._file_path) > 10:
            path = repr("..." + self._file_path[-10:])
        fallback = ""
        if self._fallback is not None:
            fallback = " with fallback"
        return "<{0} {1}{2} at 0x{3:012x}>".format(
            type(self).__name__, path, fallback, id(self)
        )

    @property
    def timeout(self):
        return self._timeout

    @property
    def num_bytes(self):
        if self._num_bytes is None:
            self._num_bytes = get_num_bytes(
                self._file_path, self.parsed_url, self._timeout
            )
        return self._num_bytes

    @property
    def parsed_url(self):
        return self._executor.workers[0].resource.parsed_url

    @property
    def executor(self):
        return self._executor

    @property
    def fallback(self):
        return self._fallback

    def _set_fallback(self):
        self._fallback = MultithreadedHTTPSource(
            self._file_path,
            **self._fallback_options    # NOTE: a comma after **fallback_options breaks Python 2
        )

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self._executor.shutdown()

    @property
    def closed(self):
        return self._executor.closed

    def chunk(self, start, stop):
        self._num_requests += 1
        self._num_requested_chunks += 1
        self._num_requested_bytes += stop - start

        future = self.ResourceClass.future(self, start, stop)
        chunk = uproot4.source.chunk.Chunk(self, start, stop, future)
        self._executor.submit(future)
        return chunk

    def chunks(self, ranges, notifications):
        if self._fallback is None:
            self._num_requests += 1
            self._num_requested_chunks += len(ranges)
            self._num_requested_bytes += sum(stop - start for start, stop in ranges)

            futures = {}
            results = {}
            chunks = []
            for start, stop in ranges:
                partfuture = self.ResourceClass.partfuture(results, start, stop)
                futures[start, stop] = partfuture
                results[start, stop] = None
                chunk = uproot4.source.chunk.Chunk(self, start, stop, partfuture)
                partfuture._set_notify(
                    uproot4.source.chunk.notifier(chunk, notifications)
                )
                chunks.append(chunk)

            self._executor.submit(
                self.ResourceClass.multifuture(self, ranges, futures, results)
            )
            return chunks

        else:
            return self._fallback.chunks(ranges, notifications)

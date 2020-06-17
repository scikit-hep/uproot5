# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

"""
Source and Resource for XRootD (pyxrootd).
"""

from __future__ import absolute_import

import os

import uproot4.source.chunk
import uproot4.source.futures


def import_pyxrootd():
    os.environ["XRD_RUNFORKHANDLER"] = "1"  # set multiprocessing flag
    try:
        import pyxrootd.client
        import XRootD.client

    except ImportError:
        raise ImportError(
            """Install pyxrootd package with:

    conda install -c conda-forge xrootd

(or download from http://xrootd.org/dload.html and manually compile with """
            """cmake; setting PYTHONPATH and LD_LIBRARY_PATH appropriately)."""
        )

    else:
        return pyxrootd, XRootD


def get_server_config(file_path):
    """
    Query a XRootD server for its configuration

    Args:
        file_path (str): The full URl to the requested resource

    Returns:
        readv_iov_max (int): The maximum number of elements that can be
            requested in a single vector read
        readv_ior_max (int): The maximum number of bytes that can be requested
            per **element** in a vector read
    """
    pyxrootd, XRootD = import_pyxrootd()

    url = XRootD.client.URL(file_path)
    fs = XRootD.client.FileSystem("{0}://{1}/".format(url.protocol, url.hostid))

    status, readv_iov_max = fs.query(
        XRootD.client.flags.QueryCode.CONFIG, "readv_iov_max"
    )
    if not status.ok:
        raise OSError(status.message)
    readv_iov_max = int(readv_iov_max)

    status, readv_ior_max = fs.query(
        XRootD.client.flags.QueryCode.CONFIG, "readv_ior_max"
    )
    if not status.ok:
        raise OSError(status.message)
    readv_ior_max = int(readv_ior_max)

    return readv_iov_max, readv_ior_max


class XRootDResource(uproot4.source.chunk.Resource):
    """
    Resource wrapping a pyxrootd.File.
    """

    __slots__ = ["_file_path", "_file"]

    def __init__(self, file_path, timeout):
        """
        Args:
            file_path (str): URL starting with "root://".
            timeout (int): Number of seconds (loosely interpreted by XRootD)
                before giving up on a remote file.
        """

        pyxrootd, XRootD = import_pyxrootd()
        self._file_path = file_path
        self._timeout = timeout
        self._file = pyxrootd.client.File()

        status, dummy = self._file.open(
            self._file_path, timeout=(0 if timeout is None else timeout)
        )

        if status.get("error", None):
            self._file.close(timeout=(0 if self._timeout is None else self._timeout))
            raise OSError(
                """XRootD error: {0}
in file {1}""".format(
                    status["message"], self._file_path
                )
            )

    @property
    def file_path(self):
        """
        URL starting with "root://".
        """
        return self._file_path

    @property
    def timeout(self):
        """
        Number of seconds (loosely interpreted by XRootD) before giving up on a
        remote file.
        """
        return self._timeout

    @property
    def file(self):
        """
        The pyxrootd.File handle.
        """
        return self._file

    def __enter__(self):
        """
        Does nothing and returns self.
        """
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """
        Closes the pyxrootd.File.
        """
        self._file.close(timeout=(0 if self._timeout is None else self._timeout))

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
        status, data = self._file.read(
            start, stop - start, timeout=(0 if self._timeout is None else self._timeout)
        )
        if status.get("error", None):
            self._file.close(timeout=(0 if self._timeout is None else self._timeout))
            raise OSError(
                """XRootD error: {0}
in file {1}""".format(
                    status["message"], self._file_path
                )
            )
        return data

    @property
    def closed(self):
        """
        True if the associated file/connection/thread pool is closed; False
        otherwise.
        """
        return not self._file.is_open()


class XRootDSource(uproot4.source.chunk.Source):
    """
    Source managing data access using XRootD vector reads.
    """

    __slots__ = ["_file_path", "_max_num_elements", "_resource"]

    def __init__(self, file_path, **options):
        """
        Args:
            file_path (str): URL starting with "root://".
            timeout (int): Number of seconds (loosely interpreted by XRootD)
                before giving up on a remote file.
            max_num_elements (int): Maximum number of reads to batch into a
                single request. May be reduced to match the server's
                capabilities.
        """
        self._timeout = options["timeout"]
        max_num_elements = options["max_num_elements"]
        self._num_requests = 0
        self._num_requested_chunks = 0
        self._num_requested_bytes = 0

        self._file_path = file_path
        self._num_bytes = None

        # important: construct this first because it raises an error for nonexistent hosts
        self._resource = XRootDResource(file_path, self._timeout)

        # this comes after because it HANGS for nonexistent hosts
        self._max_num_elements, self._max_element_size = get_server_config(file_path)
        if max_num_elements:
            self._max_num_elements = min(self._max_num_elements, max_num_elements)

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
        self._resource.__exit__(exception_type, exception_value, traceback)

    @property
    def num_bytes(self):
        """
        The number of bytes in the file.
        """
        if self._num_bytes is None:
            status, info = self._resource._file.stat(
                timeout=(0 if self._timeout is None else self._timeout)
            )
            if not status["ok"]:
                raise OSError(
                    """XRootD error: {0}
in file {1}""".format(
                        status["message"], self._file_path
                    )
                )
            self._num_bytes = info["size"]

        return self._num_bytes

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

        Returns a single Chunk that has already been filled synchronously.
        """
        self._num_requests += 1
        self._num_requested_chunks += 1
        self._num_requested_bytes += stop - start

        data = self._resource.get(start, stop)
        future = uproot4.source.futures.TrivialFuture(data)
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

        Returns a list of Chunks that will be filled asynchronously by the
        one or more XRootD vector reads.
        """
        self._num_requests += 1
        self._num_requested_chunks += len(ranges)
        self._num_requested_bytes += sum(stop - start for start, stop in ranges)

        all_request_ranges = [[]]
        for start, stop in ranges:
            if stop - start > self._max_element_size:
                raise NotImplementedError(
                    "TODO: Probably need to fall back to a non-vector read"
                )
            if len(all_request_ranges[-1]) > self._max_num_elements:
                all_request_ranges.append([])
            all_request_ranges[-1].append((start, stop - start))

        chunks = []
        for i, request_ranges in enumerate(all_request_ranges):
            futures = {}
            for start, size in request_ranges:
                future = future = uproot4.source.futures.TaskFuture(None)
                futures[(start, size)] = future
                chunk = uproot4.source.chunk.Chunk(
                    self, start, start + size, future, exact
                )
                if notifications is not None:
                    future.add_done_callback(
                        uproot4.source.chunk.Resource.notifier(chunk, notifications)
                    )
                chunks.append(chunk)

            def _callback(status, response, hosts, futures=futures):
                for chunk in response["chunks"]:
                    future = futures[(chunk["offset"], chunk["length"])]
                    future._result = chunk["buffer"]
                    future._set_finished()

            status = self._resource._file.vector_read(
                chunks=request_ranges, callback=_callback
            )
            if not status["ok"]:
                raise OSError(
                    """XRootD error: {0}
in file {1}""".format(
                        status["message"], self._file_path
                    )
                )

        return chunks

    @property
    def closed(self):
        """
        True if the associated file/connection/thread pool is closed; False
        otherwise.
        """
        return self._resource.closed


class MultithreadedXRootDSource(uproot4.source.chunk.MultithreadedSource):
    """
    Source managing one synchronous or multiple asynchronous XRootD handles as
    a context manager.
    """

    __slots__ = ["_file_path", "_executor"]

    def __init__(self, file_path, **options):
        """
        Args:
            file_path (str): URL starting with "root://".
            num_workers (int): If 0, one synchronous ResourceExecutor is
                created; if 1 or more, a collection of asynchronous
                ThreadResourceExecutors are created.
            timeout (int): Number of seconds (loosely interpreted by XRootD)
                before giving up on a remote file.
        """
        timeout = options["timeout"]
        num_workers = options["num_workers"]
        self._num_requests = 0
        self._num_requested_chunks = 0
        self._num_requested_bytes = 0

        self._file_path = file_path
        self._resource = XRootDResource(file_path, timeout)
        self._num_bytes = None

        if num_workers == 0:
            self._executor = uproot4.source.futures.ResourceExecutor(self._resource)
        else:
            self._executor = uproot4.source.futures.ThreadResourceExecutor(
                [XRootDResource(file_path, timeout) for x in range(num_workers)]
            )

        self._timeout = timeout

    @property
    def timeout(self):
        """
        Number of seconds (loosely interpreted by XRootD) before giving up on a
        remote file.
        """
        return self._timeout

    @property
    def num_bytes(self):
        """
        The number of bytes in the file.
        """
        if self._num_bytes is None:
            status, info = self._resource._file.stat(
                timeout=(0 if self._timeout is None else self._timeout)
            )
            if not status["ok"]:
                raise OSError(
                    """XRootD error: {0}
in file {1}""".format(
                        status["message"], self._file_path
                    )
                )
            self._num_bytes = info["size"]

        return self._num_bytes

    @property
    def closed(self):
        """
        True if the associated file/connection/thread pool is closed; False
        otherwise.
        """
        return self._resource.closed

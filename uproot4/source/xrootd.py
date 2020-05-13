# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

"""
Source and Resource for XRootD (pyxrootd).
"""

from __future__ import absolute_import

import os

import uproot4.source.chunk
import uproot4.source.futures


def get_pyxrootd():
    os.environ["XRD_RUNFORKHANDLER"] = "1"  # set multiprocessing flag
    try:
        import pyxrootd.client

        return pyxrootd

    except ImportError:
        raise ImportError(
            """Install pyxrootd package with:

    conda install -c conda-forge xrootd

(or download from http://xrootd.org/dload.html and manually compile with """
            """cmake; setting PYTHONPATH and LD_LIBRARY_PATH appropriately)."""
        )


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
        pyxrootd = get_pyxrootd()
        self._file_path = file_path
        self._timeout = timeout

        self._file = pyxrootd.client.File()
        status, dummy = self._file.open(
            self._file_path, timeout=(0 if timeout is None else timeout)
        )
        if status.get("error", None):
            raise OSError(status["message"])

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
            raise OSError(status["message"])
        return data


class XRootDSource(uproot4.source.chunk.MultiThreadedSource):
    """
    Source managing one synchronous or multiple asynchronous XRootD handles as
    a context manager.
    """

    __slots__ = ["_file_path", "_executor"]

    def __init__(self, file_path, num_workers=0, timeout=None):
        """
        Args:
            file_path (str): URL starting with "root://".
            num_workers (int): If 0, one synchronous ResourceExecutor is
                created; if 1 or more, a collection of asynchronous
                ThreadResourceExecutors are created.
            timeout (int): Number of seconds (loosely interpreted by XRootD)
                before giving up on a remote file.
        """
        self._file_path = file_path

        if num_workers == 0:
            self._executor = uproot4.source.futures.ResourceExecutor(
                XRootDResource(file_path, timeout)
            )
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

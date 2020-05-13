# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

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
    __slots__ = ["_file_path", "_file"]

    def __init__(self, file_path, timeout):
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
        return self._file_path

    @property
    def timeout(self):
        return self._timeout

    @property
    def file(self):
        return self._file

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self._file.close(timeout=(0 if self._timeout is None else self._timeout))

    def get(self, start, stop):
        status, data = self._file.read(
            start, stop - start, timeout=(0 if self._timeout is None else self._timeout)
        )
        if status.get("error", None):
            raise OSError(status["message"])
        return data


class XRootDSource(uproot4.source.chunk.MultiThreadedSource):
    __slots__ = ["_file_path", "_executor"]

    def __init__(self, file_path, num_workers=0, timeout=None):
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
        return self._timeout

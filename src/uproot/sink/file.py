# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
This module defines a physical layer for writing local files.

Defines a :doc:`uproot.sink.file.FileSink`, which can wrap a Python file-like object
that has ``read``, ``write``, ``seek``, ``tell``, and ``flush`` methods and has a
context manager (Python's ``with`` statement) to ensure that files are properly closed
(although files are flushed after every object-write).
"""

from __future__ import annotations

import numbers
import os
from pathlib import Path
from typing import IO

import fsspec

import uproot._util


class FileSink:
    """
    Args:
        urlpath_or_file_like (str, Path, or file-like object): If a string or Path, a
            filesystem URL that specifies the file to open. If a file-like object, it
            must have ``read``, ``write``, ``seek``, ``tell``, and ``flush`` methods.

    An object that can write (and read) files on a local filesystem, which either opens
    a new file from a ``file_path`` in ``"r+b"`` mode or wraps a file-like object
    with the :ref:`uproot.sink.file.FileSink.from_object` constructor.

    With the ``file_path``-based constructor, the file is opened upon first read or
    write.
    """

    def __init__(self, urlpath_or_file_like: str | Path | IO, **storage_options):
        self._open_file = None
        self._file = None

        if uproot._util.is_file_like(
            urlpath_or_file_like, readable=False, writable=False, seekable=False
        ):
            self._file = urlpath_or_file_like

            if not uproot._util.is_file_like(
                self._file, readable=True, writable=True, seekable=True
            ):
                raise TypeError(
                    """writable file can only be created from a file path or an object that supports reading and writing"""
                )
        else:
            urlpath = str(urlpath_or_file_like)
            fs, local_path = fsspec.core.url_to_fs(urlpath, **storage_options)
            if not fs.exists(local_path):
                parent_directory = fs.sep.join(local_path.split(fs.sep)[:-1])
                fs.mkdirs(parent_directory, exist_ok=True)
                fs.touch(local_path, truncate=True)

            self._open_file = fsspec.open(urlpath, mode="r+b", **storage_options)

    @property
    def file_path(self) -> str | None:
        """
        A path to the file, which is None if constructed with a file-like object.
        """
        return self._open_file.path if self._open_file else None

    def _ensure(self):
        if not self._file:
            self._file = self._open_file.open()

        self._file.seek(0)

    def __getstate__(self):
        state = dict(self.__dict__)
        state.pop("_file")
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self._file = None

    def tell(self):
        """
        Calls the file or file-like object's ``tell`` method.
        """
        self._ensure()
        return self._file.tell()

    def flush(self):
        """
        Calls the file or file-like object's ``flush`` method.
        """
        self._ensure()
        return self._file.flush()

    @property
    def closed(self) -> bool:
        """
        True if the file is closed; False otherwise.
        """
        return self._file is None

    def close(self):
        """
        Closes the file (calls ``close`` if it has such a method) and sets it to
        None so that closure is permanent.
        """
        if self._file is not None:
            if hasattr(self._file, "close"):
                self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()

    @property
    def in_path(self) -> str:
        return f"\n\nin path: {self.file_path}" if self.file_path is not None else ""

    def write(self, location, serialization):
        """
        Args:
            location (int): Position in the file to write.
            serialization (bytes or NumPy array): Data to write to the file.

        Writes data to the file at a specific location, calling the file-like
        object's ``seek`` and ``write`` methods.
        """
        self._ensure()
        self._file.seek(location)
        self._file.write(serialization)

    def set_file_length(self, length):
        """
        Sets the file's length to ``length``, truncating with zeros if necessary.

        Calls ``seek``, ``tell``, and possibly ``write``.
        """
        self._ensure()
        # self._file.truncate(length)

        self._file.seek(0, os.SEEK_END)
        missing = length - self._file.tell()
        if missing > 0:
            self._file.write(b"\x00" * missing)

    def read(self, location, num_bytes, insist=True) -> bytes:
        """
        Args:
            location (int): Position in the file to read.
            num_bytes (int): Number of bytes to read from the file.
            insist (bool): If True, raise an OSError if ``num_bytes`` cannot be read
                from the file. Otherwise, this function may return data with fewer
                than ``num_bytes``.

        Returns a bytes object of data from the file by calling the file-like
        object's ``seek`` and ``read`` methods. The ``insist`` parameter can be
        used to ensure that the output has the requested length.
        """
        self._ensure()
        self._file.seek(location)
        out = self._file.read(num_bytes)
        if insist is True:
            if len(out) != num_bytes:
                raise OSError(
                    "could not read {} bytes from the file at position {}{}".format(
                        num_bytes,
                        location,
                        self.in_path,
                    )
                )
        elif isinstance(insist, numbers.Integral) and len(out) < insist:
            raise OSError(
                "could not read {} bytes from the file at position {}{}".format(
                    insist,
                    location,
                    self.in_path,
                )
            )
        return out

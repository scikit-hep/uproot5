# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

"""
This module defines a physical layer for writing local files.

Defines a :doc:`uproot.sink.file.FileSink`, which can wrap a Python file-like object
that has ``read``, ``write``, ``seek``, ``tell``, and ``flush`` methods and has a
context manager (Python's ``with`` statement) to ensure that files are properly closed
(although files are flushed after every object-write).
"""


import numbers
import os


class FileSink:
    """
    Args:
        file_path (str): The filesystem path of the file to open.

    An object that can write (and read) files on a local filesystem, which either opens
    a new file from a ``file_path`` in ``"r+b"`` mode or wraps a file-like object
    with the :ref:`uproot.sink.file.FileSink.from_object` constructor.

    With the ``file_path``-based constructor, the file is opened upon first read or
    write.
    """

    @classmethod
    def from_object(cls, obj):
        """
        Args:
            obj (file-like object): An object with ``read``, ``write``, ``seek``,
                ``tell``, and ``flush`` methods.

        Creates a :doc:`uproot.sink.file.FileSink` from a file-like object, such
        as ``io.BytesIO``. The object must be readable, writable, and seekable
        with ``"r+b"`` mode semantics.
        """
        if (
            callable(getattr(obj, "read", None))
            and callable(getattr(obj, "write", None))
            and callable(getattr(obj, "seek", None))
            and callable(getattr(obj, "tell", None))
            and callable(getattr(obj, "flush", None))
            and (not hasattr(obj, "readable") or obj.readable())
            and (not hasattr(obj, "writable") or obj.writable())
            and (not hasattr(obj, "seekable") or obj.seekable())
        ):
            self = cls(None)
            self._file = obj
        else:
            raise TypeError(
                """writable file can only be created from a file path or an object

    * that has 'read', 'write', 'seek', and 'tell' methods
    * is 'readable() and writable() and seekable()'"""
            )
        return self

    def __init__(self, file_path):
        self._file_path = file_path
        self._file = None

    @property
    def file_path(self):
        """
        A path to the file, which is None if constructed with a file-like object.
        """
        return self._file_path

    def _ensure(self):
        if self._file is None:
            if self._file_path is None:
                raise TypeError("FileSink created from an object cannot be reopened")
            self._file = open(self._file_path, "r+b")
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
    def closed(self):
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
    def in_path(self):
        if self._file_path is None:
            return ""
        else:
            return "\n\nin path: " + self._file_path

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

    def read(self, location, num_bytes, insist=True):
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
        elif isinstance(insist, numbers.Integral):
            if len(out) < insist:
                raise OSError(
                    "could not read {} bytes from the file at position {}{}".format(
                        insist,
                        location,
                        self.in_path,
                    )
                )
        return out

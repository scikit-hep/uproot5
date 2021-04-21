# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

"""
FIXME: docstring
"""

from __future__ import absolute_import

import numbers
import os


class FileSink(object):
    """
    FIXME: docstring
    """

    @classmethod
    def from_object(cls, obj):
        """
        FIXME: docstring
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

    def __init__(self, file_path):
        self._file_path = file_path
        self._file = None

    @property
    def file_path(self):
        """
        FIXME: docstring
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
        FIXME: docstring
        """
        self._ensure()
        return self._file.tell()

    def flush(self):
        """
        FIXME: docstring
        """
        self._ensure()
        return self._file.flush()

    @property
    def closed(self):
        """
        FIXME: docstring
        """
        return self._file is None

    def close(self):
        """
        FIXME: docstring
        """
        if self._file is not None:
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

    @property
    def position(self):
        """
        FIXME: docstring
        """
        self._ensure()
        return self._file.tell()

    def write(self, location, serialization):
        """
        FIXME: docstring
        """
        self._ensure()
        self._file.seek(location)
        self._file.write(serialization)

    def set_file_length(self, length):
        """
        FIXME: docstring
        """
        self._ensure()
        # self._file.truncate(length)

        self._file.seek(0, os.SEEK_END)
        missing = length - self._file.tell()
        assert missing >= 0
        if missing > 0:
            self._file.write(b"\x00" * missing)

    def read(self, location, num_bytes, insist=True):
        """
        FIXME: docstring
        """
        self._ensure()
        self._file.seek(location)
        out = self._file.read(num_bytes)
        if insist is True:
            if len(out) != num_bytes:
                raise OSError(
                    "could not read {0} bytes from the file at position {1}{2}".format(
                        num_bytes,
                        location,
                        self.in_path,
                    )
                )
        elif isinstance(insist, numbers.Integral):
            if len(out) < insist:
                raise OSError(
                    "could not read {0} bytes from the file at position {1}{2}".format(
                        insist,
                        location,
                        self.in_path,
                    )
                )
        return out

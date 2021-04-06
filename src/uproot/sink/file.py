# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

"""
FIXME: docstring
"""

from __future__ import absolute_import

import os
import struct

_string_size_format_4 = struct.Struct(">I")


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
            and obj.readable()
            and obj.writable()
            and obj.seekable()
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
            if not os.path.exists(self._file_path):
                with open(self._file_path, "a"):
                    pass
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

        (flush is only ever user-initiated)
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

    def write(self, location, serialization):
        """
        FIXME: docstring
        """
        self._ensure()
        self._file.seek(location)
        self._file.write(serialization)

    # def move_to(self, index, end=False):
    #     """
    #     FIXME: docstring
    #     """
    #     self._ensure()
    #     if end:
    #         self._file.seek(index, os.SEEK_END)
    #     else:
    #         self._file.seek(index)

    # def skip(self, num_bytes):
    #     """
    #     FIXME: docstring
    #     """
    #     self._ensure()
    #     self._file.seek(num_bytes, os.SEEK_CUR)

    # def read_raw(self, num_bytes):
    #     """
    #     FIXME: docstring
    #     """
    #     self._ensure()
    #     return self._file.read(num_bytes)

    # def write_raw(self, data):
    #     """
    #     FIXME: docstring
    #     """
    #     self._ensure()
    #     return self._file.write(data)

    # def read_fields(self, format):
    #     """
    #     FIXME: docstring
    #     """
    #     self._ensure()
    #     return format.unpack(self._file.read(format.size))

    # def write_fields(self, format, *fields):
    #     """
    #     FIXME: docstring
    #     """
    #     self._ensure()
    #     self._file.write(format.pack(fields))

    # def read_bytestring(self):
    #     """
    #     FIXME: docstring
    #     """
    #     self._ensure()
    #     num_bytes = ord(self._file.read(1))
    #     if num_bytes == 255:
    #         (num_bytes,) = _string_size_format_4.unpack(self._file.read(4))
    #     return self._file.read(num_bytes)

    # def write_bytestring(self, bytestring):
    #     """
    #     FIXME: docstring
    #     """
    #     self._ensure()
    #     num_bytes = len(bytestring)
    #     if num_bytes < 255:
    #         self._file.write(struct.pack(">B%ds" % num_bytes, num_bytes, bytestring))
    #     else:
    #         self._file.write(
    #             struct.pack(">BI%ds" % num_bytes, 255, num_bytes, bytestring)
    #         )

    # @staticmethod
    # def bytestring_footprint(bytestring):
    #     """
    #     FIXME: docstring
    #     """
    #     num_bytes = len(bytestring)
    #     if num_bytes < 255:
    #         return 1 + num_bytes
    #     else:
    #         return 5 + num_bytes

    # def read_string(self):
    #     """
    #     FIXME: docstring
    #     """
    #     if uproot._util.py2:
    #         return self.read_bytestring()
    #     else:
    #         return self.read_bytestring().decode(errors="surrogateescape")

    # def write_string(self, string):
    #     """
    #     FIXME: docstring
    #     """
    #     if uproot._util.py2:
    #         self.write_bytestring(string)
    #     else:
    #         self.write_bytestring(string.encode(errors="surrogateescape"))

    # @staticmethod
    # def string_footprint(string):
    #     """
    #     FIXME: docstring
    #     """
    #     return FileSink.bytestring_footprint(string.encode(errors="surrogateescape"))

    # def read_classname(self):
    #     """
    #     FIXME: docstring
    #     """
    #     self._ensure()
    #     char = None
    #     out = []
    #     while char != b"\x00":
    #         char = self._file.read(1)
    #         if char == b"":
    #             raise OSError(
    #                 "C-style string has no terminator (null byte)"
    #                 + (
    #                     ""
    #                     if self._file_path is None
    #                     else "\n\nin file path " + self._file_path
    #                 )
    #             )
    #         out.append(char)

    #     if uproot._util.py2:
    #         return b"".join(out)
    #     else:
    #         return b"".join(out).decode(errors="surrogateescape")

    # def write_classname(self, string):
    #     """
    #     FIXME: docstring
    #     """
    #     self._ensure()
    #     if uproot._util.py2:
    #         bytestring = string.encode(errors="surrogateescape")
    #     else:
    #         bytestring = string
    #     self._file.write(bytestring)
    #     self._file.write(b"\x00")

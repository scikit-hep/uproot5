# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

"""
FIXME: docstring
"""

from __future__ import absolute_import

import os
import struct

import uproot._util

_string_size_format_4 = struct.Struct(">I")


class FileSink(object):
    """
    FIXME: docstring
    """

    def __init__(self, file_path):
        self._file_path = file_path
        self._file = None

    @property
    def file_path(self):
        return self._file_path

    def _ensure(self):
        if self._file is None:
            self._file = open(self._file_path, "r+b")
            self._file.seek(0)

    def move_to(self, index, end=False):
        self._ensure()
        if end:
            self._file.seek(index, os.SEEK_END)
        else:
            self._file.seek(index)

    def skip(self, num_bytes):
        self._ensure()
        self._file.seek(num_bytes, os.SEEK_CUR)

    def read_fields(self, format):
        self._ensure()
        return format.unpack(self._file.read(format.size))

    def write_fields(self, format, *fields):
        self._ensure()
        self._file.write(format.pack(fields))

    def read_bytestring(self):
        self._ensure()
        num_bytes = ord(self._file.read(1))
        if num_bytes == 255:
            (num_bytes,) = _string_size_format_4.unpack(self._file.read(4))
        return self._file.read(num_bytes)

    def write_bytestring(self, bytestring):
        self._ensure()
        num_bytes = len(bytestring)
        if num_bytes < 255:
            self._file.write(struct.pack(">B%ds" % num_bytes, num_bytes, bytestring))
        else:
            self._file.write(
                struct.pack(">BI%ds" % num_bytes, 255, num_bytes, bytestring)
            )

    @staticmethod
    def bytestring_footprint(bytestring):
        num_bytes = len(bytestring)
        if num_bytes < 255:
            return 1 + num_bytes
        else:
            return 5 + num_bytes

    def read_string(self):
        if uproot._util.py2:
            return self.read_bytestring()
        else:
            return self.read_bytestring().decode(errors="surrogateescape")

    def write_string(self, string):
        if uproot._util.py2:
            self.write_bytestring(string)
        else:
            self.write_bytestring(string.encode(errors="surrogateescape"))

    @staticmethod
    def string_footprint(string):
        return FileSink.bytestring_footprint(string.encode(errors="surrogateescape"))

    def read_classname(self):
        self._ensure()
        char = None
        out = []
        while char != b"\x00":
            char = self._file.read(1)
            if char == b"":
                raise OSError(
                    """C-style string has no terminator (null byte)
in file path {0}""".format(
                        self._file_path
                    )
                )
            out.append(char)

        if uproot._util.py2:
            return b"".join(out)
        else:
            return b"".join(out).decode(errors="surrogateescape")

    def write_classname(self, string):
        self._ensure()
        if uproot._util.py2:
            bytestring = string.encode(errors="surrogateescape")
        else:
            bytestring = string
        self._file.write(bytestring)
        self._file.write(b"\x00")

    @property
    def closed(self):
        return self._file is None

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()

    def __getstate__(self):
        state = dict(self.__dict__)
        state.pop("_file")
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self._file = None

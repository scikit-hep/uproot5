# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import struct
import threading

import uproot4._util
import uproot4.compression
import uproot4.source.cursor
import uproot4.source.chunk
import uproot4.source.memmap
import uproot4.source.http
import uproot4.source.xrootd


_header_fields_small = struct.Struct(">4siiiiiiiBiii18s")
_header_fields_big = struct.Struct(">4siiqqiiiBiqi18s")


class ReadOnlyFile(object):
    defaults = {
        "file_handler": uproot4.source.memmap.MemmapSource,
        "xrootd_handler": uproot4.source.xrootd.XRootDSource,
        "http_handler": uproot4.source.http.HTTPSource,
        "timeout": 30,
        "max_num_elements": None,
        "num_workers": 10,
        "num_fallback_workers": 10,
    }

    def __init__(self, file_path, **options):
        all_options = dict(self.defaults)
        all_options.update(options)

        self._file_path = file_path

        Source = uproot4._util.path_to_source_class(file_path, all_options)
        self._source = Source(file_path, **all_options)
        chunk = self._source.chunk(0, _header_fields_big.size)

        (
            magic,
            self._fVersion,
            self._fBEGIN,
            self._fEND,
            self._fSeekFree,
            self._fNbytesFree,
            self._nfree,
            self._fNbytesName,
            self._fUnits,
            self._fCompress,
            self._fSeekInfo,
            self._fNbytesInfo,
            self._fUUID,
        ) = uproot4.source.cursor.Cursor(0).fields(chunk, _header_fields_small)

        if self._fVersion >= 1000000:
            (
                magic,
                self._fVersion,
                self._fBEGIN,
                self._fEND,
                self._fSeekFree,
                self._fNbytesFree,
                self._nfree,
                self._fNbytesName,
                self._fUnits,
                self._fCompress,
                self._fSeekInfo,
                self._fNbytesInfo,
                self._fUUID,
            ) = uproot4.source.cursor.Cursor(0).fields(chunk, _header_fields_big)

        if magic != b"root":
            raise ValueError(
                """not a ROOT file: first four bytes are {0}
in file {1}""".format(
                    repr(magic), file_path
                )
            )

        self._streamers = {}

    @property
    def file_path(self):
        return self._file_path

    @property
    def source(self):
        return self._source

    @property
    def streamers(self):
        return self._streamers

    def __repr__(self):
        return "<ReadOnlyFile {0}>".format(repr(self._file_path))

    @property
    def root_version_tuple(self):
        version = self._fVersion
        if version >= 1000000:
            version -= 1000000

        major = version // 10000
        version %= 10000
        minor = version // 100
        version %= 100

        return major, minor, version

    @property
    def root_version(self):
        return "{0}.{1:02d}/{2:02d}".format(*self.root_version_tuple)

    @property
    def compression(self):
        return uproot4.compression.Compression.from_code(self._fCompress)

    @property
    def uuid(self):
        return self._fUUID

    @property
    def fVersion(self):
        return self._fVersion

    @property
    def fBEGIN(self):
        return self._fBEGIN

    @property
    def fEND(self):
        return self._fEND

    @property
    def fSeekFree(self):
        return self._fSeekFree

    @property
    def fNbytesFree(self):
        return self._fNbytesFree

    @property
    def nfree(self):
        return self._nfree

    @property
    def fNbytesName(self):
        return self._fNbytesName

    @property
    def fUnits(self):
        return self._fUnits

    @property
    def fCompress(self):
        return self._fCompress

    @property
    def fSeekInfo(self):
        return self._fSeekInfo

    @property
    def fNbytesInfo(self):
        return self._fNbytesInfo

    @property
    def fUUID(self):
        return self._fUUID

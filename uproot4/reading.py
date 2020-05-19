# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import struct
import uuid

import uproot4._util
import uproot4.compression
import uproot4.source.cursor
import uproot4.source.chunk
import uproot4.source.memmap
import uproot4.source.http
import uproot4.source.xrootd


class ReadOnlyFile(object):
    defaults = {
        "file_handler": uproot4.source.memmap.MemmapSource,
        "xrootd_handler": uproot4.source.xrootd.XRootDSource,
        "http_handler": uproot4.source.http.HTTPSource,
        "timeout": 30,
        "max_num_elements": None,
        "num_workers": 10,
        "num_fallback_workers": 10,
        "begin_guess_bytes": 512,
        "end_guess_bytes": 64 * 1024,
        "streamer_guess_bytes": 64 * 1024,
    }

    _header_fields_small = struct.Struct(">4siiiiiiiBiiiH16s")
    _header_fields_big = struct.Struct(">4siiqqiiiBiqiH16s")

    def __init__(self, file_path, cache=None, **options):
        self._options = dict(self.defaults)
        self._options.update(options)

        self._file_path = file_path
        self._streamer_key = None
        self._streamers = None

        self.cache = cache

        self.hook_before_create_source(file_path=file_path, options=self._options)

        Source = uproot4._util.path_to_source_class(file_path, self._options)
        self._source = Source(file_path, **self._options)

        self.hook_before_get_chunks(file_path=file_path, options=self._options)

        if self._options["begin_guess_bytes"] < self._header_fields_big.size:
            raise ValueError(
                "begin_guess_bytes={0} is not enough to read the TFile header ({1})".format(
                    self._options["begin_guess_bytes"], self._header_fields_big.size
                )
            )

        self._begin_chunk, self._end_chunk = self._source.begin_end_chunks(
            self._options["begin_guess_bytes"], self._options["end_guess_bytes"]
        )

        self.hook_before_read(file_path=file_path, options=self._options)

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
            self._fUUID_version,
            self._fUUID,
        ) = uproot4.source.cursor.Cursor(0).fields(
            self._begin_chunk, self._header_fields_small
        )

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
                self._fUUID_version,
                self._fUUID,
            ) = uproot4.source.cursor.Cursor(0).fields(
                self._begin_chunk, self._header_fields_big
            )

        if magic != b"root":
            raise ValueError(
                """not a ROOT file: first four bytes are {0}
in file {1}""".format(
                    repr(magic), file_path
                )
            )

        self.hook_before_root_directory(file_path=file_path, options=self._options)

        self._root_directory = ReadOnlyDirectory(
            "/",
            uproot4.source.cursor.Cursor(self._fBEGIN + self._fNbytesName),
            self,
            self,
            self._options,
        )

        self.hook_after_root_directory(file_path=file_path, options=self._options)

    def __repr__(self):
        return "<ReadOnlyFile {0}>".format(repr(self._file_path))

    def hook_before_create_source(self, **kwargs):
        pass

    def hook_before_get_chunks(self, **kwargs):
        pass

    def hook_before_read(self, **kwargs):
        pass

    def hook_before_root_directory(self, **kwargs):
        pass

    def hook_after_root_directory(self, **kwargs):
        pass

    def hook_before_read_streamer_key(self, **kwargs):
        pass

    def hook_before_read_streamers(self, **kwargs):
        pass

    def hook_after_read_streamers(self, **kwargs):
        pass

    @property
    def options(self):
        return self._options

    @property
    def file_path(self):
        return self._file_path

    @property
    def cache(self):
        return self._cache

    @cache.setter
    def cache(self, value):
        self._cache = value

    @property
    def source(self):
        return self._source

    @property
    def begin_chunk(self):
        return self._begin_chunk

    @property
    def end_chunk(self):
        return self._end_chunk

    @property
    def streamers(self):
        if self._streamers is None:
            if self._fSeekInfo == 0:
                self._streamers = {}

            else:
                if self._options["streamer_guess_bytes"] < ReadOnlyKey._format_big.size:
                    raise ValueError(
                        "streamer_guess_bytes={0} is not enough to read the streamer TKey ({1})".format(
                            self._options["streamer_guess_bytes"],
                            ReadOnlyKey._format_big.size,
                        )
                    )
                streamer_start = self._fSeekInfo
                streamer_stop = min(
                    self._fSeekInfo + self._options["streamer_guess_bytes"], self._fEND
                )

                if (streamer_start, streamer_stop) in self._end_chunk:
                    chunk = self._end_chunk
                elif (streamer_start, streamer_stop) in self._begin_chunk:
                    chunk = self._begin_chunk
                else:
                    chunk = self._source.chunk(
                        streamer_start, streamer_stop, exact=False
                    )

                self.hook_before_read_streamer_key(chunk=chunk)

                self._streamer_key = ReadOnlyKey(
                    uproot4.source.cursor.Cursor(self._fSeekInfo),
                    chunk,
                    self,
                    self,
                    self._options,
                )

                if self._streamer_key.fNbytes > streamer_stop - streamer_start:
                    chunk = self._source.chunk(
                        streamer_start, streamer_start + self._streamer_key.fNbytes
                    )

                self.hook_before_read_streamers(
                    chunk=chunk, streamer_key=self._streamer_key
                )

                # TODO: read streamers here
                self._streamers = {}

                self.hook_after_read_streamers(
                    chunk=chunk,
                    streamer_key=self._streamer_key,
                    streamers=self._streamers,
                )

        return self._streamers

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
    def is_64bit(self):
        return self._fVersion >= 1000000

    @property
    def compression(self):
        return uproot4.compression.Compression.from_code(self._fCompress)

    @property
    def hex_uuid(self):
        if uproot4._util.py2:
            out = "".join("{0:02x}".format(ord(x)) for x in self._fUUID)
        else:
            out = "".join("{0:02x}".format(x) for x in self._fUUID)
        return "-".join([out[0:8], out[8:12], out[12:16], out[16:20], out[20:32]])

    @property
    def uuid(self):
        return uuid.UUID(self.hex_uuid.replace("-", ""))

    @property
    def streamer_key(self):
        return self._streamer_key

    @property
    def root_directory(self):
        return self._root_directory

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


class ReadOnlyKey(object):
    _encoded_classname = "ROOT_TKey"

    _format_small = struct.Struct(">ihiIhhii")
    _format_big = struct.Struct(">ihiIhhqq")

    def __init__(self, cursor, chunk, file, parent, options, read_strings=False):
        self._cursor = cursor.copy(link_refs=True)
        self._file = file
        self._parent = parent

        self.hook_before_read(
            cursor=cursor,
            chunk=chunk,
            file=file,
            parent=parent,
            options=options,
            read_strings=read_strings,
        )

        (
            self._fNbytes,
            self._fVersion,
            self._fObjlen,
            self._fDatime,
            self._fKeylen,
            self._fCycle,
            self._fSeekKey,
            self._fSeekPdir,
        ) = cursor.fields(chunk, self._format_small, move=False)

        if self._fVersion > 1000:
            (
                self._fNbytes,
                self._fVersion,
                self._fObjlen,
                self._fDatime,
                self._fKeylen,
                self._fCycle,
                self._fSeekKey,
                self._fSeekPdir,
            ) = cursor.fields(chunk, self._format_big)

        else:
            cursor.skip(self._format_small.size)

        if read_strings:
            self.hook_before_read_strings(
                cursor=cursor,
                chunk=chunk,
                file=file,
                parent=parent,
                options=options,
                read_strings=read_strings,
            )

            self._fClassName = cursor.string(chunk)
            self._fName = cursor.string(chunk)
            self._fTitle = cursor.string(chunk)

        else:
            self._fClassName = None
            self._fName = None
            self._fTitle = None

        self.hook_after_read(
            cursor=cursor,
            chunk=chunk,
            file=file,
            parent=parent,
            options=options,
            read_strings=read_strings,
        )

    def __repr__(self):
        if self._fName is None or self._fClass is None:
            nameclass = ""
        else:
            nameclass = " {0}: {1}".format(self._fName, self._fClassName)
        return "<ReadOnlyKey{0} at {1}>".format(nameclass, self.data_cursor.index)

    def hook_before_read(self, **kwargs):
        pass

    def hook_before_read_strings(self, **kwargs):
        pass

    def hook_after_read(self, **kwargs):
        pass

    @property
    def data_cursor(self):
        return uproot4.source.cursor.Cursor(self._fSeekKey + self._fKeylen)

    @property
    def data_compressed_bytes(self):
        return self._fNbytes - self._fKeylen

    @property
    def data_uncompressed_bytes(self):
        return self._fObjlen

    @property
    def is_compressed(self):
        return self.data_compressed_bytes != self.data_uncompressed_bytes

    @property
    def fNbytes(self):
        return self._fNbytes

    @property
    def fVersion(self):
        return self._fVersion

    @property
    def fObjlen(self):
        return self._fObjlen

    @property
    def fDatime(self):
        return self._fDatime

    @property
    def fKeylen(self):
        return self._fKeylen

    @property
    def fCycle(self):
        return self._fCycle

    @property
    def fSeekKey(self):
        return self._fSeekKey

    @property
    def fSeekPdir(self):
        return self._fSeekPdir

    @property
    def fClassName(self):
        return self._fClassName

    @property
    def fName(self):
        return self._fName

    @property
    def fTitle(self):
        return self._fTitle


class ReadOnlyDirectory(object):
    _encoded_classname = "ROOT_TDirectory"

    _format_small = struct.Struct(">hIIiiiii")
    _format_big = struct.Struct(">hIIiiqqq")
    _format_num_keys = struct.Struct(">i")

    def __init__(self, name, cursor, file, parent, options):
        self._name = name
        self._cursor = cursor.copy(link_refs=True)
        self._file = file
        self._parent = parent

        directory_start = cursor.index
        directory_stop = min(directory_start + self._format_big.size, file.fEND)

        if (directory_start, directory_stop) in file.begin_chunk:
            chunk = file.begin_chunk
        elif (directory_start, directory_stop) in file.end_chunk:
            chunk = file.end_chunk
        else:
            chunk = file.source.chunk(directory_start, directory_stop, exact=False)

        self.hook_before_read(
            name=name,
            cursor=cursor,
            chunk=chunk,
            file=file,
            parent=parent,
            options=options,
        )

        (
            self._fVersion,
            self._fDatimeC,
            self._fDatimeM,
            self._fNbytesKeys,
            self._fNbytesName,
            self._fSeekDir,
            self._fSeekParent,
            self._fSeekKeys,
        ) = cursor.fields(chunk, self._format_small, move=False)

        if self._fVersion > 1000:
            (
                self._fVersion,
                self._fDatimeC,
                self._fDatimeM,
                self._fNbytesKeys,
                self._fNbytesName,
                self._fSeekDir,
                self._fSeekParent,
                self._fSeekKeys,
            ) = cursor.fields(chunk, self._format_big)

        else:
            cursor.skip(self._format_small.size)

        if self._fSeekKeys == 0:
            self._header_key = None
            self._keys = []

        else:
            keys_start = self._fSeekKeys
            keys_stop = min(keys_start + self._fNbytesKeys + 8, file.fEND)

            if (keys_start, keys_stop) in file.end_chunk:
                keys_chunk = file.end_chunk
            elif (keys_start, keys_stop) in file.begin_chunk:
                keys_chunk = file.begin_chunk
            elif (keys_start, keys_stop) in chunk:
                keys_chunk = chunk
            else:
                keys_chunk = file.source.chunk(keys_start, keys_stop)

            keys_cursor = uproot4.source.cursor.Cursor(self._fSeekKeys)

            self.hook_before_header_key(
                name=name,
                cursor=cursor,
                chunk=chunk,
                file=file,
                parent=parent,
                options=options,
                keys_chunk=keys_chunk,
                keys_cursor=keys_cursor,
            )

            self._header_key = ReadOnlyKey(
                keys_cursor, keys_chunk, file, self, options, read_strings=True
            )

            num_keys = keys_cursor.field(keys_chunk, self._format_num_keys)

            self.hook_before_keys(
                name=name,
                cursor=cursor,
                chunk=chunk,
                file=file,
                parent=parent,
                options=options,
                keys_chunk=keys_chunk,
                keys_cursor=keys_cursor,
                num_keys=num_keys,
            )

            self._keys = []
            for i in range(num_keys):
                key = ReadOnlyKey(
                    keys_cursor, keys_chunk, file, self, options, read_strings=True
                )
                self._keys.append(key)

            self.hook_after_read(
                name=name,
                cursor=cursor,
                chunk=chunk,
                file=file,
                parent=parent,
                options=options,
                keys_chunk=keys_chunk,
                keys_cursor=keys_cursor,
                num_keys=num_keys,
            )

    def __repr__(self):
        return "<ReadOnlyDirectory {0}>".format(self._name)

    def hook_before_read(self, **kwargs):
        pass

    def hook_before_header_key(self, **kwargs):
        pass

    def hook_before_keys(self, **kwargs):
        pass

    def hook_after_read(self, **kwargs):
        pass

    @property
    def header_key(self):
        return self._header_key

    @property
    def keys(self):
        return self._keys

    @property
    def fVersion(self):
        return self._fVersion

    @property
    def fDatimeC(self):
        return self._fDatimeC

    @property
    def fDatimeM(self):
        return self._fDatimeM

    @property
    def fNbytesKeys(self):
        return self._fNbytesKeys

    @property
    def fNbytesName(self):
        return self._fNbytesName

    @property
    def fSeekDir(self):
        return self._fSeekDir

    @property
    def fSeekParent(self):
        return self._fSeekParent

    @property
    def fSeekKeys(self):
        return self._fSeekKeys

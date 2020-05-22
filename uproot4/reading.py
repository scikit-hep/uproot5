# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

"""
Defines the most basic functions for reading ROOT files: ReadOnlyFile,
ReadOnlyKey, and ReadOnlyDirectory, as well as the uproot.open function.
"""

from __future__ import absolute_import

import struct
import uuid

try:
    from collections.abc import Mapping
    from collections.abc import MutableMapping
except ImportError:
    from collections import Mapping
    from collections import MutableMapping

import uproot4._util
import uproot4.compression
import uproot4.source.cursor
import uproot4.source.chunk
import uproot4.source.memmap
import uproot4.source.http
import uproot4.source.xrootd
import uproot4.streamers
import uproot4.model


def open(file_path, cache=None, classes=None, **options):
    """
    Args:
        file_path (str or Path): File path or URL to open.
        cache (None or MutableMapping): FIXME not implemented.
        classes (None or MutableMapping): If None, defaults to uproot4.classes;
            otherwise, a container of class definitions that is both used to
            fill with new classes and search for dependencies.
        options: see below.

    Opens a ROOT file, possibly through a remote protocol.

    Options (type; default):

        * file_handler (Source class; uproot4.source.memmap.MemmapSource)
        * xrootd_handler (Source class; uproot4.source.xrootd.XRootDSource)
        * http_handler (Source class; uproot4.source.http.HTTPSource)
        * timeout (float for HTTP, int for XRootD; 30)
        * max_num_elements (None or int; None)
        * num_workers (int; 10)
        * num_fallback_workers (int; 10)
        * begin_guess_bytes (memory_size; 512)
        * end_guess_bytes (memory_size; "64 kB")
        * streamer_guess_bytes (memory_size; "64 kB")
    """

    file = ReadOnlyFile(file_path, cache=cache, classes=classes, **options)
    return file.root_directory


open.defaults = {
    "file_handler": uproot4.source.memmap.MemmapSource,
    "xrootd_handler": uproot4.source.xrootd.XRootDSource,
    "http_handler": uproot4.source.http.HTTPSource,
    "timeout": 30,
    "max_num_elements": None,
    "num_workers": 10,
    "num_fallback_workers": 10,
    "begin_guess_bytes": 512,
    "end_guess_bytes": "64 kB",
}


def no_filter(x):
    return True


_file_header_fields_small = struct.Struct(">4siiiiiiiBiiiH16s")
_file_header_fields_big = struct.Struct(">4siiqqiiiBiqiH16s")


class ReadOnlyFile(object):
    def __init__(self, file_path, cache=None, classes=None, **options):
        self._file_path = file_path
        self.cache = cache
        self.classes = classes

        self._options = dict(open.defaults)
        self._options.update(options)
        for option in ("begin_guess_bytes", "end_guess_bytes"):
            self._options[option] = uproot4._util.memory_size(self._options[option])

        self._streamers = None

        self.hook_before_create_source()

        Source = uproot4._util.path_to_source_class(file_path, self._options)
        self._source = Source(file_path, **self._options)

        self.hook_before_get_chunks()

        if self._options["begin_guess_bytes"] < _file_header_fields_big.size:
            raise ValueError(
                "begin_guess_bytes={0} is not enough to read the TFile header ({1})".format(
                    self._options["begin_guess_bytes"],
                    self._file_header_fields_big.size,
                )
            )

        self._begin_chunk, self._end_chunk = self._source.begin_end_chunks(
            self._options["begin_guess_bytes"], self._options["end_guess_bytes"]
        )

        self.hook_before_read()

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
            self._begin_chunk, _file_header_fields_small
        )

        if self.is_64bit:
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
                self._begin_chunk, _file_header_fields_big
            )

        self.hook_after_read(magic=magic)

        if magic != b"root":
            raise ValueError(
                """not a ROOT file: first four bytes are {0}
in file {1}""".format(
                    repr(magic), file_path
                )
            )

    def __repr__(self):
        return "<ReadOnlyFile {0}>".format(repr(self._file_path))

    def hook_before_create_source(self, **kwargs):
        pass

    def hook_before_get_chunks(self, **kwargs):
        pass

    def hook_before_read(self, **kwargs):
        pass

    def hook_after_read(self, **kwargs):
        pass

    def hook_before_read_streamer_key(self, **kwargs):
        pass

    def hook_before_decompress_streamers(self, **kwargs):
        pass

    def hook_before_read_streamers(self, **kwargs):
        pass

    def hook_after_read_streamers(self, **kwargs):
        pass

    @property
    def file_path(self):
        return self._file_path

    @property
    def cache(self):
        return self._cache

    @cache.setter
    def cache(self, value):
        if value is None or isinstance(value, MutableMapping):
            self._cache = value
        else:
            raise TypeError("cache must be None or a MutableMapping")

    @property
    def classes(self):
        return self._classes

    @classes.setter
    def classes(self, value):
        if value is None:
            self._classes = uproot4.classes
        elif isinstance(value, MutableMapping):
            self._classes = value
        else:
            raise TypeError("classes must be None or a MutableMapping")

    @property
    def options(self):
        return self._options

    @property
    def source(self):
        return self._source

    @property
    def begin_chunk(self):
        return self._begin_chunk

    @property
    def end_chunk(self):
        return self._end_chunk

    def chunk(self, start, stop):
        if (start, stop) in self._end_chunk:
            return self._end_chunk
        elif (start, stop) in self._begin_chunk:
            return self._begin_chunk
        else:
            return self._source.chunk(start, stop)

    @property
    def root_directory(self):
        return ReadOnlyDirectory(
            (),
            uproot4.source.cursor.Cursor(self._fBEGIN + self._fNbytesName),
            self,
            self,
            self._options,
        )

    @property
    def streamers(self):
        import uproot4.streamers
        import uproot4.models.TList
        import uproot4.models.TObjArray
        import uproot4.models.TObjString

        if self._streamers is None:
            if self._fSeekInfo == 0:
                self._streamers = {}

            else:
                key_cursor = uproot4.source.cursor.Cursor(self._fSeekInfo)
                key_start = self._fSeekInfo
                key_stop = min(
                    self._fSeekInfo + ReadOnlyKey._format_big.size, self._fEND
                )
                key_chunk = self.chunk(key_start, key_stop)

                self.hook_before_read_streamer_key(
                    key_chunk=key_chunk, key_cursor=key_cursor,
                )

                streamer_key = ReadOnlyKey(
                    key_chunk, key_cursor, {}, self, self, self._options
                )

                self.hook_before_decompress_streamers(
                    key_chunk=key_chunk,
                    key_cursor=key_cursor,
                    streamer_key=streamer_key,
                )

                (
                    streamer_chunk,
                    streamer_cursor,
                ) = streamer_key.get_uncompressed_chunk_cursor()

                self.hook_before_read_streamers(
                    key_chunk=key_chunk,
                    key_cursor=key_cursor,
                    streamer_key=streamer_key,
                    streamer_cursor=streamer_cursor,
                    streamer_chunk=streamer_chunk,
                )

                tlist = self._classes["TList"].read(
                    streamer_chunk, streamer_cursor, {}, self, self
                )

                self._streamers = {}
                for x in tlist:
                    if x.name not in self._streamers:
                        self._streamers[x.name] = {}
                    self._streamers[x.name][x.class_version] = x

                self.hook_after_read_streamers(
                    key_chunk=key_chunk,
                    key_cursor=key_cursor,
                    streamer_key=streamer_key,
                    streamer_cursor=streamer_cursor,
                    streamer_chunk=streamer_chunk,
                )

        return self._streamers

    def streamer_named(self, classname, version=None):
        streamer_versions = self._streamers.get(classname)
        if streamer_versions is None or len(streamer_versions) == 0:
            return None
        elif version is None:
            return streamer_versions[max(streamer_versions)]
        else:
            return streamer_versions.get(version)

    def streamers_named(self, classname):
        return list(self.streamers[classname].values())

    def class_named(self, classname, version=None):
        cls = self._classes.get(classname)

        if cls is None:
            streamers = self.streamers_named(classname)

            if len(streamers) == 0:
                unknown_cls = uproot4.unknown_classes.get(classname)
                if unknown_cls is None:
                    unknown_cls = uproot4._util.new_class(
                        uproot4.model.classname_encode(classname, unknown=True),
                        (uproot4.model.UnknownClass,),
                        {},
                    )
                    uproot4.unknown_classes[classname] = unknown_cls
                return unknown_cls

            else:
                cls = uproot4._util.new_class(
                    uproot4._util.ensure_str(uproot4.model.classname_encode(classname)),
                    (uproot4.model.DispatchByVersion,),
                    {"_known_versions": {}},
                )
                self._classes[classname] = cls

        if version is not None and isinstance(cls, uproot4.model.DispatchByVersion):
            cls = cls.class_of_version(version)

        return cls

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

    def __enter__(self):
        """
        Passes __enter__ to the file's Source and returns self.
        """
        self._source.__enter__()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """
        Passes __exit__ to the file's Source, which closes physical files
        and shuts down any other resources, such as thread pools for parallel
        reading.
        """
        self._source.__exit__(exception_type, exception_value, traceback)


class ReadOnlyKey(object):
    _format_small = struct.Struct(">ihiIhhii")
    _format_big = struct.Struct(">ihiIhhqq")

    def __init__(
        self, chunk, cursor, context, file, parent, options, read_strings=False
    ):
        self._cursor = cursor.copy()
        self._file = file
        self._parent = parent

        self.hook_before_read(
            chunk=chunk,
            cursor=cursor,
            context=context,
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

        if self.is_64bit:
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
                chunk=chunk,
                cursor=cursor,
                context=context,
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
            chunk=chunk,
            cursor=cursor,
            context=context,
            file=file,
            parent=parent,
            options=options,
            read_strings=read_strings,
        )

    def __repr__(self):
        if self._fName is None or self._fClassName is None:
            nameclass = ""
        else:
            nameclass = " {0}: {1}".format(self.name(cycle=True), self.classname())
        return "<ReadOnlyKey{0} at byte {1}>".format(nameclass, self.data_cursor.index)

    def hook_before_read(self, **kwargs):
        pass

    def hook_before_read_strings(self, **kwargs):
        pass

    def hook_after_read(self, **kwargs):
        pass

    @property
    def cursor(self):
        return self._cursor

    @property
    def file(self):
        return self._file

    @property
    def parent(self):
        return self._parent

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
    def is_64bit(self):
        return self._fVersion > 1000

    def name(self, cycle=False):
        if cycle:
            return "{0};{1}".format(self.fName, self.fCycle)
        else:
            return self.fName

    def classname(self, encoded=False, version=None):
        if encoded:
            return uproot4.model.classname_encode(self.fClassName, version=version)
        else:
            return self.fClassName

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

    @property
    def data_cursor(self):
        return uproot4.source.cursor.Cursor(self._fSeekKey + self._fKeylen)

    def get_uncompressed_chunk_cursor(self):
        data_start = self.data_cursor.index
        data_stop = data_start + self.data_compressed_bytes
        chunk = self._file.chunk(data_start, data_stop)

        cursor = uproot4.source.cursor.Cursor(0, origin=-self._fKeylen)

        if self.is_compressed:
            return (
                uproot4.compression.decompress(
                    chunk,
                    self.data_cursor,
                    {},
                    self.data_compressed_bytes,
                    self.data_uncompressed_bytes,
                ),
                cursor,
            )
        else:
            return (
                uproot4.source.chunk.Chunk.wrap(
                    chunk.source, chunk.get(data_start, data_stop)
                ),
                cursor,
            )

    def get(self):
        if isinstance(self._parent, ReadOnlyDirectory) and self._fClassName in (
            "TDirectory",
            "TDirectoryFile",
        ):
            return ReadOnlyDirectory(
                self._parent.path + (self.fName,),
                self.data_cursor,
                self._file,
                self,
                self._file.options,
            )

        else:
            chunk, cursor = self.get_uncompressed_chunk_cursor()
            cls = self._file.class_named(self._fClassName)
            return cls.read(chunk, cursor, {}, self._file, self)


class ReadOnlyDirectory(Mapping):
    _format_small = struct.Struct(">hIIiiiii")
    _format_big = struct.Struct(">hIIiiqqq")
    _format_num_keys = struct.Struct(">i")

    def __init__(self, path, cursor, file, parent, options):
        self._path = path
        self._cursor = cursor.copy()
        self._file = file
        self._parent = parent

        directory_start = cursor.index
        directory_stop = min(directory_start + self._format_big.size, file.fEND)
        chunk = file.chunk(directory_start, directory_stop)

        self.hook_before_read(
            path=path,
            chunk=chunk,
            cursor=cursor,
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

        if self.is_64bit:
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

            if (keys_start, keys_stop) in chunk:
                keys_chunk = chunk
            else:
                keys_chunk = file.chunk(keys_start, keys_stop)

            keys_cursor = uproot4.source.cursor.Cursor(self._fSeekKeys)

            self.hook_before_header_key(
                path=path,
                chunk=chunk,
                cursor=cursor,
                file=file,
                parent=parent,
                options=options,
                keys_chunk=keys_chunk,
                keys_cursor=keys_cursor,
            )

            self._header_key = ReadOnlyKey(
                keys_chunk, keys_cursor, {}, file, self, options, read_strings=True
            )

            num_keys = keys_cursor.field(keys_chunk, self._format_num_keys)

            self.hook_before_keys(
                path=path,
                chunk=chunk,
                cursor=cursor,
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
                    keys_chunk, keys_cursor, {}, file, self, options, read_strings=True
                )
                self._keys.append(key)

            self.hook_after_read(
                path=path,
                chunk=chunk,
                cursor=cursor,
                file=file,
                parent=parent,
                options=options,
                keys_chunk=keys_chunk,
                keys_cursor=keys_cursor,
                num_keys=num_keys,
            )

    def __repr__(self):
        return "<ReadOnlyDirectory {0}>".format(repr("/" + "/".join(self._path)))

    def hook_before_read(self, **kwargs):
        pass

    def hook_before_header_key(self, **kwargs):
        pass

    def hook_before_keys(self, **kwargs):
        pass

    def hook_after_read(self, **kwargs):
        pass

    @property
    def path(self):
        return self._path

    @property
    def cursor(self):
        return self._cursor

    @property
    def file(self):
        return self._file

    @property
    def parent(self):
        return self._parent

    @property
    def header_key(self):
        return self._header_key

    @property
    def is_64bit(self):
        return self._fVersion > 1000

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

    def __enter__(self):
        """
        Passes __enter__ to the directory's file and returns self.
        """
        self._file.source.__enter__()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """
        Passes __exit__ to the directory's file, which closes physical files
        and shuts down any other resources, such as thread pools for parallel
        reading.
        """
        self._file.source.__exit__(exception_type, exception_value, traceback)

    def iterkeys(
        self,
        recursive=True,
        cycle=True,
        filter_name=no_filter,
        filter_classname=no_filter,
    ):
        for key in self._keys:
            if filter_name(key.fName) and filter_classname(key.fClassName):
                yield key.name(cycle=cycle)

            if recursive and key.fClassName in ("TDirectory", "TDirectoryFile"):
                for k in key.get().iterkeys(recursive, filter_name, filter_classname):
                    yield "{0}/{1}".format(key.name(cycle=False), k)

    def iteritems(
        self,
        recursive=True,
        cycle=True,
        filter_name=no_filter,
        filter_classname=no_filter,
    ):
        for key in self._keys:
            if filter_name(key.fName) and filter_classname(key.fClassName):
                yield key.name(cycle=cycle), key.get()

            if recursive and key.fClassName in ("TDirectory", "TDirectoryFile"):
                for k, v in key.get().iteritems(
                    recursive, filter_name, filter_classname
                ):
                    yield "{0}/{1}".format(key.name(cycle=False), k), v

    def itervalues(
        self, recursive=True, filter_name=no_filter, filter_classname=no_filter,
    ):
        for k, v in self.iteritems(recursive, False, filter_name, filter_classname):
            yield v

    def keys(
        self,
        recursive=True,
        cycle=True,
        filter_name=no_filter,
        filter_classname=no_filter,
    ):
        return list(self.iterkeys(recursive, cycle, filter_name, filter_classname))

    def items(
        self,
        recursive=True,
        cycle=True,
        filter_name=no_filter,
        filter_classname=no_filter,
    ):
        return list(self.iteritems(recursive, cycle, filter_name, filter_classname))

    def values(
        self, recursive=True, filter_name=no_filter, filter_classname=no_filter,
    ):
        return list(self.itervalues(recursive, filter_name, filter_classname))

    def __len__(self):
        return len(self._keys) + sum(
            len(x.get())
            for x in self._keys
            if x.fClassName in ("TDirectory", "TDirectoryFile")
        )

    def __contains__(self, where):
        try:
            self.key(where)
        except KeyWithCycleError:
            return False
        else:
            return True

    def __iter__(self):
        return self.iterkeys()

    def _ipython_key_completions_(self):
        "Support key-completion in an IPython or Jupyter kernel."
        return self.iterkeys()

    def __getitem__(self, where):
        return self.key(where).get()

    def classname_of(self, where, encoded=False, version=None):
        key = self.key(where)
        return key.classname(encoded=encoded, version=version)

    def streamer_of(self, where, version=None):
        key = self.key(where)
        return self._file.streamer_named(key.fClassName, version=version)

    def class_of(self, where, version=None):
        key = self.key(where)
        return self._file.class_named(key.fClassName, version=version)

    def key(self, where):
        where = uproot4._util.ensure_str(where)

        if "/" in where:
            items = where.split("/")
            step = self
            for item in items[:-1]:
                if item != "":
                    step = step[item]
            return step.key(items[-1])

        if ";" in where:
            at = where.rindex(";")
            item, cycle = where[:at], where[at + 1 :]
            cycle = int(cycle)
        else:
            item, cycle = where, None

        last = None
        for key in self._keys:
            if key.fName == item:
                if cycle == key.fCycle:
                    return key
                elif cycle is None and last is None:
                    last = key
                elif cycle is None and last.fCycle < key.fCycle:
                    last = key

        if last is not None:
            return last
        elif cycle is None:
            raise KeyWithCycleError(
                """not found: {0} (with any cycle number)
in file {1}""".format(
                    repr(item), self._file.file_path
                )
            )
        else:
            raise KeyWithCycleError(
                """not found: {0} with cycle {1}
in file {2}""".format(
                    repr(item), cycle, self._file.file_path
                )
            )


class KeyWithCycleError(KeyError):
    def __str__(self):
        return self.args[0]

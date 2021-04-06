# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

"""
FIXME: docstring
"""

from __future__ import absolute_import

import datetime
import os.path
import struct
import uuid

import uproot._util
import uproot.compression
import uproot.const
import uproot.reading
import uproot.sink.file

# def create(file_path, **options):
#     """
#     FIXME: docstring
#     """
#     file_path = uproot._util.regularize_path(file_path)
#     if uproot._util.isstr(file_path):
#         if os.path.exists(file_path):
#             raise OSError(
#                 "path exists and refusing to overwrite (use 'uproot.recreate' to "
#                 "overwrite)\n\nin path {0}".format(file_path)
#             )
#         file = uproot.sink.file.FileSink()
#     else:
#         file = uproot.sink.file.FileSink.from_object(file_path)

#     compression = options.pop("compression", update.defaults["compression"])
#     initial_directory_bytes = options.pop(
#         "initial_directory_bytes", create.defaults["initial_directory_bytes"]
#     )
#     uuid_version = options.pop("uuid_version", create.defaults["uuid_version"])
#     uuid_function = options.pop("uuid", create.defaults["uuid"])
#     if len(options) != 0:
#         raise TypeError("unrecognized options: " + ", ".join(repr(x) for x in options))

#     _create_empty_root(
#         file, compression, initial_directory_bytes, uuid_version, uuid_function
#     )
#     return WritableFile(file, compression, initial_directory_bytes)


# create.defaults = {
#     "compression": uproot.compression.ZLIB(1),
#     "initial_directory_bytes": 256,
#     "uuid_version": 1,
#     "uuid_function": uuid.uuid1,
# }


# def recreate(file_path, **options):
#     """
#     FIXME: docstring
#     """
#     file_path = uproot._util.regularize_path(file_path)
#     if uproot._util.isstr(file_path):
#         file = uproot.sink.file.FileSink()
#     else:
#         file = uproot.sink.file.FileSink.from_object(file_path)

#     compression = options.pop("compression", update.defaults["compression"])
#     initial_directory_bytes = options.pop(
#         "initial_directory_bytes", recreate.defaults["initial_directory_bytes"]
#     )
#     uuid_version = options.pop("uuid_version", create.defaults["uuid_version"])
#     uuid_function = options.pop("uuid", create.defaults["uuid"])
#     if len(options) != 0:
#         raise TypeError("unrecognized options: " + ", ".join(repr(x) for x in options))

#     _create_empty_root(
#         file, compression, initial_directory_bytes, uuid_version, uuid_function
#     )
#     return WritableFile(file, compression, initial_directory_bytes)


# recreate.defaults = {
#     "compression": uproot.compression.ZLIB(1),
#     "initial_directory_bytes": 256,
#     "uuid_version": 1,
#     "uuid_function": uuid.uuid1,
# }


# def update(file_path, **options):
#     """
#     FIXME: docstring
#     """
#     file_path = uproot._util.regularize_path(file_path)
#     if uproot._util.isstr(file_path):
#         file = uproot.sink.file.FileSink()
#     else:
#         file = uproot.sink.file.FileSink.from_object(file_path)

#     compression = options.pop("compression", update.defaults["compression"])
#     initial_directory_bytes = options.pop(
#         "initial_directory_bytes", update.defaults["initial_directory_bytes"]
#     )
#     uuid_version = options.pop("uuid_version", create.defaults["uuid_version"])
#     uuid_function = options.pop("uuid", create.defaults["uuid"])
#     if len(options) != 0:
#         raise TypeError("unrecognized options: " + ", ".join(repr(x) for x in options))

#     if file.read_raw(4) != b"root":
#         _create_empty_root(
#             file, compression, initial_directory_bytes, uuid_version, uuid_function
#         )
#     return WritableFile(file, compression, initial_directory_bytes)


# update.defaults = {
#     "compression": uproot.compression.ZLIB(1),
#     "initial_directory_bytes": 256,
#     "uuid_version": 1,
#     "uuid_function": uuid.uuid1,
# }


def _create_empty_root(
    sink, compression, initial_directory_bytes, uuid_version, uuid_function
):
    filename = sink.file_path
    if filename is None:
        filename = "dynamic.root"
    else:
        filename = os.path.split(filename)[-1]
    if len(filename) >= 256:
        raise ValueError("ROOT file names must be less than 256 bytes")

    # tmp = uuid_function()
    tmp = uuid.UUID(
        bytes=b"\xf7\xfe\xef\xda\x93\xe5\x11\xeb\x95\xc8\xd2\x01\xa8\xc0\xbe\xef"
    )

    fileheader = WritableFileHeader(
        None, None, None, None, None, compression, None, None, uuid_version, tmp
    )

    freesegments_key = WritableKey(
        None,
        None,
        None,
        WritableString(None, "TFile"),
        WritableString(None, filename),
        WritableString(None, ""),
        1,
        fileheader.begin,
    )
    freesegments_data = WritableFreeSegmentsData(None, (), None)
    freesegments = FreeSegments(freesegments_key, freesegments_data, fileheader)

    streamers_key = WritableKey(
        None,
        None,
        None,
        WritableString(None, "TList"),
        WritableString(None, "StreamerInfo"),
        WritableString(None, "Doubly linked list"),
        1,
        fileheader.begin,
    )
    streamers_data = WritableStreamersData(
        None, 21
    )  # FIXME: parameterize initial_streamers_bytes
    streamers = Streamers(streamers_key, streamers_data, freesegments)

    # tmp = uuid_function()
    tmp = uuid.UUID(
        bytes=b"\xf7\xfe\xef\xda\x93\xe5\x11\xeb\x95\xc8\xd2\x01\xa8\xc0\xbe\xef"
    )

    directory_key = WritableKey(
        None,
        None,
        None,
        WritableString(None, "TFile"),
        WritableString(None, filename),
        WritableString(None, ""),
        1,
        0,
    )
    directory_name = WritableString(None, filename)
    directory_title = WritableString(None, "")
    directory_header = WritableDirectoryHeader(
        None, fileheader.begin, None, None, None, 0, uuid_version, tmp
    )
    directory_datakey = WritableKey(
        None,
        None,
        None,
        WritableString(None, "TFile"),
        WritableString(None, filename),
        WritableString(None, ""),
        1,
        fileheader.begin,
    )
    directory_data = WritableDirectoryData(None, None, [])
    rootdirectory = RootDirectory(
        directory_key,
        directory_name,
        directory_title,
        directory_header,
        directory_datakey,
        directory_data,
        freesegments,
    )

    directory_key.location = fileheader.begin
    streamers_key.location = (
        directory_key.location
        + directory_key.allocation
        + directory_name.allocation
        + directory_title.allocation
        + directory_header.allocation
    )
    directory_datakey.location = streamers_key.location + streamers.allocation
    directory_data.location = directory_datakey.location + directory_datakey.allocation
    freesegments_key.location = directory_data.location + directory_data.allocation
    fileheader.info_location = streamers_key.location
    fileheader.info_num_bytes = streamers_key.allocation + streamers_data.allocation

    rootdirectory.write(sink)
    streamers.write(sink)


class Writable(object):
    """
    FIXME: docstring
    """

    def __init__(self, location, allocation):
        self._location = location
        self._allocation = allocation
        self._file_dirty = True

    @property
    def location(self):
        return self._location

    @location.setter
    def location(self, value):
        if self._location != value:
            self._file_dirty = True
            self._location = value

    @property
    def allocation(self):
        if self._allocation is None:
            return self.num_bytes
        else:
            return self._allocation

    @property
    def num_bytes(self):
        return len(self.serialize())

    def write(self, sink):
        if self._file_dirty:
            if self._location is None:
                raise RuntimeError(
                    "can't write object because location is unknown:\n\n    "
                    + repr(self)
                )
            sink.write(self._location, self.serialize())
            self._file_dirty = False

    def serialize(self):
        raise AssertionError("Writable is abstract; 'serialize' must be overloaded")


class HasDependencies(object):
    """
    FIXME: docstring
    """

    def __init__(self, *dependencies):
        self._dependencies = dependencies

    def write(self, sink):
        for dependency in self._dependencies:
            dependency.write(sink)


class WritableString(Writable):
    """
    FIXME: docstring
    """

    def __init__(self, location, string):
        super(WritableString, self).__init__(location, None)
        self._string = string

        bytestring = self._string.encode(errors="surrogateescape")
        length = len(bytestring)
        if length < 255:
            self._serialization = struct.pack(">B%ds" % length, length, bytestring)
        else:
            self._serialization = struct.pack(
                ">BI%ds" % length, 255, length, bytestring
            )

    def __repr__(self):
        return "{0}({1}, {2})".format(
            type(self).__name__,
            self._location,
            repr(self._string),
        )

    @property
    def string(self):
        return self._string

    def serialize(self):
        return self._serialization


class WritableKey(Writable):
    """
    FIXME: docstring
    """

    class_version = 4

    def __init__(
        self,
        location,
        uncompressed_bytes,
        compressed_bytes,
        classname,
        name,
        title,
        cycle,
        parent_location,
    ):
        super(WritableKey, self).__init__(location, None)
        self._uncompressed_bytes = uncompressed_bytes
        self._compressed_bytes = compressed_bytes
        self._classname = classname
        self._name = name
        self._title = title
        self._cycle = cycle
        self._parent_location = parent_location

    def __repr__(self):
        return "{0}({1}, {2}, {3}, {4}, {5}, {6}, {7}, {8})".format(
            type(self).__name__,
            self._location,
            self._uncompressed_bytes,
            self._compressed_bytes,
            repr(self._classname),
            repr(self._name),
            repr(self._title),
            self._cycle,
            self._parent_location,
        )

    @property
    def uncompressed_bytes(self):
        return self._uncompressed_bytes

    @uncompressed_bytes.setter
    def uncompressed_bytes(self, value):
        if self._uncompressed_bytes != value:
            self._file_dirty = True
            self._uncompressed_bytes = value

    @property
    def compressed_bytes(self):
        return self._compressed_bytes

    @compressed_bytes.setter
    def compressed_bytes(self, value):
        if self._compressed_bytes != value:
            self._file_dirty = True
            self._compressed_bytes = value

    @property
    def classname(self):
        return self._classname

    @property
    def name(self):
        return self._name

    @property
    def title(self):
        return self._title

    @property
    def cycle(self):
        return self._cycle

    @property
    def parent_location(self):
        return self._parent_location

    @property
    def big(self):
        return False  # FIXME
        # return (
        #     self._location is None
        #     or self._location >= uproot.const.kStartBigFile
        #     or self._parent_location >= uproot.const.kStartBigFile
        # )

    @property
    def num_bytes(self):
        if self.big:
            return (
                uproot.reading._key_format_big.size
                + self._classname.allocation
                + self._name.allocation
                + self._title.allocation
            )
        else:
            return (
                uproot.reading._key_format_small.size
                + self._classname.allocation
                + self._name.allocation
                + self._title.allocation
            )

    def serialize(self):
        if self._location is None:
            raise RuntimeError(
                "can't serialize key because location is unknown:\n\n    " + repr(self)
            )

        if self.big:
            format = uproot.reading._key_format_big
            version = self.class_version + 1000
        else:
            format = uproot.reading._key_format_small
            version = self.class_version

        return (
            format.pack(
                self._compressed_bytes + self.num_bytes,  # fNbytes
                version,  # fVersion
                self._uncompressed_bytes,  # fObjlen
                1761927327,  # FIXME: compute fDatime
                self.num_bytes,  # fKeylen
                self._cycle,  # fCycle
                self._location,  # fSeekKey
                self._parent_location,  # fSeekPdir
            )
            + self._classname.serialize()
            + self._name.serialize()
            + self._title.serialize()
        )


_free_format_small = struct.Struct(">HII")
_free_format_big = struct.Struct(">HQQ")


class WritableFreeSegmentsData(Writable):
    """
    FIXME: docstring
    """

    class_version = 1

    def __init__(self, location, slices, end):
        super(WritableFreeSegmentsData, self).__init__(location, None)
        self._slices = slices
        self._end = end

    def __repr__(self):
        return "{0}({1}, {2}, {3})".format(
            type(self).__name__,
            self._location,
            self._slices,
            self._end,
        )

    @property
    def slices(self):
        return self._slices

    @slices.setter
    def slices(self, value):
        if self._slices != value:
            self._file_dirty = True
            self._slices = value

    @property
    def end(self):
        return self._end

    @end.setter
    def end(self, value):
        if self._end != value:
            self._file_dirty = True
            self._end = value

    @property
    def num_bytes(self):
        total = 0
        for _, stop in self._slices:
            if stop - 1 >= uproot.const.kStartBigFile:
                total += _free_format_big.size
            else:
                total += _free_format_small.size

        if self._end is None:
            if total + _free_format_small.size >= uproot.const.kStartBigFile:
                total += _free_format_big.size
            else:
                total += _free_format_small.size
        elif self._end >= uproot.const.kStartBigFile:
            total += _free_format_big.size
        else:
            total += _free_format_small.size

        return total

    def serialize(self):
        pairs = []
        for start, stop in self._slices:
            if stop - 1 < uproot.const.kStartBigFile:
                pairs.append(
                    _free_format_small.pack(self.class_version, start, stop - 1)
                )
            else:
                pairs.append(_free_format_big.pack(self.class_version, start, stop - 1))

        if self._end < uproot.const.kStartBigFile:
            pairs.append(
                _free_format_small.pack(
                    self.class_version, self._end, uproot.const.kStartBigFile
                )
            )
        else:
            infinity = uproot.const.kStartBigFile
            while not self._end < infinity:
                infinity *= 2
            pairs.append(_free_format_big.pack(self.class_version, self._end, infinity))

        return b"".join(pairs)


class FreeSegments(HasDependencies):
    """
    FIXME: docstring
    """

    # key cycle = 1, parent = 100 (fBEGIN), class = "TFile", name = filename, title = ""

    def __init__(self, key, data, fileheader):
        super(FreeSegments, self).__init__(key, data, fileheader)
        self._key = key
        self._data = data
        self._fileheader = fileheader

    def __repr__(self):
        return "{0}({1}, {2}, {3})".format(
            type(self).__name__,
            self._key,
            self._data,
            self._fileheader,
        )

    @property
    def key(self):
        return self._key

    @property
    def data(self):
        return self._data

    @property
    def fileheader(self):
        return self._fileheader

    def write(self, sink):
        self._key.uncompressed_bytes = self._data.num_bytes
        self._key.compressed_bytes = self._key.uncompressed_bytes
        self._data.location = self._key.location + self._key.allocation
        self._data.end = self._data.location + self._key.uncompressed_bytes
        self._fileheader.free_location = self._key.location
        self._fileheader.free_num_bytes = self._data.end - self._key.location
        self._fileheader.free_num_slices = len(self._data.slices)
        self._fileheader.end = self._data.end
        super(FreeSegments, self).write(sink)


class WritableStreamersData(Writable):
    """
    FIXME: docstring
    """

    # allocation = 21

    def __init__(self, location, allocation):
        super(WritableStreamersData, self).__init__(location, allocation)

        self._serialization = b"@\x00\x00\x11\x00\x05\x00\x01\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00"

    def __repr__(self):
        return "{0}({1}, {2})".format(
            type(self).__name__,
            self._location,
            self._allocation,
        )

    def serialize(self):
        return self._serialization


class Streamers(HasDependencies):
    """
    FIXME: docstring
    """

    # key cycle = 1, parent = 100 (fBEGIN), class = "TList", name = "StreamerInfo", title = "Doubly linked list"

    def __init__(self, key, data, freesegments):
        super(Streamers, self).__init__(key, data, freesegments)
        self._key = key
        self._data = data
        self._freesegments = freesegments

    def __repr__(self):
        return "{0}({1}, {2}, {3})".format(
            type(self).__name__,
            self._key,
            self._data,
            self._freesegments,
        )

    @property
    def key(self):
        return self._key

    @property
    def data(self):
        return self._data

    @property
    def freesegments(self):
        return self._freesegments

    @property
    def allocation(self):
        return self._key.allocation + self._data.allocation

    def write(self, sink):
        self._key.uncompressed_bytes = self._data.num_bytes
        self._key.compressed_bytes = self._key.uncompressed_bytes
        self._data.location = self._key.location + self._key.allocation
        self._freesegments.fileheader.info_location = self._key.location
        self._freesegments.fileheader.info_num_bytes = (
            self._key.allocation + self._data.allocation
        )
        super(Streamers, self).write(sink)


class WritableDirectoryData(Writable):
    """
    FIXME: docstring
    """

    def __init__(self, location, allocation, keys):
        super(WritableDirectoryData, self).__init__(location, allocation)
        self._keys = keys

    def __repr__(self):
        return "{0}({1}, {2}, {3})".format(
            type(self).__name__,
            self._location,
            self._allocation,
            self._keys,
        )

    @property
    def keys(self):
        return self._keys

    @property
    def num_bytes(self):
        return uproot.reading._directory_format_num_keys.size + sum(
            x.allocation for x in self._keys
        )

    def serialize(self):
        out = [uproot.reading._directory_format_num_keys.pack(len(self._keys))]
        for key in self._keys:
            out.append(key.serialize())
        return b"".join(out)


class WritableDirectoryHeader(Writable):
    """
    FIXME: docstring
    """

    class_version = 5

    # begin_location = 100 (fBEGIN), parent_location = 0

    def __init__(
        self,
        location,
        begin_location,
        begin_num_bytes,
        data_location,
        data_num_bytes,
        parent_location,
        uuid_version,
        uuid,
    ):
        super(WritableDirectoryHeader, self).__init__(
            location, uproot.reading._directory_format_big.size + 2 + len(uuid.bytes)
        )
        self._begin_location = begin_location
        self._begin_num_bytes = begin_num_bytes
        self._data_location = data_location
        self._data_num_bytes = data_num_bytes
        self._parent_location = parent_location
        self._uuid_version = uuid_version
        self._uuid = uuid
        self._created_on = datetime.datetime.now()
        self._modified_on = self._created_on

    def __repr__(self):
        return "{0}({1}, {2}, {3}, {4}, {5}, {6}, {7}, {8})".format(
            type(self).__name__,
            self._location,
            self._begin_location,
            self._begin_num_bytes,
            self._data_location,
            self._data_num_bytes,
            self._parent_location,
            self._uuid_version,
            self._uuid,
        )

    @property
    def begin_location(self):
        return self._begin_location

    @property
    def begin_num_bytes(self):
        return self._begin_num_bytes

    @begin_num_bytes.setter
    def begin_num_bytes(self, value):
        if self._begin_num_bytes != value:
            self._file_dirty = True
            self._begin_num_bytes = value

    @property
    def data_location(self):
        return self._data_location

    @data_location.setter
    def data_location(self, value):
        if self._data_location != value:
            self._file_dirty = True
            self._data_location = value

    @property
    def data_num_bytes(self):
        return self._data_num_bytes

    @data_num_bytes.setter
    def data_num_bytes(self, value):
        if self._data_num_bytes != value:
            self._file_dirty = True
            self._data_num_bytes = value

    @property
    def parent_location(self):
        return self._parent_location

    @property
    def uuid_version(self):
        return self._uuid_version

    @property
    def uuid(self):
        return self._uuid

    @property
    def big(self):
        return False  # FIXME
        # return (
        #     self._begin_location >= uproot.const.kStartBigFile
        #     or self._data_location >= uproot.const.kStartBigFile
        #     or self._parent_location >= uproot.const.kStartBigFile
        # )

    @property
    def num_bytes(self):
        if self.big:
            return uproot.reading._directory_format_big.size
        else:
            return uproot.reading._directory_format_small.size

    def serialize(self):
        if self.big:
            format = uproot.reading._directory_format_big
            version = self.class_version + 1000
            extra = b""
        else:
            format = uproot.reading._directory_format_small
            version = self.class_version
            extra = b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"

        return (
            format.pack(
                version,  # fVersion
                1761927327,  # FIXME: compute fDatimeC
                1761927327,  # FIXME: compute fDatimeM
                self._data_num_bytes,  # fNbytesKeys
                self._begin_num_bytes,  # fNbytesName
                self._begin_location,  # fSeekDir
                self._parent_location,  # fSeekParent
                self._data_location,  # fSeekKeys
            )
            + struct.pack(">H", self._uuid_version)
            + self._uuid.bytes
            + extra
        )


class RootDirectory(HasDependencies):
    """
    FIXME: docstring
    """

    def __init__(self, key, name, title, header, datakey, data, freesegments):
        super(RootDirectory, self).__init__(
            key, name, title, header, datakey, data, freesegments
        )
        self._key = key
        self._name = name
        self._title = title
        self._header = header
        self._datakey = datakey
        self._data = data
        self._freesegments = freesegments

    def __repr__(self):
        return "{0}({1}, {2}, {3}, {4}, {5}, {6}, {7})".format(
            type(self).__name__,
            self._key,
            self._name,
            self._title,
            self._header,
            self._datakey,
            self._data,
            self._freesegments,
        )

    @property
    def key(self):
        return self._key

    @property
    def name(self):
        return self._name

    @property
    def title(self):
        return self._title

    @property
    def header(self):
        return self._header

    @property
    def datakey(self):
        return self._datakey

    @property
    def data(self):
        return self._data

    @property
    def freesegments(self):
        return self._freesegments

    @property
    def begin_num_bytes(self):
        return self._key.allocation + self._name.allocation + self._title.allocation

    def write(self, sink):
        self._key.uncompressed_bytes = (
            self._name.allocation + self._title.allocation + self._header.allocation
        )
        self._key.compressed_bytes = self._key.uncompressed_bytes
        self._name.location = self._key.location + self._key.allocation
        self._title.location = self._name.location + self._name.allocation
        self._header.location = self._title.location + self._title.allocation
        self._header.begin_num_bytes = self.begin_num_bytes
        self._header.data_location = self._datakey.location
        self._header.data_num_bytes = self._datakey.num_bytes + self._data.num_bytes
        self._datakey.uncompressed_bytes = self._data.num_bytes
        self._datakey.compressed_bytes = self._datakey.uncompressed_bytes
        self._freesegments.fileheader.begin_num_bytes = self.begin_num_bytes
        super(RootDirectory, self).write(sink)


class SubDirectory(HasDependencies):
    """
    FIXME: docstring
    """

    def __init__(self, key, header, datakey, data, parent, freesegments):
        super(SubDirectory, self).__init__(key, header, datakey, data, parent)
        self._key = key
        self._header = header
        self._datakey = datakey
        self._data = data
        self._parent = parent
        self._freesegments = freesegments

    def __repr__(self):
        return "{0}({1}, {2}, {3}, {4}, {5}, {6})".format(
            type(self).__name__,
            self._key,
            self._header,
            self._datakey,
            self._data,
            self._parent,
            self._freesegments,
        )

    @property
    def key(self):
        return self._key

    @property
    def header(self):
        return self._header

    @property
    def datakey(self):
        return self._datakey

    @property
    def data(self):
        return self._data

    @property
    def parent(self):
        return self._parent

    @property
    def freesegments(self):
        return self._freesegments

    def write(self, sink):
        self._key.uncompressed_bytes = self._header.allocation
        self._key.compressed_bytes = self._key.uncompressed_bytes
        self._header.location = self._key.location + self._key.allocation
        self._header.begin_num_bytes = self._key.num_bytes
        self._header.data_location = self._datakey.location
        self._header.data_num_bytes = self._datakey.num_bytes + self._data.num_bytes
        self._datakey.uncompressed_bytes = self._data.num_bytes
        self._datakey.compressed_bytes = self._datakey.uncompressed_bytes
        super(SubDirectory, self).write(sink)


class WritableFileHeader(Writable):
    """
    FIXME: docstring
    """

    magic = b"root"
    class_version = 62206  # ROOT 6.22/06 is our model
    begin = 100

    def __init__(
        self,
        end,
        free_location,
        free_num_bytes,
        free_num_slices,
        begin_num_bytes,
        compression,
        info_location,
        info_num_bytes,
        uuid_version,
        uuid,
    ):
        super(WritableFileHeader, self).__init__(0, self.begin)
        self._end = end
        self._free_location = free_location
        self._free_num_bytes = free_num_bytes
        self._free_num_slices = free_num_slices
        self._begin_num_bytes = begin_num_bytes
        self._compression = compression
        self._info_location = info_location
        self._info_num_bytes = info_num_bytes
        self._uuid_version = uuid_version
        self._uuid = uuid

    def __repr__(self):
        return "{0}({1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10})".format(
            type(self).__name__,
            self._end,
            self._free_location,
            self._free_num_bytes,
            self._free_num_slices,
            self._begin_num_bytes,
            repr(self._compression),
            self._info_location,
            self._info_num_bytes,
            self._uuid_version,
            repr(self._uuid),
        )

    @property
    def end(self):
        return self._end

    @end.setter
    def end(self, value):
        if self._end != value:
            self._file_dirty = True
            self._end = value

    @property
    def free_location(self):
        return self._free_location

    @free_location.setter
    def free_location(self, value):
        if self._free_location != value:
            self._file_dirty = True
            self._free_location = value

    @property
    def free_num_bytes(self):
        return self._free_num_bytes

    @free_num_bytes.setter
    def free_num_bytes(self, value):
        if self._free_num_bytes != value:
            self._file_dirty = True
            self._free_num_bytes = value

    @property
    def free_num_slices(self):
        return self._free_num_slices

    @free_num_slices.setter
    def free_num_slices(self, value):
        if self._free_num_slices != value:
            self._file_dirty = True
            self._free_num_slices = value

    @property
    def begin_num_bytes(self):
        return self._begin_num_bytes

    @begin_num_bytes.setter
    def begin_num_bytes(self, value):
        if self._begin_num_bytes != value:
            self._file_dirty = True
            self._begin_num_bytes = value

    @property
    def compression(self):
        return self._compression

    @compression.setter
    def compression(self, value):
        if self._compression.code != value.code:
            self._file_dirty = True
            self._compression = value

    @property
    def info_location(self):
        return self._info_location

    @info_location.setter
    def info_location(self, value):
        if self._info_location != value:
            self._file_dirty = True
            self._info_location = value

    @property
    def info_num_bytes(self):
        return self._info_num_bytes

    @info_num_bytes.setter
    def info_num_bytes(self, value):
        if self._info_num_bytes != value:
            self._file_dirty = True
            self._info_num_bytes = value

    @property
    def uuid_version(self):
        return self._uuid_version

    @uuid_version.setter
    def uuid_version(self, value):
        if self._uuid_version != value:
            self._file_dirty = True
            self._uuid_version = value

    @property
    def uuid(self):
        return self._uuid

    @uuid.setter
    def uuid(self, value):
        if self._uuid != value:
            self._file_dirty = True
            self._uuid = value

    @property
    def big(self):
        return False  # FIXME
        # return (
        #     self._end is None
        #     or self._end >= uproot.const.kStartBigFile
        #     or self._free_location >= uproot.const.kStartBigFile
        #     or self._info_location >= uproot.const.kStartBigFile
        # )

    @property
    def num_bytes(self):
        if self.big:
            return uproot.reading._file_header_fields_big.size
        else:
            return uproot.reading._file_header_fields_small.size

    def serialize(self):
        if self.big:
            format = uproot.reading._file_header_fields_big
            version = self.class_version + 1000000
            units = 8
        else:
            format = uproot.reading._file_header_fields_small
            version = self.class_version
            units = 4

        return format.pack(
            self.magic,
            version,  # fVersion
            self.begin,  # fBEGIN
            self._end,  # fEND
            self._free_location,  # fSeekFree
            self._free_num_bytes,  # fNbytesFree
            self._free_num_slices + 1,  # nfree
            self._begin_num_bytes,  # fNbytesName
            units,  # fUnits
            self._compression.code,  # fCompress
            self._info_location,  # fSeekInfo
            self._info_num_bytes,  # fNbytesInfo
            self._uuid_version,  # fUUID_version
            self._uuid.bytes,  # fUUID
        )

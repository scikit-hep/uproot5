# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

"""
FIXME: docstring
"""

from __future__ import absolute_import

import os.path
import struct
import uuid

import uproot._util
import uproot.compression
import uproot.const
import uproot.reading
import uproot.sink.file


def create(file_path, **options):
    """
    FIXME: docstring
    """
    file_path = uproot._util.regularize_path(file_path)
    if uproot._util.isstr(file_path):
        if os.path.exists(file_path):
            raise OSError(
                "path exists and refusing to overwrite (use 'uproot.recreate' to "
                "overwrite)\n\nin path {0}".format(file_path)
            )
        file = uproot.sink.file.FileSink()
    else:
        file = uproot.sink.file.FileSink.from_object(file_path)

    compression = options.pop("compression", update.defaults["compression"])
    initial_bytes = options.pop("initial_bytes", create.defaults["initial_bytes"])
    uuid_version = options.pop("uuid_version", create.defaults["uuid_version"])
    uuid_function = options.pop("uuid", create.defaults["uuid"])
    if len(options) != 0:
        raise TypeError("unrecognized options: " + ", ".join(repr(x) for x in options))

    _create_empty_root(file, compression, initial_bytes, uuid_version, uuid_function)
    return WritableFile(file, compression, initial_bytes)


create.defaults = {
    "compression": uproot.compression.ZLIB(1),
    "initial_bytes": 256,
    "uuid": uuid.uuid1,
}


def recreate(file_path, **options):
    """
    FIXME: docstring
    """
    file_path = uproot._util.regularize_path(file_path)
    if uproot._util.isstr(file_path):
        file = uproot.sink.file.FileSink()
    else:
        file = uproot.sink.file.FileSink.from_object(file_path)

    compression = options.pop("compression", update.defaults["compression"])
    initial_bytes = options.pop("initial_bytes", recreate.defaults["initial_bytes"])
    uuid_version = options.pop("uuid_version", create.defaults["uuid_version"])
    uuid_function = options.pop("uuid", create.defaults["uuid"])
    if len(options) != 0:
        raise TypeError("unrecognized options: " + ", ".join(repr(x) for x in options))

    _create_empty_root(file, compression, initial_bytes, uuid_version, uuid_function)
    return WritableFile(file, compression, initial_bytes)


recreate.defaults = {
    "compression": uproot.compression.ZLIB(1),
    "initial_bytes": 256,
    "uuid": uuid.uuid1,
}


def update(file_path, **options):
    """
    FIXME: docstring
    """
    file_path = uproot._util.regularize_path(file_path)
    if uproot._util.isstr(file_path):
        file = uproot.sink.file.FileSink()
    else:
        file = uproot.sink.file.FileSink.from_object(file_path)

    compression = options.pop("compression", update.defaults["compression"])
    initial_bytes = options.pop("initial_bytes", update.defaults["initial_bytes"])
    uuid_version = options.pop("uuid_version", create.defaults["uuid_version"])
    uuid_function = options.pop("uuid", create.defaults["uuid"])
    if len(options) != 0:
        raise TypeError("unrecognized options: " + ", ".join(repr(x) for x in options))

    if file.read_raw(4) != b"root":
        _create_empty_root(
            file, compression, initial_bytes, uuid_version, uuid_function
        )
    return WritableFile(file, compression, initial_bytes)


update.defaults = {
    "compression": uproot.compression.ZLIB(1),
    "initial_bytes": 256,
    "uuid": uuid.uuid1,
}


def _create_empty_root(file, compression, initial_bytes, uuid_version, uuid_function):
    filename = file.file_path
    if filename is None:
        filename = "dynamic.root"
    else:
        filename = os.path.split(filename)[-1]
    if len(filename) >= 256:
        raise ValueError("ROOT file names must be less than 256 bytes")

    # temporary
    # streamerinfo = b"\x00\x00\x01\x19\x00\x04\x00\x00\x01ri\x04\xe0\xc5\x00@\x00\x01\x00\x00\x016\x00\x00\x00d\x05TList\x0cStreamerInfo\x12Doubly linked listZL\x08\xd0\x00\x00r\x01\x00x\x01uP;\n\xc2@\x14\x9c\x185h#v\x01\x9b\x14b\xe7\x1d\x92\x80\xa0U@\xa3X\x08\x12u\x95\x08&\xb2F\xc4K\x88\x07\xf00\xdeLg\xd7\x0fn\xe1\xc0\x9bb\xde\xcc\x0eo}X\x19*\xb0@\x94\x14\x11\x96\x0fk\xfa \xe2Q!E\xb2\x13r\x90\xadsP\rQ\xf3\x01\x97v\x1d\xb0\xc9\xf58Zl\xe9K\xb3\rn\x97\xfe'\xef\xea<W\x81\x94\xc9Ye\xab\xb0\xcd\x1a]\xc7\xe7\x8eFU\x98\x1c\x04\xed\x98\xc3&\x8fQ&w\xbe\x85\x00\x1cU(\x96E\x93\xcet\xe9\r\xa3(\xf6r\xadp\x19r\x0c\\[\xf7\xee\xafP\x0e\x83Q\x8f\x02oDa\x14\xbfo\xa0>C\x89<\xd1\xd5m\xa3z\xfd25N2\xd9\xef\xc5\xcaS\x1f\xa4\x0e\x07\x02\x8e\xcb\xf9\x0b\xe7\xe3}\x024bH."
    # streamerinfo = b"\x00\x00\x00U\x00\x04\x00\x00\x00\x15i\x04\xe0\x9f\x00@\x00\x01\x00\x00\x00\xd8\x00\x00\x00d\x05TList\x0cStreamerInfo\x12Doubly linked list@\x00\x00\x11\x00\x05\x00\x01\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00"

    # magic = b"root"
    # fVersion = 1062206  # ROOT 6.22/06 is my model
    # fBEGIN = 100
    # fEND = 0  # will be replaced
    # fSeekFree = 0  # will be replaced
    # nfree = 0  # will be replaced
    # fNbytesName = 0  # will be replaced
    # fUnits = 8
    # fCompress = compression.code
    # fSeekInfo = 0  # will be replaced
    # fNbytesInfo = len(streamerinfo)
    # fUUID_version = uuid_version
    # if callable(uuid_function):
    #     fUUID = uuid_function()
    # elif isinstance(uuid_function, uuid.UUID):
    #     fUUID = uuid_function.bytes
    # elif isinstance(uuid_function, bytes):
    #     fUUID = uuid_function
    # else:
    #     fUUID = uuid.UUID(str(uuid_function)).bytes

    # uproot.reading._file_header_fields_big


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
            return len(self.serialize())
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


class WritableString(Writable):
    """
    FIXME: docstring
    """

    def __init__(self, location, string):
        super(self, WritableString).__init__(location, None)
        self._string = string

        bytestring = self._string.encode(errors="surrogateescape")
        length = len(bytestring)
        if length < 255:
            self._serialization = struct.pack(">B%ds" % length, length, bytestring)
        else:
            self._serialization = struct.pack(
                ">BI%ds" % length, 255, length, bytestring
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

    class_version = 5

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
        super(self, WritableKey).__init__(location, None)
        self._uncompressed_bytes = uncompressed_bytes
        self._compressed_bytes = compressed_bytes
        self._classname = classname
        self._name = name
        self._title = title
        self._cycle = cycle
        self._parent_location = parent_location

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
        return (
            self._location is None
            or self._location >= uproot.const.kStartBigFile
            or self._parent_location >= uproot.const.kStartBigFile
        )

    @property
    def num_bytes(self):
        if self.big:
            return (
                uproot.reading._key_format_big.size
                + self._classname.num_bytes
                + self._name.num_bytes
                + self._title.num_bytes
            )
        else:
            return (
                uproot.reading._key_format_small.size
                + self._classname.num_bytes
                + self._name.num_bytes
                + self._title.num_bytes
            )

    def serialize(self):
        if self._location is None:
            raise RuntimeError(
                "can't serialize key because location is unknown:\n\n    " + repr(self)
            )

        fNbytes = self._compressed_bytes + self.num_bytes
        fVersion = self.class_version + 1000 if self.big else self.class_version
        fObjlen = self._uncompressed_bytes
        fDatime = 1761927327  # FIXME: compute fDatime
        fKeylen = self.num_bytes
        fCycle = self._cycle
        fSeekKey = self._location
        fSeekPdir = self._parent_location

        if self.big:
            format = uproot.reading._key_format_big
        else:
            format = uproot.reading._key_format_small

        return (
            format.pack(
                fNbytes,
                fVersion,
                fObjlen,
                fDatime,
                fKeylen,
                fCycle,
                fSeekKey,
                fSeekPdir,
            )
            + self._classname.num_bytes.serialize()
            + self._name.num_bytes.serialize()
            + self._title.num_bytes.serialize()
        )


_free_format_small = struct.Struct(">HII")
_free_format_big = struct.Struct(">HQQ")


class WritableFrees(Writable):
    """
    FIXME: docstring
    """

    class_version = 1

    def __init__(self, location, slices, end):
        super(self, WritableFrees).__init__(location, None)
        self._slices = slices
        self._end = end

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
            if stop - 1 < uproot.const.kStartBigFile:
                total += _free_format_small.size()
            else:
                total += _free_format_big.size()

        if self._end < uproot.const.kStartBigFile:
            total += _free_format_small.size()
        else:
            total += _free_format_big.size()

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


class WritableFreesWithKey(Writable):
    """
    FIXME: docstring
    """

    def __init__(self, location, key, frees, file):
        super(self, WritableFreesWithKey).__init__(location, None)
        self._key = key
        self._frees = frees
        self._file = file

    @property
    def key(self):
        return self._key

    @property
    def frees(self):
        return self._frees

    @property
    def file(self):
        return self._file

    @property
    def num_bytes(self):
        return self._key.num_bytes + self._frees.num_bytes

    def serialize(self):
        return self._key.serialization + self._frees.serialization

    def write(self, sink):
        self._key.location = self._location
        self._key.uncompressed_bytes = (
            self._key.compressed_bytes
        ) = self._frees.num_bytes
        self._frees.location = self._location + self._key.num_bytes
        self._frees.end = self._frees.location + self._key.uncompressed_bytes
        self._file.free_location = self._location
        self._file.free_num_bytes = self._frees.end - self._location
        self._file.free_num_slices = len(self._frees.slices)
        self._key.write()
        self._frees.write()
        self._file.write()


class WritableFile(object):
    """
    FIXME: docstring
    """

    magic = b"root"
    class_version = 1062206  # ROOT 6.22/06 is our model
    begin = 100

    def __init__(
        self,
        end,
        free_location,
        free_num_bytes,
        free_num_slices,
        rootdir_keylen,
        compression,
        info_location,
        info_num_bytes,
        uuid_version,
        uuid,
    ):
        super(self, WritableFile).__init__(0, self.begin)
        self._end = end
        self._free_location = free_location
        self._free_num_bytes = free_num_bytes
        self._free_num_slices = free_num_slices
        self._rootdir_keylen = rootdir_keylen
        self._compression = compression
        self._info_location = info_location
        self._info_num_bytes = info_num_bytes
        self._uuid_version = uuid_version
        self._uuid = uuid

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
    def rootdir_keylen(self):
        return self._rootdir_keylen

    @rootdir_keylen.setter
    def rootdir_keylen(self, value):
        if self._rootdir_keylen != value:
            self._file_dirty = True
            self._rootdir_keylen = value

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

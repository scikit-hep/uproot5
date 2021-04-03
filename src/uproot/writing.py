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
        self._serialization = None
        self._file_dirty = True

    @property
    def location(self):
        return self._location

    @location.setter
    def location(self, value):
        self._location = value
        self._file_dirty = True

    @property
    def serialization(self):
        if self._serialization is None:
            self._serialization = self.serialize()
        return self._serialization

    @property
    def num_bytes(self):
        if self._serialization is None:
            self._serialization = self.serialize()
        return len(self._serialization)

    def write(self, sink):
        if self._file_dirty:
            if self._location is None:
                raise RuntimeError(
                    "can't write object because location is unknown:\n\n    "
                    + repr(self)
                )
            if self._serialization is None:
                self._serialization = self.serialize()
        sink.write(self._location, self._serialization)

    def serialize(self):
        raise AssertionError("Writable is abstract; 'serialize' must be overloaded")


class WritableString(Writable):
    """
    FIXME: docstring
    """

    def __init__(self, location, string):
        super(self, WritableString).__init__(location, None)
        self._string = string

    @property
    def string(self):
        return self._string

    def serialize(self):
        bytestring = self._string.encode(errors="surrogateescape")
        num_bytes = len(bytestring)
        if num_bytes < 255:
            self._serialization = struct.pack(
                ">B%ds" % num_bytes, num_bytes, bytestring
            )
        else:
            self._serialization = struct.pack(
                ">BI%ds" % num_bytes, 255, num_bytes, bytestring
            )


class WritableKey(Writable):
    """
    FIXME: docstring
    """

    fVersion = 5

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
    def location(self):
        return self._location

    @location.setter
    def location(self, value):
        self._location = value
        self._serialization = None
        self._file_dirty = True

    @property
    def uncompressed_bytes(self):
        return self._uncompressed_bytes

    @uncompressed_bytes.setter
    def uncompressed_bytes(self, value):
        self._uncompressed_bytes = value
        self._serialization = None
        self._file_dirty = True

    @property
    def compressed_bytes(self):
        return self._compressed_bytes

    @compressed_bytes.setter
    def compressed_bytes(self, value):
        self._compressed_bytes = value
        self._serialization = None
        self._file_dirty = True

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
            or self._location > uproot.const.kStartBigFile
            or self._parent_location > uproot.const.kStartBigFile
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

        fKeylen = self.num_bytes
        fNbytes = self._compressed_bytes + fKeylen

        if self.big:
            return (
                uproot.reading._key_format_big.pack(
                    fNbytes,
                    self.fVersion + 1000,
                    self._uncompressed_bytes,
                    1761927327,  # FIXME: compute fDatime
                    fKeylen,
                    self._cycle,
                    self._location,
                    self._parent_location,
                )
                + self._classname.num_bytes.serialization()
                + self._name.num_bytes.serialization()
                + self._title.num_bytes.serialization()
            )
        else:
            return (
                uproot.reading._key_format_small.pack(
                    fNbytes,
                    self.fVersion,
                    self._uncompressed_bytes,
                    1761927327,  # FIXME: compute fDatime
                    fKeylen,
                    self._cycle,
                    self._location,
                    self._parent_location,
                )
                + self._classname.num_bytes.serialization()
                + self._name.num_bytes.serialization()
                + self._title.num_bytes.serialization()
            )


class WritableFile(object):
    """
    FIXME: docstring
    """

    pass

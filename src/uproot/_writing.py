# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

"""
FIXME: docstring
"""

from __future__ import absolute_import

import datetime
import math
import os.path
import struct
import uuid

import numpy

import uproot.compression
import uproot.const
import uproot.models.TList
import uproot.reading
import uproot.sink.file
import uproot.source.chunk
import uproot.source.cursor
import uproot.streamers


class CascadeLeaf(object):
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
            self._allocation = self.num_bytes
        return self._allocation

    @property
    def num_bytes(self):
        return len(self.serialize())

    def serialize(self):
        raise AssertionError("CascadeLeaf is abstract; 'serialize' must be overloaded")

    @classmethod
    def deserialize(cls, raw_bytes, location):
        raise AssertionError(
            "CascadeLeaf is abstract; 'deserialize' must be overloaded"
        )

    def write(self, sink):
        if self._file_dirty:
            if self._location is None:
                raise RuntimeError(
                    "can't write object because location is unknown:\n\n    "
                    + repr(self)
                )
            sink.write(self._location, self.serialize())
            self._file_dirty = False


class CascadeNode(object):
    """
    FIXME: docstring
    """

    def __init__(self, *dependencies):
        self._dependencies = dependencies

    def write(self, sink):
        for dependency in self._dependencies:
            dependency.write(sink)


_string_size_format_4 = struct.Struct(">I")


class String(CascadeLeaf):
    """
    FIXME: docstring
    """

    def __init__(self, location, string):
        super(String, self).__init__(location, None)
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

    def copy_to(self, location):
        return String(location, self._string)

    def serialize(self):
        return self._serialization

    @classmethod
    def deserialize(cls, raw_bytes, location):
        num_bytes = ord(raw_bytes[:1])
        position = 1
        if num_bytes == 255:
            (num_bytes,) = _string_size_format_4.unpack(raw_bytes[1:5])
            position = 5
        out = raw_bytes[position : position + num_bytes]
        if not uproot._util.py2:
            out = out.decode(errors="surrogateescape")
        return String(location, out), location + position + num_bytes


class Key(CascadeLeaf):
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
        seek_location,
    ):
        super(Key, self).__init__(location, None)
        self._uncompressed_bytes = uncompressed_bytes
        self._compressed_bytes = compressed_bytes
        self._classname = classname
        self._name = name
        self._title = title
        self._cycle = cycle
        self._parent_location = parent_location
        self._seek_location = seek_location
        self._created_on = datetime.datetime.now()
        self._big = None

    def __repr__(self):
        return "{0}({1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9})".format(
            type(self).__name__,
            self._location,
            self._uncompressed_bytes,
            self._compressed_bytes,
            repr(self._classname),
            repr(self._name),
            repr(self._title),
            self._cycle,
            self._parent_location,
            self._seek_location,
        )

    @property
    def allocation(self):
        return self.num_bytes

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

    @parent_location.setter
    def parent_location(self, value):
        if self._parent_location != value:
            self._file_dirty = True
            self._big = None
            self._parent_location = value

    @property
    def seek_location(self):
        return self._seek_location

    @seek_location.setter
    def seek_location(self, value):
        if self._seek_location != value:
            self._file_dirty = True
            self._big = None
            self._seek_location = value

    def copy_to(self, location):
        position = location + self.num_bytes
        classname = self._classname.copy_to(position)

        position += classname.num_bytes
        name = self._name.copy_to(position)

        position += name.num_bytes
        title = self._title.copy_to(position)

        if self._seek_location is not None:
            location = self._seek_location
        else:
            location = self._location

        out = Key(
            location,
            self._uncompressed_bytes,
            self._compressed_bytes,
            classname,
            name,
            title,
            self._cycle,
            self._parent_location,
            location,
        )
        out._created_on = self._created_on
        return out

    @property
    def big(self):
        if self._big is not None:
            return self._big
        elif self._seek_location is not None:
            return (
                self._seek_location >= uproot.const.kStartBigFile
                or self._parent_location >= uproot.const.kStartBigFile
            )
        else:
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
        if self._seek_location is None and self._location is None:
            raise RuntimeError(
                "can't serialize key because location is unknown:\n\n    " + repr(self)
            )

        if self.big:
            format = uproot.reading._key_format_big
            version = self.class_version + 1000
        else:
            format = uproot.reading._key_format_small
            version = self.class_version

        if self._seek_location is not None:
            location = self._seek_location
        else:
            location = self._location

        return (
            format.pack(
                self._compressed_bytes + self.num_bytes,  # fNbytes
                version,  # fVersion
                self._uncompressed_bytes,  # fObjlen
                1761927327,  # FIXME: compute fDatime
                self.num_bytes,  # fKeylen
                self._cycle,  # fCycle
                location,  # fSeekKey
                self._parent_location,  # fSeekPdir
            )
            + self._classname.serialize()
            + self._name.serialize()
            + self._title.serialize()
        )

    @classmethod
    def deserialize(cls, raw_bytes, location, in_path, is_directory_key=False):
        (
            fNbytes,
            version,
            fObjlen,
            fDatime,
            fKeylen,
            fCycle,
            fSeekKey,
            fSeekPdir,
        ) = uproot.reading._key_format_small.unpack(
            raw_bytes[: uproot.reading._key_format_small.size]
        )
        position = location + uproot.reading._key_format_small.size
        big = False

        if version >= 1000:
            (
                fNbytes,
                version,
                fObjlen,
                fDatime,
                fKeylen,
                fCycle,
                fSeekKey,
                fSeekPdir,
            ) = uproot.reading._key_format_big.unpack(
                raw_bytes[: uproot.reading._key_format_big.size]
            )
            version -= 1000
            position = location + uproot.reading._key_format_big.size
            big = True

        if version != cls.class_version:
            raise ValueError(
                "Uproot can't read TKey version {0} for writing, only version {1}{2}".format(
                    version,
                    cls.class_version,
                    in_path,
                )
            )

        assert 0 < fNbytes <= fKeylen + fObjlen
        assert fCycle > 0
        if not is_directory_key:
            assert fSeekKey == location

        classname, position = String.deserialize(
            raw_bytes[position - location :], position
        )
        name, position = String.deserialize(raw_bytes[position - location :], position)
        title, position = String.deserialize(raw_bytes[position - location :], position)

        assert fKeylen == position - location

        out = Key(
            location,
            fObjlen,  # uncompressed_bytes
            fNbytes - fKeylen,  # compressed_bytes
            classname,
            name,
            title,
            fCycle,  # cycle
            fSeekPdir,  # parent_location
            fSeekKey,  # may be location
        )
        out._created_on = datetime.datetime.now()  # FIXME: compute from fDatime
        out._big = big
        return out


_free_format_small = struct.Struct(">HII")
_free_format_big = struct.Struct(">HQQ")


class FreeSegmentsData(CascadeLeaf):
    """
    FIXME: docstring
    """

    class_version = 1

    def __init__(self, location, slices, end):
        super(FreeSegmentsData, self).__init__(location, None)
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
    def allocation(self):
        if self._allocation is None:
            self._allocation = self.num_bytes
        return self._allocation

    @allocation.setter
    def allocation(self, value):
        if self._allocation != value:
            self._file_dirty = True
            self._allocation = value

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
                pairs.append(
                    _free_format_big.pack(self.class_version + 1000, start, stop - 1)
                )

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
            pairs.append(
                _free_format_big.pack(self.class_version + 1000, self._end, infinity)
            )

        return b"".join(pairs)

    @classmethod
    def deserialize(cls, raw_bytes, location, num_bytes, num_slices, in_path):
        slices = []
        position = 0
        for _ in range(num_slices + 1):
            version, fFirst, fLast = _free_format_small.unpack(
                raw_bytes[: _free_format_small.size]
            )
            if version >= 1000:
                version, fFirst, fLast = _free_format_small.unpack(
                    raw_bytes[: _free_format_big.size]
                )
                version -= 1000
                position += _free_format_big.size
            else:
                position += _free_format_small.size

            if version != cls.class_version:
                raise ValueError(
                    "Uproot can't read TFree version {0} for writing, only version {1}{2}".format(
                        version,
                        cls.class_version,
                        in_path,
                    )
                )

            slices.append((fFirst, fLast + 1))

        end = slices.pop()[0]

        assert position == num_bytes

        out = FreeSegmentsData(location, tuple(slices), end)
        out._allocation = num_bytes
        return out


class FreeSegments(CascadeNode):
    """
    FIXME: docstring
    """

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

    @property
    def at_end(self):
        end_of_record = self._key.location + self._key.num_bytes + self._data.allocation
        assert end_of_record <= self._data.end
        return end_of_record == self._data.end

    def allocate(self, num_bytes):
        slices = self._data.slices
        for i, (start, stop) in enumerate(slices):
            if stop - start == num_bytes:
                # This will reduce the num_bytes of the FreeSegments record,
                # but the allocation can stay the same size.
                self._data.slices = tuple(
                    slices[j] for j in range(len(slices)) if i != j
                )
                return start

            elif stop - start > num_bytes:
                # This will not change the num_bytes of the FreeSegments record.
                self._data.slices = tuple(
                    slices[j] if i != j else (start + num_bytes, stop)
                    for j in range(len(slices))
                )
                return start

        if self.at_end:
            # The new object can take FreeSegments's spot; FreeSegments will
            # move to stay at the end.
            out = self._key.location
            self._key.location = self._key.location + num_bytes
            self._data.end = (
                self._key.location + self._key.allocation + self._data.allocation
            )
            return out

        else:
            # FreeSegments is not changing size and not at the end; it can
            # stay where it is.
            out = self._data.end
            self._data.end = self._data.end + num_bytes
            return out

    @staticmethod
    def _another_slice(slices, original_start, original_stop):
        for start, stop in slices:
            if start <= original_start < stop or start < original_stop <= stop:
                raise RuntimeError(
                    "segment of data to release overlaps one already marked as free: "
                    "releasing [{0}, {1}) but [{2}, {3}) is free".format(
                        original_start, original_stop, start, stop
                    )
                )

        for i, (start, stop) in enumerate(slices):
            if original_start == stop:
                # This slice needs to grow to the right.
                return tuple(
                    slices[j] if i != j else (start, original_stop)
                    for j in range(len(slices))
                )

            elif original_stop == start:
                # This slice needs to grow to the left.
                return tuple(
                    slices[j] if i != j else (original_start, stop)
                    for j in range(len(slices))
                )

        # The FreeSegments record will have to grow.
        return tuple(sorted(slices + ((original_start, original_stop),)))

    @staticmethod
    def _slices_bytes(slices):
        total = 0
        for _, stop in slices:
            if stop - 1 >= uproot.const.kStartBigFile:
                total += _free_format_big.size
            else:
                total += _free_format_small.size
        return total

    def release(self, start, stop):
        new_slices = self._another_slice(self._data.slices, start, stop)

        if self.at_end:
            self._data.slices = new_slices
            self._data.allocation = None
            self._key.uncompressed_bytes = self._data.allocation
            self._key.compressed_bytes = self._key.uncompressed_bytes
            self._data.end = (
                self._key.location + self._key.allocation + self._key.uncompressed_bytes
            )

        elif self._slices_bytes(new_slices) <= self._slices_bytes(self._data.slices):
            # Wherever the FreeSegments record is, it's not getting bigger.
            # It can stay there.
            self._data.slices = new_slices
            self._data.allocation = None
            self._key.uncompressed_bytes = self._data.allocation
            self._key.compressed_bytes = self._key.uncompressed_bytes

        else:
            # The FreeSegments record needs to move, opening up yet another slice.
            # Move it to the end (regardless of whether there's now enough room
            # to put it elsewhere; we like keeping it at the end).
            self._data.slices = self._another_slice(
                new_slices,
                self._key.location,
                self._key.location + self._key.allocation + self._data.allocation,
            )
            self._data.allocation = None
            self._key.uncompressed_bytes = self._data.allocation
            self._key.compressed_bytes = self._key.uncompressed_bytes
            self._key.location = self._data.end
            self._data.location = self._key.location + self._key.allocation
            self._data.end = self._data.location + self._key.uncompressed_bytes

    def write(self, sink):
        self._key.uncompressed_bytes = self._data.allocation
        self._key.compressed_bytes = self._key.uncompressed_bytes
        self._data.location = self._key.location + self._key.allocation
        self._fileheader.free_location = self._key.location
        self._fileheader.free_num_bytes = self._data.end - self._key.location
        self._fileheader.free_num_slices = len(self._data.slices)
        self._fileheader.end = self._data.end
        super(FreeSegments, self).write(sink)


class StreamersData(CascadeLeaf):
    """
    FIXME: docstring
    """

    def __init__(self, location, allocation):
        super(StreamersData, self).__init__(location, allocation)
        self._serialization = b"@\x00\x00\x11\x00\x05\x00\x01\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00"
        # self._serialization = b'@\x00\x01n\x00\x05\x00\x01\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x01@\x00\x01X\xff\xff\xff\xffTStreamerInfo\x00@\x00\x01B\x00\t@\x00\x00\x18\x00\x01\x00\x01\x00\x00\x00\x00\x03\x01\x00\x00\nTObjString\x00\x9c\x8eH\x00\x00\x00\x00\x01@\x00\x01\x18\xff\xff\xff\xffTObjArray\x00@\x00\x01\x06\x00\x03\x00\x01\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00@\x00\x00u\xff\xff\xff\xffTStreamerBase\x00@\x00\x00_\x00\x03@\x00\x00U\x00\x04@\x00\x00&\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x07TObject\x11Basic ROOT object\x00\x00\x00B\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x90\x1b\xc0-\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x01@\x00\x00t\xff\xff\xff\xffTStreamerString\x00@\x00\x00\\\x00\x02@\x00\x00V\x00\x04@\x00\x00$\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x07fString\x0fwrapped TString\x00\x00\x00A\x00\x00\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x07TString\x00'
        # self._serialization = b'@\x00\x02\xcb\x00\x05\x00\x01\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x02@\x00\x01X\xff\xff\xff\xffTStreamerInfo\x00@\x00\x01B\x00\t@\x00\x00\x18\x00\x01\x00\x01\x00\x00\x00\x00\x03\x01\x00\x00\nTObjString\x00\x9c\x8eH\x00\x00\x00\x00\x01@\x00\x01\x18\xff\xff\xff\xffTObjArray\x00@\x00\x01\x06\x00\x03\x00\x01\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00@\x00\x00u\xff\xff\xff\xffTStreamerBase\x00@\x00\x00_\x00\x03@\x00\x00U\x00\x04@\x00\x00&\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x07TObject\x11Basic ROOT object\x00\x00\x00B\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x90\x1b\xc0-\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x01@\x00\x00t\xff\xff\xff\xffTStreamerString\x00@\x00\x00\\\x00\x02@\x00\x00V\x00\x04@\x00\x00$\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x07fString\x0fwrapped TString\x00\x00\x00A\x00\x00\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x07TString\x00@\x00\x01X\xff\xff\xff\xffTStreamerInfo\x00@\x00\x01B\x00\t@\x00\x00\x18\x00\x01\x00\x01\x00\x00\x00\x00\x03\x01\x00\x00\nTObjString\x00\x9c\x8eH\x00\x00\x00\x00\x01@\x00\x01\x18\xff\xff\xff\xffTObjArray\x00@\x00\x01\x06\x00\x03\x00\x01\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00@\x00\x00u\xff\xff\xff\xffTStreamerBase\x00@\x00\x00_\x00\x03@\x00\x00U\x00\x04@\x00\x00&\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x07TObject\x11Basic ROOT object\x00\x00\x00B\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x90\x1b\xc0-\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x01@\x00\x00t\xff\xff\xff\xffTStreamerString\x00@\x00\x00\\\x00\x02@\x00\x00V\x00\x04@\x00\x00$\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x07fString\x0fwrapped TString\x00\x00\x00A\x00\x00\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x07TString\x00'

    def __repr__(self):
        return "{0}({1}, {2})".format(
            type(self).__name__,
            self._location,
            self._allocation,
        )

    def serialize(self):
        return self._serialization

    @classmethod
    def deserialize(cls, raw_bytes, location):
        out = StreamersData(location, len(raw_bytes))
        out._serialization = raw_bytes
        return out


_tlistheader_format = struct.Struct(">IHHIIBI")


class TListHeader(CascadeLeaf):
    """
    FIXME: docstring
    """

    class_version = 5
    tobject_version = 1

    def __init__(self, location, data_bytes, num_entries):
        super(TListHeader, self).__init__(location, _tlistheader_format.size)
        self._data_bytes = data_bytes
        self._num_entries = num_entries

    def __repr__(self):
        return "{0}({1}, {2}, {3})".format(
            type(self).__name__, self._location, self._data_bytes, self._num_entries
        )

    @property
    def data_bytes(self):
        return self._data_bytes

    @data_bytes.setter
    def data_bytes(self, value):
        if self._data_bytes != value:
            self._file_dirty = True
            self._data_bytes = value

    @property
    def num_entries(self):
        return self._num_entries

    @num_entries.setter
    def num_entries(self, value):
        if self._num_entries != value:
            self._file_dirty = True
            self._num_entries = value

    @property
    def num_bytes(self):
        return _tlistheader_format.size

    def serialize(self):
        return _tlistheader_format.pack(
            numpy.uint32(self._data_bytes - 4) | uproot.const.kByteCountMask,
            self.class_version,
            self.tobject_version,
            0,
            uproot.const.kNotDeleted,
            0,
            self._num_entries,
        )


class RawStreamerInfo(CascadeLeaf):
    """
    FIXME: docstring
    """

    def __init__(self, location, serialization, name, version):
        super(RawStreamerInfo, self).__init__(location, len(serialization))
        self._serialization = serialization
        self._name = name
        self._version = version

    @property
    def name(self):
        return self._name

    @property
    def version(self):
        return self._version

    def __repr__(self):
        return "{0}({1}, {2}, {3}, {4})".format(
            type(self).__name__,
            self._location,
            self._serialization,
            self._name,
            self._version,
        )

    def serialize(self):
        return self._serialization


class TListOfStreamers(CascadeNode):
    """
    FIXME: docstring
    """

    def __init__(self, allocation, key, header, rawstreamers, freesegments):
        super(TListOfStreamers, self).__init__(freesegments, key, header, *rawstreamers)
        self._allocation = allocation
        self._key = key
        self._header = header
        self._rawstreamers = rawstreamers
        self._freesegments = freesegments

    def __repr__(self):
        return "{0}({1}, {2}, {3}, {4}, {5})".format(
            type(self).__name__,
            self._allocation,
            self._key,
            self._header,
            self._rawstreamers,
            self._freesegments,
        )

    @property
    def allocation(self):
        return self._allocation

    @property
    def num_bytes(self):
        return self._header.num_bytes + sum(x.num_bytes for x in self._rawstreamers)

    @property
    def key(self):
        return self._key

    @property
    def header(self):
        return self._header

    @property
    def freesegments(self):
        return self._freesegments

    def _reallocate(self, self_num_bytes):
        original_start = self._key.location
        original_stop = self._key.location + self._key.allocation + self._allocation

        requested_num_bytes = self._key.num_bytes + self_num_bytes
        self._key.location = self._freesegments.allocate(requested_num_bytes)
        self._key.seek_location = self._key.location
        self._allocation = self_num_bytes

        self._freesegments.release(original_start, original_stop)

    def write(self, sink):
        self_num_bytes = self.num_bytes
        if self_num_bytes > self.allocation:
            self._reallocate(self_num_bytes)

        position = afterkey = self._key.location + self._key.num_bytes

        self._header.location = position
        position += self._header.num_bytes

        for rawstreamer in self._rawstreamers:
            rawstreamer.location = position
            position += rawstreamer.num_bytes

        self._header.data_bytes = position - afterkey
        self._header.num_entries = len(self._rawstreamers)

        self._key.uncompressed_bytes = self._allocation
        self._key.compressed_bytes = self._key.uncompressed_bytes
        self._freesegments.fileheader.info_location = self._key.location
        self._freesegments.fileheader.info_num_bytes = (
            self._key.allocation + self._allocation
        )

        super(TListOfStreamers, self).write(sink)

    @classmethod
    def deserialize(cls, raw_bytes, location, key, freesegments, file_path):
        readforupdate = _ReadForUpdate(file_path)

        chunk = uproot.source.chunk.Chunk.wrap(readforupdate, raw_bytes)

        if key.compressed_bytes == key.uncompressed_bytes:
            uncompressed = chunk
        else:
            uncompressed = uproot.compression.decompress(
                chunk,
                uproot.source.cursor.Cursor(0),
                {},
                key.compressed_bytes,
                key.uncompressed_bytes,
            )

        tlist = uproot.models.TList.Model_TList.read(
            uncompressed,
            uproot.source.cursor.Cursor(0),
            {},
            readforupdate,
            readforupdate,
            None,
        )

        header = TListHeader(location, key.uncompressed_bytes, len(tlist))

        rawstreamers = []
        for (start, stop), streamer in zip(tlist.byte_ranges, tlist):
            rawstreamers.append(
                RawStreamerInfo(
                    location + start,
                    uncompressed.raw_data[start:stop].tobytes(),
                    streamer.name,
                    streamer.class_version,
                )
            )

        return TListOfStreamers(
            key.compressed_bytes,
            key,
            header,
            rawstreamers,
            freesegments,
        )


class _ReadForUpdate(object):
    def __init__(self, file_path):
        self.file_path = file_path

    @staticmethod
    def class_named(classname):
        return uproot.classes[classname]


class Streamers(CascadeNode):
    """
    FIXME: docstring
    """

    def __init__(self, key, data, freesegments):
        super(Streamers, self).__init__(freesegments, key, data)
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
        return self._data.allocation

    def write(self, sink):
        self._key.uncompressed_bytes = self._data.allocation
        self._key.compressed_bytes = self._key.uncompressed_bytes
        self._data.location = self._key.location + self._key.allocation
        self._freesegments.fileheader.info_location = self._key.location
        self._freesegments.fileheader.info_num_bytes = (
            self._key.allocation + self._data.allocation
        )
        super(Streamers, self).write(sink)


class DirectoryData(CascadeLeaf):
    """
    FIXME: docstring
    """

    def __init__(self, location, allocation, keys):
        super(DirectoryData, self).__init__(location, allocation)
        self._keys = keys

    def __repr__(self):
        return "{0}({1}, {2}, {3})".format(
            type(self).__name__,
            self._location,
            self._allocation,
            self._keys,
        )

    @property
    def allocation(self):
        if self._allocation is None:
            self._allocation = self.num_bytes
        return self._allocation

    @allocation.setter
    def allocation(self, value):
        if self._allocation != value:
            self._file_dirty = True
            self._allocation = value

    def next_cycle(self, name):
        cycle = 1
        for key in self._keys:
            if name == key.name:
                cycle = max(cycle, key.cycle + 1)
        return cycle

    def add_key(self, key):
        self._file_dirty = True
        self._keys.append(key)

    @property
    def num_bytes(self):
        return uproot.reading._directory_format_num_keys.size + sum(
            x.allocation for x in self._keys
        )

    @property
    def next_location(self):
        return self._location + self.num_bytes

    def serialize(self):
        out = [uproot.reading._directory_format_num_keys.pack(len(self._keys))]
        for key in self._keys:
            out.append(key.serialize())
        return b"".join(out)

    @classmethod
    def deserialize(cls, raw_bytes, location, in_path):
        (num_keys,) = uproot.reading._directory_format_num_keys.unpack(raw_bytes[:4])
        position = location + 4
        keys = []
        for _ in range(num_keys):
            keys.append(
                Key.deserialize(
                    raw_bytes[position - location :],
                    position,
                    in_path,
                    is_directory_key=True,
                )
            )
        return DirectoryData(location, len(raw_bytes), keys)


class DirectoryHeader(CascadeLeaf):
    """
    FIXME: docstring
    """

    class_version = 5

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
        super(DirectoryHeader, self).__init__(
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
            repr(self._uuid),
        )

    @property
    def begin_location(self):
        return self._begin_location

    @begin_location.setter
    def begin_location(self, value):
        if self._begin_location != value:
            self._file_dirty = True
            self._begin_location = value

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
        return (
            self._begin_location >= uproot.const.kStartBigFile
            or self._data_location >= uproot.const.kStartBigFile
            or self._parent_location >= uproot.const.kStartBigFile
        )

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
            + _directory_uuid_version_format.pack(self._uuid_version)
            + self._uuid.bytes
            + extra
        )

    @classmethod
    def deserialize(cls, raw_bytes, location, in_path):
        (
            version,
            fDatimeC,
            fDatimeM,
            fNbytesKeys,
            fNbytesName,
            fSeekDir,
            fSeekParent,
            fSeekKeys,
        ) = uproot.reading._directory_format_small.unpack(
            raw_bytes[: uproot.reading._directory_format_small.size]
        )
        position = location + uproot.reading._directory_format_small.size

        if version >= 1000:
            (
                version,
                fDatimeC,
                fDatimeM,
                fNbytesKeys,
                fNbytesName,
                fSeekDir,
                fSeekParent,
                fSeekKeys,
            ) = uproot.reading._directory_format_big.unpack(
                raw_bytes[: uproot.reading._directory_format_big.size]
            )
            version -= 1000
            position = location + uproot.reading._directory_format_big.size

        if version != cls.class_version:
            raise ValueError(
                "Uproot can't read TDirectory version {0} for writing, only version {1}{2}".format(
                    version,
                    cls.class_version,
                    in_path,
                )
            )

        (uuid_version,) = _directory_uuid_version_format.unpack(
            raw_bytes[position - location : position - location + 2]
        )

        uuid_bytes = raw_bytes[position - location + 2 : position - location + 18]

        out = DirectoryHeader(
            location,
            fSeekDir,  # begin_location
            fNbytesName,  # begin_num_bytes
            fSeekKeys,  # data_location
            fNbytesKeys,  # data_num_bytes
            fSeekParent,  # parent_location
            uuid_version,
            uuid.UUID(bytes=uuid_bytes),
        )
        out._created_on = datetime.datetime.now()  # FIXME: compute from fDatimeC
        out._modified_on = out._created_on  # FIXME: compute from fDatimeM
        return out


_directory_uuid_version_format = struct.Struct(">H")


class Directory(CascadeNode):
    """
    FIXME: docstring
    """

    def _reallocate_data(self, new_data_size):
        original_start = self._datakey.location
        original_stop = original_start + self._datakey.num_bytes + self._data.allocation

        self._datakey.location = None  # let it assume the key might be big
        requested_num_bytes = self._datakey.num_bytes + new_data_size
        self._datakey.location = self._freesegments.allocate(requested_num_bytes)
        self._header.data_location = self._datakey.location
        self._data.location = self._datakey.location + self._datakey.num_bytes
        might_be_slightly_more = requested_num_bytes - self._datakey.num_bytes
        self._data.allocation = might_be_slightly_more

        self._freesegments.release(original_start, original_stop)

    def add_directory(self, sink, name, initial_directory_bytes, uuid_version, uuid):
        cycle = self._data.next_cycle(name)

        subdirectory_key = Key(
            None,
            None,
            None,
            String(None, "TDirectory"),
            String(None, name),
            String(None, name),
            cycle,
            self._key.location,
            None,
        )
        subdirectory_header = DirectoryHeader(
            None,
            None,
            None,
            None,
            None,
            self._key.location,
            uuid_version,
            uuid,
        )
        subdirectory_datakey = Key(
            None,
            None,
            None,
            String(None, "TDirectory"),
            String(None, name),
            String(None, name),
            cycle,
            self._key.location,
            None,
        )

        requested_num_bytes = (
            subdirectory_key.num_bytes
            + subdirectory_header.allocation
            + subdirectory_datakey.num_bytes
            + initial_directory_bytes
        )
        subdirectory_key.location = self._freesegments.allocate(requested_num_bytes)
        subdirectory_datakey.location = (
            subdirectory_key.location
            + subdirectory_key.num_bytes
            + subdirectory_header.allocation
        )
        might_be_slightly_more = requested_num_bytes - (
            subdirectory_key.num_bytes  # because Key.num_bytes depends on location
            + subdirectory_header.allocation
            + subdirectory_datakey.num_bytes  # including this Key, too
        )
        subdirectory_data = DirectoryData(None, might_be_slightly_more, [])

        subdirectory = SubDirectory(
            subdirectory_key,
            subdirectory_header,
            subdirectory_datakey,
            subdirectory_data,
            self,
            self._freesegments,
        )

        subdirectory_key.uncompressed_bytes = subdirectory_header.allocation
        subdirectory_key.compressed_bytes = subdirectory_key.uncompressed_bytes

        next_key = subdirectory_key.copy_to(self._data.next_location)
        if self._data.num_bytes + next_key.num_bytes > self._data.allocation:
            self._reallocate_data(
                int(math.ceil(1.5 * (self._data.allocation + next_key.num_bytes + 8)))
            )
            next_key = subdirectory_key.copy_to(self._data.next_location)
        self._data.add_key(next_key)

        self._freesegments.write(sink)
        subdirectory.write(sink)
        self.write(sink)

        return subdirectory


class RootDirectory(Directory):
    """
    FIXME: docstring
    """

    def __init__(self, key, name, title, header, datakey, data, freesegments):
        super(RootDirectory, self).__init__(
            freesegments, datakey, data, key, name, title, header
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
    def location(self):
        return self._key.location

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
        self._header.begin_location = self._key.location
        self._header.begin_num_bytes = self.begin_num_bytes
        self._header.data_location = self._datakey.location
        self._header.data_num_bytes = self._datakey.allocation + self._data.allocation
        self._datakey.uncompressed_bytes = self._data.allocation
        self._datakey.compressed_bytes = self._datakey.uncompressed_bytes
        self._data.location = self._datakey.location + self._datakey.allocation
        self._freesegments.fileheader.begin_num_bytes = self.begin_num_bytes
        super(RootDirectory, self).write(sink)


class SubDirectory(Directory):
    """
    FIXME: docstring
    """

    def __init__(self, key, header, datakey, data, parent, freesegments):
        super(SubDirectory, self).__init__(
            freesegments, datakey, data, key, header, parent
        )
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

    @property
    def location(self):
        return self._key.location

    def write(self, sink):
        self._key.uncompressed_bytes = self._header.allocation
        self._key.compressed_bytes = self._key.uncompressed_bytes
        self._header.location = self._key.location + self._key.allocation
        self._header.begin_location = self._key.location
        self._header.begin_num_bytes = self._key.num_bytes
        self._header.data_location = self._datakey.location
        self._header.data_num_bytes = self._datakey.allocation + self._data.allocation
        self._datakey.uncompressed_bytes = self._data.allocation
        self._datakey.compressed_bytes = self._datakey.uncompressed_bytes
        self._data.location = self._datakey.location + self._datakey.allocation
        super(SubDirectory, self).write(sink)


class FileHeader(CascadeLeaf):
    """
    FIXME: docstring
    """

    magic = b"root"
    class_version = 62206  # ROOT 6.22/06 is our model

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
        super(FileHeader, self).__init__(0, 100)
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
        self._version = None
        self._begin = 100

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
    def version(self):
        if self._version is None:
            return self.class_version
        else:
            return self._version

    @property
    def begin(self):
        return self._begin

    @property
    def big(self):
        return (
            self._end is None
            or self._end >= uproot.const.kStartBigFile
            or self._free_location >= uproot.const.kStartBigFile
            or self._info_location >= uproot.const.kStartBigFile
        )

    @property
    def num_bytes(self):
        if self.big:
            return uproot.reading._file_header_fields_big.size
        else:
            return uproot.reading._file_header_fields_small.size

    def serialize(self):
        if self.big:
            format = uproot.reading._file_header_fields_big
            version = self.version + 1000000
            units = 8
        else:
            format = uproot.reading._file_header_fields_small
            version = self.version
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

    @classmethod
    def deserialize(cls, raw_bytes, location):
        (
            magic,
            version,
            begin,
            end,
            free_location,
            free_num_bytes,
            free_num_slices_plus_1,
            begin_num_bytes,
            units,
            compression_code,
            info_location,
            info_num_bytes,
            uuid_version,
            uuid_bytes,
        ) = uproot.reading._file_header_fields_small.unpack(
            raw_bytes[: uproot.reading._file_header_fields_small.size]
        )
        assert begin >= uproot.reading._file_header_fields_small.size
        assert free_location >= 0
        assert free_num_bytes >= 0
        assert free_num_slices_plus_1 >= 1
        assert begin_num_bytes >= 0
        assert compression_code >= 0
        assert info_location >= 0
        assert info_num_bytes >= 0
        assert uuid_version >= 0

        if version >= 1000000:
            (
                magic,
                version,
                begin,
                end,
                free_location,
                free_num_bytes,
                free_num_slices_plus_1,
                begin_num_bytes,
                units,
                compression_code,
                info_location,
                info_num_bytes,
                uuid_version,
                uuid_bytes,
            ) = uproot.reading._file_header_fields_big.unpack(raw_bytes)
            assert units == 8
        else:
            assert units == 4

        out = FileHeader(
            end,
            free_location,
            free_num_bytes,
            free_num_slices_plus_1 - 1,
            begin_num_bytes,
            uproot.compression.Compression.from_code(compression_code),
            info_location,
            info_num_bytes,
            uuid_version,
            uuid.UUID(bytes=uuid_bytes),
        )
        out._version = version - 1000000
        out._begin = begin
        return out


class CascadingFile(object):
    """
    FIXME: docstring
    """

    def __init__(
        self,
        fileheader,
        streamers,
        freesegments,
        rootdirectory,
    ):
        self._fileheader = fileheader
        self._streamers = streamers
        self._freesegments = freesegments
        self._rootdirectory = rootdirectory

    def __repr__(self):
        return "{0}({1}, {2}, {3}, {4})".format(
            type(self).__name__,
            self._fileheader,
            self._streamers,
            self._freesegments,
            self._rootdirectory,
        )

    @property
    def fileheader(self):
        return self._fileheader

    @property
    def streamers(self):
        return self._streamers

    @property
    def freesegments(self):
        return self._freesegments

    @property
    def rootdirectory(self):
        return self._rootdirectory


def create_empty(
    sink,
    compression,
    initial_directory_bytes,
    initial_streamers_bytes,
    uuid_version,
    uuid_function,
):
    """
    FIXME: docstring
    """
    filename = sink.file_path
    if filename is None:
        filename = "dynamic.root"
    else:
        filename = os.path.split(filename)[-1]
    if len(filename) >= 256:
        raise ValueError("ROOT file names must be less than 256 bytes")

    fileheader = FileHeader(
        None,
        None,
        None,
        None,
        None,
        compression,
        None,
        None,
        uuid_version,
        uuid_function(),
    )

    freesegments_key = Key(
        None,
        None,
        None,
        String(None, "TFile"),
        String(None, filename),
        String(None, ""),
        1,
        fileheader.begin,
        None,
    )
    freesegments_data = FreeSegmentsData(None, (), None)
    freesegments = FreeSegments(freesegments_key, freesegments_data, fileheader)

    streamers_key = Key(
        None,
        None,
        None,
        String(None, "TList"),
        String(None, "StreamerInfo"),
        String(None, "Doubly linked list"),
        1,
        fileheader.begin,
        None,
    )
    streamers_header = TListHeader(None, None, None)
    streamers = TListOfStreamers(
        initial_streamers_bytes, streamers_key, streamers_header, [], freesegments
    )

    directory_key = Key(
        None,
        None,
        None,
        String(None, "TFile"),
        String(None, filename),
        String(None, ""),
        1,
        0,
        None,
    )
    directory_name = String(None, filename)
    directory_title = String(None, "")
    directory_header = DirectoryHeader(
        None, fileheader.begin, None, None, None, 0, uuid_version, uuid_function()
    )
    directory_datakey = Key(
        None,
        None,
        None,
        String(None, "TFile"),
        String(None, filename),
        String(None, ""),
        1,
        fileheader.begin,
        None,
    )
    directory_data = DirectoryData(None, initial_directory_bytes, [])
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
    directory_datakey.location = (
        streamers_key.location + streamers_key.allocation + streamers.allocation
    )
    directory_data.location = directory_datakey.location + directory_datakey.allocation
    freesegments_key.location = directory_data.location + directory_data.allocation
    freesegments_data.end = (
        freesegments_key.location
        + freesegments_key.allocation
        + freesegments_data.allocation
    )
    fileheader.info_location = streamers_key.location
    fileheader.info_num_bytes = streamers_key.allocation + streamers.allocation

    rootdirectory.write(sink)
    streamers.write(sink)

    return CascadingFile(fileheader, streamers, freesegments, rootdirectory)


def update_existing(
    sink,
    initial_directory_bytes,
    uuid_version,
    uuid_function,
):
    """
    FIXME: docstring
    """
    raw_bytes = sink.read(
        0,
        uproot.reading._file_header_fields_big.size,
        insist=uproot.reading._file_header_fields_small.size,
    )
    if raw_bytes[:4] != b"root":
        raise ValueError(
            "not a ROOT file: first four bytes are {0}{1}".format(
                repr(raw_bytes[:4]), sink.in_path
            )
        )
    fileheader = FileHeader.deserialize(raw_bytes, 0)

    raw_bytes = sink.read(fileheader.free_location, fileheader.free_num_bytes)
    freesegments_key = Key.deserialize(
        raw_bytes, fileheader.free_location, sink.in_path
    )

    freesegments_data = FreeSegmentsData.deserialize(
        raw_bytes[freesegments_key.num_bytes :],
        fileheader.free_location + freesegments_key.num_bytes,
        fileheader.free_num_bytes - freesegments_key.num_bytes,
        fileheader.free_num_slices,
        sink.in_path,
    )

    freesegments = FreeSegments(freesegments_key, freesegments_data, fileheader)

    raw_bytes = sink.read(fileheader.info_location, fileheader.info_num_bytes)
    streamers_key = Key.deserialize(raw_bytes, fileheader.info_location, sink.in_path)
    streamers = TListOfStreamers.deserialize(
        raw_bytes[streamers_key.num_bytes :],
        fileheader.info_location + streamers_key.num_bytes,
        streamers_key,
        freesegments,
        sink.file_path,
    )

    raw_bytes = sink.read(
        fileheader.begin,
        fileheader.begin_num_bytes + uproot.reading._directory_format_big.size + 18,
    )
    directory_key = Key.deserialize(raw_bytes, fileheader.begin, sink.in_path)
    position = fileheader.begin + directory_key.num_bytes

    directory_name, position = String.deserialize(
        raw_bytes[position - fileheader.begin :], position
    )
    directory_title, position = String.deserialize(
        raw_bytes[position - fileheader.begin :], position
    )

    assert fileheader.begin_num_bytes == position - fileheader.begin

    directory_header = DirectoryHeader.deserialize(
        raw_bytes[position - fileheader.begin :], position, sink.in_path
    )
    assert directory_header.begin_location == fileheader.begin
    assert directory_header.begin_num_bytes == fileheader.begin_num_bytes
    assert directory_header.parent_location == 0

    raw_bytes = sink.read(
        directory_header.data_location, directory_header.data_num_bytes
    )
    directory_datakey = Key.deserialize(
        raw_bytes, directory_header.data_location, sink.in_path
    )
    directory_data = DirectoryData.deserialize(
        raw_bytes[directory_datakey.num_bytes :],
        directory_header.data_location + directory_datakey.num_bytes,
        sink.in_path,
    )

    rootdirectory = RootDirectory(
        directory_key,
        directory_name,
        directory_title,
        directory_header,
        directory_datakey,
        directory_data,
        freesegments,
    )

    return CascadingFile(fileheader, streamers, freesegments, rootdirectory)

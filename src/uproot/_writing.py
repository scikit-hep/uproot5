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

try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping

import numpy

import uproot.compression
import uproot.const
import uproot.models.TList
import uproot.reading
import uproot.serialization
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
            tmp = self.serialize()
            # print(f"writing {self._location}:{self._location + len(tmp)} ({len(tmp)}) {type(self).__name__} {self.name if hasattr(self, 'name') else ''} {self.title if hasattr(self, 'title') else ''}")
            sink.write(self._location, tmp)
            self._file_dirty = False


class CascadeNode(object):
    """
    FIXME: docstring
    """

    def __init__(self, *dependencies):
        self._dependencies = dependencies

    def more_dependencies(self, *dependencies):
        self._dependencies = self._dependencies + dependencies

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
        self._serialization = uproot.serialization.string(self._string)

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
        created_on=None,
        big=None,
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
        self._created_on = datetime.datetime.now() if created_on is None else created_on
        self._big = big

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

        return Key(
            location,
            self._uncompressed_bytes,
            self._compressed_bytes,
            classname,
            name,
            title,
            self._cycle,
            self._parent_location,
            location,
            created_on=self._created_on,
            big=self._big,
        )

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
                uproot._util.datetime_to_code(self._created_on),  # fDatime
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
            assert fSeekKey == location, "fSeekKey {0} location {1}".format(
                fSeekKey, location
            )
            fSeekKey = None

        classname, position = String.deserialize(
            raw_bytes[position - location :], position
        )
        name, position = String.deserialize(raw_bytes[position - location :], position)
        title, position = String.deserialize(raw_bytes[position - location :], position)

        assert fKeylen == position - location

        return Key(
            location,
            fObjlen,  # uncompressed_bytes
            fNbytes - fKeylen,  # compressed_bytes
            classname,
            name,
            title,
            fCycle,  # cycle
            fSeekPdir,  # parent_location
            fSeekKey,  # may be location
            created_on=uproot._util.code_to_datetime(fDatime),
            big=big,
        )


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
                raw_bytes[position : position + _free_format_small.size]
            )
            if version >= 1000:
                version, fFirst, fLast = _free_format_big.unpack(
                    raw_bytes[position : position + _free_format_big.size]
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

    def allocate(self, num_bytes, dry_run=False):
        slices = self._data.slices
        for i, (start, stop) in enumerate(slices):
            if stop - start == num_bytes:
                # This will reduce the num_bytes of the FreeSegments record,
                # but the allocation can stay the same size.
                if not dry_run:
                    self._data.slices = tuple(
                        slices[j] for j in range(len(slices)) if i != j
                    )
                return start

            elif stop - start > num_bytes:
                # This will not change the num_bytes of the FreeSegments record.
                if not dry_run:
                    self._data.slices = tuple(
                        slices[j] if i != j else (start + num_bytes, stop)
                        for j in range(len(slices))
                    )
                return start

        if self.at_end:
            # The new object can take FreeSegments's spot; FreeSegments will
            # move to stay at the end.
            out = self._key.location
            if not dry_run:
                self._key.location = self._key.location + num_bytes
                self._data.end = (
                    self._key.location + self._key.allocation + self._data.allocation
                )
            return out

        else:
            # FreeSegments is not changing size and not at the end; it can
            # stay where it is.
            out = self._data.end
            if not dry_run:
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

        for i in range(len(slices) - 1):
            if slices[i][1] == original_start and original_stop == slices[i + 1][0]:
                # These two slices need to be merged, including the newly released interval.
                return (
                    slices[:i] + ((slices[i][0], slices[i + 1][1]),) + slices[i + 2 :]
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
        self._fileheader.free_num_bytes = self._key.allocation + self._data.allocation
        self._fileheader.free_num_slices = len(self._data.slices)
        self._fileheader.end = self._data.end
        super(FreeSegments, self).write(sink)


_tlistheader_format = struct.Struct(">IHHIIBI")


class TListHeader(CascadeLeaf):
    """
    FIXME: docstring
    """

    class_version = 5

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
            1,  # TObject version
            0,  # TObject::fUniqueID
            uproot.const.kNotDeleted,  # TObject::fBits
            0,
            self._num_entries,
        )


class RawStreamerInfo(CascadeLeaf):
    """
    FIXME: docstring
    """

    def __init__(self, location, serialization, name, class_version):
        super(RawStreamerInfo, self).__init__(location, len(serialization))
        self._serialization = serialization
        self._name = name
        self._class_version = class_version

    @property
    def name(self):
        return self._name

    @property
    def class_version(self):
        return self._class_version

    def __repr__(self):
        return "{0}({1}, {2}, {3}, {4})".format(
            type(self).__name__,
            self._location,
            self._serialization,
            repr(self._name),
            self._class_version,
        )

    def copy_to(self, location):
        return RawStreamerInfo(
            location, self._serialization, self._name, self._class_version
        )

    def serialize(self):
        return self._serialization


class RawTListOfStrings(CascadeLeaf):
    """
    FIXME: docstring
    """

    def __init__(self, location, serialization):
        super(RawTListOfStrings, self).__init__(location, len(serialization))
        self._serialization = serialization

    def __repr__(self):
        return "{0}({1}, {2})".format(
            type(self).__name__,
            self._location,
            self._serialization,
        )

    def copy_to(self, location):
        return RawTListOfStrings(location, self._serialization)

    def serialize(self):
        return self._serialization


class TListOfStreamers(CascadeNode):
    """
    FIXME: docstring
    """

    def __init__(self, allocation, key, header, rawstreamers, rawstrings, freesegments):
        super(TListOfStreamers, self).__init__(
            freesegments, key, header, *(rawstreamers + rawstrings)
        )
        self._allocation = allocation
        self._key = key
        self._header = header
        self._rawstreamers = rawstreamers
        self._rawstrings = rawstrings
        self._freesegments = freesegments
        self._lookup = set([(x.name, x.class_version) for x in self._rawstreamers])

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

    def update_streamers(self, sink, streamers, flush=True):
        where = len(self._rawstreamers)

        for streamer in streamers:
            pair = (streamer.name, streamer.class_version)
            if pair not in self._lookup:
                self._lookup.add(pair)

                if isinstance(streamer, uproot.streamers.Model_TStreamerInfo):
                    self._rawstreamers.append(
                        RawStreamerInfo(
                            self._key.location + self.num_bytes,
                            uproot.serialization.serialize_object_any(
                                streamer, streamer.name
                            )
                            + b"\x00",
                            streamer.name,
                            streamer.class_version,
                        )
                    )

                elif isinstance(streamer, uproot._writing.RawStreamerInfo):
                    self._rawstreamers.append(
                        streamer.copy_to(self._key.location + self.num_bytes)
                    )

        self.more_dependencies(*self._rawstreamers[where:])

        if flush:
            self.write(sink)
            sink.flush()

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

        for rawstring in self._rawstrings:
            rawstring.location = position
            position += rawstring.num_bytes

        self._header.data_bytes = position - afterkey
        self._header.num_entries = len(self._rawstreamers) + len(self._rawstrings)

        self._key.uncompressed_bytes = self._allocation
        self._key.compressed_bytes = self._key.uncompressed_bytes
        self._freesegments.fileheader.info_location = self._key.location
        self._freesegments.fileheader.info_num_bytes = (
            self._key.allocation + self._allocation
        )

        super(TListOfStreamers, self).write(sink)

    @classmethod
    def deserialize(cls, raw_bytes, location, key, freesegments, file_path, uuid):
        readforupdate = _ReadForUpdate(file_path, uuid)

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
            uproot.source.cursor.Cursor(0, origin=-key.num_bytes),
            {},
            readforupdate,
            readforupdate,
            None,
        )

        header = TListHeader(location, key.uncompressed_bytes, len(tlist))

        rawstreamers = []
        rawstrings = []

        for (start, stop), streamer in zip(tlist.byte_ranges, tlist):
            if isinstance(streamer, uproot.streamers.Model_TStreamerInfo):
                rawstreamers.append(
                    RawStreamerInfo(
                        location + start,
                        uproot._util.tobytes(uncompressed.raw_data[start:stop]),
                        streamer.name,
                        streamer.class_version,
                    )
                )

            elif isinstance(streamer, uproot.models.TList.Model_TList):
                rawstrings.append(
                    RawTListOfStrings(
                        location + start,
                        uproot._util.tobytes(uncompressed.raw_data[start:stop]),
                    )
                )

        return (
            TListOfStreamers(
                key.compressed_bytes,
                key,
                header,
                rawstreamers,
                rawstrings,
                freesegments,
            ),
            tlist,
        )


class _ReadForUpdate(object):
    def __init__(self, file_path, uuid, get_chunk=None, tlist_of_streamers=None):
        self.file_path = file_path
        self.uuid = uuid
        self.object_cache = None
        self._get_chunk = get_chunk
        self._tlist_of_streamers = tlist_of_streamers
        self._custom_classes = None

    @property
    def detached(self):
        return self

    def chunk(self, start, stop):
        return self._get_chunk(start, stop)

    def class_named(self, classname, version=None):
        return uproot.reading.ReadOnlyFile.class_named(self, classname, version=version)

    def streamers_named(self, classname):
        if self._tlist_of_streamers is None:
            return []
        else:
            return [x for x in self._tlist_of_streamers if x.name == classname]

    def streamer_named(self, classname, version="max"):
        out = None
        if self._tlist_of_streamers is not None:
            for x in self._tlist_of_streamers:
                if x.name == classname:
                    if version == "max":
                        if out is None or x.class_version > out.class_version:
                            out = x
                    elif version == "min":
                        if out is None or x.class_version < out.class_version:
                            out = x
                    elif x.class_version == version:
                        return x
        return out

    @property
    def custom_classes(self):
        return self._custom_classes


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
            if name == key.name.string:
                cycle = max(cycle, key.cycle + 1)
        return cycle

    def add_key(self, key):
        self._file_dirty = True
        self._keys.append(key)

    def replace_key(self, key):
        self._file_dirty = True
        for i in range(len(self._keys)):
            old = self._keys[i]
            if old.name.string == key.name.string and old.cycle == key.cycle:
                self._keys[i] = key
                return
        else:
            raise AssertionError

    def remove_key(self, key):
        self._file_dirty = True
        for i in range(len(self._keys)):
            old = self._keys[i]
            if old.name.string == key.name.string and old.cycle == key.cycle:
                del self._keys[i]
                return
        else:
            raise AssertionError

    @property
    def num_keys(self):
        return len(self._keys)

    def haskey(self, name):
        for key in self._keys:
            if key.name.string == name:
                return True
        return False

    def get_key(self, name, cycle=None):
        out = None
        for key in self._keys:
            if key.name.string == name and cycle is None:
                if out is None or key.cycle > out.cycle:
                    out = key
            elif key.name.string == name and key.cycle == cycle:
                return key
        return out  # None if a match wasn't found

    @property
    def key_names(self):
        return [x.name.string for x in self._keys]

    @property
    def key_triples(self):
        return [(x.name.string, x.cycle, x.classname.string) for x in self._keys]

    @property
    def dir_names(self):
        return [
            x.name.string
            for x in self._keys
            if x.classname.string in ("TDirectory", "TDirectoryFile")
        ]

    def classname_of(self, name):
        for key in self._keys:
            if key.name.string == name:
                return key.classname.string
        return None

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
            position += keys[-1].num_bytes

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
        self._uuid = uuid
        self._created_on = datetime.datetime.now()
        self._modified_on = self._created_on

    def __repr__(self):
        return "{0}({1}, {2}, {3}, {4}, {5}, {6}, {7})".format(
            type(self).__name__,
            self._location,
            self._begin_location,
            self._begin_num_bytes,
            self._data_location,
            self._data_num_bytes,
            self._parent_location,
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
    def uuid(self):
        return self._uuid

    @property
    def modified_on(self):
        return self._modified_on

    @modified_on.setter
    def modified_on(self, value):
        if self._modified_on != value:
            self._file_dirty = True
            self._modified_on = value

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
                uproot._util.datetime_to_code(self._created_on),  # fDatimeC
                uproot._util.datetime_to_code(self._modified_on),  # fDatimeM
                self._data_num_bytes,  # fNbytesKeys
                self._begin_num_bytes,  # fNbytesName
                self._begin_location,  # fSeekDir
                self._parent_location,  # fSeekParent
                self._data_location,  # fSeekKeys
            )
            + b"\x00\x01"  # TUUID version 1
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

        # TUUID version 1
        assert raw_bytes[position - location : position - location + 2] == b"\x00\x01"
        uuid_bytes = raw_bytes[position - location + 2 : position - location + 18]

        out = DirectoryHeader(
            location,
            fSeekDir,  # begin_location
            fNbytesName,  # begin_num_bytes
            fSeekKeys,  # data_location
            fNbytesKeys,  # data_num_bytes
            fSeekParent,  # parent_location
            uuid.UUID(bytes=uuid_bytes),
        )
        out._created_on = uproot._util.code_to_datetime(fDatimeC)
        out._modified_on = uproot._util.code_to_datetime(fDatimeM)
        return out


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

    def add_object(
        self, sink, classname, name, title, raw_data, uncompressed_bytes, replaces=None, big=None
    ):
        if replaces is None:
            cycle = self._data.next_cycle(name)
        else:
            cycle = replaces.cycle

        strings_size = 0
        strings_size += (1 if len(classname) < 255 else 5) + len(classname)
        strings_size += (1 if len(name) < 255 else 5) + len(name)
        strings_size += (1 if len(title) < 255 else 5) + len(title)

        parent_location = self._key.location  # FIXME: is this correct?

        location = None
        if not big and parent_location < uproot.const.kStartBigFile:
            requested_bytes = (
                uproot.reading._key_format_small.size + strings_size + len(raw_data)
            )
            location = self._freesegments.allocate(requested_bytes, dry_run=True)
            position = location + uproot.reading._key_format_small.size
            if location < uproot.const.kStartBigFile:
                self._freesegments.allocate(requested_bytes, dry_run=False)
            else:
                location = None

        if location is None:
            requested_bytes = (
                uproot.reading._key_format_big.size + strings_size + len(raw_data)
            )
            location = self._freesegments.allocate(requested_bytes, dry_run=False)
            position = location + uproot.reading._key_format_big.size

        writable_classname = String(position, classname)
        position += writable_classname.num_bytes

        writable_name = String(position, name)
        position += writable_name.num_bytes

        writable_title = String(position, title)
        position += writable_title.num_bytes

        key = Key(
            location,
            uncompressed_bytes,
            len(raw_data),
            writable_classname,
            writable_name,
            writable_title,
            cycle,
            parent_location,
            location,
            big=big,
        )

        if replaces is None:
            next_key = key.copy_to(self._data.next_location)
            if self._data.num_bytes + next_key.num_bytes > self._data.allocation:
                self._reallocate_data(
                    int(
                        math.ceil(
                            1.5 * (self._data.allocation + next_key.num_bytes + 8)
                        )
                    )
                )
                next_key = key.copy_to(self._data.next_location)
            self._data.add_key(next_key)

        else:
            original_key = self._data.get_key(replaces.name.string, replaces.cycle)
            assert original_key is not None
            new_key = key.copy_to(original_key.location)
            if (
                self._data.num_bytes + new_key.num_bytes - original_key.num_bytes
                > self._data.allocation
            ):
                self._reallocate_data(
                    int(
                        math.ceil(1.5 * (self._data.allocation + new_key.num_bytes + 8))
                    )
                )
                original_key = self._data.get_key(replaces.name.string, replaces.cycle)
                assert original_key is not None
                new_key = key.copy_to(original_key.location)
            self._data.replace_key(new_key)

        self._header.modified_on = datetime.datetime.now()

        key.write(sink)
        sink.write(location + key.num_bytes, raw_data)
        self.write(sink)
        sink.set_file_length(self._freesegments.fileheader.end)
        sink.flush()

        return key

    def add_directory(self, sink, name, initial_directory_bytes, uuid, flush=True):
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

        self._header.modified_on = datetime.datetime.now()

        if flush:
            subdirectory.write(sink)
            self.write(sink)
            sink.set_file_length(self._freesegments.fileheader.end)
            sink.flush()

        return subdirectory

    def add_tree(
        self,
        sink,
        name,
        title,
        branch_types,
        counter_name,
        field_name,
        initial_basket_capacity,
        resize_factor,
    ):
        tree = Tree(
            self,
            name,
            title,
            branch_types,
            self._freesegments,
            counter_name,
            field_name,
            initial_basket_capacity,
            resize_factor,
        )
        tree.write_anew(sink)
        return tree


_dtype_to_char = {
    numpy.dtype("bool"): "O",
    numpy.dtype(">i1"): "B",
    numpy.dtype(">u1"): "b",
    numpy.dtype(">i2"): "S",
    numpy.dtype(">u2"): "s",
    numpy.dtype(">i4"): "I",
    numpy.dtype(">u4"): "i",
    numpy.dtype(">i8"): "L",
    numpy.dtype(">u8"): "l",
    numpy.dtype(">f4"): "F",
    numpy.dtype(">f8"): "D",
}


class Tree(object):
    """
    FIXME: docstring
    """

    def __init__(
        self,
        directory,
        name,
        title,
        branch_types,
        freesegments,
        counter_name,
        field_name,
        initial_basket_capacity,
        resize_factor,
    ):
        self._directory = directory
        self._name = name
        self._title = title
        self._freesegments = freesegments
        self._counter_name = counter_name
        self._field_name = field_name
        self._basket_capacity = initial_basket_capacity
        self._resize_factor = resize_factor

        if isinstance(branch_types, dict):
            branch_types_items = branch_types.items()
        else:
            branch_types_items = branch_types

        if len(branch_types) == 0:
            raise ValueError("TTree must have at least one branch")

        self._branch_data = []
        self._branch_lookup = {}
        for branch_name, branch_type in branch_types_items:
            branch_dict = None
            branch_dtype = None
            branch_datashape = None

            if isinstance(branch_type, Mapping) and all(
                uproot._util.isstr(x) for x in branch_type
            ):
                branch_dict = branch_type

            else:
                try:
                    if type(branch_type).__module__.startswith("awkward."):
                        raise TypeError
                    if (
                        uproot._util.isstr(branch_type)
                        and branch_type.strip() == "bytes"
                    ):
                        raise TypeError
                    branch_dtype = numpy.dtype(branch_type)

                except TypeError:
                    try:
                        import awkward
                    except ImportError:
                        raise TypeError(
                            "not a NumPy dtype and 'awkward' cannot be imported: {0}".format(
                                repr(branch_type)
                            )
                        )
                    if isinstance(branch_type, awkward.types.Type):
                        branch_datashape = branch_type
                    else:
                        try:
                            branch_datashape = awkward.types.from_datashape(branch_type)
                        except Exception:
                            raise TypeError(
                                "not a NumPy dtype or an Awkward datashape: {0}".format(
                                    repr(branch_type)
                                )
                            )
                    # checking by class name to be Awkward v1/v2 insensitive
                    if type(branch_datashape).__name__ == "ArrayType":
                        if hasattr(branch_datashape, "content"):
                            branch_datashape = branch_datashape.content
                        else:
                            branch_datashape = branch_datashape.type
                    branch_dtype = self._branch_ak_to_np(branch_datashape)

            if branch_dict is not None:
                self._branch_lookup[branch_name] = len(self._branch_data)
                self._branch_data.append(
                    {"kind": "record", "name": branch_name, "keys": list(branch_dict)}
                )

                for key, content in branch_dict.items():
                    subname = self._field_name(branch_name, key)
                    try:
                        dtype = numpy.dtype(content)
                    except Exception:
                        raise TypeError(
                            "values of a dict must be NumPy types\n\n    key {0} has type {1}".format(
                                repr(key), repr(content)
                            )
                        )
                    self._branch_lookup[subname] = len(self._branch_data)
                    self._branch_data.append(self._branch_np(subname, content, dtype))

            elif branch_dtype is not None:
                self._branch_lookup[branch_name] = len(self._branch_data)
                self._branch_data.append(
                    self._branch_np(branch_name, branch_type, branch_dtype)
                )

            else:
                parameters = branch_datashape.parameters
                if parameters is None:
                    parameters = {}

                if parameters.get("__array__") == "string":
                    raise NotImplementedError("array of strings")

                elif parameters.get("__array__") == "bytes":
                    raise NotImplementedError("array of bytes")

                # checking by class name to be Awkward v1/v2 insensitive
                elif type(branch_datashape).__name__ == "ListType":
                    if hasattr(branch_datashape, "content"):
                        content = branch_datashape.content
                    else:
                        content = branch_datashape.type

                    counter_name = self._counter_name(branch_name)
                    counter_dtype = numpy.dtype(numpy.int32)
                    counter = self._branch_np(
                        counter_name, counter_dtype, counter_dtype, kind="counter"
                    )
                    self._branch_lookup[counter_name] = len(self._branch_data)
                    self._branch_data.append(counter)

                    if type(content).__name__ == "RecordType":
                        if hasattr(content, "contents"):
                            contents = content.contents
                        else:
                            contents = content.fields()
                        keys = content.keys
                        if callable(keys):
                            keys = keys()
                        if keys is None:
                            keys = [str(x) for x in range(len(contents))]

                        self._branch_lookup[branch_name] = len(self._branch_data)
                        self._branch_data.append(
                            {"kind": "record", "name": branch_name, "keys": keys}
                        )

                        for key, cont in zip(keys, contents):
                            subname = self._field_name(branch_name, key)
                            dtype = self._branch_ak_to_np(cont)
                            if dtype is None:
                                raise TypeError(
                                    "fields of a record must be NumPy types, though the record itself may be in a jagged array\n\n    field {0} has type {1}".format(
                                        repr(key), str(cont)
                                    )
                                )
                            self._branch_lookup[subname] = len(self._branch_data)
                            self._branch_data.append(
                                self._branch_np(subname, cont, dtype, counter=counter)
                            )

                    else:
                        dt = self._branch_ak_to_np(content)
                        if dt is None:
                            raise TypeError(
                                "cannot write Awkward Array type to ROOT file:\n\n    {0}".format(
                                    str(branch_datashape)
                                )
                            )
                        self._branch_lookup[branch_name] = len(self._branch_data)
                        self._branch_data.append(
                            self._branch_np(branch_name, dt, dt, counter=counter)
                        )

                elif type(branch_datashape).__name__ == "RecordType":
                    if hasattr(branch_datashape, "contents"):
                        contents = branch_datashape.contents
                    else:
                        contents = branch_datashape.fields()
                    keys = branch_datashape.keys
                    if callable(keys):
                        keys = keys()
                    if keys is None:
                        keys = [str(x) for x in range(len(contents))]

                    self._branch_lookup[branch_name] = len(self._branch_data)
                    self._branch_data.append(
                        {"kind": "record", "name": branch_name, "keys": keys}
                    )

                    for key, content in zip(keys, contents):
                        subname = self._field_name(branch_name, key)
                        dtype = self._branch_ak_to_np(content)
                        if dtype is None:
                            raise TypeError(
                                "fields of a record must be NumPy types, though the record itself may be in a jagged array\n\n    field {0} has type {1}".format(
                                    repr(key), str(content)
                                )
                            )
                        self._branch_lookup[subname] = len(self._branch_data)
                        self._branch_data.append(
                            self._branch_np(subname, content, dtype)
                        )

                else:
                    raise TypeError(
                        "cannot write Awkward Array type to ROOT file:\n\n    {0}".format(
                            str(branch_datashape)
                        )
                    )

        self._num_entries = 0
        self._num_baskets = 0

        self._metadata_start = None
        self._metadata = {
            "fTotBytes": 0,
            "fZipBytes": 0,
            "fSavedBytes": 0,
            "fFlushedBytes": 0,
            "fWeight": 1.0,
            "fTimerInterval": 0,
            "fScanField": 25,
            "fUpdate": 0,
            "fDefaultEntryOffsetLen": 1000,
            "fNClusterRange": 0,
            "fMaxEntries": 1000000000000,
            "fMaxEntryLoop": 1000000000000,
            "fMaxVirtualSize": 0,
            "fAutoSave": -300000000,
            "fAutoFlush": -30000000,
            "fEstimate": 1000000,
        }
        self._key = None

    def _branch_ak_to_np(self, branch_datashape):
        # checking by class name to be Awkward v1/v2 insensitive
        if type(branch_datashape).__name__ == "NumpyType":
            return numpy.dtype(branch_datashape.primitive)
        elif type(branch_datashape).__name__ == "PrimitiveType":
            return numpy.dtype(branch_datashape.dtype)
        elif type(branch_datashape).__name__ == "RegularType":
            if hasattr(branch_datashape, "content"):
                content = self._branch_ak_to_np(branch_datashape.content)
            else:
                content = self._branch_ak_to_np(branch_datashape.type)
            if content is None:
                return None
            elif content.subdtype is None:
                dtype, shape = content, ()
            else:
                dtype, shape = content.subdtype
            return numpy.dtype((dtype, (branch_datashape.size,) + shape))
        else:
            return None

    def _branch_np(
        self, branch_name, branch_type, branch_dtype, counter=None, kind="normal"
    ):
        branch_dtype = branch_dtype.newbyteorder(">")

        if branch_dtype.subdtype is None:
            branch_shape = ()
        else:
            branch_dtype, branch_shape = branch_dtype.subdtype

        letter = _dtype_to_char.get(branch_dtype)
        if letter is None:
            raise TypeError(
                "cannot write NumPy dtype {0} in TTree".format(branch_dtype)
            )

        if branch_shape == ():
            dims = ""
        else:
            dims = "".join("[" + str(x) + "]" for x in branch_shape)

        title = "{0}{1}/{2}".format(branch_name, dims, letter)

        return {
            "fName": branch_name,
            "branch_type": branch_type,
            "kind": kind,
            "counter": counter,
            "dtype": branch_dtype,
            "shape": branch_shape,
            "fTitle": title,
            "compression": self._directory.freesegments.fileheader.compression,
            "fBasketSize": 32000,
            "fEntryOffsetLen": 0 if counter is None else 1000,
            "fOffset": 0,
            "fSplitLevel": 0,
            "fFirstEntry": 0,
            "fTotBytes": 0,
            "fZipBytes": 0,
            "fBasketBytes": numpy.zeros(
                self._basket_capacity, uproot.models.TBranch._tbranch13_dtype1
            ),
            "fBasketEntry": numpy.zeros(
                self._basket_capacity, uproot.models.TBranch._tbranch13_dtype2
            ),
            "fBasketSeek": numpy.zeros(
                self._basket_capacity, uproot.models.TBranch._tbranch13_dtype3
            ),
            "metadata_start": None,
            "basket_metadata_start": None,
            "tleaf_reference_number": None,
            "tleaf_maximum_value": 0,
            "tleaf_special_struct": None,
        }

    def __repr__(self):
        return "{0}({1}, {2}, {3}, {4}, {5}, {6}, {7})".format(
            type(self).__name__,
            self._directory,
            self._name,
            self._title,
            [(datum["fName"], datum["branch_type"]) for datum in self._branch_data],
            self._freesegments,
            self._basket_capacity,
            self._resize_factor,
        )

    @property
    def directory(self):
        return self._directory

    @property
    def key(self):
        return self._key

    @property
    def name(self):
        return self._key.name

    @property
    def title(self):
        return self._key.title

    @property
    def branch_types(self):
        return self._branch_types

    @property
    def freesegments(self):
        return self._freesegments

    @property
    def counter_name(self):
        return self._counter_name

    @property
    def field_name(self):
        return self._field_name

    @property
    def basket_capacity(self):
        return self._basket_capacity

    @property
    def resize_factor(self):
        return self._resize_factor

    @property
    def location(self):
        return self._key.location

    @property
    def num_entries(self):
        return self._num_entries

    @property
    def num_baskets(self):
        return self._num_baskets

    def extend(self, file, sink, data):
        # expand capacity if this would REACH (not EXCEED) the existing capacity
        # that's because completely a full fBasketEntry has nowhere to put the
        # number of entries in the last basket (it's a fencepost principle thing),
        # forcing ROOT and Uproot to look it up from the basket header.
        if self._num_baskets >= self._basket_capacity - 1:
            self._basket_capacity = max(
                self._basket_capacity + 1,
                int(math.ceil(self._basket_capacity * self._resize_factor)),
            )

            for datum in self._branch_data:
                fBasketBytes = datum["fBasketBytes"]
                fBasketEntry = datum["fBasketEntry"]
                fBasketSeek = datum["fBasketSeek"]
                datum["fBasketBytes"] = numpy.zeros(
                    self._basket_capacity, uproot.models.TBranch._tbranch13_dtype1
                )
                datum["fBasketEntry"] = numpy.zeros(
                    self._basket_capacity, uproot.models.TBranch._tbranch13_dtype2
                )
                datum["fBasketSeek"] = numpy.zeros(
                    self._basket_capacity, uproot.models.TBranch._tbranch13_dtype3
                )
                datum["fBasketBytes"][: len(fBasketBytes)] = fBasketBytes
                datum["fBasketEntry"][: len(fBasketEntry)] = fBasketEntry
                datum["fBasketSeek"][: len(fBasketSeek)] = fBasketSeek
                datum["fBasketEntry"][len(fBasketEntry)] = self._num_entries

            oldloc = start = self._key.location
            stop = start + self._key.num_bytes + self._key.compressed_bytes

            self.write_anew(sink)

            newloc = self._key.seek_location
            file._move_tree(oldloc, newloc)

            self._freesegments.release(start, stop)
            sink.set_file_length(self._freesegments.fileheader.end)
            sink.flush()

        provided = None
        module_name = type(data).__module__

        if module_name == "pandas" or module_name.startswith("pandas."):
            import pandas

            if isinstance(data, pandas.DataFrame) and data.index.is_numeric():
                provided = dataframe_to_dict(data)

        if module_name == "awkward" or module_name.startswith("awkward."):
            import awkward

            if isinstance(data, awkward.Array):
                provided = dict(zip(awkward.fields(data), awkward.unzip(data)))

        if isinstance(data, numpy.ndarray) and data.dtype.fields is not None:
            provided = recarray_to_dict(data)

        if provided is None:
            if not isinstance(data, Mapping) or not all(
                uproot._util.isstr(x) for x in data
            ):
                raise TypeError(
                    "'extend' requires a mapping from branch name (str) to arrays"
                )

            provided = {}
            for k, v in data.items():
                module_name = type(v).__module__
                if module_name == "awkward" or module_name.startswith("awkward."):
                    import awkward

                    if (
                        isinstance(v, awkward.Array)
                        and v.ndim > 1
                        and not v.layout.purelist_isregular
                    ):
                        provided[self._counter_name(k)] = awkward.num(v, axis=1)

                provided[k] = v

        actual_branches = {}
        for datum in self._branch_data:
            if datum["kind"] == "record":
                if datum["name"] in provided:
                    recordarray = provided.pop(datum["name"])

                    module_name = type(recordarray).__module__
                    if module_name == "pandas" or module_name.startswith("pandas."):
                        import pandas

                        if isinstance(recordarray, pandas.DataFrame):
                            tmp = {"index": recordarray.index.values}
                            for column in recordarray.columns:
                                tmp[column] = recordarray[column]
                            recordarray = tmp

                    for key in datum["keys"]:
                        provided[self._field_name(datum["name"], key)] = recordarray[
                            key
                        ]
                else:
                    raise ValueError(
                        "'extend' must be given an array for every branch; missing {0}".format(
                            repr(datum["name"])
                        )
                    )

            else:
                if datum["fName"] in provided:
                    actual_branches[datum["fName"]] = provided.pop(datum["fName"])
                else:
                    raise ValueError(
                        "'extend' must be given an array for every branch; missing {0}".format(
                            repr(datum["fName"])
                        )
                    )

        if len(provided) != 0:
            raise ValueError(
                "'extend' was given data that do not correspond to any branch: {0}".format(
                    ", ".join(repr(x) for x in provided)
                )
            )

        tofill = []
        num_entries = None
        for branch_name, branch_array in actual_branches.items():
            if num_entries is None:
                num_entries = len(branch_array)
            elif num_entries != len(branch_array):
                raise ValueError(
                    "'extend' must fill every branch with the same number of entries; {0} has {1} entries".format(
                        repr(branch_name),
                        len(branch_array),
                    )
                )

            datum = self._branch_data[self._branch_lookup[branch_name]]
            if datum["kind"] == "record":
                continue

            if datum["counter"] is None:
                big_endian = numpy.asarray(branch_array, dtype=datum["dtype"])
                if big_endian.shape != (len(branch_array),) + datum["shape"]:
                    raise ValueError(
                        "'extend' must fill branches with a consistent shape: has {0}, trying to fill with {1}".format(
                            datum["shape"],
                            big_endian.shape[1:],
                        )
                    )
                tofill.append((branch_name, big_endian, None))

                if datum["kind"] == "counter":
                    datum["tleaf_maximum_value"] = max(
                        big_endian.max(), datum["tleaf_maximum_value"]
                    )

            else:
                import awkward

                layout = branch_array.layout
                while not isinstance(
                    layout,
                    (
                        awkward.layout.ListOffsetArray32,
                        awkward.layout.ListOffsetArrayU32,
                        awkward.layout.ListOffsetArray64,
                    ),
                ):
                    if isinstance(layout, awkward.partition.PartitionedArray):
                        layout = awkward.concatenate(layout.partitions, highlevel=False)

                    elif isinstance(
                        layout,
                        (
                            awkward.layout.IndexedArray32,
                            awkward.layout.IndexedArrayU32,
                            awkward.layout.IndexedArray64,
                        ),
                    ):
                        layout = layout.project()

                    elif isinstance(
                        layout,
                        (
                            awkward.layout.ListArray32,
                            awkward.layout.ListArrayU32,
                            awkward.layout.ListArray64,
                        ),
                    ):
                        layout = layout.toListOffsetArray64(False)

                    else:
                        raise AssertionError(
                            "how did this pass the type check?\n\n" + repr(layout)
                        )

                content = layout.content
                offsets = numpy.asarray(layout.offsets)
                if offsets[0] != 0:
                    content = content[offsets[0] :]
                    offsets -= offsets[0]
                if len(content) > offsets[-1]:
                    content = content[: offsets[-1]]

                shape = [len(content)]
                while not isinstance(content, awkward.layout.NumpyArray):
                    if isinstance(
                        content,
                        (
                            awkward.layout.IndexedArray32,
                            awkward.layout.IndexedArrayU32,
                            awkward.layout.IndexedArray64,
                        ),
                    ):
                        content = content.project()

                    elif isinstance(content, awkward.layout.EmptyArray):
                        content = content.toNumpyArray()

                    elif isinstance(content, awkward.layout.RegularArray):
                        shape.append(content.size)
                        content = content.content

                    else:
                        raise AssertionError(
                            "how did this pass the type check?\n\n" + repr(content)
                        )

                big_endian = numpy.asarray(content, dtype=datum["dtype"])
                shape = tuple(shape) + big_endian.shape[1:]

                if shape[1:] != datum["shape"]:
                    raise ValueError(
                        "'extend' must fill branches with a consistent shape: has {0}, trying to fill with {1}".format(
                            datum["shape"],
                            shape[1:],
                        )
                    )
                big_endian_offsets = offsets.astype(">i4", copy=True)

                tofill.append((branch_name, big_endian.reshape(-1), big_endian_offsets))

        # actually write baskets into the file
        uncompressed_bytes = 0
        compressed_bytes = 0
        for branch_name, big_endian, big_endian_offsets in tofill:
            datum = self._branch_data[self._branch_lookup[branch_name]]

            if big_endian_offsets is None:
                totbytes, zipbytes, location = self.write_np_basket(
                    sink, branch_name, big_endian
                )
            else:
                totbytes, zipbytes, location = self.write_jagged_basket(
                    sink, branch_name, big_endian, big_endian_offsets
                )
                datum["fEntryOffsetLen"] = 4 * (len(big_endian_offsets) - 1)
            uncompressed_bytes += totbytes
            compressed_bytes += zipbytes

            datum["fTotBytes"] += totbytes
            datum["fZipBytes"] += zipbytes

            datum["fBasketBytes"][self._num_baskets] = zipbytes

            if self._num_baskets + 1 < self._basket_capacity:
                fBasketEntry = datum["fBasketEntry"]
                i = self._num_baskets
                fBasketEntry[i + 1] = num_entries + fBasketEntry[i]

            datum["fBasketSeek"][self._num_baskets] = location

        # update TTree metadata in file
        self._num_entries += num_entries
        self._num_baskets += 1
        self._metadata["fTotBytes"] += uncompressed_bytes
        self._metadata["fZipBytes"] += compressed_bytes

        self.write_updates(sink)

    def write_anew(self, sink):
        key_num_bytes = uproot.reading._key_format_big.size + 6
        name_asbytes = self._name.encode(errors="surrogateescape")
        title_asbytes = self._title.encode(errors="surrogateescape")
        key_num_bytes += (1 if len(name_asbytes) < 255 else 5) + len(name_asbytes)
        key_num_bytes += (1 if len(title_asbytes) < 255 else 5) + len(title_asbytes)

        out = [None]
        ttree_header_index = 0

        tobject = uproot.models.TObject.Model_TObject.empty()
        tnamed = uproot.models.TNamed.Model_TNamed.empty()
        tnamed._bases.append(tobject)
        tnamed._members["fTitle"] = self._title
        tnamed._serialize(out, True, self._name, uproot.const.kMustCleanup)

        # TAttLine v2, fLineColor: 602 fLineStyle: 1 fLineWidth: 1
        # TAttFill v2, fFillColor: 0, fFillStyle: 1001
        # TAttMarker v2, fMarkerColor: 1, fMarkerStyle: 1, fMarkerSize: 1.0
        out.append(
            b"@\x00\x00\x08\x00\x02\x02Z\x00\x01\x00\x01"
            + b"@\x00\x00\x06\x00\x02\x00\x00\x03\xe9"
            + b"@\x00\x00\n\x00\x02\x00\x01\x00\x01?\x80\x00\x00"
        )

        metadata_out_index = len(out)
        out.append(
            uproot.models.TTree._ttree20_format1.pack(
                self._num_entries,
                self._metadata["fTotBytes"],
                self._metadata["fZipBytes"],
                self._metadata["fSavedBytes"],
                self._metadata["fFlushedBytes"],
                self._metadata["fWeight"],
                self._metadata["fTimerInterval"],
                self._metadata["fScanField"],
                self._metadata["fUpdate"],
                self._metadata["fDefaultEntryOffsetLen"],
                self._metadata["fNClusterRange"],
                self._metadata["fMaxEntries"],
                self._metadata["fMaxEntryLoop"],
                self._metadata["fMaxVirtualSize"],
                self._metadata["fAutoSave"],
                self._metadata["fAutoFlush"],
                self._metadata["fEstimate"],
            )
        )

        # speedbump (0), fClusterRangeEnd (empty array),
        # speedbump (0), fClusterSize (empty array)
        # fIOFeatures (TIOFeatures)
        out.append(b"\x00\x00@\x00\x00\x07\x00\x00\x1a\xa1/\x10\x00")

        tleaf_reference_numbers = []

        tobjarray_of_branches_index = len(out)
        out.append(None)

        num_branches = sum(
            0 if datum["kind"] == "record" else 1 for datum in self._branch_data
        )

        # TObjArray header with fName: ""
        out.append(b"\x00\x01\x00\x00\x00\x00\x03\x00@\x00\x00")
        out.append(
            uproot.models.TObjArray._tobjarray_format1.pack(
                num_branches,  # TObjArray fSize
                0,  # TObjArray fLowerBound
            )
        )

        for datum in self._branch_data:
            if datum["kind"] == "record":
                continue

            any_tbranch_index = len(out)
            out.append(None)
            out.append(b"TBranch\x00")

            tbranch_index = len(out)
            out.append(None)

            tbranch_tobject = uproot.models.TObject.Model_TObject.empty()
            tbranch_tnamed = uproot.models.TNamed.Model_TNamed.empty()
            tbranch_tnamed._bases.append(tbranch_tobject)
            tbranch_tnamed._members["fTitle"] = datum["fTitle"]
            tbranch_tnamed._serialize(
                out, True, datum["fName"], numpy.uint32(0x00400000)
            )

            # TAttFill v2, fFillColor: 0, fFillStyle: 1001
            out.append(b"@\x00\x00\x06\x00\x02\x00\x00\x03\xe9")

            assert sum(1 if x is None else 0 for x in out) == 4
            datum["metadata_start"] = (6 + 6 + 8 + 6) + sum(
                len(x) for x in out if x is not None
            )

            if datum["compression"] is None:
                fCompress = uproot.compression.ZLIB(0).code
            else:
                fCompress = datum["compression"].code

            out.append(
                uproot.models.TBranch._tbranch13_format1.pack(
                    fCompress,
                    datum["fBasketSize"],
                    datum["fEntryOffsetLen"],
                    self._num_baskets,  # fWriteBasket
                    self._num_entries,  # fEntryNumber
                )
            )

            # fIOFeatures (TIOFeatures)
            out.append(b"@\x00\x00\x07\x00\x00\x1a\xa1/\x10\x00")

            out.append(
                uproot.models.TBranch._tbranch13_format2.pack(
                    datum["fOffset"],
                    self._basket_capacity,  # fMaxBaskets
                    datum["fSplitLevel"],
                    self._num_entries,  # fEntries
                    datum["fFirstEntry"],
                    datum["fTotBytes"],
                    datum["fZipBytes"],
                )
            )

            # empty TObjArray of TBranches
            out.append(
                b"@\x00\x00\x15\x00\x03\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            )

            subtobjarray_of_leaves_index = len(out)
            out.append(None)

            # TObjArray header with fName: "", fSize: 1, fLowerBound: 0
            out.append(
                b"\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00"
            )

            absolute_location = key_num_bytes + sum(
                len(x) for x in out if x is not None
            )
            absolute_location += 8 + 6 * (sum(1 if x is None else 0 for x in out) - 1)
            datum["tleaf_reference_number"] = absolute_location + 2
            tleaf_reference_numbers.append(datum["tleaf_reference_number"])

            subany_tleaf_index = len(out)
            out.append(None)

            letter = _dtype_to_char[datum["dtype"]]
            letter_upper = letter.upper()
            out.append(("TLeaf" + letter_upper).encode() + b"\x00")
            if letter_upper == "O":
                special_struct = uproot.models.TLeaf._tleafO1_format1
            elif letter_upper == "B":
                special_struct = uproot.models.TLeaf._tleafb1_format1
            elif letter_upper == "S":
                special_struct = uproot.models.TLeaf._tleafs1_format1
            elif letter_upper == "I":
                special_struct = uproot.models.TLeaf._tleafi1_format1
            elif letter_upper == "L":
                special_struct = uproot.models.TLeaf._tleafl1_format0
            elif letter_upper == "F":
                special_struct = uproot.models.TLeaf._tleaff1_format1
            elif letter_upper == "D":
                special_struct = uproot.models.TLeaf._tleafd1_format1
            fLenType = datum["dtype"].itemsize
            fIsUnsigned = letter != letter_upper

            if datum["shape"] == ():
                dims = ""
            else:
                dims = "".join("[" + str(x) + "]" for x in datum["shape"])

            # single TLeaf
            leaf_name = datum["fName"].encode(errors="surrogateescape")
            leaf_title = (datum["fName"] + dims).encode(errors="surrogateescape")
            leaf_name_length = (1 if len(leaf_name) < 255 else 5) + len(leaf_name)
            leaf_title_length = (1 if len(leaf_title) < 255 else 5) + len(leaf_title)

            leaf_header = numpy.array(
                [64, 0, 0, 76, 0, 1, 64, 0, 0, 54, 0, 2, 64, 0]
                + [0, 30, 0, 1, 0, 1, 0, 0, 0, 0, 3, 0, 0, 0],
                numpy.uint8,
            )
            tmp = leaf_header[0:4].view(">u4")
            tmp[:] = (
                numpy.uint32(
                    42 + leaf_name_length + leaf_title_length + special_struct.size
                )
                | uproot.const.kByteCountMask
            )
            tmp = leaf_header[6:10].view(">u4")
            tmp[:] = (
                numpy.uint32(36 + leaf_name_length + leaf_title_length)
                | uproot.const.kByteCountMask
            )
            tmp = leaf_header[12:16].view(">u4")
            tmp[:] = (
                numpy.uint32(12 + leaf_name_length + leaf_title_length)
                | uproot.const.kByteCountMask
            )

            out.append(uproot._util.tobytes(leaf_header))
            if len(leaf_name) < 255:
                out.append(
                    struct.pack(">B%ds" % len(leaf_name), len(leaf_name), leaf_name)
                )
            else:
                out.append(
                    struct.pack(
                        ">BI%ds" % len(leaf_name), 255, len(leaf_name), leaf_name
                    )
                )
            if len(leaf_title) < 255:
                out.append(
                    struct.pack(">B%ds" % len(leaf_title), len(leaf_title), leaf_title)
                )
            else:
                out.append(
                    struct.pack(
                        ">BI%ds" % len(leaf_title), 255, len(leaf_title), leaf_title
                    )
                )

            fLen = 1
            for item in datum["shape"]:
                fLen *= item

            # generic TLeaf members
            out.append(
                uproot.models.TLeaf._tleaf2_format0.pack(
                    fLen,
                    fLenType,
                    0,  # fOffset
                    datum["kind"] == "counter",  # fIsRange
                    fIsUnsigned,
                )
            )

            if datum["counter"] is None:
                # null fLeafCount
                out.append(b"\x00\x00\x00\x00")
            else:
                # reference to fLeafCount
                out.append(
                    uproot.deserialization._read_object_any_format1.pack(
                        datum["counter"]["tleaf_reference_number"]
                    )
                )

            # specialized TLeaf* members (fMinimum, fMaximum)
            out.append(special_struct.pack(0, 0))
            datum["tleaf_special_struct"] = special_struct

            out[
                subany_tleaf_index
            ] = uproot.serialization._serialize_object_any_format1.pack(
                numpy.uint32(sum(len(x) for x in out[subany_tleaf_index + 1 :]) + 4)
                | uproot.const.kByteCountMask,
                uproot.const.kNewClassTag,
            )

            out[subtobjarray_of_leaves_index] = uproot.serialization.numbytes_version(
                sum(len(x) for x in out[subtobjarray_of_leaves_index + 1 :]),
                3,  # TObjArray
            )

            # empty TObjArray of fBaskets (embedded)
            out.append(
                b"@\x00\x00\x15\x00\x03\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            )

            assert sum(1 if x is None else 0 for x in out) == 4
            datum["basket_metadata_start"] = (6 + 6 + 8 + 6) + sum(
                len(x) for x in out if x is not None
            )

            # speedbump and fBasketBytes
            out.append(b"\x01")
            out.append(uproot._util.tobytes(datum["fBasketBytes"]))

            # speedbump and fBasketEntry
            out.append(b"\x01")
            out.append(uproot._util.tobytes(datum["fBasketEntry"]))

            # speedbump and fBasketSeek
            out.append(b"\x01")
            out.append(uproot._util.tobytes(datum["fBasketSeek"]))

            # empty fFileName
            out.append(b"\x00")

            out[tbranch_index] = uproot.serialization.numbytes_version(
                sum(len(x) for x in out[tbranch_index + 1 :]), 13  # TBranch
            )

            out[
                any_tbranch_index
            ] = uproot.serialization._serialize_object_any_format1.pack(
                numpy.uint32(sum(len(x) for x in out[any_tbranch_index + 1 :]) + 4)
                | uproot.const.kByteCountMask,
                uproot.const.kNewClassTag,
            )

        out[tobjarray_of_branches_index] = uproot.serialization.numbytes_version(
            sum(len(x) for x in out[tobjarray_of_branches_index + 1 :]), 3  # TObjArray
        )

        # TObjArray of TLeaf references
        tleaf_reference_bytes = uproot._util.tobytes(
            numpy.array(tleaf_reference_numbers, ">u4")
        )
        out.append(
            struct.pack(
                ">I13sI4s",
                (21 + len(tleaf_reference_bytes)) | uproot.const.kByteCountMask,
                b"\x00\x03\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x00",
                len(tleaf_reference_numbers),
                b"\x00\x00\x00\x00",
            )
        )

        tleaf_reference_start = len(out)
        out.append(tleaf_reference_bytes)

        # null fAliases (b"\x00\x00\x00\x00")
        # empty fIndexValues array (4-byte length is zero)
        # empty fIndex array (4-byte length is zero)
        # null fTreeIndex (b"\x00\x00\x00\x00")
        # null fFriends (b"\x00\x00\x00\x00")
        # null fUserInfo (b"\x00\x00\x00\x00")
        # null fBranchRef (b"\x00\x00\x00\x00")
        out.append(b"\x00" * 28)

        out[ttree_header_index] = uproot.serialization.numbytes_version(
            sum(len(x) for x in out[ttree_header_index + 1 :]), 20  # TTree
        )

        self._metadata_start = sum(len(x) for x in out[:metadata_out_index])

        raw_data = b"".join(out)
        self._key = self._directory.add_object(
            sink,
            "TTree",
            self._name,
            self._title,
            raw_data,
            len(raw_data),
            replaces=self._key,
            big=True,
        )

    def write_updates(self, sink):
        base = self._key.seek_location + self._key.num_bytes

        sink.write(
            base + self._metadata_start,
            uproot.models.TTree._ttree20_format1.pack(
                self._num_entries,
                self._metadata["fTotBytes"],
                self._metadata["fZipBytes"],
                self._metadata["fSavedBytes"],
                self._metadata["fFlushedBytes"],
                self._metadata["fWeight"],
                self._metadata["fTimerInterval"],
                self._metadata["fScanField"],
                self._metadata["fUpdate"],
                self._metadata["fDefaultEntryOffsetLen"],
                self._metadata["fNClusterRange"],
                self._metadata["fMaxEntries"],
                self._metadata["fMaxEntryLoop"],
                self._metadata["fMaxVirtualSize"],
                self._metadata["fAutoSave"],
                self._metadata["fAutoFlush"],
                self._metadata["fEstimate"],
            ),
        )
        sink.flush()

        for datum in self._branch_data:
            if datum["kind"] == "record":
                continue

            position = base + datum["metadata_start"]

            if datum["compression"] is None:
                fCompress = uproot.compression.ZLIB(0).code
            else:
                fCompress = datum["compression"].code

            sink.write(
                position,
                uproot.models.TBranch._tbranch13_format1.pack(
                    fCompress,
                    datum["fBasketSize"],
                    datum["fEntryOffsetLen"],
                    self._num_baskets,  # fWriteBasket
                    self._num_entries,  # fEntryNumber
                ),
            )

            position += uproot.models.TBranch._tbranch13_format1.size + 11
            sink.write(
                position,
                uproot.models.TBranch._tbranch13_format2.pack(
                    datum["fOffset"],
                    self._basket_capacity,  # fMaxBaskets
                    datum["fSplitLevel"],
                    self._num_entries,  # fEntries
                    datum["fFirstEntry"],
                    datum["fTotBytes"],
                    datum["fZipBytes"],
                ),
            )

            fBasketBytes = uproot._util.tobytes(datum["fBasketBytes"])
            fBasketEntry = uproot._util.tobytes(datum["fBasketEntry"])
            fBasketSeek = uproot._util.tobytes(datum["fBasketSeek"])

            position = base + datum["basket_metadata_start"] + 1
            sink.write(position, fBasketBytes)
            position += len(fBasketBytes) + 1
            sink.write(position, fBasketEntry)
            position += len(fBasketEntry) + 1
            sink.write(position, fBasketSeek)

            if datum["kind"] == "counter":
                position = (
                    base
                    + datum["basket_metadata_start"]
                    - 25  # empty TObjArray of fBaskets (embedded)
                    - datum["tleaf_special_struct"].size
                )
                sink.write(
                    position,
                    datum["tleaf_special_struct"].pack(
                        0,
                        datum["tleaf_maximum_value"],
                    ),
                )

    def write_np_basket(self, sink, branch_name, array):
        fClassName = uproot.serialization.string("TBasket")
        fName = uproot.serialization.string(branch_name)
        fTitle = uproot.serialization.string(self._name)

        fKeylen = (
            uproot.reading._key_format_big.size
            + len(fClassName)
            + len(fName)
            + len(fTitle)
            + uproot.models.TBasket._tbasket_format2.size
            + 1
        )

        raw_array = uproot._util.tobytes(array)
        itemsize = array.dtype.itemsize
        for item in array.shape[1:]:
            itemsize *= item

        fObjlen = len(raw_array)

        fNbytes = fKeylen + fObjlen  # FIXME: no compression yet

        parent_location = self._directory.key.location  # FIXME: is this correct?

        location = self._freesegments.allocate(fNbytes, dry_run=False)

        out = []
        out.append(
            uproot.reading._key_format_big.pack(
                fNbytes,
                1004,  # fVersion
                fObjlen,
                uproot._util.datetime_to_code(datetime.datetime.now()),  # fDatime
                fKeylen,
                0,  # fCycle
                location,  # fSeekKey
                parent_location,  # fSeekPdir
            )
        )
        out.append(fClassName)
        out.append(fName)
        out.append(fTitle)
        out.append(
            uproot.models.TBasket._tbasket_format2.pack(
                3,  # fVersion
                32000,  # fBufferSize
                itemsize,  # fNevBufSize
                len(array),  # fNevBuf
                fKeylen + len(raw_array),  # fLast
            )
        )
        out.append(b"\x00")  # part of the Key (included in fKeylen, at least)

        out.append(raw_array)

        sink.write(location, b"".join(out))
        sink.set_file_length(self._freesegments.fileheader.end)
        sink.flush()

        return fNbytes, fNbytes, location

    def write_jagged_basket(self, sink, branch_name, array, offsets):
        fClassName = uproot.serialization.string("TBasket")
        fName = uproot.serialization.string(branch_name)
        fTitle = uproot.serialization.string(self._name)

        fKeylen = (
            uproot.reading._key_format_big.size
            + len(fClassName)
            + len(fName)
            + len(fTitle)
            + uproot.models.TBasket._tbasket_format2.size
            + 1
        )

        raw_array = uproot._util.tobytes(array)
        itemsize = array.dtype.itemsize
        for item in array.shape[1:]:
            itemsize *= item

        # offsets became a *copy* of the Awkward Array's offsets
        # when it was converted to big-endian (astype with copy=True)
        offsets *= itemsize
        offsets += fKeylen
        fLast = offsets[-1]
        offsets[-1] = 0
        raw_offsets = uproot._util.tobytes(offsets)

        fObjlen = len(raw_array) + 4 + len(raw_offsets)

        fNbytes = fKeylen + fObjlen  # FIXME: no compression yet

        parent_location = self._directory.key.location  # FIXME: is this correct?

        location = self._freesegments.allocate(fNbytes, dry_run=False)

        out = []
        out.append(
            uproot.reading._key_format_big.pack(
                fNbytes,
                1004,  # fVersion
                fObjlen,
                uproot._util.datetime_to_code(datetime.datetime.now()),  # fDatime
                fKeylen,
                0,  # fCycle
                location,  # fSeekKey
                parent_location,  # fSeekPdir
            )
        )
        out.append(fClassName)
        out.append(fName)
        out.append(fTitle)
        out.append(
            uproot.models.TBasket._tbasket_format2.pack(
                3,  # fVersion
                32000,  # fBufferSize
                len(offsets) + 1,  # fNevBufSize
                len(offsets) - 1,  # fNevBuf
                fLast,
            )
        )
        out.append(b"\x00")  # part of the Key (included in fKeylen, at least)

        out.append(raw_array)
        out.append(_tbasket_offsets_length.pack(len(offsets)))
        out.append(raw_offsets)

        sink.write(location, b"".join(out))
        sink.set_file_length(self._freesegments.fileheader.end)
        sink.flush()

        return fNbytes, fNbytes, location


_tbasket_offsets_length = struct.Struct(">I")


def dataframe_to_dict(df):
    """
    FIXME: docstring
    """
    out = {"index": df.index.values}
    for column_name in df.columns:
        out[str(column_name)] = df[column_name].values
    return out


def recarray_to_dict(array):
    """
    FIXME: docstring
    """
    out = {}
    for field_name in array.dtype.fields:
        field = array[field_name]
        if field.dtype.fields is not None:
            for subfield_name, subfield in recarray_to_dict(field):
                out[field_name + "." + subfield_name] = subfield
        else:
            out[field_name] = field
    return out


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
    class_version = 62400  # ROOT 6.24/00 is our model

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
        self._uuid = uuid
        self._version = None
        self._begin = 100

    def __repr__(self):
        return "{0}({1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9})".format(
            type(self).__name__,
            self._end,
            self._free_location,
            self._free_num_bytes,
            self._free_num_slices,
            self._begin_num_bytes,
            repr(self._compression),
            self._info_location,
            self._info_num_bytes,
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
        if self._compression is None and value is None:
            pass
        elif (
            self._compression is None
            or self.value is None
            or self._compression.code != value.code
        ):
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

        if self._compression is None:
            fCompress = uproot.compression.ZLIB(0).code
        else:
            fCompress = self._compression.code

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
            fCompress,
            self._info_location,  # fSeekInfo
            self._info_num_bytes,  # fNbytesInfo
            1,  # TUUID version
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

        assert begin >= uproot.reading._file_header_fields_small.size
        assert free_location >= 0
        assert free_num_bytes >= 0
        assert free_num_slices_plus_1 >= 1
        assert begin_num_bytes >= 0
        assert compression_code >= 0
        assert info_location >= 0
        assert info_num_bytes >= 0
        assert uuid_version == 1

        out = FileHeader(
            end,
            free_location,
            free_num_bytes,
            free_num_slices_plus_1 - 1,
            begin_num_bytes,
            uproot.compression.Compression.from_code(compression_code),
            info_location,
            info_num_bytes,
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
        tlist_of_streamers,
    ):
        self._fileheader = fileheader
        self._streamers = streamers
        self._freesegments = freesegments
        self._rootdirectory = rootdirectory
        self._tlist_of_streamers = tlist_of_streamers

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

    @property
    def tlist_of_streamers(self):
        return self._tlist_of_streamers


def create_empty(
    sink,
    compression,
    initial_directory_bytes,
    initial_streamers_bytes,
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
        initial_streamers_bytes, streamers_key, streamers_header, [], [], freesegments
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
        None, fileheader.begin, None, None, None, 0, uuid_function()
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

    return CascadingFile(fileheader, streamers, freesegments, rootdirectory, None)


def update_existing(sink, initial_directory_bytes, uuid_function):
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
    streamers, tlist_of_streamers = TListOfStreamers.deserialize(
        raw_bytes[streamers_key.num_bytes :],
        fileheader.info_location + streamers_key.num_bytes,
        streamers_key,
        freesegments,
        sink.file_path,
        fileheader.uuid,
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

    streamers.write(sink)
    sink.flush()

    return CascadingFile(
        fileheader, streamers, freesegments, rootdirectory, tlist_of_streamers
    )


_serialize_string_small = struct.Struct(">B")
_serialize_string_big = struct.Struct(">BI")


def serialize_string(string):
    """
    FIXME: docstring
    """
    as_bytes = string.encode(errors="surrogateescape")
    if len(as_bytes) < 255:
        return _serialize_string_small.pack(len(as_bytes)) + as_bytes
    else:
        return _serialize_string_big.pack(255, len(as_bytes)) + as_bytes

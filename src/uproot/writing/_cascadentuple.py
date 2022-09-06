# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

"""
This is an internal module for writing RNTuple in the "cascading" file writer.

The implementation in this module does not use the RNTuple infrastructure in
:doc:`uproot.models.RNTuple`.

See :doc:`uproot.writing._cascade` for a general overview of the cascading writer concept.
"""


import struct
import warnings
import zlib

import numpy

import uproot
import uproot.compression
import uproot.const
import uproot.reading
import uproot.serialization
from uproot.models.RNTuple import (
    _rntuple_feature_flag_format,
    _rntuple_field_description,
    _rntuple_format1,
    _rntuple_frame_format,
    _rntuple_num_bytes_fields,
)
from uproot.writing._cascade import CascadeLeaf, CascadeNode


def serialize_rntuple_string(content):
    aloc = len(content)
    return struct.Struct("<I").pack(aloc) + str.encode(content)


def serialize_rntuple_list_frame(items):
    n_items = len(items)
    payload_bytes = b"".join([x.serialize() for x in items])
    size = 4 + 4 + len(payload_bytes)
    # n.b last byte of `n_item bytes` is reserved as of Sep 2022
    return b"".join(
        [size.to_bytes(4, "little"), n_items.to_bytes(4, "little"), payload_bytes]
    )


# https://github.com/root-project/root/blob/master/tree/ntuple/v7/doc/specifications.md#field-description
class NTuple_Field_Description:
    def __init__(
        self,
        field_version,
        type_version,
        parent_field_id,
        struct_role,
        flags,
        field_name,
        type_name,
        type_alias,
        field_description,
    ):
        self.field_version = field_version
        self.type_version = type_version
        self.parent_field_id = parent_field_id
        self.struct_role = struct_role
        self.flags = flags
        self.field_name = field_name
        self.type_name = type_name
        self.type_alias = type_alias
        self.field_description = field_description

    def serialize(self):
        header_bytes = _rntuple_field_description.pack(
            self.field_version,
            self.type_version,
            self.parent_field_id,
            self.struct_role,
            self.flags,
        )
        string_bytes = b"".join(
            [
                serialize_rntuple_string(x)
                for x in (
                    self.field_name,
                    self.type_name,
                    self.type_alias,
                    self.field_description,
                )
            ]
        )
        return string_bytes


"""
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|        Envelope Version       |        Minimum Version        |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
                         ENVELOPE PAYLOAD
    Feature flag
    UInt32: Release candidate tag
    String: name of the ntuple
    String: description of the ntuple
    String: identifier of the library or program that writes the data
    List frame: list of field record frames
    List frame: list of column record frames
    List frame: list of alias column record frames
    List frame: list of extra type information
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                             CRC32                             |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
"""


class NTuple_Header(CascadeLeaf):
    def __init__(self, location, name, ntuple_description, akform):

        self._name = name
        self._ntuple_description = ntuple_description
        self._akform = akform

        aloc = len(self.serialize())
        super().__init__(location, aloc)

    def __repr__(self):
        return "{}({}, {})".format(
            type(self).__name__,
            self._location,
            ", ".join([repr(x) for x in self._members]),
        )

    def serialize(self):
        env_header = _rntuple_frame_format.pack(1, 1)
        feature_flag = _rntuple_feature_flag_format.pack(0)
        rc_tag = struct.Struct("I").pack(1)
        name = serialize_rntuple_string(self._name)
        description = serialize_rntuple_string(self._ntuple_description)
        writer = serialize_rntuple_string("uproot " + uproot.__version__)

        field_records = [
            NTuple_Field_Description(
                0, 0, 0, 0, 0, "one_integer", "std::int32_t", "", ""
            )
        ]

        header_data_bytes = b"".join(
            [env_header, feature_flag, rc_tag, name, description, writer]
        )
        crc32 = zlib.crc32(header_data_bytes)

        header_bytes = b"".join([header_data_bytes, crc32.to_bytes(4, "little")])
        return header_bytes


class NTuple_Anchor(CascadeLeaf):
    """
    A :doc:`uproot.writing._cascade.CascadeLeaf` for writing a string, such
    as a name, a title, or a class name.

    If the string's byte representation (UTF-8) has fewer than 255 bytes, it
    is preceded by a 1-byte length; otherwise, it is preceded by ``b'\xff'`` and a
    4-byte length.
    """

    def __init__(
        self,
        location,
        fCheckSum,
        fVersion,
        fSize,
        fSeekHeader,
        fNBytesHeader,
        fLenHeader,
        fSeekFooter,
        fNBytesFooter,
        fLenFooter,
        fReserved,
    ):

        aloc = _rntuple_format1.size
        super().__init__(location, aloc)
        self.fCheckSum = fCheckSum
        self.fVersion = fVersion
        self.fSize = fSize
        self.fSeekHeader = fSeekHeader
        self.fNBytesHeader = fNBytesHeader
        self.fLenHeader = fLenHeader
        self.fSeekFooter = fSeekFooter
        self.fNBytesFooter = fNBytesFooter
        self.fLenFooter = fLenFooter
        self.fReserved = fReserved

    @property
    def _members(self):
        return [
            self.fCheckSum,
            self.fVersion,
            self.fSize,
            self.fSeekHeader,
            self.fNBytesHeader,
            self.fLenHeader,
            self.fSeekFooter,
            self.fNBytesFooter,
            self.fLenFooter,
            self.fReserved,
        ]

    def __repr__(self):
        return "{}({}, {})".format(
            type(self).__name__,
            self._location,
            ", ".join([repr(x) for x in self._members]),
        )

    def serialize(self):
        # hardcoded unless version changes
        # version = 0
        # aloc = _rntuple_format1.size
        # uproot.serialization.numbytes_version(aloc, version)
        return b"@\x00\x006\x00\x00" + _rntuple_format1.pack(*self._members)


class NTuple(CascadeNode):
    """
    Writes a RNTuple, including all fields, columns, and (upon ``extend``) pages.

    The ``write_anew`` method writes the whole ntuple, possibly for the first time, possibly
    because it has been moved (exceeded its initial allocation of TBasket pointers).

    The ``write_updates`` method rewrites the parts that change when new TBaskets are
    added.

    The ``extend`` method adds a page cluster to every column.

    See `ROOT RNTuple specification <https://github.com/root-project/root/blob/master/tree/ntuple/v7/doc/specifications.md>`__.
    """

    def __init__(
        self,
        directory,
        name,
        title,
        ak_form,
        freesegments,
        header,
        footer,
        cluster_metadata,
        anchor,
    ):
        super().__init__(footer, anchor, freesegments)
        self._directory = directory
        self._name = name
        self._title = title
        self._header = header
        self._footer = footer
        self._cluster_metadata = cluster_metadata
        self._anchor = anchor
        self._freesegments = freesegments

        self._key = None
        self._header_key = None

    def __repr__(self):
        return "{}({}, {}, {}, {}, {}, {}, {})".format(
            type(self).__name__,
            self._directory,
            self._name,
            self._title,
            [(datum["fName"], datum["akform"]) for datum in self._branch_data],
            self._freesegments,
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
    def field_name(self):
        return self._field_name

    @property
    def location(self):
        return self._key.location

    @property
    def num_entries(self):
        return self._num_entries

    def extend(self, file, sink, data):
        pass

    def write(self, sink):
        header_raw_data = self._header.serialize()
        self._header_key = self._directory.add_object(
            sink,
            "RBlob",
            self._name,
            self._title,
            header_raw_data,
            len(header_raw_data),
            replaces=self._header_key,
            big=False,
        )

        self._anchor.fSeekHeader = self._header_key.seek_location
        self._anchor.fNBytesHeader = len(header_raw_data)
        self._anchor.fLenHeader = len(header_raw_data)

        anchor_raw_data = self._anchor.serialize()
        self._key = self._directory.add_object(
            sink,
            "ROOT::Experimental::RNTuple",
            self._name,
            self._title,
            anchor_raw_data,
            len(anchor_raw_data),
            replaces=self._key,
            big=True,
        )
        self._anchor.location = self._key.location + self._key.allocation

    def write_updates(self, sink):
        sink.flush()

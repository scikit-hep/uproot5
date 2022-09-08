# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

"""
This is an internal module for writing RNTuple in the "cascading" file writer.

The implementation in this module does not use the RNTuple infrastructure in
:doc:`uproot.models.RNTuple`.

See :doc:`uproot.writing._cascade` for a general overview of the cascading writer concept.
"""


import datetime
import struct
import zlib

import awkward
import numpy

import uproot
import uproot.compression
import uproot.const
import uproot.reading
import uproot.serialization
from uproot.models.RNTuple import (
    _rntuple_cluster_group_format,
    _rntuple_column_record_format,
    _rntuple_feature_flag_format,
    _rntuple_field_description,
    _rntuple_format1,
    _rntuple_locator_format,
    _rntuple_record_size_format,
)
from uproot.writing._cascade import CascadeLeaf, CascadeNode, Key, String

_ak_primitive_to_typename_dict = {
    "i64": "std::int64_t",
    "i32": "std::int32_t",
    # "switch": 3,
    # "byte": 4,
    # "char": 5,
    "bool": "bit",
    "float64": "double",
    "float32": "float",
    # "float16": 9,
    "int64": "std::int64_t",
    "int32": "std::int32_t",
    "int16": "std::int16_t",
    "int8": "std::int8_t",
    # "splitindex64": 14,
    # "splitindex32": 15,
    # "splitreal64": 16,
    # "splitreal32": 17,
    # "splitreal16": 18,
    # "splitin64": 19,
    # "splitint32": 20,
    # "splitint16": 21,
}
_ak_primitive_to_num_dict = {
    "i64": 1,
    "i32": 2,
    # "switch": 3,
    # "byte": 4,
    # "char": 5,
    "bool": 6,
    "float64": 7,
    "float32": 8,
    "float16": 9,
    "int64": 10,
    "int32": 11,
    "int16": 12,
    "int8": 13,
    # "splitindex64": 14,
    # "splitindex32": 15,
    # "splitreal64": 16,
    # "splitreal32": 17,
    # "splitreal16": 18,
    # "splitin64": 19,
    # "splitint32": 20,
    # "splitint16": 21,
}


class RBlob_Key(Key):
    def __init__(
        self,
        location,
        uncompressed_bytes,
        compressed_bytes,
        created_on=None,
        big=None,
    ):
        super().__init__(
            location,
            uncompressed_bytes,
            compressed_bytes,
            String(None, "RBlob"),
            String(None, ""),
            String(None, ""),
            0,
            13,
            location,
            created_on=created_on,
            big=big,
        )


def _serialize_rntuple_string(content):
    return _record_frame_wrap(str.encode(content), False)


def _record_frame_wrap(payload, includeself=True):
    aloc = len(payload)
    if includeself:
        aloc += 4
    raw_bytes = _rntuple_record_size_format.pack(aloc) + payload
    return raw_bytes


def _serialize_rntuple_list_frame(items):
    # when items is [], b'\xf8\xff\xff\xff\x00\x00\x00\x00'
    n_items = len(items)
    payload_bytes = b"".join([_record_frame_wrap(x.serialize(), True) for x in items])
    size = 4 + 4 + len(payload_bytes)
    size_bytes = struct.Struct("<i").pack(-size)  # negative size means list
    # n.b last byte of `n_item bytes` is reserved as of Sep 2022
    raw_bytes = b"".join([size_bytes, n_items.to_bytes(4, "little"), payload_bytes])
    return raw_bytes


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

    def __repr__(self):
        return "{}({}, {}, {}, {}, {}, {}, {}, {}, {})".format(
            type(self).__name__,
            repr(self.field_version),
            repr(self.type_version),
            repr(self.parent_field_id),
            repr(self.struct_role),
            repr(self.flags),
            repr(self.field_name),
            repr(self.type_name),
            repr(self.type_alias),
            repr(self.field_description),
        )

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
                _serialize_rntuple_string(x)
                for x in (
                    self.field_name,
                    self.type_name,
                    self.type_alias,
                    self.field_description,
                )
            ]
        )
        return b"".join([header_bytes, string_bytes])


# https://github.com/root-project/root/blob/master/tree/ntuple/v7/doc/specifications.md#column-description
class NTuple_Column_Description:
    def __init__(self, type_num, bits_on_disk, field_id, flags):
        self.type_num = type_num
        self.bits_on_disk = bits_on_disk
        self.field_id = field_id
        self.flags = flags

    def serialize(self):
        header_bytes = _rntuple_column_record_format.pack(
            self.type_num,
            self.bits_on_disk,
            self.field_id,
            self.flags,
        )
        return header_bytes


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

        self._serialize = None
        self._crc32 = None
        aloc = len(self.serialize())
        super().__init__(location, aloc)

    def __repr__(self):
        return "{}({}, {})".format(
            type(self).__name__,
            self._location,
            ", ".join([repr(x) for x in self._akform]),
        )

    def generate_field_col_records(self):
        akform = self._akform
        field_names = akform.fields
        contents = akform.contents
        field_records = []
        column_records = []

        for field_id, (field_name, ak_col) in enumerate(zip(field_names, contents)):
            if not isinstance(ak_col, awkward._v2.forms.NumpyForm):
                raise NotImplementedError("only flat column is supported")
            ak_primitive = ak_col.primitive
            type_name = _ak_primitive_to_typename_dict[ak_primitive]
            parent_field_id = field_id
            field = NTuple_Field_Description(
                0, 0, parent_field_id, 0, 0, field_name, type_name, "", ""
            )
            type_num = _ak_primitive_to_num_dict[ak_primitive]
            type_size = uproot.const.rntuple_col_num_to_size_dict[type_num]
            col = NTuple_Column_Description(type_num, type_size, field_id, 0)

            field_records.append(field)
            column_records.append(col)

        return field_records, column_records

    def serialize(self):
        if self._serialize:
            return self._serialize
        env_header = uproot.const.rntuple_env_header
        feature_flag = _rntuple_feature_flag_format.pack(0)
        rc_tag = struct.Struct("I").pack(1)
        name = _serialize_rntuple_string(self._name)
        description = _serialize_rntuple_string(self._ntuple_description)
        # writer = _serialize_rntuple_string("uproot " + uproot.__version__)
        writer = _serialize_rntuple_string("ROOT v6.26/06")

        out = []
        out.extend([env_header, feature_flag, rc_tag, name, description, writer])

        field_records, column_records = self.generate_field_col_records()
        alias_records = []
        extra_type_info = []

        out.append(_serialize_rntuple_list_frame(field_records))
        out.append(_serialize_rntuple_list_frame(column_records))
        out.append(_serialize_rntuple_list_frame(alias_records))
        out.append(_serialize_rntuple_list_frame(extra_type_info))
        out_string = b"".join(out)
        self._crc32 = zlib.crc32(out_string)

        header_bytes = b"".join([out_string, self._crc32.to_bytes(4, "little")])
        self._serialize = header_bytes
        return self._serialize


"""
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|        Envelope Version       |        Minimum Version        |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
                         ENVELOPE PAYLOAD
    Feature flags
    Header checksum (CRC32)
    List frame of extension header envelope links
    List frame of column group record frames
    List frame of cluster summary record frames
    List frame of cluster group record frames
    List frame of meta-data block envelope links
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                             CRC32                             |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
"""


class NTuple_Footer(CascadeLeaf):
    def __init__(self, location, feature_flags, header_crc32, akform):

        self._feature_flags = feature_flags
        self._header_crc32 = header_crc32
        self._akform = akform
        self._page_list_link = NTuple_EnvLink(16, None)

        super().__init__(location, None)

    def __repr__(self):
        return "{}({}, {}, {}, {})".format(
            type(self).__name__,
            self._location,
            self._feature_flags,
            self._header_crc32,
            ", ".join([repr(x) for x in self._akform]),
        )

    def serialize(self):
        env_header = uproot.const.rntuple_env_header
        out = []
        feature_flag = _rntuple_feature_flag_format.pack(0)
        out.extend([env_header, feature_flag, self._header_crc32.to_bytes(4, "little")])

        extension_header_envelope_links = []
        column_group_record_frames = []
        cluster_summary_record_frames = []
        cluster_group_record_frames = [
            NTuple_ClusterGroupRecord(0, self._page_list_link)
        ]
        metadata_block_envelope_links = []

        out.append(_serialize_rntuple_list_frame(extension_header_envelope_links))
        out.append(_serialize_rntuple_list_frame(column_group_record_frames))
        out.append(_serialize_rntuple_list_frame(cluster_summary_record_frames))
        out.append(
            _serialize_rntuple_list_frame(cluster_group_record_frames)
        )  # never empty
        out.append(_serialize_rntuple_list_frame(metadata_block_envelope_links))
        out_bytes = b"".join(out)
        crc32 = zlib.crc32(out_bytes)
        return out_bytes + crc32.to_bytes(4, "little")


class NTuple_Locator:
    def __init__(self, num_bytes, offset):
        self._num_bytes = num_bytes
        self._offset = offset

    def serialize(self):
        outbytes = _rntuple_locator_format.pack(self._num_bytes, self._offset)
        return outbytes


class NTuple_EnvLink:
    def __init__(self, uncomp_size, locator):
        self.uncomp_size = uncomp_size
        self.locator = locator

    def serialize(self):
        out = [struct.Struct("<I").pack(self.uncomp_size), self.locator.serialize()]
        return b"".join(out)


class NTuple_ClusterGroupRecord:
    def __init__(self, num_clusters, page_list_link):
        self.num_clusters = num_clusters
        self.page_list_link = page_list_link

    def serialize(self):
        header_bytes = _rntuple_cluster_group_format.pack(self.num_clusters)
        page_list_link_bytes = self.page_list_link.serialize()
        return header_bytes + page_list_link_bytes


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
    def _fields(self):
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
            ", ".join([repr(x) for x in self._fields]),
        )

    def serialize(self):
        # hardcoded unless version changes
        # version = 0
        # aloc = _rntuple_format1.size
        # uproot.serialization.numbytes_version(aloc, version)
        out = _rntuple_format1.pack(*self._fields)
        crc32 = zlib.crc32(out[4:])
        self.fCheckSum = crc32
        out = _rntuple_format1.pack(*self._fields)
        return b"@\x00\x006\x00\x00" + out


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
        return "{}({}, {}, {}, {}, {}, {}, {}, {})".format(
            type(self).__name__,
            self._directory,
            self._name,
            self._title,
            self._header,
            self._footer,
            self._cluster_metadata,
            self._anchor,
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

    def add_rblob(
        self,
        sink,
        raw_data,
        uncompressed_bytes,
        big=None,
    ):

        strings_size = 8

        location = None
        if not big:
            requested_bytes = (
                uproot.reading._key_format_small.size + strings_size + len(raw_data)
            )
            location = self._freesegments.allocate(requested_bytes, dry_run=True)
            if location < uproot.const.kStartBigFile:
                self._freesegments.allocate(requested_bytes, dry_run=False)
            else:
                location = None

        if location is None:
            requested_bytes = (
                uproot.reading._key_format_big.size + strings_size + len(raw_data)
            )
            location = self._freesegments.allocate(requested_bytes, dry_run=False)

        key = RBlob_Key(
            location,
            uncompressed_bytes,
            len(raw_data),
            created_on=datetime.datetime.now(),
            big=big,
        )

        key.write(sink)
        sink.write(location + key.num_bytes, raw_data)
        sink.set_file_length(self._freesegments.fileheader.end)
        sink.flush()
        return key

    def write(self, sink):
        #### Header ##############################
        header_raw_data = self._header.serialize()
        self._header_key = self.add_rblob(
            sink,
            header_raw_data,
            len(header_raw_data),
            big=False,
        )
        self._anchor.fSeekHeader = (
            self._header_key.location + self._header_key.allocation
        )
        self._anchor.fNBytesHeader = len(header_raw_data)
        self._anchor.fLenHeader = len(header_raw_data)
        #### Header end ##############################

        #### Footer ##############################
        footer_raw_data = self._footer.serialize()
        self._footer_key = self.add_rblob(
            sink,
            footer_raw_data,
            len(footer_raw_data),
            big=False,
        )
        self._anchor.fSeekFooter = (
            self._footer_key.location + self._footer_key.allocation
        )
        self._anchor.fNBytesFooter = len(footer_raw_data)
        self._anchor.fLenFooter = len(footer_raw_data)
        #### Footer end ##############################

        #### Anchor ##############################
        anchor_raw_data = self._anchor.serialize()
        self._key = self._directory.add_object(
            sink,
            "ROOT::Experimental::RNTuple",
            self._name,
            self._title,
            anchor_raw_data,
            len(anchor_raw_data),
            replaces=self._key,
            big=False,
        )
        self._anchor.location = self._key.location + self._key.allocation
        #### Anchor end ##############################

        self._freesegments.write(sink)

    def write_updates(self, sink):
        sink.flush()

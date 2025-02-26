# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
This is an internal module for writing RNTuple in the "cascading" file writer.

The implementation in this module does not use the RNTuple infrastructure in
:doc:`uproot.models.RNTuple`.

See :doc:`uproot.writing._cascade` for a general overview of the cascading writer concept.
"""
from __future__ import annotations

import datetime
import struct

import awkward
import numpy
import xxhash

import uproot
import uproot.compression
import uproot.const
import uproot.reading
import uproot.serialization
from uproot.models.RNTuple import (
    _rntuple_anchor_checksum_format,
    _rntuple_anchor_format,
    _rntuple_checksum_format,
    _rntuple_cluster_group_format,
    _rntuple_cluster_summary_format,
    _rntuple_column_compression_settings_format,
    _rntuple_column_element_offset_format,
    _rntuple_column_record_format,
    _rntuple_env_header_format,
    _rntuple_envlink_size_format,
    _rntuple_feature_flag_format,
    _rntuple_field_description_format,
    _rntuple_frame_num_items_format,
    _rntuple_frame_size_format,
    _rntuple_locator_offset_format,
    _rntuple_locator_size_format,
    _rntuple_page_num_elements_format,
)
from uproot.writing._cascade import CascadeLeaf, CascadeNode, Key, String

_rntuple_string_length_format = struct.Struct("<I")

_ak_primitive_to_typename_dict = {
    "i64": "std::int64_t",
    "i32": "std::int32_t",
    # "switch": 3,
    # "byte": 4,
    # "char": 5,
    "bool": "bool",  # check
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
    "bool": 0x00,
    "int8": 0x03,
    "uint8": 0x04,
    "int16": 0x05,
    "uint16": 0x06,
    "int32": 0x07,
    "uint32": 0x08,
    "int64": 0x09,
    "uint64": 0x0A,
    "float16": 0x0B,
    "float32": 0x0C,
    "float64": 0x0D,
    "i32": 0x0E,
    "i64": 0x0F,
}


def _cpp_typename(akform, subcall=False):
    if isinstance(akform, awkward.forms.NumpyForm):
        ak_primitive = akform.primitive
        typename = _ak_primitive_to_typename_dict[ak_primitive]
    elif isinstance(akform, awkward.forms.ListOffsetForm):
        content_typename = _cpp_typename(akform.content, subcall=True)
        typename = f"std::vector<{content_typename}>"
    elif isinstance(akform, awkward.forms.RecordForm):
        typename = "UntypedRecord"
    else:
        raise NotImplementedError(f"Form type {type(akform)} cannot be written yet")
    if not subcall and "UntypedRecord" in typename:
        typename = ""  # empty types for anything that contains UntypedRecord
    return typename


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


def _serialize_string(content):
    content_bytes = str.encode(content, encoding="utf-8")
    return _rntuple_string_length_format.pack(len(content_bytes)) + content_bytes


def _record_frame_wrap(payload, includeself=True):
    aloc = len(payload)
    if includeself:
        aloc += _rntuple_frame_size_format.size
    raw_bytes = _rntuple_frame_size_format.pack(aloc) + payload
    return raw_bytes


def _serialize_rntuple_list_frame(items, wrap=True, rawinput=False, extra_payload=None):
    # when items is [], b'\xf4\xff\xff\xff\xff\xff\xff\xff\x00\x00\x00\x00'
    n_items = len(items)
    if wrap and rawinput:
        payload_bytes = b"".join([_record_frame_wrap(x) for x in items])
    elif rawinput:
        payload_bytes = b"".join(items)
    elif wrap:
        payload_bytes = b"".join([_record_frame_wrap(x.serialize()) for x in items])
    else:
        payload_bytes = b"".join([x.serialize() for x in items])
    if extra_payload is not None:
        payload_bytes += extra_payload
    size = (
        _rntuple_frame_size_format.size
        + _rntuple_frame_num_items_format.size
        + len(payload_bytes)
    )
    raw_bytes = _rntuple_frame_size_format.pack(-size)  # negative size means list
    raw_bytes += _rntuple_frame_num_items_format.pack(n_items)
    raw_bytes += payload_bytes
    return raw_bytes


def _serialize_envelope_header(type, length):
    assert type in uproot.const.RNTupleEnvelopeType
    assert 0 <= length < 1 << 48
    data = length
    data <<= 16
    data |= type
    return _rntuple_env_header_format.pack(data)


def _serialize_rntuple_page_innerlist(items):  # TODO: check
    n_items = len(items)
    payload_bytes = b"".join([x.serialize() for x in items])
    offset = (0).to_bytes(8, "little")
    compression_setting = (0).to_bytes(4, "little")
    payload_bytes = b"".join([payload_bytes, offset, compression_setting])
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
        return f"{type(self).__name__}({self.field_version!r}, {self.type_version!r}, {self.parent_field_id!r}, {self.struct_role!r}, {self.flags!r}, {self.field_name!r}, {self.type_name!r}, {self.type_alias!r}, {self.field_description!r})"

    def serialize(self):
        header_bytes = _rntuple_field_description_format.pack(
            self.field_version,
            self.type_version,
            self.parent_field_id,
            self.struct_role,
            self.flags,
        )
        string_bytes = b"".join(
            [
                _serialize_string(x)
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
    def __init__(self, type_num, bits_on_disk, field_id, flags, repr_index):
        self.type_num = type_num
        self.bits_on_disk = bits_on_disk
        self.field_id = field_id
        self.flags = flags
        self.repr_index = repr_index

    def serialize(self):
        header_bytes = _rntuple_column_record_format.pack(
            self.type_num,
            self.bits_on_disk,
            self.field_id,
            self.flags,
            self.repr_index,
        )
        return header_bytes


# https://github.com/root-project/root/blob/8cd9eed6f3a32e55ef1f0f1df8e5462e753c735d/tree/ntuple/v7/doc/BinaryFormatSpecification.md#header-envelope
class NTuple_Header(CascadeLeaf):
    def __init__(self, location, name, ntuple_description, akform):
        self._name = name
        self._ntuple_description = ntuple_description
        self._akform = akform

        self._serialize = None
        self._checksum = None
        self._field_records = []
        self._column_records = []
        self._column_keys = []
        self._ak_node_count = 0
        aloc = len(self.serialize())
        super().__init__(location, aloc)

    def __repr__(self):
        return "{}({}, {})".format(
            type(self).__name__,
            self._location,
            ", ".join([repr(x) for x in self._akform]),
        )

    def _build_field_col_records(self, akform, field_name=None, parent_fid=None):
        field_id = len(self._field_records)
        if parent_fid is None:
            parent_fid = field_id
        if field_name is None:
            field_name = f"_{field_id}"
        self._ak_node_count += 1
        if isinstance(akform, awkward.forms.NumpyForm):
            type_name = _cpp_typename(akform)
            field = NTuple_Field_Description(
                0,
                0,
                parent_fid,
                uproot.const.RNTupleFieldRole.LEAF,
                0,
                field_name,
                type_name,
                "",
                "",
            )
            self._field_records.append(field)
            type_num = _ak_primitive_to_num_dict[akform.primitive]
            type_size = uproot.const.rntuple_col_num_to_size_dict[type_num]
            col = NTuple_Column_Description(type_num, type_size, field_id, 0, 0)
            self._column_records.append(col)
            self._column_keys.append(f"node{self._ak_node_count}-data")
        elif isinstance(akform, awkward.forms.ListOffsetForm):
            type_name = _cpp_typename(akform)
            field = NTuple_Field_Description(
                0,
                0,
                parent_fid,
                uproot.const.RNTupleFieldRole.COLLECTION,
                0,
                field_name,
                type_name,
                "",
                "",
            )
            self._field_records.append(field)
            ak_offset = akform.offsets
            type_num = _ak_primitive_to_num_dict[ak_offset]
            type_size = uproot.const.rntuple_col_num_to_size_dict[type_num]
            col = NTuple_Column_Description(type_num, type_size, field_id, 0, 0)
            self._column_records.append(col)
            self._column_keys.append(f"node{self._ak_node_count}-offsets")
            # content data
            self._build_field_col_records(
                akform.content,
                parent_fid=field_id,
            )
        elif isinstance(akform, awkward.forms.RecordForm):
            field = NTuple_Field_Description(
                0,
                0,
                parent_fid,
                uproot.const.RNTupleFieldRole.RECORD,
                0,
                field_name,
                "",
                "",
                "",
            )
            self._field_records.append(field)
            for subfield_name, subakform in zip(akform.fields, akform.contents):
                self._build_field_col_records(
                    subakform,
                    field_name=subfield_name,
                    parent_fid=field_id,
                )
        else:
            raise NotImplementedError(f"Form type {type(akform)} cannot be written yet")

    def generate_field_col_records(self):
        akform = self._akform
        for field_name, topakform in zip(akform.fields, akform.contents):
            self._build_field_col_records(
                topakform,
                field_name=field_name,
            )

    def serialize(self):
        if self._serialize:
            return self._serialize

        feature_flag = _rntuple_feature_flag_format.pack(0)
        name = _serialize_string(self._name)
        description = _serialize_string(self._ntuple_description)
        writer = _serialize_string(f"Uproot {uproot.__version__}")

        out = []
        out.extend([feature_flag, name, description, writer])

        self.generate_field_col_records()
        alias_records = []
        extra_type_info = []

        out.append(_serialize_rntuple_list_frame(self._field_records))
        out.append(_serialize_rntuple_list_frame(self._column_records))
        out.append(_serialize_rntuple_list_frame(alias_records))
        out.append(_serialize_rntuple_list_frame(extra_type_info))
        payload = b"".join(out)

        env_header = _serialize_envelope_header(
            uproot.const.RNTupleEnvelopeType.HEADER,
            len(payload)
            + _rntuple_env_header_format.size
            + _rntuple_checksum_format.size,
        )
        header_and_payload = b"".join([env_header, payload])
        self._checksum = xxhash.xxh3_64_intdigest(header_and_payload)
        checksum_bytes = _rntuple_checksum_format.pack(self._checksum)

        final_bytes = b"".join([header_and_payload, checksum_bytes])
        self._serialize = final_bytes
        return self._serialize


# https://github.com/root-project/root/blob/8cd9eed6f3a32e55ef1f0f1df8e5462e753c735d/tree/ntuple/v7/doc/BinaryFormatSpecification.md#footer-envelope
class NTuple_Footer(CascadeLeaf):
    def __init__(self, location, header_checksum):
        self._header_checksum = header_checksum

        self.extension_field_record_frames = []
        self.extension_column_record_frames = []
        self.extension_alias_record_frames = []
        self.extension_extra_type_info = []

        self.cluster_group_record_frames = [
            #    NTuple_ClusterGroupRecord(0, NTuple_EnvLink(48, None))
        ]
        self._checksum = None

        super().__init__(location, None)

    def __repr__(self):
        return f"{type(self).__name__}(extension_field_record_frames = {self.extension_field_record_frames}, column_group_record_frames = {self.extension_column_record_frames}, cluster_summary_record_frames={self.extension_column_record_frames}, cluster_group_record_frames{self.cluster_group_record_frames}, metadata_block_envelope_link = {self.metadata_block_envelope_links})"

    def serialize(self):
        out = []
        out.extend(
            [
                _rntuple_feature_flag_format.pack(0),
                _rntuple_checksum_format.pack(self._header_checksum),
            ]
        )
        schema_extension_payload = b"".join(
            [
                _serialize_rntuple_list_frame(self.extension_field_record_frames),
                _serialize_rntuple_list_frame(self.extension_column_record_frames),
                _serialize_rntuple_list_frame(self.extension_alias_record_frames),
                _serialize_rntuple_list_frame(self.extension_extra_type_info),
            ]
        )
        out.append(_record_frame_wrap(schema_extension_payload))

        out.append(_serialize_rntuple_list_frame(self.cluster_group_record_frames))
        payload = b"".join(out)

        env_header = _serialize_envelope_header(
            uproot.const.RNTupleEnvelopeType.FOOTER,
            len(payload)
            + _rntuple_env_header_format.size
            + _rntuple_checksum_format.size,
        )
        header_and_payload = b"".join([env_header, payload])
        self._checksum = xxhash.xxh3_64_intdigest(header_and_payload)
        checksum_bytes = _rntuple_checksum_format.pack(self._checksum)

        final_bytes = b"".join([header_and_payload, checksum_bytes])
        return final_bytes


class NTuple_Locator:
    def __init__(self, num_bytes, offset):
        # approximate 2^16 - size of locator itself
        assert num_bytes < 32768
        self.num_bytes = num_bytes
        self.offset = offset

    def serialize(self):
        outbytes = _rntuple_locator_size_format.pack(
            self.num_bytes
        ) + _rntuple_locator_offset_format.pack(self.offset)
        return outbytes

    def __repr__(self):
        return f"{type(self).__name__}({self.num_bytes}, {self.offset})"


class NTuple_EnvLink:
    def __init__(self, uncomp_size, locator):
        self.uncomp_size = uncomp_size
        self.locator = locator

    def serialize(self):
        out = [
            _rntuple_envlink_size_format.pack(self.uncomp_size),
            self.locator.serialize(),
        ]
        return b"".join(out)

    def __repr__(self):
        return f"{type(self).__name__}({self.uncomp_size}, {self.locator})"


class NTuple_PageListEnvelope:
    def __init__(
        self, header_checksum, cluster_summaries, page_data, compression_settings=0
    ):
        self.header_checksum = header_checksum
        self.cluster_summaries = cluster_summaries
        self.page_data = page_data
        self.compression_settings = compression_settings
        self._checksum = None
        assert len(cluster_summaries) == len(page_data)

    def serialize(self):
        # For now we, only support one cluster per page list envelope
        nested_pagelist_rawbytes = _serialize_rntuple_list_frame(
            [  # list of clusters
                _serialize_rntuple_list_frame(
                    [  # list of columns
                        _serialize_rntuple_list_frame(
                            [  # list of pages
                                NTuple_PageDescription(page[1], page[0]) for page in col
                            ],
                            wrap=False,
                            extra_payload=b"".join(
                                [
                                    _rntuple_column_element_offset_format.pack(
                                        col[0][2]
                                    ),
                                    _rntuple_column_compression_settings_format.pack(
                                        self.compression_settings
                                    ),
                                ]
                            ),
                        )
                        for col in cluster_page_locations
                    ],
                    rawinput=True,
                    wrap=False,
                )
                for cluster_page_locations in self.page_data
            ],
            rawinput=True,
            wrap=False,
        )
        out = [
            _rntuple_checksum_format.pack(self.header_checksum),
            _serialize_rntuple_list_frame(self.cluster_summaries),
            nested_pagelist_rawbytes,
        ]
        payload = b"".join(out)

        env_header = _serialize_envelope_header(
            uproot.const.RNTupleEnvelopeType.PAGELIST,
            len(payload)
            + _rntuple_env_header_format.size
            + _rntuple_checksum_format.size,
        )
        header_and_payload = b"".join([env_header, payload])
        self._checksum = xxhash.xxh3_64_intdigest(header_and_payload)
        checksum_bytes = _rntuple_checksum_format.pack(self._checksum)

        final_bytes = b"".join([header_and_payload, checksum_bytes])
        return final_bytes


class NTuple_ClusterGroupRecord:
    def __init__(self, min_entry, entry_span, num_clusters, page_list_envlink):
        self.min_entry = min_entry
        self.entry_span = entry_span
        self.num_clusters = num_clusters
        self.page_list_envlink = page_list_envlink

    def serialize(self):
        header_bytes = _rntuple_cluster_group_format.pack(
            self.min_entry, self.entry_span, self.num_clusters
        )
        page_list_link_bytes = self.page_list_envlink.serialize()
        return header_bytes + page_list_link_bytes

    def __repr__(self):
        return f"{type(self).__name__}({self.num_clusters}, {self.page_list_envlink})"


class NTuple_ClusterSummary:
    def __init__(self, num_first_entry, num_entries, flags=0):
        self.num_first_entry = num_first_entry
        self.num_entries = num_entries
        self.flags = flags

    def serialize(self):
        # Highest 8 bits are flags reserved for future use
        assert 0 <= self.num_first_entry < 2**56
        assert 0 <= self.flags < 2**8
        num_entries = (self.flags << 56) | self.num_entries
        payload_bytes = _rntuple_cluster_summary_format.pack(
            self.num_first_entry, num_entries
        )
        return payload_bytes

    def __repr__(self):
        return f"{type(self).__name__}({self.num_first_entry}, {self.num_entries}, {self.flags})"


class NTuple_InnerListLocator:
    def __init__(self, page_descs):
        self.page_descs = page_descs

    def serialize(self):
        # from RNTuple spec:
        # to save space, the page descriptions (inner items) are not in a record frame.
        raw_bytes = _serialize_rntuple_page_innerlist(self.page_descs)
        return raw_bytes

    def __repr__(self):
        return f"{type(self).__name__}({self.page_descs})"


class NTuple_PageDescription:
    def __init__(self, num_entries, locator):
        assert num_entries <= 65536
        self.num_entries = num_entries
        self.locator = locator

    def serialize(self):
        out = [
            _rntuple_page_num_elements_format.pack(self.num_entries),
            self.locator.serialize(),
        ]
        return b"".join(out)

    def __repr__(self):
        return f"{type(self).__name__}({self.num_entries}, {self.locator})"


class NTuple_Anchor(CascadeLeaf):
    def __init__(
        self,
        location,
        version_epoch,
        version_major,
        version_minor,
        version_patch,
        seek_header,
        nbytes_header,
        len_header,
        seek_footer,
        nbytes_footer,
        len_footer,
        max_key_size,
    ):
        aloc = (
            _rntuple_anchor_format.size
            + _rntuple_checksum_format.size
            + 6  # 6 bytes from header
        )
        super().__init__(location, aloc)
        self.version_epoch = version_epoch
        self.version_major = version_major
        self.version_minor = version_minor
        self.version_patch = version_patch
        self.seek_header = seek_header
        self.nbytes_header = nbytes_header
        self.len_header = len_header
        self.seek_footer = seek_footer
        self.nbytes_footer = nbytes_footer
        self.len_footer = len_footer
        self.max_key_size = max_key_size
        self.checksum = None

    @property
    def _fields(self):
        return [
            self.version_epoch,
            self.version_major,
            self.version_minor,
            self.version_patch,
            self.seek_header,
            self.nbytes_header,
            self.len_header,
            self.seek_footer,
            self.nbytes_footer,
            self.len_footer,
            self.max_key_size,
        ]

    def __repr__(self):
        return "{}({}, {})".format(
            type(self).__name__,
            self._location,
            ", ".join([repr(x) for x in self._fields]),
        )

    def serialize(self):
        out = _rntuple_anchor_format.pack(*self._fields)
        self.checksum = xxhash.xxh3_64_intdigest(out)
        checksum_bytes = _rntuple_anchor_checksum_format.pack(self.checksum)
        header = uproot.serialization.numbytes_version(
            _rntuple_anchor_format.size, 2
        )  # TODO: check
        out = b"".join([header, out, checksum_bytes])
        return out


class Ntuple_PageLink:
    def __init__(self, num_bytes, offset):
        self.num_bytes = num_bytes
        self.offset = offset

    def serialize(self):
        outbytes = _rntuple_locator_size_format.pack(self.num_bytes, self.offset)
        return outbytes

    def __repr__(self):
        return f"{type(self).__name__}({self.num_bytes}, {self.offset})"


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
        ak_form,
        freesegments,
        header,
        footer,
        cluster_metadata,
        anchor,
    ):
        super().__init__(footer, anchor, freesegments)
        self._directory = directory
        self._header = header
        self._footer = footer
        self._cluster_metadata = cluster_metadata
        self._anchor = anchor
        self._freesegments = freesegments

        self._key = None
        self._header_key = None
        self._num_entries = 0

        self._column_counts = numpy.zeros(len(self._header._column_keys), dtype=int)
        self._column_offsets = numpy.zeros(len(self._header._column_keys), dtype=int)

    def __repr__(self):
        return f"{type(self).__name__}({self._directory}, {self._header}, {self._footer}, {self._cluster_metadata}, {self._anchor}, {self._freesegments})"

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
        """
        1. Write pages
        2. Write page list for new cluster group
        3. Relocate footer
        4. Update anchor's foot metadata values in-place
        """

        if data.layout.form != self._header._akform:
            raise ValueError("data is not compatible with this RNTuple")

        # 1. Write pages
        # We write a single page for each column for now

        cluster_page_data = []  # list of list of (locator, len, offset)
        data_buffers = awkward.to_buffers(data)[2]
        for idx, key in enumerate(self._header._column_keys):
            col_data = data_buffers[key]
            if "offsets" in key:
                col_data = (
                    col_data[1:] + self._column_offsets[idx]
                )  # TODO: check if there is a better way to do this
                self._column_offsets[idx] = col_data[-1]
            raw_data = col_data.view("uint8")
            page_key = self.add_rblob(sink, raw_data, len(raw_data), big=False)
            page_locator = NTuple_Locator(
                len(raw_data), page_key.location + page_key.allocation  # probably wrong
            )
            cluster_page_data.append(
                [(page_locator, len(col_data), self._column_counts[idx])]
            )
            self._column_counts[idx] += len(col_data)
        page_data = [
            cluster_page_data
        ]  # list of list of list of (locator, len, offset)

        # 2. Write page list envelope for new cluster group

        # only a single cluster for now
        cluster_summaries = [NTuple_ClusterSummary(self._num_entries, len(data))]
        self._num_entries += len(data)

        pagelistenv = NTuple_PageListEnvelope(
            self._header._checksum,
            cluster_summaries,
            page_data,
        )
        pagelistenv_rawdata = pagelistenv.serialize()
        pagelistenv_key = self.add_rblob(
            sink, pagelistenv_rawdata, len(pagelistenv_rawdata), big=False
        )
        pagelistenv_locator = NTuple_Locator(
            len(pagelistenv_rawdata),
            pagelistenv_key.location + pagelistenv_key.allocation,
        )  # check
        pagelistenv_envlink = NTuple_EnvLink(
            len(pagelistenv_rawdata), pagelistenv_locator
        )

        cluster_group = NTuple_ClusterGroupRecord(
            self._num_entries - len(data), len(data), 1, pagelistenv_envlink
        )

        self._footer.cluster_group_record_frames.append(cluster_group)

        # 3. Relocate footer

        old_footer_key = self._footer_key
        self._freesegments.release(
            old_footer_key.location, old_footer_key.location + old_footer_key.allocation
        )
        footer_raw_data = self._footer.serialize()
        self._footer_key = self.add_rblob(
            sink,
            footer_raw_data,
            len(footer_raw_data),
            big=False,
        )

        # 4. Update anchor's foot metadata values in-place

        self._anchor.seek_footer = (
            self._footer_key.location + self._footer_key.allocation
        )
        self._anchor.nbytes_footer = len(footer_raw_data)
        self._anchor.len_footer = self._anchor.nbytes_footer

        anchor_raw_data = self._anchor.serialize()
        sink.write(self._anchor.location, anchor_raw_data)
        self._freesegments.write(sink)

        sink.flush()

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
        self._anchor.seek_header = (
            self._header_key.location + self._header_key.allocation
        )
        self._anchor.nbytes_header = len(header_raw_data)
        self._anchor.len_header = len(header_raw_data)
        #### Header end ##############################

        #### Footer ##############################
        footer_raw_data = self._footer.serialize()
        self._footer_key = self.add_rblob(
            sink,
            footer_raw_data,
            len(footer_raw_data),
            big=False,
        )
        self._anchor.seek_footer = (
            self._footer_key.location + self._footer_key.allocation
        )
        self._anchor.nbytes_footer = len(footer_raw_data)
        self._anchor.len_footer = len(footer_raw_data)
        #### Footer end ##############################

        #### Anchor ##############################
        anchor_raw_data = self._anchor.serialize()
        self._key = self._directory.add_object(
            sink,
            "ROOT::RNTuple",
            self._header._name,
            self._header._name,
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

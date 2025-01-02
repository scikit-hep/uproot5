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
import zlib

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
    _rntuple_column_record_format,
    _rntuple_env_header_format,
    _rntuple_envlink_size_format,
    _rntuple_feature_flag_format,
    _rntuple_field_description_format,
    _rntuple_frame_num_items_format,
    _rntuple_frame_size_format,
    _rntuple_locator_offset_format,
    _rntuple_locator_size_format,
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


def _serialize_string(content):
    return _rntuple_string_length_format.pack(len(content)) + str.encode(content)


def _record_frame_wrap(payload, includeself=True):
    aloc = len(payload)
    if includeself:
        aloc += _rntuple_frame_size_format.size
    raw_bytes = _rntuple_frame_size_format.pack(aloc) + payload
    return raw_bytes


def _serialize_rntuple_list_frame(items, wrap=True):
    # when items is [], b'\xf4\xff\xff\xff\xff\xff\xff\xff\x00\x00\x00\x00'
    n_items = len(items)
    if wrap:
        payload_bytes = b"".join([_record_frame_wrap(x.serialize()) for x in items])
    else:
        payload_bytes = b"".join([x.serialize() for x in items])
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
            if not isinstance(ak_col, awkward.forms.NumpyForm):
                raise NotImplementedError("only flat column is supported")
            ak_primitive = ak_col.primitive
            type_name = _ak_primitive_to_typename_dict[ak_primitive]
            parent_field_id = field_id
            field = NTuple_Field_Description(
                0, 0, parent_field_id, 0, 0, field_name, type_name, "", ""
            )
            type_num = _ak_primitive_to_num_dict[ak_primitive]
            type_size = uproot.const.rntuple_col_num_to_size_dict[type_num]
            col = NTuple_Column_Description(type_num, type_size, field_id, 0, 0)

            field_records.append(field)
            column_records.append(col)

        return field_records, column_records

    def serialize(self):
        if self._serialize:
            return self._serialize

        feature_flag = _rntuple_feature_flag_format.pack(0)
        name = _serialize_string(self._name)
        description = _serialize_string(self._ntuple_description)
        writer = _serialize_string(f"Uproot {uproot.__version__}")

        out = []
        out.extend([feature_flag, name, description, writer])

        field_records, column_records = self.generate_field_col_records()
        alias_records = []
        extra_type_info = []

        out.append(_serialize_rntuple_list_frame(field_records))
        out.append(_serialize_rntuple_list_frame(column_records))
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
    def __init__(self, location, feature_flags, header_checksum, akform):
        self._feature_flags = feature_flags
        self._header_checksum = header_checksum
        self._akform = akform

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
                _rntuple_feature_flag_format.pack(self._feature_flags),
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

        out.append(
            _serialize_rntuple_list_frame(self.cluster_group_record_frames)
        )  # never empty
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


class NTuple_ClusterGroupRecord:
    def __init__(self, num_clusters, page_list_envlink):
        self.num_clusters = num_clusters
        self.page_list_envlink = page_list_envlink

    def serialize(self):
        header_bytes = _rntuple_cluster_group_format.pack(0, 1, self.num_clusters)
        page_list_link_bytes = self.page_list_envlink.serialize()
        return header_bytes + page_list_link_bytes

    def __repr__(self):
        return f"{type(self).__name__}({self.num_clusters}, {self.page_list_envlink})"


class NTuple_ClusterSummary:
    def __init__(self, num_first_entry, num_entries):
        self.num_first_entry = num_first_entry
        self.num_entries = num_entries

    def serialize(self):
        # from spec:
        # to save space, the page descriptions (inner items) are not in a record frame.
        payload_bytes = _rntuple_cluster_summary_format.pack(
            self.num_first_entry, self.num_entries
        )
        return payload_bytes

    def __repr__(self):
        return f"{type(self).__name__}({self.num_first_entry}, {self.num_entries})"


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
    def __init__(self, num_elements, locator):
        assert num_elements <= 65536
        self.num_elements = num_elements
        self.locator = locator

    def serialize(self):
        return struct.Struct("<I").pack(self.num_elements) + self.locator.serialize()

    def __repr__(self):
        return f"{type(self).__name__}({self.num_elements}, {self.locator})"


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
        self._num_entries = 0

    def __repr__(self):
        return f"{type(self).__name__}({self._directory}, {self._name}, {self._title}, {self._header}, {self._footer}, {self._cluster_metadata}, {self._anchor}, {self._freesegments})"

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

    def actually_use(self, array):
        pass
        # print(type(array))
        # print(f"using {array!r}")

    def array_to_type(self, array, type):
        if isinstance(type, awkward.types.ArrayType):
            type = type.content
        # type is unknown
        if isinstance(type, awkward.types.UnknownType):
            raise TypeError("cannot write data of unknown type to RNTuple")

        # type is primitive (e.g. "float32")
        elif isinstance(type, awkward.types.NumpyType):
            if isinstance(array, awkward.contents.IndexedArray):
                self.array_to_type(array.project(), type)  # always project IndexedArray
                return
            elif isinstance(array, awkward.contents.EmptyArray):
                self.array_to_type(
                    array.to_NumpyArray(
                        awkward.types.numpytype.primitive_to_dtype(type.primitive)
                    ),
                    type,
                )
                return
            elif isinstance(array, awkward.contents.NumpyArray):
                if array.form.type != type:
                    raise TypeError(f"expected {type!s}, found {array.form.type!s}")
                else:
                    self.actually_use(array.data)
                    return
            else:
                raise TypeError(f"expected {type!s}, found {array.form.type!s}")

        # type is regular-length lists (e.g. "3 * float32")
        elif isinstance(type, awkward.types.RegularType):
            if isinstance(array, awkward.contents.IndexedArray):
                self.array_to_type(array.project(), type)  # always project IndexedArray
                return
            elif isinstance(array, awkward.contents.RegularArray):
                if array.size != type.size:
                    raise TypeError(f"expected {type!s}, found {array.form.type!s}")
                else:
                    if type.parameter("__array__") == "string":
                        # maybe the fact that this is a string changes how it's used
                        self.actually_use(f"regular strings of length {type.size}")
                    else:
                        self.actually_use(f"regular lists of length {type.size}")
                    self.array_to_type(array.content, type.content)
                    return
            else:
                raise TypeError(f"expected {type!s}, found {array.form.type!s}")

        # type is variable-length lists (e.g. "var * float32")
        elif isinstance(type, awkward.types.ListType):
            if isinstance(array, awkward.contents.IndexedArray):
                self.array_to_type(array.project(), type)  # always project IndexedArray
                return
            elif isinstance(array, awkward.contents.ListArray):
                self.array_to_type(array.toListOffsetArray64(True), type)
                return
            elif isinstance(array, awkward.contents.ListOffsetArray):
                if type.parameter("__array__") == "string":
                    # maybe the fact that this is a string changes how it's used
                    self.actually_use("variable-length strings")
                else:
                    self.actually_use("variable-length lists")
                self.actually_use(array.offsets.data)
                self.array_to_type(array.content, type.content)
                return
            else:
                raise TypeError(f"expected {type!s}, found {array.form.type!s}")

        # type is potentially missing data (e.g. "?float32")
        elif isinstance(type, awkward.types.OptionType):
            raise NotImplementedError("RNTuple does not yet have an option-type")

        # type is struct-like records (e.g. "{x: float32, y: var * int64}")
        elif isinstance(type, awkward.types.RecordType):
            if isinstance(array, awkward.contents.IndexedArray):
                self.array_to_type(array.project(), type)  # always project IndexedArray
                return
            elif isinstance(array, awkward.contents.RecordArray):
                self.actually_use("begin record")
                for field, subtype in zip(type.fields, type.contents):
                    self.actually_use(f"field {field}")
                    self.array_to_type(array[field], subtype)
                self.actually_use("end record")
                return
            else:
                raise TypeError(f"expected {type!s}, found {array.form.type!s}")

        # type is heterogeneous unions/variants (e.g. "union[float32, var * int64]")
        elif isinstance(type, awkward.types.UnionType):
            if isinstance(array, awkward.contents.IndexedArray):
                self.array_to_type(array.project(), type)  # always project IndexedArray
                return
            elif isinstance(array, awkward.contents.UnionArray):
                self.actually_use("begin union")
                self.actually_use(array.tags.data)
                self.actually_use(array.index.data)
                for index, subtype in enumerate(type.contents):
                    self.actually_use(f"index {index}")
                    self.array_to_type(array.project(index), subtype)
                self.actually_use("end union")
                return
            else:
                raise TypeError(f"expected {type!s}, found {array.form.type!s}")

        else:
            raise AssertionError(f"type must be an Awkward Type, not {type!r}")

    def extend(self, file, sink, data):
        """
        1. pages(data)
        2. page inner list locator
        3. page list envelopes
        4. relocate footer
        5. update anchor's foot metadata values in-place
        """

        # DUMMY, replace with real `data` later
        data = numpy.array([5, 4, 3, 2, 1], dtype="int32")
        #######################################

        cluster_summary = NTuple_ClusterSummary(self._num_entries, len(data))
        self._num_entries += len(data)
        self._footer.cluster_summary_record_frames.append(cluster_summary)
        data_bytes = data.view("uint8")
        page_key = self.add_rblob(sink, data_bytes, len(data_bytes), big=False)
        page_locator = NTuple_Locator(
            len(data_bytes), page_key.location + page_key.allocation
        )
        # FIXME use this
        # self.array_to_type(data.layout, data.type)

        # we always add one more `list of list` into the `footer.cluster_group_records`, because we always make a new
        # cluster
        page_desc = NTuple_PageDescription(len(data), page_locator)
        inner_page_list = NTuple_InnerListLocator([page_desc])
        inner_page_list_bytes = _serialize_rntuple_list_frame([inner_page_list], False)
        inner_size_bytes = struct.Struct("<i").pack(
            -len(inner_page_list_bytes) - 8
        )  # negative size means list
        # we always extend one cluster at a time
        outer_page_list_bytes = b"".join(
            [inner_size_bytes, struct.Struct("<i").pack(1), inner_page_list_bytes]
        )

        pagelist_bytes = uproot.const.rntuple_env_header + outer_page_list_bytes
        _crc32 = zlib.crc32(pagelist_bytes)

        pagelist_bytes += struct.Struct("<I").pack(_crc32)

        pagelist_key = self.add_rblob(
            sink, pagelist_bytes, len(pagelist_bytes), big=False
        )
        pagelist_locator = NTuple_Locator(
            len(pagelist_bytes), pagelist_key.location + pagelist_key.allocation
        )
        new_page_list_envlink = NTuple_EnvLink(len(pagelist_bytes), pagelist_locator)

        new_cluster_group_record = NTuple_ClusterGroupRecord(1, new_page_list_envlink)
        self._footer.cluster_group_record_frames[0] = new_cluster_group_record

        #### relocate Footer ##############################
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

        ### update anchor
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

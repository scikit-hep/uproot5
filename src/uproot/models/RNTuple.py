# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

"""
This module defines a versionless model for ``ROOT::Experimental::RNTuple``.
"""


import queue
import struct

import uproot, numpy

# https://github.com/root-project/root/blob/e9fa243af91217e9b108d828009c81ccba7666b5/tree/ntuple/v7/inc/ROOT/RMiniFile.hxx#L65
_rntuple_format1 = struct.Struct(">iIIQIIQIIQ")

# https://github.com/jblomer/root/blob/ntuple-binary-format-v1/tree/ntuple/v7/doc/specifications.md#envelopes
_rntuple_frame_format = struct.Struct("<HH")
_rntuple_feature_flag_format = struct.Struct("<Q")
_rntuple_num_bytes_fields = struct.Struct("<II")

def _renamemeA(chunk, cursor, context):
    env_version, min_version = cursor.fields(
        chunk, _rntuple_frame_format, context
    )
    return {"env_version": env_version, "min_version": min_version}


class LocatorReader:
    _locator_format = struct.Struct("<iQ")
    def read(self, chunk, cursor, context):
        out = MetaData("Locator")
        out.size, out.offset = cursor.fields(
            chunk, self._locator_format, context
        )
        return out

class EnvLinkReader:
    def read(self, chunk, cursor, context):
        out = MetaData("EnvLink")
        out.env_size = cursor.field(
            chunk, struct.Struct("<I"), context
        )
        out.locator = LocatorReader().read(chunk, cursor, context)
        return out
    
class MetaData:
    def __init__(self, name, **kwargs):
        self.__dict__["_name"] = name
        self.__dict__["_fields"] = kwargs

    @property
    def name(self):
        return self.__dict__["_name"]

    def __repr__(self):
        kwargs = ", ".join(f"{k}={v!r}" for k, v in self.__dict__["_fields"].items())
        return f"MetaData({self.name!r}, {kwargs})"
    def __getattr__(self, name):
        if not name.startswith("_"):
            return self.__dict__["_fields"][name]
        else:
            return self.__dict__[name]
    def __setattr__(self, name, val):
        self.__dict__["_fields"][name] = val
        
class ListFrameReader:
    _frame_header = struct.Struct("<ii")

    def __init__(self, payload):
        self.payload = payload

    def read(self, chunk, cursor, context):
        local_cursor = cursor.copy()
        num_bytes, num_items = local_cursor.fields(
            chunk, self._frame_header, context
        )
        assert num_bytes < 0, f"{num_bytes= !r}"
        cursor.skip(-num_bytes)
        return [self.payload.read(chunk, local_cursor, context) for _ in range(num_items)]

# https://github.com/jblomer/root/blob/ntuple-binary-format-v1/tree/ntuple/v7/doc/specifications.md#field-description
class FieldRecordFrameReader():
    _field_description = struct.Struct("<IIIHH")

    def read(self, chunk, cursor, context):
        out = MetaData("FieldRecordFrame")
        local_cursor = cursor.copy()
        num_bytes = local_cursor.field(
            chunk, struct.Struct('<i'), context
        )
        assert num_bytes > 0, f"{num_bytes= !r}"
        cursor.skip(num_bytes)

        out.field_record_frame = local_cursor.fields(
            chunk, self._field_description, context
        )
        out.f_name, out.t_name, out.t_alias, out.f_desc = [local_cursor.rntuple_string(chunk, context) for i in range(4)]
        return out

# https://github.com/jblomer/root/blob/ntuple-binary-format-v1/tree/ntuple/v7/doc/specifications.md#column-description
class ColumnRecordFrameReader:
    _column_description = struct.Struct("<HHII")

    def read(self, chunk, cursor, context):
        out = MetaData("ColumnRecordFrame")
        local_cursor = cursor.copy()
        num_bytes = local_cursor.field(
            chunk, struct.Struct('<i'), context
        )
        assert num_bytes > 0, f"{num_bytes= !r}"
        cursor.skip(num_bytes)

        out.column_header = local_cursor.fields(
            chunk, self._column_description, context
        )
        return out

# class ColumnRecordFrame:
#     def __init__(self, header):
#         self.header = header

class AliasColumnReader:
    _alias_column_= struct.Struct("<II")

    def read(self, chunk, cursor, context):
        out = MetaData("AliasColumn")
        local_cursor = cursor.copy()
        num_bytes = local_cursor.field(
            chunk, struct.Struct('<i'), context
        )
        assert num_bytes > 0, f"{num_bytes= !r}"
        cursor.skip(num_bytes)

        out.physical_id, out.field_id = local_cursor.fields(
            chunk, self._alias_column, context
        )
        return out

class ExtraTypeInfoReader:
    _extra_type_info = struct.Struct("<III")

    def read(self, chunk, cursor, context):
        out = MetaData("ExtraTypeInfoReader")
        local_cursor = cursor.copy()
        num_bytes = local_cursor.field(
            chunk, struct.Struct('<i'), context
        )
        assert num_bytes > 0, f"{num_bytes= !r}"
        cursor.skip(num_bytes)

        out.type_ver_from, out.type_ver_to, out.content_id = local_cursor.fields(
            chunk, self._extra_type_info, context
        )
        out.type_name = local_cursor.rntuple_string(chunk, context)
        return out

class HeaderReader:
    def __init__(self):
        self.list_field_record_frames = ListFrameReader(FieldRecordFrameReader())
        self.list_column_record_frames = ListFrameReader(ColumnRecordFrameReader())
        self.list_alias_column_frames = ListFrameReader(AliasColumnReader())
        self.list_extra_type_info_reader = ListFrameReader(ExtraTypeInfoReader())
        
    def read(self, chunk, cursor, context):
        out = MetaData("Header")
        out.header_frame = _renamemeA(chunk, cursor, context)
        out.feature_flag = cursor.field(
            chunk, _rntuple_feature_flag_format, context
        )
        out.rc_tag = cursor.field(
            chunk, struct.Struct("I"), context
        )
        out.name, out.ntuple_description, out.writer_identifier = [cursor.rntuple_string(chunk, context) for _ in range(3)]

        out.field_records = self.list_field_record_frames.read(chunk, cursor, context)
        out.column_records = self.list_column_record_frames.read(chunk, cursor, context)
        out.alias_columns = self.list_alias_column_frames.read(chunk, cursor, context)
        out.extra_type_infos = self.list_extra_type_info_reader.read(chunk, cursor, context)
        out.crc32 = cursor.field(chunk, struct.Struct("<I"), context)

        return out

# class Header:
#     def __init__(self, header_frame, rc_tag, feature_flag, ntuple_name, ntuple_description, writer_identifier, list_field_record_frames,
#             list_column_record_frames, list_alias_columns, list_extra_type_info, crc32):
#         self.header_frame = header_frame
#         self.rc_tag = rc_tag
#         self.feature_flag = feature_flag
#         self.ntuple_name = ntuple_name
#         self.ntuple_description = ntuple_description
#         self.writer_identifier = writer_identifier
#         self.list_field_record_frames = list_field_record_frames
#         self.list_column_record_frames = list_column_record_frames
#         self.list_alias_columns = list_alias_columns
#         self.list_extra_type_info = list_extra_type_info
#         self.crc32 = crc32

#     def __repr__(self):
#         return f"RNTuple Header: {self.rc_tag=}, {self.feature_flag=}, {self.ntuple_name=}, {self.crc32=}"

#     @property
#     def field_names(self):
#         return [x.field_name for x in self.list_field_record_frames]
#     @property
#     def field_type_names(self):
#         return [x.type_name for x in self.list_field_record_frames]

class ColumnGroupRecordReader:
    def read(self, chunk, cursor, context):
        out = MetaData("ClusterSummaryRecord")
        out.num_first_entry, out.num_entries = cursor.fields(
            chunk, self._cluster_summary_format, context
        )
        return out


class ClusterSummaryRecordReader:
    _cluster_summary_format = struct.Struct("<QQ")

    def read(self, chunk, cursor, context):
        out = MetaData("ClusterSummaryRecord")
        out.num_first_entry, out.num_entries = cursor.fields(
            chunk, self._cluster_summary_format, context
        )
        return out

class ClusterGroupRecordReader:
    _cluster_group_format = struct.Struct("<I")

    def read(self, chunk, cursor, context):
        out = MetaData("ClusterGroupRecord")
        out.num_clusters = cursor.field(
            chunk, self._cluster_group_format, context
        )
        out.page_list_link = EnvLinkReader().read(chunk, cursor, context)
        return out

class FooterReader:
    def __init__(self):
        self.extension_header_links = ListFrameReader(EnvLinkReader())
        self.column_group_record_frames = ListFrameReader(ColumnGroupRecordReader())
        self.cluster_summary_frames = ListFrameReader(ClusterSummaryRecordReader())
        self.cluster_group_record_frames = ListFrameReader(ClusterGroupRecordReader())
        self.meta_data_links = ListFrameReader(EnvLinkReader())

    def read(self, chunk, cursor, context):
        out = MetaData("Footer")
        out.footer_frame = _renamemeA(chunk, cursor, context)
        out.feature_flag = cursor.field(chunk, _rntuple_feature_flag_format, context)
        out.crc32 = cursor.field(chunk, struct.Struct("<I"), context)

        out.list_extension_links = self.extension_header_links.read(chunk, cursor, context)
        out.list_col_group_records = self.column_group_record_frames.read(chunk, cursor, context)
        out.list_cluster_records = self.cluster_summary_frames.read(chunk, cursor, context)
        out.list_cluster_records = self.cluster_group_record_frames.read(chunk, cursor, context)
        out.list_meta_block_links = self.meta_data_links.read(chunk, cursor, context)
        return out


class Model_ROOT_3a3a_Experimental_3a3a_RNTuple(uproot.model.Model):
    """
    A versionless :doc:`uproot.model.Model` for ``ROOT::Experimental::RNTuple``.
    """
    @property
    def keys(self):
        return self.header.field_names

    def read_members(self, chunk, cursor, context, file):
        if self.is_memberwise:
            raise NotImplementedError(
                """memberwise serialization of {}
in file {}""".format(
                    type(self).__name__, self.file.file_path
                )
            )

        (
            self._members["fCheckSum"],
            self._members["fVersion"],
            self._members["fSize"],
            self._members["fSeekHeader"],
            self._members["fNBytesHeader"],
            self._members["fLenHeader"],
            self._members["fSeekFooter"],
            self._members["fNBytesFooter"],
            self._members["fLenFooter"],
            self._members["fReserved"],
        ) = cursor.fields(chunk, _rntuple_format1, context)

        seek, nbytes = self._members["fSeekHeader"], self._members["fNBytesHeader"]
        header_range = (seek, seek + nbytes)

        seek, nbytes = self._members["fSeekFooter"], self._members["fNBytesFooter"]
        footer_range = (seek, seek + nbytes)

        notifications = queue.Queue()
        compressed_header_chunk, compressed_footer_chunk = file.source.chunks(
            [header_range, footer_range], notifications=notifications
        )

        if self._members["fNBytesHeader"] == self._members["fLenHeader"]:
            self._header_chunk = compressed_header_chunk
            self._header_cursor = uproot.source.cursor.Cursor(
                self._members["fSeekHeader"]
            )
        else:
            self._header_chunk = uproot.compression.decompress(
                compressed_header_chunk,
                uproot.source.cursor.Cursor(self._members["fSeekHeader"]),
                context,
                self._members["fNBytesHeader"],
                self._members["fLenHeader"],
            )
            self._header_cursor = uproot.source.cursor.Cursor(0)

        if self._members["fNBytesFooter"] == self._members["fLenFooter"]:
            self._footer_chunk = compressed_footer_chunk
            self._footer_cursor = uproot.source.cursor.Cursor(
                self._members["fSeekFooter"]
            )
        else:
            self._footer_chunk = uproot.compression.decompress(
                compressed_footer_chunk,
                uproot.source.cursor.Cursor(self._members["fSeekFooter"]),
                context,
                self._members["fNBytesFooter"],
                self._members["fLenFooter"],
            )
            self._footer_cursor = uproot.source.cursor.Cursor(0)

        self._header, self._footer = None, None

    @property
    def header(self):
        if self._header is None:
            cursor = self._header_cursor.copy()
            context = {}

            h = HeaderReader().read(self._header_chunk, cursor, context)
            self._header = h

        return self._header

    @property
    def footer(self):
        if self._footer is None:
            cursor = self._footer_cursor.copy()
            context = {}

            f = FooterReader().read(self._footer_chunk, cursor, context)
            assert f.crc32 == self.header.crc32, f"{self.header.crc32=}, {f.crc32=}"
            self._footer = f

        return self._footer



uproot.classes[
    "ROOT::Experimental::RNTuple"
] = Model_ROOT_3a3a_Experimental_3a3a_RNTuple

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

"""
This module defines a versionless model for ``ROOT::Experimental::RNTuple``.
"""


import queue
import struct

import uproot

# https://github.com/root-project/root/blob/e9fa243af91217e9b108d828009c81ccba7666b5/tree/ntuple/v7/inc/ROOT/RMiniFile.hxx#L65
_rntuple_format1 = struct.Struct(">iIIQIIQIIQ")

# https://github.com/jblomer/root/blob/ntuple-binary-format-v1/tree/ntuple/v7/doc/specifications.md#envelopes
_rntuple_frame_format = struct.Struct("<HHI")
_rntuple_feature_flag_format = struct.Struct("<Q")
_rntuple_num_bytes_fields = struct.Struct("<II")

def _renamemeA(chunk, cursor, context):
    version, min_version, num_bytes = cursor.fields(
        chunk, _rntuple_frame_format, context
    )
    return {"version": version, "min_version": min_version, "num_bytes": num_bytes}

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
        local_cursor = cursor.copy()
        num_bytes = local_cursor.field(
            chunk, struct.Struct('<i'), context
        )
        assert num_bytes > 0, f"{num_bytes= !r}"
        cursor.skip(num_bytes)

        field_record_frame = local_cursor.fields(
            chunk, self._field_description, context
        )
        f_name, t_name, t_alias, f_desc = [local_cursor.rntuple_string(chunk, context) for i in range(4)]
        return FieldRecordFrame(field_record_frame, f_name, t_name, t_alias, f_desc)

class FieldRecordFrame:
    def __init__(self, header, field_name, type_name, type_alias, field_description):
        self.header = header
        self.field_name = field_name
        self.type_name = type_name
        self.type_alias = type_alias
        self.field_description = field_description

# https://github.com/jblomer/root/blob/ntuple-binary-format-v1/tree/ntuple/v7/doc/specifications.md#column-description
class ColumnRecordFrameReader():
    _column_description = struct.Struct("<HHII")

    def read(self, chunk, cursor, context):
        local_cursor = cursor.copy()
        num_bytes = local_cursor.field(
            chunk, struct.Struct('<i'), context
        )
        assert num_bytes > 0, f"{num_bytes= !r}"
        cursor.skip(num_bytes)

        column_header = local_cursor.fields(
            chunk, self._column_description, context
        )
        return ColumnRecordFrame(column_header)

class ColumnRecordFrame:
    def __init__(self, header):
        self.header = header

class AliasColumnReader():
    _alias_column_= struct.Struct("<II")

    def read(self, chunk, cursor, context):
        local_cursor = cursor.copy()
        num_bytes = local_cursor.field(
            chunk, struct.Struct('<i'), context
        )
        assert num_bytes > 0, f"{num_bytes= !r}"
        cursor.skip(num_bytes)

        physical_id, field_id = local_cursor.fields(
            chunk, self._alias_column, context
        )
        return AliasColumn(physical_id, field_id)

class AliasColumn:
    def __init__(self, physical_id, field_id):
        self.physical_id = physical_id
        self.field_id = field_id

class ExtraTypeInfoReader():
    _extra_type_info = struct.Struct("<III")

    def read(self, chunk, cursor, context):
        local_cursor = cursor.copy()
        num_bytes = local_cursor.field(
            chunk, struct.Struct('<i'), context
        )
        assert num_bytes > 0, f"{num_bytes= !r}"
        cursor.skip(num_bytes)

        type_ver_from, type_ver_to, content_id = local_cursor.fields(
            chunk, self._extra_type_info, context
        )
        type_name, = local_cursor.rntuple_string(chunk, context)
        return ExtraTypeInfo(type_ver_from, type_ver_to, content_id, type_name)

class ExtraTypeInfo:
    def __init__(fr, to, tid, tname):
        self.type_ver_from = fr
        self.type_ver_to = to
        self.type_id = tid
        self.type_name = tname

class HeaderReader:
    def __init__(self):
        self.list_field_record_frames_reader = ListFrameReader(FieldRecordFrameReader())
        self.list_column_record_frames_reader = ListFrameReader(ColumnRecordFrameReader())
        self.list_alias_columns_reader = ListFrameReader(AliasColumnReader())
        self.list_extra_type_info_reader = ListFrameReader(ExtraTypeInfoReader())
        
    def read(self, chunk, cursor, context):
        header_frame = _renamemeA(chunk, cursor, context)
        feature_flag = cursor.field(
            chunk, _rntuple_feature_flag_format, context
        )
        name, ntuple_description, writer_identifier = [cursor.rntuple_string(chunk, context) for _ in range(3)]

        list_field_record_frames = self.list_field_record_frames_reader.read(chunk, cursor, context)
        list_column_record_frames = self.list_column_record_frames_reader.read(chunk, cursor, context)
        list_alias_columns = self.list_alias_columns_reader.read(chunk, cursor, context)
        list_extra_type_info = self.list_extra_type_info_reader.read(chunk, cursor, context)

        return Header(header_frame, feature_flag, name, ntuple_description, writer_identifier, list_field_record_frames,
                list_column_record_frames, list_alias_columns, list_extra_type_info)

class Header:
    def __init__(self, header_frame, feature_flag, ntuple_name, ntuple_description, writer_identifier, list_field_record_frames,
            list_column_record_frames, list_alias_columns, list_extra_type_info):
        self.header_frame = header_frame
        self.feature_flag = feature_flag
        self.ntuple_name = ntuple_name
        self.ntuple_description = ntuple_description
        self.writer_identifier = writer_identifier
        self.list_field_record_frames = list_field_record_frames
        self.list_column_record_frames = list_column_record_frames
        self.list_alias_columns = list_alias_columns
        self.list_extra_type_info = list_extra_type_info

    @property
    def field_names(self):
        return [x.field_name for x in self.list_field_record_frames]
    @property
    def field_type_names(self):
        return [x.type_name for x in self.list_field_record_frames]


class FooterReader:
    def __init__(self):
        self.extension_reader = ListFrameReader(LinkExtensionReader())
        self.col_group_reader = ListFrameReader(ColumnGroupReader())
        self.cluster_reader = ListFrameReader(ClusterReader())
        self.meta_reader = ListFrameReader(MetaBlockReader())
        
    def read(self, chunk, cursor, context):
        footer_frame = _renamemeA(chunk, cursor, context)
        feature_flag = cursor.field(chunk, _rntuple_feature_flag_format, context)
        crc32 = cursor.field(chunk, struct.Struct("<i"), context)

        list_extension_links = self.extension_reader.read()
        list_col_group_records = self.col_group_reader.read()
        list_cluster_records = self.cluster_reader.read()
        list_meta_block_links = self.meta_reader.read()
        return Footer(self, footer_frame, feature_flag, list_extension_links, 
                list_col_group_records, list_cluster_records, list_meta_block_links)

class Footer:
    def __init__(self, header_frame, feature_flag, list_extension_links, list_col_group_records,
            list_cluster_records, list_meta_block_links):
        self.header_frame = header_frame
        self.feature_flag = feature_flag
        self.list_extension_links = list_extension_links
        self.list_col_group_records = list_col_group_records
        self.list_cluster_records = list_cluster_records
        self.list_meta_block_links = list_meta_block_links

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

            self._footer = {}
            self._footer["frame"] = _renamemeA(self._footer_chunk, cursor, context)
            self._footer["feature_flag"] = cursor.field(
                self._footer_chunk, _rntuple_feature_flag_format, context
            )
            self._footer["CRC32"] = cursor.field(self._footer_chunk, struct.Struct('<i'), context)
            # self.read_list_of_extension_header_envelope_links(self._footer_chunk, cursor, context)
            # self.read_list_of_column_record_frames(self._header_chunk, cursor, context)
            # self.read_list_of_alias_column_record_frames(self._header_chunk, cursor, context)
            # self.read_list_of_extra_type_info(self._header_chunk, cursor, context)

        return self._footer



uproot.classes[
    "ROOT::Experimental::RNTuple"
] = Model_ROOT_3a3a_Experimental_3a3a_RNTuple

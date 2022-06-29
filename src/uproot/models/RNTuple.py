# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

"""
This module defines a versionless model for ``ROOT::Experimental::RNTuple``.
"""


import queue
import struct

import numpy

import uproot

# https://github.com/root-project/root/blob/e9fa243af91217e9b108d828009c81ccba7666b5/tree/ntuple/v7/inc/ROOT/RMiniFile.hxx#L65
_rntuple_format1 = struct.Struct(">iIIQIIQIIQ")

# https://github.com/jblomer/root/blob/ntuple-binary-format-v1/tree/ntuple/v7/doc/specifications.md#envelopes
_rntuple_frame_format = struct.Struct("<HH")
_rntuple_feature_flag_format = struct.Struct("<Q")
_rntuple_num_bytes_fields = struct.Struct("<II")


def _renamemeA(chunk, cursor, context):
    env_version, min_version = cursor.fields(chunk, _rntuple_frame_format, context)
    return {"env_version": env_version, "min_version": min_version}


class LocatorReader:
    _locator_format = struct.Struct("<iQ")

    def read(self, chunk, cursor, context):
        out = MetaData("Locator")
        out.num_bytes, out.offset = cursor.fields(chunk, self._locator_format, context)
        return out


class EnvLinkReader:
    def read(self, chunk, cursor, context):
        out = MetaData("EnvLink")
        out.env_uncomp_size = cursor.field(chunk, struct.Struct("<I"), context)
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


class RecordFrameReader:
    _record_size_format = struct.Struct("<I")

    def __init__(self, payload):
        self.payload = payload

    def read(self, chunk, cursor, context):
        local_cursor = cursor.copy()
        num_bytes = local_cursor.field(chunk, self._record_size_format, context)
        cursor.skip(num_bytes)
        return self.payload.read(chunk, local_cursor, context)


class ListFrameReader:
    _frame_header = struct.Struct("<ii")

    def __init__(self, payload):
        self.payload = payload

    def read(self, chunk, cursor, context):
        local_cursor = cursor.copy()
        num_bytes, num_items = local_cursor.fields(chunk, self._frame_header, context)
        assert num_bytes < 0, f"{num_bytes= !r}"
        cursor.skip(-num_bytes)
        return [self.payload.read(chunk, local_cursor, context) for _ in range(num_items)]


# https://github.com/jblomer/root/blob/ntuple-binary-format-v1/tree/ntuple/v7/doc/specifications.md#field-description
class FieldRecordReader:
    _field_description = struct.Struct("<IIIHH")

    def read(self, chunk, cursor, context):
        out = MetaData("FieldRecordFrame")

        out.field_record_frame = cursor.fields(chunk, self._field_description, context)
        out.f_name, out.t_name, out.t_alias, out.f_desc = (
            cursor.rntuple_string(chunk, context) for i in range(4)
        )
        return out


# https://github.com/jblomer/root/blob/ntuple-binary-format-v1/tree/ntuple/v7/doc/specifications.md#column-description
class ColumnRecordReader:
    _column_description = struct.Struct("<HHII")

    def read(self, chunk, cursor, context):
        out = MetaData("ColumnRecordFrame")
        out.column_header = cursor.fields(chunk, self._column_description, context)
        return out


class AliasColumnReader:
    _alias_column_ = struct.Struct("<II")

    def read(self, chunk, cursor, context):
        out = MetaData("AliasColumn")

        out.physical_id, out.field_id = cursor.fields(
            chunk, self._alias_column, context
        )
        return out


class ExtraTypeInfoReader:
    _extra_type_info = struct.Struct("<III")

    def read(self, chunk, cursor, context):
        out = MetaData("ExtraTypeInfoReader")

        out.type_ver_from, out.type_ver_to, out.content_id = cursor.fields(
            chunk, self._extra_type_info, context
        )
        out.type_name = cursor.rntuple_string(chunk, context)
        return out


class HeaderReader:
    def __init__(self):
        self.list_field_record_frames = ListFrameReader(
            RecordFrameReader(FieldRecordReader())
        )
        self.list_column_record_frames = ListFrameReader(
            RecordFrameReader(ColumnRecordReader())
        )
        self.list_alias_column_frames = ListFrameReader(
            RecordFrameReader(AliasColumnReader())
        )
        self.list_extra_type_info_reader = ListFrameReader(
            RecordFrameReader(ExtraTypeInfoReader())
        )

    def read(self, chunk, cursor, context):
        out = MetaData("Header")
        out.env_header = _renamemeA(chunk, cursor, context)
        out.feature_flag = cursor.field(chunk, _rntuple_feature_flag_format, context)
        out.rc_tag = cursor.field(chunk, struct.Struct("I"), context)
        out.name, out.ntuple_description, out.writer_identifier = (
            cursor.rntuple_string(chunk, context) for _ in range(3)
        )

        out.field_records = self.list_field_record_frames.read(chunk, cursor, context)
        out.column_records = self.list_column_record_frames.read(chunk, cursor, context)
        out.alias_columns = self.list_alias_column_frames.read(chunk, cursor, context)
        out.extra_type_infos = self.list_extra_type_info_reader.read(
            chunk, cursor, context
        )
        out.crc32 = cursor.field(chunk, struct.Struct("<I"), context)

        return out


class ColumnGroupRecordReader:
    def read(self, chunk, cursor, context):
        out = MetaData("ClusterSummaryRecord")
        out.num_first_entry, out.num_entries = cursor.fields(
            chunk, self._cluster_summary_format, context
        )
        return out


class ClusterSummaryReader:
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
        out.num_clusters = cursor.field(chunk, self._cluster_group_format, context)
        out.page_list_link = EnvLinkReader().read(chunk, cursor, context)
        return out


class FooterReader:
    def __init__(self):
        self.extension_header_links = ListFrameReader(EnvLinkReader())
        self.column_group_record_frames = ListFrameReader(
            RecordFrameReader(ColumnGroupRecordReader())
        )
        self.cluster_summary_frames = ListFrameReader(
            RecordFrameReader(ClusterSummaryReader())
        )
        self.cluster_group_record_frames = ListFrameReader(
            RecordFrameReader(ClusterGroupRecordReader())
        )
        self.meta_data_links = ListFrameReader(EnvLinkReader())

    def read(self, chunk, cursor, context):
        out = MetaData("Footer")
        out.env_header = _renamemeA(chunk, cursor, context)
        out.feature_flag = cursor.field(chunk, _rntuple_feature_flag_format, context)
        out.header_crc32 = cursor.field(chunk, struct.Struct("<I"), context)

        out.extension_links = self.extension_header_links.read(
            chunk, cursor, context
        )
        out.col_group_records = self.column_group_record_frames.read(
            chunk, cursor, context
        )
        out.cluster_summaries = self.cluster_summary_frames.read(
            chunk, cursor, context
        )
        out.cluster_records = self.cluster_group_record_frames.read(
            chunk, cursor, context
        )
        out.meta_block_links = self.meta_data_links.read(chunk, cursor, context)
        return out


class Model_ROOT_3a3a_Experimental_3a3a_RNTuple(uproot.model.Model):
    """
    A versionless :doc:`uproot.model.Model` for ``ROOT::Experimental::RNTuple``.
    """

    @property
    def keys(self):
        return self.header.column_names

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

        self._column_names = None

        # back link from column name to page inner list
        self.column_innerlist_dict = {}

    @property
    def header(self):
        if self._header is None:
            cursor = self._header_cursor.copy()
            context = {}

            h = HeaderReader().read(self._header_chunk, cursor, context)
            self._header = h

        return self._header

    @property
    def column_names(self):
        if self._column_names == None:
            self._column_names = [r.f_name for r in self.header.field_records]
        return self._column_names

    @property
    def footer(self):
        if self._footer is None:
            cursor = self._footer_cursor.copy()
            context = {}

            f = FooterReader().read(self._footer_chunk, cursor, context)
            assert f.header_crc32 == self.header.crc32, f"{self.header.crc32=}, {f.header_crc32=}"
            self._footer = f

        return self._footer

    # i-th cluster group
    def cluster_group(self, i):
        return self.page_list_envelopes[i]

    # i-th cluster group, j-th cluster
    def cluster_group_cluster(self, i, j):
        return self.cluster_group(i).top_list[j]

    # read_pages at i-th cluster group, j-th cluster, k-th column
    def pages_read(self, i, j, k):
        page_locs = self.cluster_group_cluster(i, j)

    def read_locator(self, loc, uncomp_size, cursor, context):
        chunk = self.file.source.chunk(loc.offset, loc.offset+loc.num_bytes)
        if loc.num_bytes < uncomp_size:
            decomp_chunk = uproot.compression.decompress(chunk, cursor, context, loc.num_bytes, uncomp_size, block_info=None)
            cursor.move_to(0)
        else:
            decomp_chunk = chunk
        return decomp_chunk

    @property
    def page_list_envelopes(self):
        context = {}
        context["column_innerlist_dict"] = self.column_innerlist_dict
        context["column_names"] = self.column_names 
        context["cluster_summaries"] = self.footer.cluster_summaries

        if not self.column_innerlist_dict:
            for record in self.footer.cluster_records:
                link = record.page_list_link
                loc = link.locator
                cursor = uproot.source.cursor.Cursor(loc.offset)
                decomp_chunk = self.read_locator(loc, link.env_uncomp_size, cursor, context)
                PageLink().read(decomp_chunk, cursor, context)
        return self.column_innerlist_dict

    # @property
    # def cluster_records(self):
    #     record = self.footer.cluster_records[0]
    #     link = record.page_list_link
    #     loc = link.locator
    #     cursor = uproot.source.cursor.Cursor(loc.offset)
    #     chunk = self.file.source.chunk(loc.offset, loc.offset+loc.num_bytes)
    #     context = {}
    #     if loc.num_bytes < link.env_uncomp_size:
    #         decomp_chunk = uproot.compression.decompress(chunk, cursor, context, loc.num_bytes, link.env_uncomp_size, block_info=None)
    #         cursor.move_to(0)
    #     else:
    #         decomp_chunk = chunk
    #     cursor.debug(decomp_chunk, dtype=numpy.int32)

# https://github.com/jblomer/root/blob/ntuple-binary-format-v1/tree/ntuple/v7/doc/specifications.md#page-list-envelope
class PageDescription:
    def read(self, chunk, cursor, context):
        out = MetaData(type(self).__name__)
        # print(f"PageDescription: {cursor}")
        # print(f"PageDescription: {cursor.bytes(chunk, 4, context=context, move=False)}")
        out.num_elements = cursor.field(chunk, struct.Struct("<I"), context)
        out.locator = LocatorReader().read(chunk, cursor, context)
        return out


# class PageLinkInnerEager:
#     def read(self, chunk, cursor, context):
#         out = MetaData(type(self).__name__)
#         out.page_descriptions = ListFrameReader(PageDescription()).read(chunk, cursor, context, skip = True)
#         # n.b the size of the inner list frame includes the element offset and compression settings
#         cursor.skip(-12)
#         out.ele_offset = cursor.field(chunk, struct.Struct("<Q"), context)
#         out.flags = cursor.field(chunk, struct.Struct("<I"), context)
#         return out

class InnerListLocator:
    def __init__(self, chunk, cursor, context, num_pages, cluster_summary):
        self.chunk = chunk
        self.cursor = cursor
        self.context = context
        self.num_pages = num_pages
        self.cluster_summary = cluster_summary
        self.reader = ListFrameReader(PageDescription())
    def __repr__(self):
        return f"InnerListLocator({self.chunk}, {self.cursor}, num_pages={self.num_pages}, {self.cluster_summary})"
    def read(self):
        return self.reader.read(self.chunk, self.cursor, self.context)

class PageLinkInner:
    _frame_header = struct.Struct("<ii")

    def read(self, chunk, cursor, context):
        out = MetaData(type(self).__name__)
        local_cursor = cursor.copy()
        out.local_cursor = local_cursor
        num_bytes, out.num_pages = local_cursor.fields(chunk, self._frame_header, context)
        out.num_bytes = -num_bytes
        assert out.num_bytes >= 0, f"{num_bytes= !r}"
        cursor.skip(out.num_bytes)
        d = context["column_innerlist_dict"]
        for col_name, summary in zip(context["column_names"], context["cluster_summaries"]):
            locator = InnerListLocator(chunk, cursor, context, out.num_pages, summary)
            if col_name in d.keys():
                d[col_name].append(locator)
            else:
                d[col_name] = [locator]


class PageLink:
    def __init__(self):
        self.top_most_list = ListFrameReader( #top-most list
                ListFrameReader( #outer list
                    PageLinkInner() #inner list
                    )
                )

    def read(self, chunk, cursor, context):
        out = MetaData(type(self).__name__)
        # print(f"PageLink: {cursor}")
        # print(f"PageLink: {cursor.bytes(chunk, 4, context=context, move=False)}")
        out.env_header = _renamemeA(chunk, cursor, context)

        # the follwoing mutates `RNTuple.column_innerlist_dict`
        self.top_most_list.read(chunk, cursor, context)
        out.crc32 = cursor.field(chunk, struct.Struct("<I"), context)
        return out


uproot.classes[
    "ROOT::Experimental::RNTuple"
] = Model_ROOT_3a3a_Experimental_3a3a_RNTuple

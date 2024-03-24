# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
This module defines a versionless model for ``ROOT::Experimental::RNTuple``.
"""
from __future__ import annotations

import struct
from collections import defaultdict

import numpy

import uproot

# https://github.com/root-project/root/blob/aa513463b0b512517370cb91cca025e53a8b13a2/tree/ntuple/v7/inc/ROOT/RNTupleAnchor.hxx#L69
_rntuple_anchor_format = struct.Struct(">HHHHQQQQQQQ")

# https://github.com/root-project/root/blob/aa513463b0b512517370cb91cca025e53a8b13a2/tree/ntuple/v7/doc/specifications.md#envelopes
_rntuple_feature_flag_format = struct.Struct("<Q")
_rntuple_env_header_format = struct.Struct("<Q")
_rntuple_field_description_format = struct.Struct("<IIIHH")
_rntuple_repetition_format = struct.Struct("<Q")
_rntuple_column_record_format = struct.Struct("<HHII")
_rntuple_alias_column_format = struct.Struct("<II")
_rntuple_extra_type_info_format = struct.Struct("<III")
_rntuple_record_size_format = struct.Struct("<q")
_rntuple_frame_header_format = struct.Struct("<qi")
_rntuple_cluster_group_format = struct.Struct("<qqi")
_rntuple_locator_format = struct.Struct("<iQ")
_rntuple_cluster_summary_format = struct.Struct("<QQ")
_rntuple_checksum_format = struct.Struct("<Q")
_rntuple_envlink_size_format = struct.Struct("<Q")
_rntuple_page_num_elements_format = struct.Struct("<I")
_rntuple_column_group_id_format = struct.Struct("<I")


def from_zigzag(n):
    return n >> 1 ^ -(n & 1)


def _envelop_header(chunk, cursor, context):
    env_data = cursor.field(chunk, _rntuple_env_header_format, context)
    env_type_id = env_data & 0xFFFF
    env_length = env_data >> 16
    return {"env_type_id": env_type_id, "env_length": env_length}


class Model_ROOT_3a3a_Experimental_3a3a_RNTuple(uproot.model.Model):
    """
    A versionless :doc:`uproot.model.Model` for ``ROOT::Experimental::RNTuple``.
    """

    @property
    def _keys(self):
        keys = []
        field_records = self.header.field_records
        for i, fr in enumerate(field_records):
            if fr.parent_field_id == i and fr.type_name != "":
                keys.append(fr.field_name)
        return keys

    def keys(self):
        return self._keys

    def read_members(self, chunk, cursor, context, file):
        if uproot._awkwardforth.get_forth_obj(context) is not None:
            raise uproot.interpretation.objects.CannotBeForth()
        if self.is_memberwise:
            raise NotImplementedError(
                f"""memberwise serialization of {type(self).__name__}
in file {self.file.file_path}"""
            )

        (
            self.members["fVersionEpoch"],
            self.members["fVersionMajor"],
            self.members["fVersionMinor"],
            self.members["fVersionPatch"],
            self._members["fSeekHeader"],
            self._members["fNBytesHeader"],
            self._members["fLenHeader"],
            self._members["fSeekFooter"],
            self._members["fNBytesFooter"],
            self._members["fLenFooter"],
            self._members["fChecksum"],
        ) = cursor.fields(chunk, _rntuple_anchor_format, context)

        self._header_chunk_ready = False
        self._footer_chunk_ready = False
        self._header, self._footer = None, None

        self._field_names = None
        self._column_records = None

        self._page_list_envelopes = []

    def _prepare_header_chunk(self):
        context = {}
        seek, nbytes = self._members["fSeekHeader"], self._members["fNBytesHeader"]

        compressed_header_chunk = self.file.source.chunk(seek, seek + nbytes)

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
        self._header_chunk_ready = True

    def _prepare_footer_chunk(self):
        context = {}
        seek, nbytes = self._members["fSeekFooter"], self._members["fNBytesFooter"]

        compressed_footer_chunk = self.file.source.chunk(seek, seek + nbytes)

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
        self._footer_chunk_ready = True

    @property
    def header(self):
        if self._header is None:
            if not self._header_chunk_ready:
                self._prepare_header_chunk()
            context = {}
            cursor = self._header_cursor.copy()

            h = HeaderReader().read(self._header_chunk, cursor, context)
            self._header = h

        return self._header

    @property
    def field_names(self):
        if self._field_names is None:
            self._field_names = [r.field_name for r in self.header.field_records]
        return self._field_names

    @property
    def column_records(self):
        return self.header.column_records

    @property
    def footer(self):
        if self._footer is None:
            if not self._footer_chunk_ready:
                self._prepare_footer_chunk()
            cursor = self._footer_cursor.copy()
            context = {}

            f = FooterReader().read(self._footer_chunk, cursor, context)
            assert (
                f.header_checksum == self.header.checksum
            ), f"checksum={self.header.checksum}, header_checksum={f.header_checksum}"
            self._footer = f

        return self._footer

    @property
    def cluster_summaries(self):
        return self.page_list_envelopes.cluster_summaries

    # FIXME
    @property
    def _length(self):
        return sum(x.num_entries for x in self.cluster_summaries)

    def __len__(self):
        return self._length

    def read_locator(self, loc, uncomp_size, context):
        cursor = uproot.source.cursor.Cursor(loc.offset)
        chunk = self.file.source.chunk(loc.offset, loc.offset + loc.num_bytes)
        if loc.num_bytes < uncomp_size:
            decomp_chunk = uproot.compression.decompress(
                chunk, cursor, context, loc.num_bytes, uncomp_size, block_info=None
            )
            cursor.move_to(0)
        else:
            decomp_chunk = chunk
        return decomp_chunk, cursor

    @property
    def page_list_envelopes(self):
        context = {}

        if not self._page_list_envelopes:
            for record in self.footer.cluster_group_records:
                link = record.page_list_link
                loc = link.locator
                decomp_chunk, cursor = self.read_locator(
                    loc, link.env_uncomp_size, context
                )
                self._page_list_envelopes = PageLink().read(
                    decomp_chunk, cursor, context
                )

        return self._page_list_envelopes

    def base_col_form(self, cr, col_id, parameters=None):
        ak = uproot.extras.awkward()

        form_key = f"column-{col_id}"
        dtype_byte = cr.type
        if dtype_byte == uproot.const.rntuple_role_union:
            return form_key
        elif dtype_byte > uproot.const.rntuple_role_struct:
            dt_str = uproot.const.rntuple_col_num_to_dtype_dict[dtype_byte]
            if dt_str == "bit":
                dt_str = "bool"
            return ak.forms.NumpyForm(
                dt_str,
                form_key=form_key,
                parameters=parameters,
            )
        else:  # offset index column
            return form_key

    def col_form(self, field_id):
        ak = uproot.extras.awkward()

        if field_id in self._column_records_dict:
            rel_crs = self._column_records_dict[field_id]["rel_crs"]
            rel_crs_idxs = self._column_records_dict[field_id]["rel_crs_idxs"]
        elif field_id in self._alias_columns_dict:
            rel_crs = self._column_records_dict[self._alias_columns_dict[field_id]][
                "rel_crs"
            ]
            rel_crs_idxs = self._column_records_dict[
                self._alias_columns_dict[field_id]
            ]["rel_crs_idxs"]
        else:
            raise (
                RuntimeError(
                    f"The filed_id: {field_id} is missing both from the columns records and the alias columns."
                )
            )

        if len(rel_crs) == 1:  # base case
            return self.base_col_form(rel_crs[0], rel_crs_idxs[0])
        elif (
            len(rel_crs_idxs) == 2
            and rel_crs[1].type == uproot.const.rntuple_col_type_to_num_dict["char"]
        ):
            # string field splits->2 in col records
            inner = self.base_col_form(
                rel_crs[1], rel_crs_idxs[-1], parameters={"__array__": "char"}
            )
            form_key = f"column-{rel_crs_idxs[0]}"
            return ak.forms.ListOffsetForm(
                "i64", inner, form_key=form_key, parameters={"__array__": "string"}
            )
        else:
            raise (RuntimeError(f"Missing special case: {field_id}"))

    def field_form(self, this_id, seen):
        ak = uproot.extras.awkward()

        field_records = self.header.field_records
        this_record = field_records[this_id]
        seen.add(this_id)
        structural_role = this_record.struct_role
        if (
            structural_role == uproot.const.rntuple_role_leaf
            and this_record.repetition == 0
        ):
            # base case of recursion
            # n.b. the split may happen in column
            return self.col_form(this_id)
        elif structural_role == uproot.const.rntuple_role_leaf:
            # std::array it only has one child
            if this_id in self._related_ids:
                child_id = self._related_ids[this_id][0]

            inner = self.field_form(child_id, seen)
            return ak.forms.RegularForm(inner, this_record.repetition)
        elif structural_role == uproot.const.rntuple_role_vector:
            keyname = f"column-{this_id}"
            #  this only has one child
            if this_id in self._related_ids:
                child_id = self._related_ids[this_id][0]
            inner = self.field_form(child_id, seen)
            return ak.forms.ListOffsetForm("u32", inner, form_key=keyname)
        elif structural_role == uproot.const.rntuple_role_struct:
            newids = []
            if this_id in self._related_ids:
                newids = self._related_ids[this_id]
            # go find N in the rest, N is the # of fields in struct
            recordlist = [self.field_form(i, seen) for i in newids]
            namelist = [field_records[i].field_name for i in newids]
            return ak.forms.RecordForm(recordlist, namelist, form_key="whatever")
        elif structural_role == uproot.const.rntuple_role_union:
            keyname = self.col_form(this_id)
            newids = []
            if this_id in self._related_ids:
                newids = self._related_ids[this_id]
            recordlist = [self.field_form(i, seen) for i in newids]
            return ak.forms.UnionForm("i8", "i64", recordlist, form_key=keyname)
        else:
            # everything should recurse above this branch
            raise AssertionError("this should be unreachable")

    def to_akform(self):
        ak = uproot.extras.awkward()

        field_records = self.header.field_records
        recordlist = []
        topnames = self.keys()
        seen = set()
        for i in range(len(field_records)):
            if i not in seen:
                recordlist.append(self.field_form(i, seen))

        form = ak.forms.RecordForm(recordlist, topnames, form_key="toplevel")
        return form

    def read_pagedesc(self, destination, desc, dtype_str, dtype, nbits, split):
        loc = desc.locator
        context = {}
        # bool in RNTuple is always stored as bits
        isbit = dtype_str == "bit"
        len_divider = 8 if isbit else 1
        num_elements = len(destination)
        num_elements_toread = int(numpy.ceil(num_elements / len_divider))
        uncomp_size = num_elements_toread * dtype.itemsize
        decomp_chunk, cursor = self.read_locator(loc, uncomp_size, context)
        content = cursor.array(
            decomp_chunk, num_elements_toread, dtype, context, move=False
        )

        if split:
            content = content.view(numpy.uint8)

            if nbits == 16:
                # AAAAABBBBB needs to become
                # ABABABABAB
                res = numpy.empty(len(content), numpy.uint8)
                res[0::2] = content[len(res) * 0 // 2 : len(res) * 1 // 2]
                res[1::2] = content[len(res) * 1 // 2 : len(res) * 2 // 2]
                res = res.view(numpy.uint16)

            elif nbits == 32:
                # AAAAABBBBBCCCCCDDDDD needs to become
                # ABCDABCDABCDABCDABCD
                res = numpy.empty(len(content), numpy.uint8)
                res[0::4] = content[len(res) * 0 // 4 : len(res) * 1 // 4]
                res[1::4] = content[len(res) * 1 // 4 : len(res) * 2 // 4]
                res[2::4] = content[len(res) * 2 // 4 : len(res) * 3 // 4]
                res[3::4] = content[len(res) * 3 // 4 : len(res) * 4 // 4]
                res = res.view(numpy.uint32)

            elif nbits == 64:
                # AAAAABBBBBCCCCCDDDDDEEEEEFFFFFGGGGGHHHHH needs to become
                # ABCDEFGHABCDEFGHABCDEFGHABCDEFGHABCDEFGH
                res = numpy.empty(len(content), numpy.uint8)
                res[0::8] = content[len(res) * 0 // 8 : len(res) * 1 // 8]
                res[1::8] = content[len(res) * 1 // 8 : len(res) * 2 // 8]
                res[2::8] = content[len(res) * 2 // 8 : len(res) * 3 // 8]
                res[3::8] = content[len(res) * 3 // 8 : len(res) * 4 // 8]
                res[4::8] = content[len(res) * 4 // 8 : len(res) * 5 // 8]
                res[5::8] = content[len(res) * 5 // 8 : len(res) * 6 // 8]
                res[6::8] = content[len(res) * 6 // 8 : len(res) * 7 // 8]
                res[7::8] = content[len(res) * 7 // 8 : len(res) * 8 // 8]
                res = res.view(numpy.uint64)

            content = res

        if isbit:
            content = (
                numpy.unpackbits(content.view(dtype=numpy.uint8))
                .reshape(-1, 8)[:, ::-1]
                .reshape(-1)
            )

        # needed to chop off extra bits incase we used `unpackbits`
        destination[:] = content[:num_elements]

    def read_col_pages(self, ncol, cluster_range):
        return numpy.concatenate(
            [self.read_col_page(ncol, i) for i in cluster_range], axis=0
        )

    def read_col_page(self, ncol, cluster_i):
        linklist = self.page_list_envelopes.pagelinklist[cluster_i]
        pagelist = linklist[ncol]
        dtype_byte = self.column_records[ncol].type
        dtype_str = uproot.const.rntuple_col_num_to_dtype_dict[dtype_byte]
        dtype = numpy.dtype("bool") if dtype_str == "bit" else numpy.dtype(dtype_str)

        # FIXME vector read
        # n.b. it's possible pagelist is empty
        if not pagelist:
            return numpy.empty(0, dtype)
        total_len = numpy.sum([desc.num_elements for desc in pagelist])
        res = numpy.empty(total_len, dtype)
        tracker = 0
        split = 14 <= dtype_byte <= 21 or 26 <= dtype_byte <= 28
        nbits = uproot.const.rntuple_col_num_to_size_dict[dtype_byte]
        for page_desc in pagelist:
            n_elements = page_desc.num_elements
            tracker_end = tracker + n_elements
            self.read_pagedesc(
                res[tracker:tracker_end], page_desc, dtype_str, dtype, nbits, split
            )
            tracker = tracker_end

        if (
            dtype_byte <= uproot.const.rntuple_col_type_to_num_dict["index32"]
            or 14 <= dtype_byte <= 15
        ):
            res = numpy.insert(res, 0, 0)  # for offsets
        zigzag = 26 <= dtype_byte <= 28
        delta = 14 <= dtype_byte <= 15
        if zigzag:
            res = from_zigzag(res)
        elif delta:
            res = numpy.cumsum(res)
        return res

    def arrays(
        self,
        filter_names="*",
        filter_typenames=None,
        entry_start=0,
        entry_stop=None,
        decompression_executor=None,
        array_cache=None,
    ):
        ak = uproot.extras.awkward()

        entry_stop = entry_stop or self._length

        clusters = self.cluster_summaries
        cluster_starts = numpy.array([c.num_first_entry for c in clusters])

        start_cluster_idx = (
            numpy.searchsorted(cluster_starts, entry_start, side="right") - 1
        )
        stop_cluster_idx = numpy.searchsorted(cluster_starts, entry_stop, side="right")
        cluster_num_entries = numpy.sum(
            [c.num_entries for c in clusters[start_cluster_idx:stop_cluster_idx]]
        )

        self._alias_columns_dict = {
            el.field_id: el.physical_id for el in self.header.alias_columns
        }
        self._column_records_dict = {}
        self._column_records_idx_to_id = {}
        for i, cr in enumerate(self.header.column_records):
            if cr.field_id not in self._column_records_dict:
                self._column_records_dict[cr.field_id] = {
                    "rel_crs": [cr],
                    "rel_crs_idxs": [i],
                }
            else:
                self._column_records_dict[cr.field_id]["rel_crs"].append(cr)
                self._column_records_dict[cr.field_id]["rel_crs_idxs"].append(i)
            self._column_records_idx_to_id[i] = cr.field_id

        self._related_ids = defaultdict(list)
        for i, el in enumerate(self.header.field_records):
            if el.parent_field_id != i:
                self._related_ids[el.parent_field_id].append(i)

        form = self.to_akform().select_columns(filter_names)
        # only read columns mentioned in the awkward form
        target_cols = []
        container_dict = {}
        _recursive_find(form, target_cols)
        for key in target_cols:
            if "column" in key:
                key_nr = int(key.split("-")[1])
                key_fid = self._column_records_idx_to_id[key_nr]
                if key_fid in self._column_records_dict:
                    id = key_fid
                elif key_nr in self._alias_columns_dict:
                    id = self._alias_columns_dict[key_fid]
                else:
                    raise (
                        RuntimeError(
                            f"The key: {key} is missing both from the columns records and the alias columns."
                        )
                    )

                dtype_byte = self._column_records_dict[id]["rel_crs"][0].type
                content = self.read_col_pages(
                    key_nr, range(start_cluster_idx, stop_cluster_idx)
                )
                if dtype_byte == uproot.const.rntuple_col_type_to_num_dict["switch"]:
                    kindex, tags = _split_switch_bits(content)
                    container_dict[f"{key}-index"] = kindex
                    container_dict[f"{key}-tags"] = tags
                else:
                    # don't distinguish data and offsets
                    container_dict[f"{key}-data"] = content
                    container_dict[f"{key}-offsets"] = content
        cluster_offset = cluster_starts[start_cluster_idx]
        entry_start -= cluster_offset
        entry_stop -= cluster_offset
        return ak.from_buffers(form, cluster_num_entries, container_dict)[
            entry_start:entry_stop
        ]


# Supporting function and classes
def _split_switch_bits(content):
    kindex = numpy.bitwise_and(content, numpy.int64(0x00000000000FFFFF))
    tags = (content >> 44).astype("int8") - 1
    return kindex, tags


def _recursive_find(form, res):
    ak = uproot.extras.awkward()

    if hasattr(form, "form_key"):
        res.append(form.form_key)
    if hasattr(form, "contents"):
        for c in form.contents:
            _recursive_find(c, res)
    if hasattr(form, "content") and issubclass(type(form.content), ak.forms.Form):
        _recursive_find(form.content, res)


class PageDescription:
    def read(self, chunk, cursor, context):
        out = MetaData(type(self).__name__)
        out.num_elements = cursor.field(
            chunk, _rntuple_page_num_elements_format, context
        )
        out.locator = LocatorReader().read(chunk, cursor, context)
        return out


class PageLink:
    def __init__(self):
        self.list_cluster_summaries = ListFrameReader(
            RecordFrameReader(ClusterSummaryReader())
        )
        self.nested_page_locations = ListFrameReader(
            ListFrameReader(ListFrameReader(PageDescription()))
        )

    def read(self, chunk, cursor, context):
        out = MetaData(type(self).__name__)
        out.env_header = _envelop_header(chunk, cursor, context)
        out.header_checksum = cursor.field(chunk, _rntuple_checksum_format, context)
        out.cluster_summaries = self.list_cluster_summaries.read(chunk, cursor, context)
        out.pagelinklist = self.nested_page_locations.read(chunk, cursor, context)
        out.checksum = cursor.field(chunk, _rntuple_checksum_format, context)
        return out


class LocatorReader:
    def read(self, chunk, cursor, context):
        out = MetaData("Locator")
        out.num_bytes, out.offset = cursor.fields(
            chunk, _rntuple_locator_format, context
        )
        return out


class EnvLinkReader:
    def read(self, chunk, cursor, context):
        out = MetaData("EnvLink")
        out.env_uncomp_size = cursor.field(chunk, _rntuple_envlink_size_format, context)
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
    def __init__(self, payload):
        self.payload = payload

    def read(self, chunk, cursor, context):
        local_cursor = cursor.copy()
        num_bytes = local_cursor.field(chunk, _rntuple_record_size_format, context)
        assert num_bytes >= 0, f"num_bytes={num_bytes}"
        cursor.skip(num_bytes)
        return self.payload.read(chunk, local_cursor, context)


class ListFrameReader:
    def __init__(self, payload):
        self.payload = payload

    def read(self, chunk, cursor, context):
        local_cursor = cursor.copy()
        num_bytes, num_items = local_cursor.fields(
            chunk, _rntuple_frame_header_format, context
        )
        assert num_bytes < 0, f"num_bytes={num_bytes}"
        cursor.skip(-num_bytes)
        return [
            self.payload.read(chunk, local_cursor, context) for _ in range(num_items)
        ]


# https://github.com/root-project/root/blob/aa513463b0b512517370cb91cca025e53a8b13a2/tree/ntuple/v7/doc/specifications.md#field-description
class FieldRecordReader:
    def read(self, chunk, cursor, context):
        out = MetaData("FieldRecordFrame")
        (
            out.field_version,
            out.type_version,
            out.parent_field_id,
            out.struct_role,
            out.flags,
        ) = cursor.fields(chunk, _rntuple_field_description_format, context)
        if out.flags == 0x0001:
            out.repetition = cursor.field(chunk, _rntuple_repetition_format, context)
        else:
            out.repetition = 0
        out.field_name, out.type_name, out.type_alias, out.field_desc = (
            cursor.rntuple_string(chunk, context) for _ in range(4)
        )
        return out


# https://github.com/root-project/root/blob/aa513463b0b512517370cb91cca025e53a8b13a2/tree/ntuple/v7/doc/specifications.md#column-description
class ColumnRecordReader:
    def read(self, chunk, cursor, context):
        out = MetaData("ColumnRecordFrame")
        out.type, out.nbits, out.field_id, out.flags = cursor.fields(
            chunk, _rntuple_column_record_format, context
        )
        return out


class AliasColumnReader:
    def read(self, chunk, cursor, context):
        out = MetaData("AliasColumn")

        out.physical_id, out.field_id = cursor.fields(
            chunk, _rntuple_alias_column_format, context
        )
        return out


class ExtraTypeInfoReader:
    def read(self, chunk, cursor, context):
        out = MetaData("ExtraTypeInfoReader")

        out.type_ver_from, out.type_ver_to, out.content_id = cursor.fields(
            chunk, _rntuple_extra_type_info_format, context
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
        out = MetaData(type(self).__name__)
        out.env_header = _envelop_header(chunk, cursor, context)
        out.feature_flag = cursor.field(chunk, _rntuple_feature_flag_format, context)
        out.name, out.ntuple_description, out.writer_identifier = (
            cursor.rntuple_string(chunk, context) for _ in range(3)
        )

        out.field_records = self.list_field_record_frames.read(chunk, cursor, context)
        out.column_records = self.list_column_record_frames.read(chunk, cursor, context)
        out.alias_columns = self.list_alias_column_frames.read(chunk, cursor, context)
        out.extra_type_infos = self.list_extra_type_info_reader.read(
            chunk, cursor, context
        )
        out.checksum = cursor.field(chunk, _rntuple_checksum_format, context)

        return out

    def read_extension_header(self, out, chunk, cursor, context):
        out.field_records = self.list_field_record_frames.read(chunk, cursor, context)
        out.column_records = self.list_column_record_frames.read(chunk, cursor, context)
        out.alias_columns = self.list_alias_column_frames.read(chunk, cursor, context)
        out.extra_type_infos = self.list_extra_type_info_reader.read(
            chunk, cursor, context
        )
        return out


class ColumnGroupIDReader:
    def read(self, chunk, cursor, context):
        out = MetaData("ColumnGroupID")
        out.col_id = cursor.field(chunk, _rntuple_column_group_id_format, context)
        return out


class ColumnGroupRecordReader:
    def read(self, chunk, cursor, context):
        out = MetaData("ColumnGroupRecord")
        out.column_ids = ListFrameReader(RecordFrameReader(ColumnGroupIDReader())).read(
            chunk, cursor, context
        )
        return out


class ClusterSummaryReader:
    def read(self, chunk, cursor, context):
        out = MetaData("ClusterSummaryRecord")
        out.num_first_entry, out.num_entries = cursor.fields(
            chunk, _rntuple_cluster_summary_format, context
        )
        return out


class ClusterGroupRecordReader:
    def read(self, chunk, cursor, context):
        out = MetaData("ClusterGroupRecord")
        out.min_entry_num, out.entry_span, out.num_clusters = cursor.fields(
            chunk, _rntuple_cluster_group_format, context
        )
        out.page_list_link = EnvLinkReader().read(chunk, cursor, context)
        return out


class RNTupleSchemaExtension:
    def read(self, chunk, cursor, context):
        out = MetaData(type(self).__name__)
        out.size = cursor.field(chunk, _rntuple_record_size_format, context)
        assert out.size >= 0, f"size={out.size}"
        out.field_records = ListFrameReader(
            RecordFrameReader(FieldRecordReader())
        ).read(chunk, cursor, context)
        out.column_records = ListFrameReader(
            RecordFrameReader(ColumnRecordReader())
        ).read(chunk, cursor, context)
        out.alias_records = ListFrameReader(
            RecordFrameReader(AliasColumnReader())
        ).read(chunk, cursor, context)
        out.extra_type_info = ListFrameReader(
            RecordFrameReader(ExtraTypeInfoReader())
        ).read(chunk, cursor, context)
        return out


class FooterReader:
    def __init__(self):
        self.extension_header_links = RNTupleSchemaExtension()
        self.column_group_record_frames = ListFrameReader(
            RecordFrameReader(ColumnGroupRecordReader())
        )
        self.cluster_summary_frames = ListFrameReader(
            RecordFrameReader(ClusterSummaryReader())
        )
        self.cluster_group_record_frames = ListFrameReader(
            RecordFrameReader(ClusterGroupRecordReader())
        )
        self.meta_data_links = ListFrameReader(RecordFrameReader(EnvLinkReader()))

    def read(self, chunk, cursor, context):
        out = MetaData("Footer")
        out.env_header = _envelop_header(chunk, cursor, context)
        out.feature_flag = cursor.field(chunk, _rntuple_feature_flag_format, context)
        out.header_checksum = cursor.field(chunk, _rntuple_checksum_format, context)
        out.extension_links = self.extension_header_links.read(chunk, cursor, context)
        out.col_group_records = self.column_group_record_frames.read(
            chunk, cursor, context
        )
        out.cluster_group_records = self.cluster_group_record_frames.read(
            chunk, cursor, context
        )
        out.meta_block_links = self.meta_data_links.read(chunk, cursor, context)
        out.checksum = cursor.field(chunk, _rntuple_checksum_format, context)
        return out


uproot.classes["ROOT::Experimental::RNTuple"] = (
    Model_ROOT_3a3a_Experimental_3a3a_RNTuple
)

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
This module defines a versionless model for ``ROOT::RNTuple``.
"""
from __future__ import annotations

import struct
from collections import defaultdict
from itertools import accumulate

import numpy
import xxhash

import uproot
import uproot.const

# https://github.com/root-project/root/blob/8cd9eed6f3a32e55ef1f0f1df8e5462e753c735d/tree/ntuple/v7/doc/BinaryFormatSpecification.md#anchor-schema
_rntuple_anchor_format = struct.Struct(">HHHHQQQQQQQ")
_rntuple_anchor_checksum_format = struct.Struct(">Q")
# https://github.com/root-project/root/blob/8cd9eed6f3a32e55ef1f0f1df8e5462e753c735d/tree/ntuple/v7/doc/BinaryFormatSpecification.md#feature-flags
_rntuple_feature_flag_format = struct.Struct("<Q")
# https://github.com/root-project/root/blob/8cd9eed6f3a32e55ef1f0f1df8e5462e753c735d/tree/ntuple/v7/doc/BinaryFormatSpecification.md#frames
_rntuple_frame_size_format = struct.Struct("<q")
_rntuple_frame_num_items_format = struct.Struct("<I")
# https://github.com/root-project/root/blob/8cd9eed6f3a32e55ef1f0f1df8e5462e753c735d/tree/ntuple/v7/doc/BinaryFormatSpecification.md#locators-and-envelope-links
_rntuple_locator_size_format = struct.Struct("<i")
_rntuple_large_locator_size_format = struct.Struct("<Q")
_rntuple_locator_offset_format = struct.Struct("<Q")
_rntuple_envlink_size_format = struct.Struct("<Q")
# https://github.com/root-project/root/blob/8cd9eed6f3a32e55ef1f0f1df8e5462e753c735d/tree/ntuple/v7/doc/BinaryFormatSpecification.md#envelopes
_rntuple_env_header_format = struct.Struct("<Q")
_rntuple_checksum_format = struct.Struct("<Q")
# https://github.com/root-project/root/blob/8cd9eed6f3a32e55ef1f0f1df8e5462e753c735d/tree/ntuple/v7/doc/BinaryFormatSpecification.md#field-description
_rntuple_field_description_format = struct.Struct("<IIIHH")
_rntuple_repetition_format = struct.Struct("<Q")
_rntuple_source_field_id_format = struct.Struct("<I")
_rntuple_root_streamer_checksum_format = struct.Struct("<I")
# https://github.com/root-project/root/blob/8cd9eed6f3a32e55ef1f0f1df8e5462e753c735d/tree/ntuple/v7/doc/BinaryFormatSpecification.md#column-description
_rntuple_column_record_format = struct.Struct("<HHIHH")
_rntuple_first_element_index_format = struct.Struct("<Q")
_rntuple_column_range_format = struct.Struct("<dd")
# https://github.com/root-project/root/blob/8cd9eed6f3a32e55ef1f0f1df8e5462e753c735d/tree/ntuple/v7/doc/BinaryFormatSpecification.md#alias-columns
_rntuple_alias_column_format = struct.Struct("<II")
# https://github.com/root-project/root/blob/8cd9eed6f3a32e55ef1f0f1df8e5462e753c735d/tree/ntuple/v7/doc/BinaryFormatSpecification.md#extra-type-information
_rntuple_extra_type_info_format = struct.Struct("<II")
# https://github.com/root-project/root/blob/8cd9eed6f3a32e55ef1f0f1df8e5462e753c735d/tree/ntuple/v7/doc/BinaryFormatSpecification.md#cluster-group-record-frame
_rntuple_cluster_group_format = struct.Struct("<QQI")
# https://github.com/root-project/root/blob/8cd9eed6f3a32e55ef1f0f1df8e5462e753c735d/tree/ntuple/v7/doc/BinaryFormatSpecification.md#cluster-summary-record-frame
_rntuple_cluster_summary_format = struct.Struct("<QQ")
# https://github.com/root-project/root/blob/8cd9eed6f3a32e55ef1f0f1df8e5462e753c735d/tree/ntuple/v7/doc/BinaryFormatSpecification.md#page-locations
_rntuple_page_num_elements_format = struct.Struct("<i")
_rntuple_column_element_offset_format = struct.Struct("<q")
_rntuple_column_compression_settings_format = struct.Struct("<I")


def _from_zigzag(n):
    return n >> 1 ^ -(n & 1)


def _envelop_header(chunk, cursor, context):
    env_data = cursor.field(chunk, _rntuple_env_header_format, context)
    env_type_id = env_data & 0xFFFF
    env_length = env_data >> 16
    return {"env_type_id": env_type_id, "env_length": env_length}


def _arrays(
    in_ntuple,
    filter_name="*",
    filter_typename=None,
    entry_start=0,
    entry_stop=None,
    decompression_executor=None,
    array_cache=None,
):
    ak = uproot.extras.awkward()

    entry_stop = entry_stop or in_ntuple.ntuple.num_entries

    clusters = in_ntuple.ntuple.cluster_summaries
    cluster_starts = numpy.array([c.num_first_entry for c in clusters])

    start_cluster_idx = (
        numpy.searchsorted(cluster_starts, entry_start, side="right") - 1
    )
    stop_cluster_idx = numpy.searchsorted(cluster_starts, entry_stop, side="right")
    cluster_num_entries = numpy.sum(
        [c.num_entries for c in clusters[start_cluster_idx:stop_cluster_idx]]
    )

    form = in_ntuple.to_akform().select_columns(
        filter_name, prune_unions_and_records=False
    )
    # only read columns mentioned in the awkward form
    target_cols = []
    container_dict = {}
    _recursive_find(form, target_cols)
    for key in target_cols:
        if "column" in key and "union" not in key:
            key_nr = int(key.split("-")[1])
            dtype_byte = in_ntuple.ntuple.column_records[key_nr].type

            content = in_ntuple.ntuple.read_col_pages(
                key_nr,
                range(start_cluster_idx, stop_cluster_idx),
                dtype_byte=dtype_byte,
                pad_missing_element=True,
            )
            if "cardinality" in key:
                content = numpy.diff(content)
            if dtype_byte == uproot.const.rntuple_col_type_to_num_dict["switch"]:
                kindex, tags = _split_switch_bits(content)
                # Find invalid variants and adjust buffers accordingly
                invalid = numpy.flatnonzero(tags == -1)
                if len(invalid) > 0:
                    kindex = numpy.delete(kindex, invalid)
                    tags = numpy.delete(tags, invalid)
                    invalid -= numpy.arange(len(invalid))
                    optional_index = numpy.insert(
                        numpy.arange(len(kindex), dtype=numpy.int64), invalid, -1
                    )
                else:
                    optional_index = numpy.arange(len(kindex), dtype=numpy.int64)
                container_dict[f"{key}-index"] = optional_index
                container_dict[f"{key}-union-index"] = kindex
                container_dict[f"{key}-union-tags"] = tags
            else:
                # don't distinguish data and offsets
                container_dict[f"{key}-data"] = content
                container_dict[f"{key}-offsets"] = content
    cluster_offset = cluster_starts[start_cluster_idx]
    entry_start -= cluster_offset
    entry_stop -= cluster_offset
    return ak.from_buffers(
        form, cluster_num_entries, container_dict, allow_noncanonical_form=True
    )[entry_start:entry_stop]


def _num_entries_for(in_ntuple, target_num_bytes, filter_name):
    # TODO: part of this is also done in _arrays, so we should refactor this
    # TODO: there might be a better way to estimate the number of entries
    entry_stop = in_ntuple.ntuple.num_entries

    clusters = in_ntuple.ntuple.cluster_summaries
    cluster_starts = numpy.array([c.num_first_entry for c in clusters])

    start_cluster_idx = numpy.searchsorted(cluster_starts, 0, side="right") - 1
    stop_cluster_idx = numpy.searchsorted(cluster_starts, entry_stop, side="right")

    form = in_ntuple.to_akform().select_columns(
        filter_name, prune_unions_and_records=False
    )
    target_cols = []
    _recursive_find(form, target_cols)

    total_bytes = 0
    for key in target_cols:
        if "column" in key and "union" not in key:
            key_nr = int(key.split("-")[1])
            for cluster in range(start_cluster_idx, stop_cluster_idx):
                pages = in_ntuple.ntuple.page_list_envelopes.pagelinklist[cluster][
                    key_nr
                ].pages
                total_bytes += sum(page.locator.num_bytes for page in pages)

    total_entries = entry_stop
    if total_bytes == 0:
        num_entries = 0
    else:
        num_entries = int(round(target_num_bytes * total_entries / total_bytes))
    if num_entries <= 0:
        return 1
    else:
        return num_entries


def _regularize_step_size(in_ntuple, step_size, filter_name):
    if uproot._util.isint(step_size):
        return step_size
    target_num_bytes = uproot._util.memory_size(
        step_size,
        "number of entries or memory size string with units "
        f"(such as '100 MB') required, not {step_size!r}",
    )
    return _num_entries_for(in_ntuple, target_num_bytes, filter_name)


class Model_ROOT_3a3a_RNTuple(uproot.model.Model):
    """
    A versionless :doc:`uproot.model.Model` for ``ROOT::RNTuple``.
    """

    @property
    def _keys(self):
        keys = []
        field_records = self.field_records
        for i, fr in enumerate(field_records):
            if fr.parent_field_id == i and fr.type_name != "":
                keys.append(fr.field_name)
        return keys

    def keys(
        self,
        *,
        filter_name=None,
        filter_typename=None,
        recursive=False,
        full_paths=True,
        # TODO: some arguments might be missing when compared with TTree. Solve when blocker is present in dask/coffea.
    ):
        if filter_name:
            # Return keys from the filter_name list:
            return [key for key in self._keys if key in filter_name]
        else:
            return self._keys

    @property
    def _key_indices(self):
        indices = []
        field_records = self.field_records
        for i, fr in enumerate(field_records):
            if fr.parent_field_id == i and fr.type_name != "":
                indices.append(i)
        return indices

    @property
    def _key_to_index(self):
        d = {}
        field_records = self.field_records
        for i, fr in enumerate(field_records):
            if fr.parent_field_id == i and fr.type_name != "":
                d[fr.field_name] = i
        return d

    def read_members(self, chunk, cursor, context, file):
        if uproot._awkwardforth.get_forth_obj(context) is not None:
            raise uproot.interpretation.objects.CannotBeForth()
        if self.is_memberwise:
            raise NotImplementedError(
                f"""memberwise serialization of {type(self).__name__}
in file {self.file.file_path}"""
            )

        (
            self._members["fVersionEpoch"],
            self._members["fVersionMajor"],
            self._members["fVersionMinor"],
            self._members["fVersionPatch"],
            self._members["fSeekHeader"],
            self._members["fNBytesHeader"],
            self._members["fLenHeader"],
            self._members["fSeekFooter"],
            self._members["fNBytesFooter"],
            self._members["fLenFooter"],
            self._members["fMaxKeySize"],
        ) = cursor.fields(chunk, _rntuple_anchor_format, context)

        self._anchor_checksum = cursor.field(
            chunk, _rntuple_anchor_checksum_format, context
        )
        assert self._anchor_checksum == xxhash.xxh3_64_intdigest(
            chunk.raw_data[
                -_rntuple_anchor_format.size
                - _rntuple_anchor_checksum_format.size : -_rntuple_anchor_checksum_format.size
            ]
        )
        cursor.skip(-_rntuple_anchor_checksum_format.size)

        self._header_chunk_ready = False
        self._footer_chunk_ready = False
        self._header, self._footer = None, None

        self._field_records = None
        self._field_names = None
        self._column_records = None
        self._alias_column_records = None
        self._alias_columns_dict_ = None
        self._related_ids_ = None
        self._column_records_dict_ = None
        self._num_entries = None
        self._length = None

        self._page_list_envelopes = []

        self.ntuple = self

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
            assert h.checksum == xxhash.xxh3_64_intdigest(
                self._header_chunk.raw_data[: -_rntuple_checksum_format.size]
            )

        return self._header

    @property
    def field_records(self):
        if self._field_records is None:
            self._field_records = list(self.header.field_records)
            self._field_records.extend(self.footer.extension_links.field_records)
        return self._field_records

    @property
    def field_names(self):
        if self._field_names is None:
            self._field_names = [r.field_name for r in self.field_records]
        return self._field_names

    @property
    def column_records(self):
        if self._column_records is None:
            self._column_records = list(self.header.column_records)
            self._column_records.extend(self.footer.extension_links.column_records)
            for i, cr in enumerate(self._column_records):
                cr.idx = i
        return self._column_records

    @property
    def alias_column_records(self):
        if self._alias_column_records is None:
            self._alias_column_records = list(self.header.alias_column_records)
            self._alias_column_records.extend(
                self.footer.extension_links.alias_column_records
            )
        return self._alias_column_records

    @property
    def _alias_columns_dict(self):
        if self._alias_columns_dict_ is None:
            self._alias_columns_dict_ = {
                el.field_id: el.physical_id for el in self.alias_column_records
            }
        return self._alias_columns_dict_

    @property
    def _column_records_dict(self):
        if self._column_records_dict_ is None:
            self._column_records_dict_ = {}
            for cr in self.column_records:
                if cr.field_id not in self._column_records_dict_:
                    self._column_records_dict_[cr.field_id] = [cr]
                else:
                    self._column_records_dict_[cr.field_id].append(cr)
        return self._column_records_dict_

    @property
    def _related_ids(self):
        if self._related_ids_ is None:
            self._related_ids_ = defaultdict(list)
            for i, el in enumerate(self.field_records):
                if el.parent_field_id != i:
                    self._related_ids_[el.parent_field_id].append(i)
        return self._related_ids_

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
            assert f.checksum == xxhash.xxh3_64_intdigest(
                self._footer_chunk.raw_data[: -_rntuple_checksum_format.size]
            )

        return self._footer

    @property
    def cluster_summaries(self):
        return self.page_list_envelopes.cluster_summaries

    @property
    def num_entries(self):
        if self._num_entries is None:
            self._num_entries = sum(x.num_entries for x in self.cluster_summaries)
        return self._num_entries

    def __len__(self):
        if self._length is None:
            self._length = len(self.keys())
        return self._length

    def __repr__(self):
        if len(self) == 0:
            return f"<RNTuple {self.name!r} at 0x{id(self):012x}>"
        else:
            return (
                f"<RNTuple {self.name!r} ({len(self)} top fields) at 0x{id(self):012x}>"
            )

    def __getitem__(self, where):
        # original_where = where

        if uproot._util.isint(where):
            index = self._key_indices[where]
        elif isinstance(where, str):
            where = uproot._util.ensure_str(where)
            index = self._key_to_index[where]
        else:
            raise TypeError(f"where must be an integer or a string, not {where!r}")

        # TODO: Implement path support

        return RNTupleField(index, self)

    @property
    def name(self):
        """
        Name of the ``RNTuple``.
        """
        return self.parent.fName

    @property
    def object_path(self):
        """
        Object path of the ``RNTuple``.
        """
        return self.parent.object_path

    @property
    def cache_key(self):
        """
        String that uniquely specifies this ``RNTuple`` in its path, to use as
        part of object and array cache keys.
        """
        return f"{self.parent.cache_key}{self.name};{self.parent.fCycle}"

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

    def base_col_form(self, cr, col_id, parameters=None, cardinality=False):
        ak = uproot.extras.awkward()

        form_key = f"column-{col_id}" + ("-cardinality" if cardinality else "")
        dtype_byte = cr.type
        if dtype_byte == uproot.const.rntuple_col_type_to_num_dict["switch"]:
            return form_key
        elif dtype_byte in uproot.const.rntuple_index_types and not cardinality:
            return form_key
        dt_str = uproot.const.rntuple_col_num_to_dtype_dict[dtype_byte]
        if dt_str == "bit":
            dt_str = "bool"
        elif dtype_byte in uproot.const.rntuple_custom_float_types:
            dt_str = "float32"
        return ak.forms.NumpyForm(
            dt_str,
            form_key=form_key,
            parameters=parameters,
        )

    def col_form(self, field_id):
        ak = uproot.extras.awkward()

        cfid = field_id
        if self.field_records[cfid].source_field_id is not None:
            cfid = self.field_records[cfid].source_field_id
        if cfid in self._alias_columns_dict:
            cfid = self._alias_columns_dict[cfid]
        if cfid not in self._column_records_dict:
            raise (
                RuntimeError(
                    f"The field_id: {cfid} is missing from the columns records."
                )
            )

        rel_crs = self._column_records_dict[cfid]

        if len(rel_crs) == 1:  # base case
            cardinality = "RNTupleCardinality" in self.field_records[field_id].type_name
            return self.base_col_form(
                rel_crs[0], rel_crs[0].idx, cardinality=cardinality
            )
        elif (
            len(rel_crs) == 2
            and rel_crs[1].type == uproot.const.rntuple_col_type_to_num_dict["char"]
        ):
            # string field splits->2 in col records
            inner = self.base_col_form(
                rel_crs[1], rel_crs[1].idx, parameters={"__array__": "char"}
            )
            form_key = f"column-{rel_crs[0].idx}"
            return ak.forms.ListOffsetForm(
                "i64", inner, form_key=form_key, parameters={"__array__": "string"}
            )
        else:
            raise (RuntimeError(f"Missing special case: {field_id}"))

    def field_form(self, this_id, seen):
        ak = uproot.extras.awkward()

        field_records = self.field_records
        this_record = field_records[this_id]
        seen.add(this_id)
        structural_role = this_record.struct_role
        if (
            structural_role == uproot.const.RNTupleFieldRole.LEAF
            and this_record.repetition == 0
        ):
            # deal with std::atomic
            # they have no associated column, but exactly one subfield containing the underlying data
            tmp_id = self._alias_columns_dict.get(this_id, this_id)
            if (
                tmp_id not in self._column_records_dict
                and len(self._related_ids[tmp_id]) == 1
            ):
                this_id = self._related_ids[tmp_id][0]
                seen.add(this_id)
            # base case of recursion
            # n.b. the split may happen in column
            return self.col_form(this_id)
        elif structural_role == uproot.const.RNTupleFieldRole.LEAF:
            if this_id in self._related_ids:
                # std::array has only one subfield
                child_id = self._related_ids[this_id][0]
                inner = self.field_form(child_id, seen)
            else:
                # std::bitset has no subfields, so we use it directly
                inner = self.col_form(this_id)
            keyname = f"RegularForm-{this_id}"
            return ak.forms.RegularForm(inner, this_record.repetition, form_key=keyname)
        elif structural_role == uproot.const.RNTupleFieldRole.COLLECTION:
            if this_id not in self._related_ids or len(self._related_ids[this_id]) != 1:
                keyname = f"vector-{this_id}"
                newids = self._related_ids.get(this_id, [])
                # go find N in the rest, N is the # of fields in vector
                recordlist = [self.field_form(i, seen) for i in newids]
                namelist = [field_records[i].field_name for i in newids]
                return ak.forms.RecordForm(recordlist, namelist, form_key="whatever")
            cfid = this_id
            if self.field_records[cfid].source_field_id is not None:
                cfid = self.field_records[cfid].source_field_id
            if cfid in self._alias_columns_dict:
                cfid = self._alias_columns_dict[cfid]
            if cfid not in self._column_records_dict:
                raise (
                    RuntimeError(
                        f"The field_id: {cfid} is missing from the columns records."
                    )
                )
            col_id = self._column_records_dict[cfid][0].idx
            keyname = f"column-{col_id}"
            #  this only has one child
            if this_id in self._related_ids:
                child_id = self._related_ids[this_id][0]
            inner = self.field_form(child_id, seen)
            return ak.forms.ListOffsetForm("i64", inner, form_key=keyname)
        elif structural_role == uproot.const.RNTupleFieldRole.RECORD:
            newids = []
            if this_id in self._related_ids:
                newids = self._related_ids[this_id]
            # go find N in the rest, N is the # of fields in struct
            recordlist = [self.field_form(i, seen) for i in newids]
            namelist = [field_records[i].field_name for i in newids]
            return ak.forms.RecordForm(recordlist, namelist, form_key="whatever")
        elif structural_role == uproot.const.RNTupleFieldRole.VARIANT:
            keyname = self.col_form(this_id)
            newids = []
            if this_id in self._related_ids:
                newids = self._related_ids[this_id]
            recordlist = [self.field_form(i, seen) for i in newids]
            inner = ak.forms.UnionForm(
                "i8", "i64", recordlist, form_key=keyname + "-union"
            )
            return ak.forms.IndexedOptionForm("i64", inner, form_key=keyname)
        elif structural_role == uproot.const.RNTupleFieldRole.STREAMER:
            raise NotImplementedError(
                f"Unsplit fields are not supported. {this_record}"
            )
        else:
            # everything should recurse above this branch
            raise AssertionError("this should be unreachable")

    def to_akform(self):
        ak = uproot.extras.awkward()

        field_records = self.field_records
        recordlist = []
        topnames = self.keys()
        seen = set()
        for i in range(len(field_records)):
            if i not in seen:
                ff = self.field_form(i, seen)
                if field_records[i].type_name != "":
                    recordlist.append(ff)

        form = ak.forms.RecordForm(recordlist, topnames, form_key="toplevel")
        return form

    def read_pagedesc(self, destination, desc, dtype_str, dtype, nbits, split):
        loc = desc.locator
        context = {}
        # bool in RNTuple is always stored as bits
        isbit = dtype_str == "bit"
        num_elements = len(destination)
        if isbit:
            num_elements_toread = int(numpy.ceil(num_elements / 8))
        elif dtype_str in ("real32trunc", "real32quant"):
            num_elements_toread = int(numpy.ceil((num_elements * 4 * nbits) / 32))
            dtype = numpy.dtype("uint8")
        else:
            num_elements_toread = num_elements
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

            elif nbits == 32:
                # AAAAABBBBBCCCCCDDDDD needs to become
                # ABCDABCDABCDABCDABCD
                res = numpy.empty(len(content), numpy.uint8)
                res[0::4] = content[len(res) * 0 // 4 : len(res) * 1 // 4]
                res[1::4] = content[len(res) * 1 // 4 : len(res) * 2 // 4]
                res[2::4] = content[len(res) * 2 // 4 : len(res) * 3 // 4]
                res[3::4] = content[len(res) * 3 // 4 : len(res) * 4 // 4]

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

            content = res.view(dtype)

        if isbit:
            content = (
                numpy.unpackbits(content.view(dtype=numpy.uint8))
                .reshape(-1, 8)[:, ::-1]
                .reshape(-1)
            )
        elif dtype_str in ("real32trunc", "real32quant"):
            if nbits == 32:
                content = content.view(numpy.uint32)
            elif nbits % 8 == 0:
                new_content = numpy.empty(num_elements, numpy.uint32)
                nbytes = nbits // 8
                new_content[:] = content[nbytes - 1 : num_elements * nbytes : nbytes]
                for i in range(1, nbytes):
                    new_content <<= 8
                    new_content += content[
                        nbytes - 1 - i : num_elements * nbytes : nbytes
                    ]
                content = new_content
            else:
                ak = uproot.extras.awkward()
                vm = ak.forth.ForthMachine32(
                    f"""input x output y uint32 {num_elements} x #{nbits}bit-> y"""
                )
                vm.run({"x": content})
                content = vm["y"]
            if dtype_str == "real32trunc":
                content <<= 32 - nbits

        # needed to chop off extra bits incase we used `unpackbits`
        destination[:] = content[:num_elements]

    def read_col_pages(
        self, ncol, cluster_range, dtype_byte, pad_missing_element=False
    ):
        arrays = [self.read_col_page(ncol, i) for i in cluster_range]

        # Check if column stores offset values for jagged arrays (splitindex64) (applies to cardinality cols too):
        if dtype_byte in uproot.const.rntuple_delta_types:
            # Extract the last offset values:
            last_elements = [
                arr[-1] for arr in arrays[:-1]
            ]  # First value always zero, therefore skip first arr.
            # Compute cumulative sum using itertools.accumulate:
            last_offsets = list(accumulate(last_elements))
            # Add the offsets to each array
            for i in range(1, len(arrays)):
                arrays[i] += last_offsets[i - 1]
            # Remove the first element from every sub-array except for the first one:
            arrays = [arrays[0]] + [arr[1:] for arr in arrays[1:]]

        res = numpy.concatenate(arrays, axis=0)

        if pad_missing_element:
            first_element_index = self.column_records[ncol].first_element_index
            res = numpy.pad(res, (first_element_index, 0))
        return res

    def read_col_page(self, ncol, cluster_i):
        linklist = self.page_list_envelopes.pagelinklist[cluster_i]
        pagelist = linklist[ncol].pages if ncol < len(linklist) else []
        dtype_byte = self.column_records[ncol].type
        dtype_str = uproot.const.rntuple_col_num_to_dtype_dict[dtype_byte]
        total_len = numpy.sum([desc.num_elements for desc in pagelist], dtype=int)
        if dtype_str == "switch":
            dtype = numpy.dtype([("index", "int64"), ("tag", "int32")])
        elif dtype_str == "bit":
            dtype = numpy.dtype("bool")
        elif dtype_byte in uproot.const.rntuple_custom_float_types:
            dtype = numpy.dtype("uint32")  # for easier bit manipulation
        else:
            dtype = numpy.dtype(dtype_str)
        res = numpy.empty(total_len, dtype)
        split = dtype_byte in uproot.const.rntuple_split_types
        zigzag = dtype_byte in uproot.const.rntuple_zigzag_types
        delta = dtype_byte in uproot.const.rntuple_delta_types
        index = dtype_byte in uproot.const.rntuple_index_types
        nbits = (
            self.column_records[ncol].nbits
            if ncol < len(self.column_records)
            else uproot.const.rntuple_col_num_to_size_dict[dtype_byte]
        )
        tracker = 0
        cumsum = 0
        for page_desc in pagelist:
            n_elements = page_desc.num_elements
            tracker_end = tracker + n_elements
            self.read_pagedesc(
                res[tracker:tracker_end], page_desc, dtype_str, dtype, nbits, split
            )
            if delta:
                res[tracker] -= cumsum
                cumsum += numpy.sum(res[tracker:tracker_end])
            tracker = tracker_end

        if index:
            res = numpy.insert(res, 0, 0)  # for offsets
        if zigzag:
            res = _from_zigzag(res)
        elif delta:
            res = numpy.cumsum(res)
        elif dtype_str == "real32trunc":
            res = res.view(numpy.float32)
        elif dtype_str == "real32quant" and ncol < len(self.column_records):
            min_value = self.column_records[ncol].min_value
            max_value = self.column_records[ncol].max_value
            res = min_value + res.astype(numpy.float32) * (max_value - min_value) / (
                (1 << nbits) - 1
            )
        return res

    def arrays(
        self,
        filter_name="*",
        filter_typename=None,
        entry_start=0,
        entry_stop=None,
        decompression_executor=None,
        array_cache=None,
    ):
        return _arrays(
            self,
            filter_name=filter_name,
            filter_typename=filter_typename,
            entry_start=entry_start,
            entry_stop=entry_stop,
            decompression_executor=decompression_executor,
            array_cache=array_cache,
        )

    def iterate(self, filter_name="*", *args, step_size="100 MB", **kwargs):
        step_size = _regularize_step_size(self, step_size, filter_name)
        for start in range(0, self.num_entries, step_size):
            yield self.arrays(
                *args, entry_start=start, entry_stop=start + step_size, **kwargs
            )


# Supporting function and classes
def _split_switch_bits(content):
    tags = content["tag"].astype(numpy.dtype("int8")) - 1
    kindex = content["index"]
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


# https://github.com/root-project/root/blob/8cd9eed6f3a32e55ef1f0f1df8e5462e753c735d/tree/ntuple/v7/doc/BinaryFormatSpecification.md#page-locations
class PageDescription:
    def read(self, chunk, cursor, context):
        out = MetaData(type(self).__name__)
        num_elements = cursor.field(chunk, _rntuple_page_num_elements_format, context)
        out.has_checksum = num_elements < 0
        out.num_elements = abs(num_elements)
        out.locator = LocatorReader().read(chunk, cursor, context)
        return out


# https://github.com/root-project/root/blob/8cd9eed6f3a32e55ef1f0f1df8e5462e753c735d/tree/ntuple/v7/doc/BinaryFormatSpecification.md#page-locations
class ColumnPageListFrameReader:
    def read(self, chunk, cursor, context):
        local_cursor = cursor.copy()
        num_bytes = local_cursor.field(chunk, _rntuple_frame_size_format, context)
        assert num_bytes < 0, f"num_bytes={num_bytes}"
        num_items = local_cursor.field(chunk, _rntuple_frame_num_items_format, context)
        cursor.skip(-num_bytes)
        out = MetaData("ColumnPages")
        out.pages = [
            PageDescription().read(chunk, local_cursor, context)
            for _ in range(num_items)
        ]
        out.element_offset = local_cursor.field(
            chunk, _rntuple_column_element_offset_format, context
        )
        out.suppressed = out.element_offset < 0
        if not out.suppressed:
            out.compression_settings = local_cursor.field(
                chunk, _rntuple_column_compression_settings_format, context
            )
        else:
            out.compression_settings = None
        return out


# https://github.com/root-project/root/blob/8cd9eed6f3a32e55ef1f0f1df8e5462e753c735d/tree/ntuple/v7/doc/BinaryFormatSpecification.md#page-list-envelope
class PageLink:
    def __init__(self):
        self.list_cluster_summaries = ListFrameReader(
            RecordFrameReader(ClusterSummaryReader())
        )
        self.nested_page_locations = ListFrameReader(
            ListFrameReader(ColumnPageListFrameReader())
        )

    def read(self, chunk, cursor, context):
        out = MetaData(type(self).__name__)
        out.env_header = _envelop_header(chunk, cursor, context)
        assert (
            out.env_header["env_type_id"] == uproot.const.RNTupleEnvelopeType.PAGELIST
        ), f"env_type_id={out.env_header['env_type_id']}"
        out.header_checksum = cursor.field(chunk, _rntuple_checksum_format, context)
        out.cluster_summaries = self.list_cluster_summaries.read(chunk, cursor, context)
        out.pagelinklist = self.nested_page_locations.read(chunk, cursor, context)
        out.checksum = cursor.field(chunk, _rntuple_checksum_format, context)
        return out


# https://github.com/root-project/root/blob/8cd9eed6f3a32e55ef1f0f1df8e5462e753c735d/tree/ntuple/v7/doc/BinaryFormatSpecification.md#locators-and-envelope-links
class LocatorReader:
    def read(self, chunk, cursor, context):
        out = MetaData("Locator")
        out.num_bytes = cursor.field(chunk, _rntuple_locator_size_format, context)
        if out.num_bytes < 0:
            out.type = -out.num_bytes >> 24
            if out.type == uproot.const.RNTupleLocatorType.LARGE:
                out.num_bytes = cursor.field(
                    chunk, _rntuple_large_locator_size_format, context
                )
                out.offset = cursor.field(
                    chunk, _rntuple_locator_offset_format, context
                )
            else:
                raise NotImplementedError(f"Unknown locator type: {out.type}")
        else:
            out.type = uproot.const.RNTupleLocatorType.STANDARD
            out.offset = cursor.field(chunk, _rntuple_locator_offset_format, context)
        return out


# https://github.com/root-project/root/blob/8cd9eed6f3a32e55ef1f0f1df8e5462e753c735d/tree/ntuple/v7/doc/BinaryFormatSpecification.md#locators-and-envelope-links
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


# https://github.com/root-project/root/blob/8cd9eed6f3a32e55ef1f0f1df8e5462e753c735d/tree/ntuple/v7/doc/BinaryFormatSpecification.md#frames
class RecordFrameReader:
    def __init__(self, payload):
        self.payload = payload

    def read(self, chunk, cursor, context):
        local_cursor = cursor.copy()
        num_bytes = local_cursor.field(chunk, _rntuple_frame_size_format, context)
        assert num_bytes >= 0, f"num_bytes={num_bytes}"
        cursor.skip(num_bytes)
        return self.payload.read(chunk, local_cursor, context)


# https://github.com/root-project/root/blob/8cd9eed6f3a32e55ef1f0f1df8e5462e753c735d/tree/ntuple/v7/doc/BinaryFormatSpecification.md#frames
class ListFrameReader:
    def __init__(self, payload):
        self.payload = payload

    def read(self, chunk, cursor, context):
        local_cursor = cursor.copy()
        num_bytes = local_cursor.field(chunk, _rntuple_frame_size_format, context)
        assert num_bytes < 0, f"num_bytes={num_bytes}"
        num_items = local_cursor.field(chunk, _rntuple_frame_num_items_format, context)
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
        out.field_name, out.type_name, out.type_alias, out.field_desc = (
            cursor.rntuple_string(chunk, context) for _ in range(4)
        )

        if out.flags & uproot.const.RNTupleFieldFlag.REPETITIVE:
            out.repetition = cursor.field(chunk, _rntuple_repetition_format, context)
        else:
            out.repetition = 0

        if out.flags & uproot.const.RNTupleFieldFlag.PROJECTED:
            out.source_field_id = cursor.field(
                chunk, _rntuple_source_field_id_format, context
            )
        else:
            out.source_field_id = None

        if out.flags & uproot.const.RNTupleFieldFlag.CHECKSUM:
            out.checksum = cursor.field(
                chunk, _rntuple_root_streamer_checksum_format, context
            )
        else:
            out.checksum = None

        return out


# https://github.com/root-project/root/blob/aa513463b0b512517370cb91cca025e53a8b13a2/tree/ntuple/v7/doc/specifications.md#column-description
class ColumnRecordReader:
    def read(self, chunk, cursor, context):
        out = MetaData("ColumnRecordFrame")
        out.type, out.nbits, out.field_id, out.flags, out.repr_idx = cursor.fields(
            chunk, _rntuple_column_record_format, context
        )
        if out.flags & uproot.const.RNTupleColumnFlag.DEFERRED:
            out.first_element_index = cursor.field(
                chunk, _rntuple_first_element_index_format, context
            )
        else:
            out.first_element_index = 0
        if out.flags & uproot.const.RNTupleColumnFlag.RANGE:
            out.min_value, out.max_value = cursor.fields(
                chunk, _rntuple_column_range_format, context
            )
        else:
            out.min_value, out.max_value = None, None
        return out


# https://github.com/root-project/root/blob/8cd9eed6f3a32e55ef1f0f1df8e5462e753c735d/tree/ntuple/v7/doc/BinaryFormatSpecification.md#alias-columns
class AliasColumnReader:
    def read(self, chunk, cursor, context):
        out = MetaData("AliasColumn")

        out.physical_id, out.field_id = cursor.fields(
            chunk, _rntuple_alias_column_format, context
        )
        return out


# https://github.com/root-project/root/blob/8cd9eed6f3a32e55ef1f0f1df8e5462e753c735d/tree/ntuple/v7/doc/BinaryFormatSpecification.md#extra-type-information
class ExtraTypeInfoReader:
    def read(self, chunk, cursor, context):
        out = MetaData("ExtraTypeInfoReader")

        out.content_id, out.type_ver = cursor.fields(
            chunk, _rntuple_extra_type_info_format, context
        )
        out.type_name = cursor.rntuple_string(chunk, context)
        return out


# https://github.com/root-project/root/blob/8cd9eed6f3a32e55ef1f0f1df8e5462e753c735d/tree/ntuple/v7/doc/BinaryFormatSpecification.md#header-envelope
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
        assert (
            out.env_header["env_type_id"] == uproot.const.RNTupleEnvelopeType.HEADER
        ), f"env_type_id={out.env_header['env_type_id']}"
        out.feature_flag = cursor.field(chunk, _rntuple_feature_flag_format, context)
        out.name, out.ntuple_description, out.writer_identifier = (
            cursor.rntuple_string(chunk, context) for _ in range(3)
        )

        out.field_records = self.list_field_record_frames.read(chunk, cursor, context)
        out.column_records = self.list_column_record_frames.read(chunk, cursor, context)
        out.alias_column_records = self.list_alias_column_frames.read(
            chunk, cursor, context
        )
        out.extra_type_infos = self.list_extra_type_info_reader.read(
            chunk, cursor, context
        )
        out.checksum = cursor.field(chunk, _rntuple_checksum_format, context)

        return out


# https://github.com/root-project/root/blob/8cd9eed6f3a32e55ef1f0f1df8e5462e753c735d/tree/ntuple/v7/doc/BinaryFormatSpecification.md#cluster-summary-record-frame
class ClusterSummaryReader:
    def read(self, chunk, cursor, context):
        out = MetaData("ClusterSummaryRecord")
        out.num_first_entry, out.num_entries = cursor.fields(
            chunk, _rntuple_cluster_summary_format, context
        )
        out.flags = out.num_entries >> 56
        out.num_entries &= 0xFFFFFFFFFFFFFF
        if out.flags & uproot.const.RNTupleClusterFlag.SHARDED:
            raise NotImplementedError("Sharded clusters are not supported.")
        return out


# https://github.com/root-project/root/blob/8cd9eed6f3a32e55ef1f0f1df8e5462e753c735d/tree/ntuple/v7/doc/BinaryFormatSpecification.md#page-locations
class ClusterGroupRecordReader:
    def read(self, chunk, cursor, context):
        out = MetaData("ClusterGroupRecord")
        out.min_entry_num, out.entry_span, out.num_clusters = cursor.fields(
            chunk, _rntuple_cluster_group_format, context
        )
        out.page_list_link = EnvLinkReader().read(chunk, cursor, context)
        return out


# https://github.com/root-project/root/blob/8cd9eed6f3a32e55ef1f0f1df8e5462e753c735d/tree/ntuple/v7/doc/BinaryFormatSpecification.md#schema-extension-record-frame
class RNTupleSchemaExtension:
    def read(self, chunk, cursor, context):
        out = MetaData(type(self).__name__)
        out.size = cursor.field(chunk, _rntuple_frame_size_format, context)
        assert out.size >= 0, f"size={out.size}"
        out.field_records = ListFrameReader(
            RecordFrameReader(FieldRecordReader())
        ).read(chunk, cursor, context)
        out.column_records = ListFrameReader(
            RecordFrameReader(ColumnRecordReader())
        ).read(chunk, cursor, context)
        out.alias_column_records = ListFrameReader(
            RecordFrameReader(AliasColumnReader())
        ).read(chunk, cursor, context)
        out.extra_type_info = ListFrameReader(
            RecordFrameReader(ExtraTypeInfoReader())
        ).read(chunk, cursor, context)
        return out


# https://github.com/root-project/root/blob/8cd9eed6f3a32e55ef1f0f1df8e5462e753c735d/tree/ntuple/v7/doc/BinaryFormatSpecification.md#footer-envelope
class FooterReader:
    def __init__(self):
        self.extension_header_links = RNTupleSchemaExtension()
        self.cluster_summary_frames = ListFrameReader(
            RecordFrameReader(ClusterSummaryReader())
        )
        self.cluster_group_record_frames = ListFrameReader(
            RecordFrameReader(ClusterGroupRecordReader())
        )

    def read(self, chunk, cursor, context):
        out = MetaData("Footer")
        out.env_header = _envelop_header(chunk, cursor, context)
        assert (
            out.env_header["env_type_id"] == uproot.const.RNTupleEnvelopeType.FOOTER
        ), f"env_type_id={out.env_header['env_type_id']}"
        out.feature_flag = cursor.field(chunk, _rntuple_feature_flag_format, context)
        out.header_checksum = cursor.field(chunk, _rntuple_checksum_format, context)
        out.extension_links = self.extension_header_links.read(chunk, cursor, context)
        out.cluster_group_records = self.cluster_group_record_frames.read(
            chunk, cursor, context
        )
        out.checksum = cursor.field(chunk, _rntuple_checksum_format, context)
        return out


class RNTupleField:
    def __init__(self, index, ntuple):
        self.index = index
        self.ntuple = ntuple
        self._length = None

    @property
    def _keys(self):
        keys = []
        for i, fr in enumerate(self.ntuple.field_records):
            if i == self.index:
                continue
            if (
                fr.parent_field_id == self.index
                and fr.type_name != ""
                and not fr.field_name.startswith("_")
                and not fr.field_name.startswith(":_")
            ):
                keys.append(fr.field_name)
        return keys

    def keys(self):
        return self._keys

    @property
    def name(self):
        """
        Name of the ``Field``.
        """
        return self.ntuple.field_records[self.index].field_name

    def __len__(self):
        if self._length is None:
            self._length = len(self.keys())
        return self._length

    def __repr__(self):
        if len(self) == 0:
            return f"<Field {self.name!r} in RNTuple {self.ntuple.name!r} at 0x{id(self):012x}>"
        else:
            return f"<Field {self.name!r} ({len(self)} subfields) in RNTuple {self.ntuple.name!r} at 0x{id(self):012x}>"

    @property
    def _key_indices(self):
        indices = []
        field_records = self.ntuple.field_records
        for i, fr in enumerate(field_records):
            if fr.parent_field_id == self.index and fr.type_name != "":
                indices.append(i)
        return indices

    @property
    def _key_to_index(self):
        d = {}
        field_records = self.ntuple.field_records
        for i, fr in enumerate(field_records):
            if fr.parent_field_id == self.index and fr.type_name != "":
                d[fr.field_name] = i
        return d

    def __getitem__(self, where):
        # original_where = where

        if uproot._util.isint(where):
            index = self._key_indices[where]
        elif isinstance(where, str):
            where = uproot._util.ensure_str(where)
            index = self._key_to_index[where]
        else:
            raise TypeError(f"where must be an integer or a string, not {where!r}")

        # TODO: Implement path support

        return RNTupleField(index, self.ntuple)

    def to_akform(self):
        ak = uproot.extras.awkward()

        field_records = self.ntuple.field_records
        recordlist = []
        topnames = self.keys()
        if len(topnames) == 0:
            topnames = [self.name]
            recordlist.append(self.ntuple.field_form(self.index, set()))
        else:
            seen = set()
            for i in range(len(field_records)):
                if (
                    i not in seen
                    and field_records[i].parent_field_id == self.index
                    and i != self.index
                    and not field_records[i].field_name.startswith("_")
                    and not field_records[i].field_name.startswith(":_")
                ):
                    ff = self.ntuple.field_form(i, seen)
                    if field_records[i].type_name != "":
                        recordlist.append(ff)

        form = ak.forms.RecordForm(recordlist, topnames, form_key="toplevel")
        return form

    def arrays(
        self,
        filter_name="*",
        filter_typename=None,
        entry_start=0,
        entry_stop=None,
        decompression_executor=None,
        array_cache=None,
    ):
        return _arrays(
            self,
            filter_name=filter_name,
            filter_typename=filter_typename,
            entry_start=entry_start,
            entry_stop=entry_stop,
            decompression_executor=decompression_executor,
            array_cache=array_cache,
        )

    def array(self, **kwargs):
        if len(self.keys()) == 0:
            return self.arrays(**kwargs)[self.name]
        return self.arrays(**kwargs)

    def __array__(self, *args, **kwargs):
        out = self.array()
        if args == () and kwargs == {}:
            return out
        else:
            return numpy.array(out, *args, **kwargs)

    def iterate(self, filter_name="*", *args, step_size="100 MB", **kwargs):
        step_size = _regularize_step_size(self, step_size, filter_name)
        for start in range(0, self.ntuple.num_entries, step_size):
            yield self.array(
                *args, entry_start=start, entry_stop=start + step_size, **kwargs
            )


uproot.classes["ROOT::RNTuple"] = Model_ROOT_3a3a_RNTuple

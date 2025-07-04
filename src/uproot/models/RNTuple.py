# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
This module defines a versionless model for ``ROOT::RNTuple``.
"""
from __future__ import annotations

import dataclasses
import struct
import sys
from collections import defaultdict

import numpy
import xxhash

import uproot
import uproot.behaviors.RNTuple
import uproot.const
from uproot.source.cufile_interface import CuFileSource

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

# https://github.com/root-project/root/blob/6dc4ff848329eaa3ca433985e709b12321098fe2/core/zip/inc/Compression.h#L93-L105
compression_settings_dict = {
    uproot.const.kZLIB: "ZLIB",  # nvCOMP unsupported
    uproot.const.kLZMA: "LZMA",  # nvCOMP unsupported
    uproot.const.kOldCompressionAlgo: "deflate",  # nvCOMP support
    uproot.const.kLZ4: "LZ4",  # nvCOMP support
    uproot.const.kZSTD: "zstd",  # nvCOMP support
}


def _from_zigzag(n):
    return n >> 1 ^ -(n & 1)


def _envelop_header(chunk, cursor, context):
    env_data = cursor.field(chunk, _rntuple_env_header_format, context)
    env_type_id = uproot.const.RNTupleEnvelopeType(env_data & 0xFFFF)
    env_length = env_data >> 16
    return {"env_type_id": env_type_id, "env_length": env_length}


class Model_ROOT_3a3a_RNTuple(uproot.behaviors.RNTuple.RNTuple, uproot.model.Model):
    """
    A versionless :doc:`uproot.model.Model` for ``ROOT::RNTuple``.
    """

    def read_members(self, chunk, cursor, context, file):
        if uproot._awkwardforth.get_forth_obj(context) is not None:
            raise uproot.interpretation.objects.CannotBeForth()
        if self.is_memberwise:
            raise NotImplementedError(
                f"""memberwise serialization of {type(self).__name__}
in file {self.file.file_path}"""
            )
        # Probably no one will encounter this, but just in case something doesn't work correctly
        if sys.byteorder != "little":
            raise NotImplementedError(
                "RNTuple reading is only supported on little-endian systems"
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
        ), "Anchor checksum does not match! File is corrupted or incompatible."
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
        self._cluster_summaries = None
        self._page_link_list = None

        self._ntuple = self
        self._fields = None
        self._all_fields = None
        self._lookup = None

    @property
    def all_fields(self):
        """
        The full list of fields in the RNTuple.

        The fields are sorted in the same way they appear in the
        file, so the field at index n corresponds to the field with ``field_id==n``.
        """
        if self._all_fields is None:
            self._all_fields = [RField(i, self) for i in range(len(self.field_records))]
        return self._all_fields

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
        """
        The header of the RNTuple.

        This provides low level access to all the metadata contained in the header.
        """
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
        """
        The complete list of field records in the RNTuple.

        This includes the fields from the header and from schema extensions in the footer.
        """
        if self._field_records is None:
            self._field_records = list(self.header.field_records)
            self._field_records.extend(self.footer.extension_links.field_records)
        return self._field_records

    @property
    def field_names(self):
        """
        The list of names of the fields in the RNTuple.
        """
        if self._field_names is None:
            self._field_names = [r.field_name for r in self.field_records]
        return self._field_names

    @property
    def column_records(self):
        """
        The complete list of column records in the RNTuple.

        This includes the columns from the header and from schema extensions in the footer.
        """
        if self._column_records is None:
            self._column_records = list(self.header.column_records)
            self._column_records.extend(self.footer.extension_links.column_records)
            for i, cr in enumerate(self._column_records):
                cr.idx = i
        return self._column_records

    @property
    def alias_column_records(self):
        """
        The list of alias column records in the RNTuple.
        """
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
        """
        The footer of the RNTuple.

        This provides low level access to all the metadata contained in the footer.
        """
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
        """
        The list of cluster summaries in the RNTuple.
        """
        if self._cluster_summaries is None:
            self._cluster_summaries = []
            for pl in self.page_list_envelopes:
                self._cluster_summaries.extend(pl.cluster_summaries)
        return self._cluster_summaries

    @property
    def page_link_list(self):
        """
        The list of page links in the RNTuple.
        """
        if self._page_link_list is None:
            self._page_link_list = []
            for pl in self.page_list_envelopes:
                self._page_link_list.extend(pl.pagelinklist)
        return self._page_link_list

    def read_locator(self, loc, uncomp_size, context):
        """
        Args:
            loc (:doc:`uproot.models.RNTuple.MetaData`): The locator of the page.
            uncomp_size (int): The size in bytes of the uncompressed data.
            context (dict): Auxiliary data used in deserialization.

        Returns a tuple of the decompressed chunk and the updated cursor.
        """
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
        """
        The list of page list envelopes in the RNTuple.
        """
        context = {}

        if not self._page_list_envelopes:
            for record in self.footer.cluster_group_records:
                link = record.page_list_link
                loc = link.locator
                decomp_chunk, cursor = self.read_locator(
                    loc, link.env_uncomp_size, context
                )
                self._page_list_envelopes.append(
                    PageLink().read(decomp_chunk, cursor, context)
                )

        return self._page_list_envelopes

    def base_col_form(self, cr, col_id, parameters=None, cardinality=False):
        """
        Args:
            cr (:doc:`uproot.models.RNTuple.MetaData`): The column record.
            col_id (int): The column id.
            parameters (dict): The parameters to pass to the ``NumpyForm``.
            cardinality (bool): Whether the column is a cardinality column.

        Returns an Awkward Form describing the column if applicable, or a form key otherwise.
        """
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

    def col_form(self, field_id, extra_parameters=None):
        """
        Args:
            field_id (int): The field id.

        Returns an Awkward Form describing the column if applicable, or a form key otherwise.
        """
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
        # for this part we can use the default (zeroth) representation
        rel_crs = [c for c in rel_crs if c.repr_idx == 0]

        if len(rel_crs) == 1:  # base case
            cardinality = "RNTupleCardinality" in self.field_records[field_id].type_name
            return self.base_col_form(
                rel_crs[0],
                rel_crs[0].idx,
                parameters=extra_parameters,
                cardinality=cardinality,
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
            parameters = {"__array__": "string"}
            if extra_parameters is not None:
                parameters.update(extra_parameters)
            return ak.forms.ListOffsetForm(
                "i64", inner, form_key=form_key, parameters=parameters
            )
        else:
            raise (RuntimeError(f"Missing special case: {field_id}"))

    def field_form(self, this_id, keys, ak_add_doc=False):
        """
        Args:
            this_id (int): The field id.
            keys (list): The list of keys to search for.
            ak_add_doc (bool | dict ): If True and ``library="ak"``, add the RField ``description``
                to the Awkward ``__doc__`` parameter of the array.
                if dict = {key:value} and ``library="ak"``, add the RField ``value`` to the
                Awkward ``key`` parameter of the array.

        Returns an Awkward Form describing the field.
        """
        ak = uproot.extras.awkward()

        field_records = self.field_records
        this_record = field_records[this_id]
        structural_role = this_record.struct_role

        parameters = None
        if isinstance(ak_add_doc, bool) and ak_add_doc and this_record.field_desc != "":
            parameters = {"__doc__": this_record.field_desc}
        elif isinstance(ak_add_doc, dict):
            parameters = {
                key: self.ntuple.all_fields[this_id].__getattribute__(value)
                for key, value in ak_add_doc.items()
            }
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
            # base case of recursion
            # n.b. the split may happen in column
            return self.col_form(this_id, extra_parameters=parameters)
        elif structural_role == uproot.const.RNTupleFieldRole.LEAF:
            if this_id in self._related_ids:
                # std::array has only one subfield
                child_id = self._related_ids[this_id][0]
                inner = self.field_form(child_id, keys, ak_add_doc=ak_add_doc)
            else:
                # std::bitset has no subfields, so we use it directly
                inner = self.col_form(this_id)
            keyname = f"RegularForm-{this_id}"
            return ak.forms.RegularForm(
                inner, this_record.repetition, form_key=keyname, parameters=parameters
            )
        elif structural_role == uproot.const.RNTupleFieldRole.COLLECTION:
            if this_id not in self._related_ids or len(self._related_ids[this_id]) != 1:
                keyname = f"vector-{this_id}"
                newids = self._related_ids.get(this_id, [])
                # go find N in the rest, N is the # of fields in vector
                recordlist = []
                namelist = []
                for i in newids:
                    if any(key.startswith(self.all_fields[i].path) for key in keys):
                        recordlist.append(
                            self.field_form(i, keys, ak_add_doc=ak_add_doc)
                        )
                        namelist.append(field_records[i].field_name)
                if all(name == f"_{i}" for i, name in enumerate(namelist)):
                    namelist = None
                return ak.forms.RecordForm(
                    recordlist, namelist, form_key="whatever", parameters=parameters
                )
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
            inner = self.field_form(child_id, keys, ak_add_doc=ak_add_doc)
            return ak.forms.ListOffsetForm(
                "i64", inner, form_key=keyname, parameters=parameters
            )
        elif structural_role == uproot.const.RNTupleFieldRole.RECORD:
            newids = []
            if this_id in self._related_ids:
                newids = self._related_ids[this_id]
            # go find N in the rest, N is the # of fields in struct
            recordlist = []
            namelist = []
            for i in newids:
                if any(key.startswith(self.all_fields[i].path) for key in keys):
                    recordlist.append(self.field_form(i, keys, ak_add_doc=ak_add_doc))
                    namelist.append(field_records[i].field_name)
            if all(name == f"_{i}" for i, name in enumerate(namelist)):
                namelist = None
            return ak.forms.RecordForm(
                recordlist, namelist, form_key="whatever", parameters=parameters
            )
        elif structural_role == uproot.const.RNTupleFieldRole.VARIANT:
            keyname = self.col_form(this_id)
            newids = []
            if this_id in self._related_ids:
                newids = self._related_ids[this_id]
            recordlist = [
                self.field_form(i, keys, ak_add_doc=ak_add_doc) for i in newids
            ]
            inner = ak.forms.UnionForm(
                "i8", "i64", recordlist, form_key=keyname + "-union"
            )
            return ak.forms.IndexedOptionForm(
                "i64", inner, form_key=keyname, parameters=parameters
            )
        elif structural_role == uproot.const.RNTupleFieldRole.STREAMER:
            raise NotImplementedError(
                f"Unsplit fields are not supported. {this_record}"
            )
        else:
            # everything should recurse above this branch
            raise AssertionError("this should be unreachable")

    def read_pagedesc(self, destination, desc, dtype_str, dtype, nbits, split):
        """
        Args:
            destination (numpy.ndarray): The array to fill.
            desc (:doc:`uproot.models.RNTuple.MetaData`): The page description.
            dtype_str (str): The data type as a string.
            dtype (numpy.dtype): The data type.
            nbits (int): The number of bits.
            split (bool): Whether the data is split.

        Fills the destination array with the data from the page.
        """
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
        destination.view(dtype)[:num_elements_toread] = content[:num_elements_toread]
        self.deserialize_page_decompressed_buffer(
            destination, desc, dtype_str, dtype, nbits, split
        )

    def read_col_pages(
        self, ncol, cluster_range, dtype_byte, pad_missing_element=False
    ):
        """
        Args:
            ncol (int): The column id.
            cluster_range (range): The range of cluster indices.
            dtype_byte (int): The data type.
            pad_missing_element (bool): Whether to pad the missing elements.

        Returns a numpy array with the data from the column.
        """
        arrays = [self.read_col_page(ncol, i) for i in cluster_range]

        # Check if column stores offset values
        if dtype_byte in uproot.const.rntuple_index_types:
            # Extract the last offset values:
            last_elements = [
                (arr[-1] if len(arr) > 0 else numpy.zeros((), dtype=arr.dtype))
                for arr in arrays[:-1]
            ]  # First value always zero, therefore skip first arr.
            last_offsets = numpy.cumsum(last_elements)
            for i in range(1, len(arrays)):
                arrays[i] += last_offsets[i - 1]

        res = numpy.concatenate(arrays, axis=0)

        # No longer needed; free memory
        del arrays

        dtype_byte = self.column_records[ncol].type
        if dtype_byte in uproot.const.rntuple_index_types:
            res = numpy.insert(res, 0, 0)  # for offsets

        if pad_missing_element:
            first_element_index = self.column_records[ncol].first_element_index
            res = numpy.pad(res, (first_element_index, 0))
        return res

    def read_col_page(self, ncol, cluster_i):
        """
        Args:
            ncol (int): The column id.
            cluster_i (int): The cluster index.

        Returns a numpy array with the data from the column.
        """
        linklist = self._ntuple.page_link_list[cluster_i]
        # Check if the column is suppressed and pick the non-suppressed one if so
        if ncol < len(linklist) and linklist[ncol].suppressed:
            rel_crs = self._column_records_dict[self.column_records[ncol].field_id]
            ncol = next(cr.idx for cr in rel_crs if not linklist[cr.idx].suppressed)
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
            res = res.astype(numpy.float32)
        return res

    def gpu_read_clusters(self, columns, start_cluster_idx, stop_cluster_idx):
        """
        Args:
            columns (list): The target columns to read.
            start_cluster_idx (int): The first cluster index containing entries
            in the range requested.
            stop_cluster_idx (int): The last cluster index containing entries
            in the range requested.

        Returns a ClusterRefs containing ColRefs_Cluster for each cluster. Each
        ColRefs_Cluster contains all ColBuffersCluster for each column in
        columns. Each ColBuffersCluster contains the page buffers, decompression
        target buffers, and compression metadata for a column in a given cluster.

        The ClusterRefs object contains all information needed for and performs
        decompression in parallel over all compressed buffers.
        """
        cluster_range = range(start_cluster_idx, stop_cluster_idx)
        clusters_datas = ClusterRefs()
        # Open filehandle and read columns for clusters
        filehandle = CuFileSource(self.file.source.file_path, "rb")

        # Iterate through each cluster
        for cluster_i in cluster_range:
            colrefs_cluster = ColRefsCluster(cluster_i)
            for key in columns:
                if "column" in key and "union" not in key:
                    key_nr = int(key.split("-")[1])
                    if key_nr not in colrefs_cluster.colbuffersclusters.keys():
                        Col_ClusterBuffers = self.gpu_read_col_cluster_pages(
                            key_nr, cluster_i, filehandle
                        )
                        colrefs_cluster._add_Col(Col_ClusterBuffers)
            clusters_datas._add_cluster(colrefs_cluster)

        filehandle.get_all()
        return clusters_datas

    def gpu_read_col_cluster_pages(self, ncol, cluster_i, filehandle):
        """
        Args:
            ncol (int): The target column's key number.
            cluster_i (int): The cluster to read column data from.
            filehandle (uproot.source.cufile_interface.CuFileSource): CuFile
            filehandle interface which performs CuFile API calls.

        Returns a ColBuffersCluster containing raw page buffers, decompression
        target buffers, and compression metadata.
        """
        # Get cluster and pages metadatas
        cupy = uproot.extras.cupy()
        linklist = self.page_link_list[cluster_i]
        ncol_orig = ncol
        if ncol < len(linklist):
            if linklist[ncol].suppressed:
                rel_crs = self._column_records_dict[self.column_records[ncol].field_id]
                ncol = next(cr.idx for cr in rel_crs if not linklist[cr.idx].suppressed)
            linklist_col = linklist[ncol]
            pagelist = linklist_col.pages
            compression = linklist_col.compression_settings
            compression_level = compression % 100
            algorithm = compression // 100
            algorithm_str = compression_settings_dict[algorithm]
        else:
            pagelist = []
            algorithm_str = None
            compression_level = None

        dtype_byte = self.column_records[ncol].type

        dtype_str = uproot.const.rntuple_col_num_to_dtype_dict[dtype_byte]
        isbit = dtype_str == "bit"
        # Prepare full output buffer
        total_len = numpy.sum([desc.num_elements for desc in pagelist], dtype=int)
        if dtype_str == "switch":
            dtype = numpy.dtype([("index", "int64"), ("tag", "int32")])
        elif dtype_str == "bit":
            dtype = numpy.dtype("bool")
        elif dtype_byte in uproot.const.rntuple_custom_float_types:
            dtype = numpy.dtype("uint32")  # for easier bit manipulation
        else:
            dtype = numpy.dtype(dtype_str)

        full_output_buffer = cupy.empty(total_len, dtype=dtype)

        nbits = (
            self.column_records[ncol].nbits
            if ncol < len(self.column_records)
            else uproot.const.rntuple_col_num_to_size_dict[dtype_byte]
        )
        # Check if col compressed/decompressed
        if isbit:  # Need to correct length when dtype = bit
            total_len = int(numpy.ceil(total_len / 8))
        elif dtype_str in ("real32trunc", "real32quant"):
            total_len = int(numpy.ceil((total_len * 4 * nbits) / 32))
            dtype = numpy.dtype("uint8")

        total_bytes = numpy.sum([desc.locator.num_bytes for desc in pagelist])

        isCompressed = total_bytes != total_len * dtype.itemsize

        Cluster_Contents = ColBuffersCluster(
            ncol_orig,
            full_output_buffer,
            isCompressed,
            algorithm_str,
            compression_level,
        )
        tracker = 0
        for page_desc in pagelist:
            num_elements = page_desc.num_elements
            loc = page_desc.locator
            n_bytes = loc.num_bytes
            if isbit:  # Need to correct length when dtype = bit
                num_elements = int(numpy.ceil(num_elements / 8))

            tracker_end = tracker + num_elements
            out_buff = full_output_buffer[tracker:tracker_end]

            # If compressed, skip 9 byte header
            if isCompressed:
                # If LZ4, page contains additional 8-byte checksum
                offset = (
                    int(loc.offset + 9)
                    if algorithm_str != "LZ4"
                    else int(loc.offset + 9 + 8)
                )
                comp_buff = cupy.empty(n_bytes - 9, dtype="b")
                filehandle.pread(comp_buff, size=int(n_bytes - 9), file_offset=offset)

            # If uncompressed, read directly into out_buff
            else:
                comp_buff = None
                filehandle.pread(
                    out_buff, size=int(n_bytes), file_offset=int(loc.offset)
                )

            Cluster_Contents._add_page(comp_buff)
            Cluster_Contents._add_output(out_buff)

            tracker = tracker_end

        return Cluster_Contents

    def gpu_deserialize_decompressed_content(
        self, clusters_datas, start_cluster_idx, stop_cluster_idx
    ):
        """
        Args:
            clusters_datas (ClusterRefs): The target column's key number.
            start_cluster_idx (int): The first cluster index containing entries
            in the range requested.
            stop_cluster_idx (int): The last cluster index containing entries
            in the range requested.

        Returns a dictionary containing contiguous buffers of deserialized data
        across requested clusters organized by column key.
        """
        cupy = uproot.extras.cupy()
        cluster_range = range(start_cluster_idx, stop_cluster_idx)

        col_arrays = {}  # collect content for each col
        for key_nr in clusters_datas.columns:
            ncol = int(key_nr)
            # Get uncompressed array for key for all clusters
            col_decompressed_buffers = clusters_datas._grab_ColOutput(ncol)
            dtype_byte = self.ntuple.column_records[ncol].type
            arrays = []

            for cluster_i in cluster_range:
                # Get decompressed buffer corresponding to cluster i
                cluster_buffer = col_decompressed_buffers[cluster_i]

                self.gpu_deserialize_pages(cluster_buffer, ncol, cluster_i, arrays)

            if dtype_byte in uproot.const.rntuple_delta_types:
                # Extract the last offset values:
                last_elements = [
                    arr[-1].get() for arr in arrays[:-1]
                ]  # First value always zero, therefore skip first arr.
                # Compute cumulative sum using itertools.accumulate:
                last_offsets = numpy.cumsum(last_elements)

                # Add the offsets to each array
                for i in range(1, len(arrays)):
                    arrays[i] += last_offsets[i - 1]
                # Remove the first element from every sub-array except for the first one:
                arrays = [arrays[0]] + [arr[1:] for arr in arrays[1:]]

            res = cupy.concatenate(arrays, axis=0)
            del arrays
            if True:
                first_element_index = self.column_records[ncol].first_element_index
                res = cupy.pad(res, (first_element_index, 0))

            col_arrays[key_nr] = res

        return col_arrays

    def gpu_deserialize_pages(self, cluster_buffer, ncol, cluster_i, arrays):
        """
        Args:
            cluster_buffer (cupy.ndarray): Buffer to deserialize.
            ncol (int): The column's key number cluster_buffer originates from.
            cluster_i (int): The cluster cluster_buffer originates from.
            arrays (list): Container for storing results of deserialization
            across clusters.

        Returns nothing. Appends deserialized data buffer for ncol from cluster_i
        to arrays.
        """
        # Get pagelist and metadatas
        cupy = uproot.extras.cupy()
        linklist = self.page_link_list[cluster_i]
        if ncol < len(linklist):
            if linklist[ncol].suppressed:
                rel_crs = self._column_records_dict[self.column_records[ncol].field_id]
                ncol = next(cr.idx for cr in rel_crs if not linklist[cr.idx].suppressed)
            linklist_col = linklist[ncol]
            pagelist = linklist_col.pages
        else:
            pagelist = []

        dtype_byte = self.column_records[ncol].type
        dtype_str = uproot.const.rntuple_col_num_to_dtype_dict[dtype_byte]

        if dtype_str == "switch":
            dtype = numpy.dtype([("index", "int64"), ("tag", "int32")])
        elif dtype_str == "bit":
            dtype = numpy.dtype("bool")
        elif dtype_byte in uproot.const.rntuple_custom_float_types:
            dtype = numpy.dtype("uint32")  # for easier bit manipulation
        else:
            dtype = numpy.dtype(dtype_str)
        split = dtype_byte in uproot.const.rntuple_split_types
        zigzag = dtype_byte in uproot.const.rntuple_zigzag_types
        delta = dtype_byte in uproot.const.rntuple_delta_types
        index = dtype_byte in uproot.const.rntuple_index_types
        nbits = (
            self.column_records[ncol].nbits
            if ncol < len(self.column_records)
            else uproot.const.rntuple_col_num_to_size_dict[dtype_byte]
        )

        # Begin looping through pages
        tracker = 0
        cumsum = 0
        for page_desc in pagelist:
            num_elements = page_desc.num_elements
            tracker_end = tracker + num_elements

            # Get content associated with page
            page_buffer = cluster_buffer[tracker:tracker_end]
            self.deserialize_page_decompressed_buffer(
                page_buffer, page_desc, dtype_str, dtype, nbits, split
            )
            if delta:
                cluster_buffer[tracker] -= cumsum
                cumsum += cupy.sum(cluster_buffer[tracker:tracker_end])
            tracker = tracker_end

        if index:
            cluster_buffer = _cupy_insert0(cluster_buffer)  # for offsets
        if zigzag:
            cluster_buffer = _from_zigzag(cluster_buffer)
        elif delta:
            cluster_buffer = cupy.cumsum(cluster_buffer)
        elif dtype_str == "real32trunc":
            cluster_buffer = cluster_buffer.view(cupy.float32)
        elif dtype_str == "real32quant" and ncol < len(self.column_records):
            min_value = self.column_records[ncol].min_value
            max_value = self.column_records[ncol].max_value
            cluster_buffer = min_value + cluster_buffer.astype(cupy.float32) * (
                max_value - min_value
            ) / ((1 << nbits) - 1)
            cluster_buffer = cluster_buffer.astype(cupy.float32)

        arrays.append(cluster_buffer)

    def deserialize_page_decompressed_buffer(
        self, destination, desc, dtype_str, dtype, nbits, split
    ):
        """
        Args:
            destination (cupy.ndarray): The array to fill.
            desc (:doc:`uproot.models.RNTuple.MetaData`): The page description.
            dtype_str (str): The data type as a string.
            dtype (cupy.dtype): The data type.
            nbits (int): The number of bits.
            split (bool): Whether the data is split.

        Returns nothing. Edits destination buffer in-place with deserialized
        data.
        """
        array_library_string = uproot._util.get_array_library(destination)
        library = numpy if array_library_string == "numpy" else uproot.extras.cupy()

        # bool in RNTuple is always stored as bits
        isbit = dtype_str == "bit"
        num_elements = desc.num_elements
        content = library.copy(destination)
        if split:
            content = content.view(library.uint8)
            length = content.shape[0]
            if nbits == 16:
                # AAAAABBBBB needs to become
                # ABABABABAB
                res = library.empty(length, library.uint8)
                res[0::2] = content[length * 0 // 2 : length * 1 // 2]
                res[1::2] = content[length * 1 // 2 : length * 2 // 2]

            elif nbits == 32:
                # AAAAABBBBBCCCCCDDDDD needs to become
                # ABCDABCDABCDABCDABCD
                res = library.empty(length, library.uint8)
                res[0::4] = content[length * 0 // 4 : length * 1 // 4]
                res[1::4] = content[length * 1 // 4 : length * 2 // 4]
                res[2::4] = content[length * 2 // 4 : length * 3 // 4]
                res[3::4] = content[length * 3 // 4 : length * 4 // 4]

            elif nbits == 64:
                # AAAAABBBBBCCCCCDDDDDEEEEEFFFFFGGGGGHHHHH needs to become
                # ABCDEFGHABCDEFGHABCDEFGHABCDEFGHABCDEFGH
                res = library.empty(length, library.uint8)
                res[0::8] = content[length * 0 // 8 : length * 1 // 8]
                res[1::8] = content[length * 1 // 8 : length * 2 // 8]
                res[2::8] = content[length * 2 // 8 : length * 3 // 8]
                res[3::8] = content[length * 3 // 8 : length * 4 // 8]
                res[4::8] = content[length * 4 // 8 : length * 5 // 8]
                res[5::8] = content[length * 5 // 8 : length * 6 // 8]
                res[6::8] = content[length * 6 // 8 : length * 7 // 8]
                res[7::8] = content[length * 7 // 8 : length * 8 // 8]
            content = res.view(dtype)

        if isbit:
            content = library.unpackbits(
                destination.view(dtype=library.uint8), bitorder="little"
            )
        elif dtype_str in ("real32trunc", "real32quant"):
            if nbits == 32:
                content = library.copy(destination).view(library.uint32)
            else:
                content = library.copy(destination)
                content = _extract_bits(content, nbits)
            if dtype_str == "real32trunc":
                content <<= 32 - nbits

        # needed to chop off extra bits incase we used `unpackbits`
        destination[:] = content[:num_elements]


def _extract_bits(packed, nbits):
    """
    Args:
        packed (library.ndarray): The packed array of 32-bit integers.
        nbits (int): The bit width of the original truncated data.

    Returns:
        library.ndarray: The unpacked data.
    """
    array_library_string = uproot._util.get_array_library(packed)
    library = numpy if array_library_string == "numpy" else uproot.extras.cupy()

    packed = packed.view(dtype=library.uint32)
    total_bits = packed.size * 32
    n_values = total_bits // nbits
    result = library.empty(n_values, dtype=library.uint32)

    # Indices into packed array
    bit_positions = library.arange(n_values, dtype=library.uint32) * nbits
    word_idx = bit_positions // 32
    offset = bit_positions % 32

    # Pad packed array by one element to avoid out-of-bounds access
    packed_padded = library.concatenate(
        [packed, library.zeros(1, dtype=library.uint32)]
    )

    # Extract words
    current_word = packed_padded[word_idx]
    next_word = packed_padded[word_idx + 1]

    # Compute bitmask and offsets
    mask = (1 << nbits) - 1
    bits_left = 32 - offset
    needs_second_word = offset + nbits > 32

    # Extract bits from current and next word
    first_part = (current_word >> offset) & mask
    second_part = (next_word << bits_left) & mask

    # Combine parts where needed
    result = library.where(needs_second_word, first_part | second_part, first_part)
    return result


# Supporting function and classes
def _split_switch_bits(content):
    tags = content["tag"].astype(numpy.dtype("int8")) - 1
    kindex = content["index"]
    return kindex, tags


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
            out.type = uproot.const.RNTupleLocatorType(-out.num_bytes >> 24)
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
        out.struct_role = uproot.const.RNTupleFieldRole(out.struct_role)
        out.flags = uproot.const.RNTupleFieldFlags(out.flags)
        out.field_name, out.type_name, out.type_alias, out.field_desc = (
            cursor.rntuple_string(chunk, context) for _ in range(4)
        )

        if out.flags & uproot.const.RNTupleFieldFlags.REPETITIVE:
            out.repetition = cursor.field(chunk, _rntuple_repetition_format, context)
        else:
            out.repetition = 0

        if out.flags & uproot.const.RNTupleFieldFlags.PROJECTED:
            out.source_field_id = cursor.field(
                chunk, _rntuple_source_field_id_format, context
            )
        else:
            out.source_field_id = None

        if out.flags & uproot.const.RNTupleFieldFlags.CHECKSUM:
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
        out.flags = uproot.const.RNTupleColumnFlags(out.flags)
        if out.flags & uproot.const.RNTupleColumnFlags.DEFERRED:
            out.first_element_index = cursor.field(
                chunk, _rntuple_first_element_index_format, context
            )
        else:
            out.first_element_index = 0
        if out.flags & uproot.const.RNTupleColumnFlags.RANGE:
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
        out.content_id = uproot.const.RNTupleExtraTypeIdentifier(out.content_id)
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
        out.ntuple_name, out.ntuple_description, out.writer_identifier = (
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
        out.flags = uproot.const.RNTupleClusterFlags(out.num_entries >> 56)
        out.num_entries &= 0xFFFFFFFFFFFFFF
        if out.flags & uproot.const.RNTupleClusterFlags.SHARDED:
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


class RField(uproot.behaviors.RNTuple.HasFields):
    def __init__(self, fid, ntuple):
        self._fid = fid
        self._ntuple = ntuple
        self._length = None
        self._fields = None
        self._lookup = None
        self._path = None

    def __repr__(self):
        if len(self) == 0:
            return f"<RField {self.name!r} in RNTuple {self.ntuple.name!r} at 0x{id(self):012x}>"
        else:
            return f"<RField {self.name!r} ({len(self)} subfields) in RNTuple {self.ntuple.name!r} at 0x{id(self):012x}>"

    @property
    def name(self):
        """
        Name of the ``RField``.
        """
        return self._ntuple.field_records[self._fid].field_name

    @property
    def description(self):
        """
        Description of the ``RField``.
        """
        return self._ntuple.field_records[self._fid].field_desc

    @property
    def typename(self):
        """
        The C++ typename of the ``RField``.
        """
        return self._ntuple.field_records[self._fid].type_name

    @property
    def parent(self):
        """
        The parent of this ``RField``.
        """
        rntuple = self.ntuple
        parent_fid = rntuple.field_records[self._fid].parent_field_id
        if parent_fid == self._fid:
            return rntuple
        return rntuple.all_fields[parent_fid]

    @property
    def index(self):
        """
        Integer position of this ``RField`` in its parent's list of fields.
        """
        for i, field in enumerate(self.parent.fields):
            if field is self:
                return i
        else:
            raise AssertionError

    @property
    def field_id(self):
        """
        The field ID of this ``RField`` in the RNTuple.
        """
        return self._fid

    @property
    def top_level(self):
        """
        True if this is a top-level field, False otherwise.
        """
        return self.parent is self.ntuple

    def array(
        self,
        entry_start=None,
        entry_stop=None,
        *,
        decompression_executor=None,  # TODO: Not implemented yet
        array_cache="inherit",  # TODO: Not implemented yet
        library="ak",
        ak_add_doc=False,
        # For compatibility reasons we also accepts kwargs meant for TTrees
        interpretation=None,
        interpretation_executor=None,
        use_GDS=False,
        backend="cpu",
    ):
        """
        Args:
            entry_start (None or int): The first entry to include. If None, start
                at zero. If negative, count from the end, like a Python slice.
            entry_stop (None or int): The first entry to exclude (i.e. one greater
                than the last entry to include). If None, stop at
                :ref:`uproot.behaviors.TTree.TTree.num_entries`. If negative,
                count from the end, like a Python slice.
            decompression_executor (None or Executor with a ``submit`` method): The
                executor that is used to decompress ``RPages``; if None, the
                file's :ref:`uproot.reading.ReadOnlyFile.decompression_executor`
                is used. (Not implemented yet.)
            array_cache ("inherit", None, MutableMapping, or memory size): Cache of arrays;
                if "inherit", use the file's cache; if None, do not use a cache;
                if a memory size, create a new cache of this size. (Not implemented yet.)
            library (str or :doc:`uproot.interpretation.library.Library`): The library
                that is used to represent arrays. Options are ``"np"`` for NumPy,
                ``"ak"`` for Awkward Array, and ``"pd"`` for Pandas.
            ak_add_doc (bool | dict ): If True and ``library="ak"``, add the RField ``description``
                to the Awkward ``__doc__`` parameter of the array.
                if dict = {key:value} and ``library="ak"``, add the RField ``value`` to the
                Awkward ``key`` parameter of the array.
            interpretation (None): This argument is not used and is only included for now
                for compatibility with software that was used for :doc:`uproot.behaviors.TBranch.TBranch`. This argument should not be used
                and will be removed in a future version.
            interpretation_executor (None): This argument is not used and is only included for now
                for compatibility with software that was used for :doc:`uproot.behaviors.TBranch.TBranch`. This argument should not be used
                and will be removed in a future version.

        Returns the ``RField`` data as an array.

        For example:

        .. code-block:: python

            >>> field = ntuple["my_field"]
            >>> array = field.array()
            >>> array
            <Array [-41.2, 35.1, 35.1, ... 32.4, 32.5] type='2304 * float64'>

        See also :ref:`uproot.behaviors.RNTuple.HasFields.arrays` to read
        multiple ``RFields`` into a group of arrays or an array-group.
        """
        return self.arrays(
            entry_start=entry_start,
            entry_stop=entry_stop,
            library=library,
            ak_add_doc=ak_add_doc,
            use_GDS=use_GDS,
            backend=backend,
        )[self.name]


# No cupy version of numpy.insert() provided
def _cupy_insert0(arr):
    cupy = uproot.extras.cupy()
    # Intended for flat cupy arrays
    array_len = arr.shape[0]
    array_dtype = arr.dtype
    out_arr = cupy.empty(array_len + 1, dtype=array_dtype)
    cupy.copyto(out_arr[1:], arr)
    out_arr[0] = 0
    return out_arr


CupyArray = any


@dataclasses.dataclass
class ColBuffersCluster:
    """
    A ColBuffersCluster contains the compressed and decompression target output
    buffers for a particular column in a particular cluster of all pages. It
    contains pointers to portions of the cluster data which correspond to the
    different pages of that cluster.
    """

    key: str
    data: CupyArray  # Type: ignore
    isCompressed: bool
    algorithm: str
    compression_level: int
    pages: list[CupyArray] = dataclasses.field(default_factory=list)
    output: list[CupyArray] = dataclasses.field(default_factory=list)

    def _add_page(self, page: CupyArray):
        self.pages.append(page)

    def _add_output(self, buffer: CupyArray):
        self.output.append(buffer)

    def _decompress(self):
        if self.isCompressed and self.algorithm is not None:
            kvikio_nvcomp_codec = uproot.extras.kvikio_nvcomp_codec()
            codec = kvikio_nvcomp_codec.NvCompBatchCodec(self.algorithm)
            codec.decode_batch(self.pages, self.output)


@dataclasses.dataclass
class ColRefsCluster:
    """
    A ColRefsCluster contains the ColBuffersCluster for all requested columns
    in a given cluster. Columns are separated by whether they are compressed or
    uncompressed. Compressed columns can be decompressed.
    """

    cluster_i: int
    colbuffersclusters: dict[str, ColBuffersCluster] = dataclasses.field(
        default_factory=dict
    )

    def _add_Col(self, ColBuffersCluster):
        self.colbuffersclusters[ColBuffersCluster.key] = ColBuffersCluster


@dataclasses.dataclass
class ClusterRefs:
    """ "
    A ClusterRefs contains the ColRefs_Cluster for multiple clusters.
    """

    clusters: [int] = dataclasses.field(default_factory=list)
    columns: list[str] = dataclasses.field(default_factory=list)
    refs: dict[int:ColRefsCluster] = dataclasses.field(default_factory=dict)

    def _add_cluster(self, Cluster):
        for nCol in Cluster.colbuffersclusters.keys():
            if nCol not in self.columns:
                self.columns.append(nCol)
        self.refs[Cluster.cluster_i] = Cluster

    def _grab_ColOutput(self, nCol):
        output_list = []
        for cluster in self.refs.values():
            colbuffer = cluster.colbuffersclusters[nCol].data
            output_list.append(colbuffer)

        return output_list

    def _decompress(self):
        to_decompress = {}
        target = {}
        # organize data by compression algorithm
        for cluster in self.refs.values():
            for colbuffers in cluster.colbuffersclusters.values():
                if colbuffers.algorithm is not None:
                    if colbuffers.algorithm not in to_decompress.keys():
                        to_decompress[colbuffers.algorithm] = []
                        target[colbuffers.algorithm] = []
                    if colbuffers.isCompressed:
                        to_decompress[colbuffers.algorithm].extend(colbuffers.pages)
                        target[colbuffers.algorithm].extend(colbuffers.output)

        # Batch decompress
        for algorithm, batch in to_decompress.items():
            kvikio_nvcomp_codec = uproot.extras.kvikio_nvcomp_codec()
            codec = kvikio_nvcomp_codec.NvCompBatchCodec(algorithm)
            codec.decode_batch(batch, target[algorithm])

        # Clean up compressed buffers from memory after decompression
        for cluster in self.refs.values():
            for colbuffers in cluster.colbuffersclusters.values():
                # Clear python references to GPU memory
                del colbuffers.pages
                colbuffers.pages = []
                # Tell GPU to free unused memory blocks
                cupy = uproot.extras.cupy()
                mempool = cupy.get_default_memory_pool()
                mempool.free_all_blocks()


uproot.classes["ROOT::RNTuple"] = Model_ROOT_3a3a_RNTuple

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
This module defines a versionless model for ``ROOT::RNTuple``.
"""
from __future__ import annotations

import dataclasses
import re
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

    def base_col_form(self, cr, parameters=None, is_cardinality=False):
        """
        Args:
            cr (:doc:`uproot.models.RNTuple.MetaData`): The column record.
            parameters (dict): The parameters to pass to the ``NumpyForm``.
            is_cardinality (bool): Whether the column is a cardinality column.

        Returns an Awkward Form describing the column if applicable, or a form key otherwise.
        """
        ak = uproot.extras.awkward()

        form_key = f"column-{cr.idx}" + ("-cardinality" if is_cardinality else "")
        dtype_byte = cr.type
        if dtype_byte == uproot.const.rntuple_col_type_to_num_dict["switch"]:
            return form_key
        elif dtype_byte in uproot.const.rntuple_index_types and not is_cardinality:
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

    def col_form(self, field_id, extra_parameters=None, is_cardinality=False):
        """
        Args:
            field_id (int): The field id.

        Returns an Awkward Form describing the column if applicable, or a form key otherwise.
        """
        ak = uproot.extras.awkward()

        cfid = field_id
        if self.field_records[cfid].source_field_id is not None:
            cfid = self.field_records[cfid].source_field_id
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
            return self.base_col_form(
                rel_crs[0],
                parameters=extra_parameters,
                is_cardinality=is_cardinality,
            )
        elif (
            len(rel_crs) == 2
            and rel_crs[1].type == uproot.const.rntuple_col_type_to_num_dict["char"]
        ):
            # string field splits->2 in col records
            inner = self.base_col_form(rel_crs[1], parameters={"__array__": "char"})
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
            is_cardinality = "RNTupleCardinality" in this_record.type_name
            if self.field_records[this_id].source_field_id is not None:
                this_id = self.field_records[this_id].source_field_id
            # deal with std::atomic
            # they have no associated column, but exactly one subfield containing the underlying data
            if (
                this_id not in self._column_records_dict
                and len(self._related_ids[this_id]) == 1
            ):
                this_id = self._related_ids[this_id][0]
            # base case of recursion
            # n.b. the split may happen in column
            return self.col_form(
                this_id, extra_parameters=parameters, is_cardinality=is_cardinality
            )
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
                    if (
                        any(
                            key.startswith(f"{self.all_fields[i].path}.")
                            or key == self.all_fields[i].path
                            for key in keys
                        )
                        or self.all_fields[i].is_anonymous
                    ):
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
                if any(
                    key.startswith(f"{self.all_fields[i].path}.")
                    or key == self.all_fields[i].path
                    for key in keys
                ):
                    recordlist.append(self.field_form(i, keys, ak_add_doc=ak_add_doc))
                    namelist.append(field_records[i].field_name)
            if all(re.fullmatch(r"_[0-9]+", name) is not None for name in namelist):
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

    def read_pagedesc(self, destination, desc, field_metadata):
        """
        Args:
            destination (numpy.ndarray): The array to fill.
            desc (:doc:`uproot.models.RNTuple.MetaData`): The page description.
            field_metadata (:doc:`uproot.models.RNTuple.FieldClusterMetadata`):
                The metadata needed to deserialize destination.

        Fills the destination array with the data from the page.
        """
        loc = desc.locator
        context = {}
        num_elements = len(destination)
        # Pages storing bits, real32trunc, and real32quant need num_elements
        # corrected
        if field_metadata.isbit:
            num_elements_toread = int(numpy.ceil(num_elements / 8))
        elif field_metadata.dtype_str in ("real32trunc", "real32quant"):
            num_elements_toread = int(
                numpy.ceil((num_elements * 4 * field_metadata.nbits) / 32)
            )
        else:
            num_elements_toread = num_elements

        uncomp_size = num_elements_toread * field_metadata.dtype_toread.itemsize
        decomp_chunk, cursor = self.read_locator(loc, uncomp_size, context)
        content = cursor.array(
            decomp_chunk,
            num_elements_toread,
            field_metadata.dtype_toread,
            context,
            move=False,
        )
        destination.view(field_metadata.dtype_toread)[:num_elements_toread] = content[
            :num_elements_toread
        ]
        self.deserialize_page_decompressed_buffer(destination, field_metadata)

    def read_col_pages(self, ncol, cluster_range, pad_missing_element=False):
        """
        Args:
            ncol (int): The column id.
            cluster_range (range): The range of cluster indices.
            pad_missing_element (bool): Whether to pad the missing elements.

        Returns a numpy array with the data from the column.
        """
        field_metadata = self.get_field_metadata(ncol)
        arrays = [self.read_col_page(ncol, i, field_metadata) for i in cluster_range]
        res = self.combine_cluster_arrays(arrays, field_metadata, pad_missing_element)

        return res

    def read_col_page(self, ncol, cluster_i, field_metadata):
        """
        Args:
            ncol (int): The column id.
            cluster_i (int): The cluster index.
            field_metadata (:doc:`uproot.models.RNTuple.FieldClusterMetadata`):
                The metadata needed to read the field's pages.

        Returns a numpy array with the data from the column.
        """
        linklist = self._ntuple.page_link_list[cluster_i]
        # Check if the column is suppressed and pick the non-suppressed one if so
        if ncol < len(linklist) and linklist[ncol].suppressed:
            rel_crs = self._column_records_dict[self.column_records[ncol].field_id]
            ncol = next(cr.idx for cr in rel_crs if not linklist[cr.idx].suppressed)
            field_metadata = self.get_field_metadata(
                ncol
            )  # Update metadata if suppressed
        pagelist = (
            linklist[field_metadata.ncol].pages
            if field_metadata.ncol < len(linklist)
            else []
        )
        total_len = numpy.sum([desc.num_elements for desc in pagelist], dtype=int)
        res = numpy.empty(total_len, field_metadata.dtype)

        tracker = 0
        cumsum = 0
        for page_desc in pagelist:
            n_elements = page_desc.num_elements
            tracker_end = tracker + n_elements
            self.read_pagedesc(
                res[tracker:tracker_end],
                page_desc,
                field_metadata,
            )
            if field_metadata.delta:
                res[tracker] -= cumsum
                cumsum += numpy.sum(res[tracker:tracker_end])
            tracker = tracker_end

        res = self.post_process(res, field_metadata)
        return res

    def gpu_read_clusters(self, fields, start_cluster_idx, stop_cluster_idx):
        """
        Args:
            fields (list: str): The target fields to read.
            start_cluster_idx (int): The first cluster index containing entries
            in the range requested.
            stop_cluster_idx (int): The last cluster index containing entries
            in the range requested.

        Returns a ClusterRefs containing FieldRefsCluster for each cluster. Each
        FieldRefsCluster contains all FieldPayload objects for each field in
        fields. Each FieldPayload contains the page buffers, decompression
        target buffers, and compression metadata for a field in a given cluster.
        """
        cluster_range = range(start_cluster_idx, stop_cluster_idx)
        clusters_datas = ClusterRefs()
        filehandle = CuFileSource(self.file.source.file_path, "rb")

        # Iterate through each cluster
        for cluster_i in cluster_range:
            colrefs_cluster = FieldRefsCluster(cluster_i)
            for key in fields:
                if "column" in key and "union" not in key:
                    ncol = int(key.split("-")[1])
                    field_metadata = self.get_field_metadata(ncol)
                    if ncol not in colrefs_cluster.fieldpayloads.keys():
                        Col_ClusterBuffers = self.gpu_read_col_cluster_pages(
                            ncol, cluster_i, filehandle, field_metadata
                        )
                        colrefs_cluster._add_field(Col_ClusterBuffers)
            clusters_datas._add_cluster(colrefs_cluster)

        filehandle.get_all()
        return clusters_datas

    def gpu_read_col_cluster_pages(self, ncol, cluster_i, filehandle, field_metadata):
        """
        Args:
            ncol (int): The target column's key number.
            cluster_i (int): The cluster to read column data from.
            filehandle (uproot.source.cufile_interface.CuFileSource): CuFile
            filehandle interface which performs CuFile API calls.
            field_metadata (:doc:`uproot.models.RNTuple.FieldClusterMetadata`):
                The metadata needed to read the fields pages.

        Returns a FieldPayload containing raw page buffers, decompression
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
                field_metadata = self.get_field_metadata(ncol)
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

        # Prepare full output buffer
        total_len = numpy.sum([desc.num_elements for desc in pagelist], dtype=int)
        full_output_buffer = cupy.empty(total_len, dtype=field_metadata.dtype)

        # Check if field compressed/decompressed
        if field_metadata.isbit:  # Need to correct length when dtype = bit
            total_len = int(numpy.ceil(total_len / 8))
        elif field_metadata.dtype_str in ("real32trunc", "real32quant"):
            total_len = int(numpy.ceil((total_len * 4 * field_metadata.nbits) / 32))
        total_raw_bytes = numpy.sum([desc.locator.num_bytes for desc in pagelist])
        page_is_compressed = (
            total_raw_bytes != total_len * field_metadata.dtype_toread.itemsize
        )

        cluster_contents = FieldPayload(
            ncol_orig,
            full_output_buffer,
            page_is_compressed,
            algorithm_str,
            compression_level,
        )
        tracker = 0
        for page_desc in pagelist:
            num_elements = page_desc.num_elements
            loc = page_desc.locator
            n_bytes = loc.num_bytes
            if field_metadata.isbit:  # Need to correct length when dtype = bit
                num_elements = int(numpy.ceil(num_elements / 8))

            tracker_end = tracker + num_elements
            out_buff = full_output_buffer[tracker:tracker_end]

            # If compressed, skip 9 byte header
            if page_is_compressed:
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

            cluster_contents._add_page(comp_buff)
            cluster_contents._add_output(out_buff)

            tracker = tracker_end

        return cluster_contents

    def gpu_deserialize_decompressed_content(
        self,
        clusters_datas,
        start_cluster_idx,
        stop_cluster_idx,
        pad_missing_element=False,
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
        cluster_range = range(start_cluster_idx, stop_cluster_idx)

        col_arrays = {}  # collect content for each col
        for key_nr in clusters_datas.columns:
            ncol = int(key_nr)
            field_metadata = self.get_field_metadata(ncol)
            # Get uncompressed array for key for all clusters
            col_decompressed_buffers = clusters_datas._grab_field_output(ncol)
            arrays = []
            for cluster_i in cluster_range:
                cluster_buffer = col_decompressed_buffers[cluster_i]
                cluster_buffer = self.gpu_deserialize_pages(
                    cluster_buffer, ncol, cluster_i, field_metadata
                )
                arrays.append(cluster_buffer)
            res = self.combine_cluster_arrays(
                arrays, field_metadata, pad_missing_element
            )
            col_arrays[key_nr] = res

        return col_arrays

    def gpu_deserialize_pages(self, cluster_buffer, ncol, cluster_i, field_metadata):
        """
        Args:
            cluster_buffer (cupy.ndarray): Buffer to deserialize.
            ncol (int): The column's key number cluster_buffer originates from.
            cluster_i (int): The cluster cluster_buffer originates from.
            field_metadata (:doc:`uproot.models.RNTuple.FieldClusterMetadata`):
                The metadata needed to deserialize the field's pages.

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
                field_metadata = self.get_field_metadata(ncol)
            linklist_col = linklist[ncol]
            pagelist = linklist_col.pages
        else:
            pagelist = []

        # Begin looping through pages
        tracker = 0
        cumsum = 0
        for page_desc in pagelist:
            num_elements = page_desc.num_elements
            tracker_end = tracker + num_elements
            # Get content associated with page
            page_buffer = cluster_buffer[tracker:tracker_end]
            self.deserialize_page_decompressed_buffer(page_buffer, field_metadata)
            if field_metadata.delta:
                cluster_buffer[tracker] -= cumsum
                cumsum += cupy.sum(cluster_buffer[tracker:tracker_end])
            tracker = tracker_end

        cluster_buffer = self.post_process(cluster_buffer, field_metadata)
        return cluster_buffer

    def post_process(self, buffer, field_metadata):
        """
        Args:
            buffer (library.ndarray): The buffer to post-process.
            field_metadata (:doc:`uproot.models.RNTuple.FieldClusterMetadata`):
                The metadata needed to post_process buffer.

        Returns post-processed buffer.
        """
        array_library_string = uproot._util.get_array_library(buffer)
        library = numpy if array_library_string == "numpy" else uproot.extras.cupy()
        if field_metadata.zigzag:
            buffer = _from_zigzag(buffer)
        elif field_metadata.delta:
            buffer = library.cumsum(buffer)
        elif field_metadata.dtype_str == "real32trunc":
            buffer = buffer.view(library.float32)
        elif field_metadata.dtype_str == "real32quant" and field_metadata.ncol < len(
            self.column_records
        ):
            min_value = self.column_records[field_metadata.ncol].min_value
            max_value = self.column_records[field_metadata.ncol].max_value
            buffer = min_value + buffer.astype(library.float32) * (
                max_value - min_value
            ) / ((1 << field_metadata.nbits) - 1)
            buffer = buffer.astype(library.float32)
        return buffer

    def deserialize_page_decompressed_buffer(self, destination, field_metadata):
        """
        Args:
            destination (cupy.ndarray): The array to fill.
            field_metadata (:doc:`uproot.models.RNTuple.FieldClusterMetadata`):
                The metadata needed to deserialize destination.


        Returns nothing. Edits destination buffer in-place with deserialized
        data.
        """
        array_library_string = uproot._util.get_array_library(destination)
        library = numpy if array_library_string == "numpy" else uproot.extras.cupy()
        num_elements = len(destination)

        content = library.copy(destination)
        if field_metadata.split:
            content = content.view(library.uint8)
            length = content.shape[0]
            if field_metadata.nbits == 16:
                # AAAAABBBBB needs to become
                # ABABABABAB
                res = library.empty(length, library.uint8)
                res[0::2] = content[length * 0 // 2 : length * 1 // 2]
                res[1::2] = content[length * 1 // 2 : length * 2 // 2]

            elif field_metadata.nbits == 32:
                # AAAAABBBBBCCCCCDDDDD needs to become
                # ABCDABCDABCDABCDABCD
                res = library.empty(length, library.uint8)
                res[0::4] = content[length * 0 // 4 : length * 1 // 4]
                res[1::4] = content[length * 1 // 4 : length * 2 // 4]
                res[2::4] = content[length * 2 // 4 : length * 3 // 4]
                res[3::4] = content[length * 3 // 4 : length * 4 // 4]

            elif field_metadata.nbits == 64:
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
            content = res.view(field_metadata.dtype)

        if field_metadata.isbit:
            content = library.unpackbits(
                destination.view(dtype=library.uint8), bitorder="little"
            )
        elif field_metadata.dtype_str in ("real32trunc", "real32quant"):
            if field_metadata.nbits == 32:
                content = library.copy(destination).view(library.uint32)
            else:
                content = library.copy(destination)
                content = _extract_bits(content, field_metadata.nbits)
            if field_metadata.dtype_str == "real32trunc":
                content <<= 32 - field_metadata.nbits

        # needed to chop off extra bits incase we used `unpackbits`
        destination[:] = content[:num_elements]

    def get_field_metadata(self, ncol):
        """
        Args:
            ncol (int): The column id.

        Returns a uproot.models.RNTuple.FieldClusterMetadata which provides
        metadata needed for processing payload data associated with column ncol.
        """
        dtype_byte = self.column_records[ncol].type
        dtype_str = uproot.const.rntuple_col_num_to_dtype_dict[dtype_byte]
        isbit = dtype_str == "bit"
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
        nbits = (
            self.column_records[ncol].nbits
            if ncol < len(self.column_records)
            else uproot.const.rntuple_col_num_to_size_dict[dtype_byte]
        )
        if dtype_str in ("real32trunc", "real32quant"):
            dtype_toread = numpy.dtype("uint8")
        else:
            dtype_toread = dtype
        field_metadata = FieldClusterMetadata(
            ncol,
            dtype_byte,
            dtype_str,
            dtype,
            dtype_toread,
            split,
            zigzag,
            delta,
            isbit,
            nbits,
        )
        return field_metadata

    def combine_cluster_arrays(self, arrays, field_metadata, pad_missing_element):
        """
        Args:
            arrays (list): A list of arrays to combine.
            field_metadata (:doc:`uproot.models.RNTuple.FieldClusterMetadata`):
                The metadata needed to combine arrays.
            pad_missing_element (bool): Whether to pad for missing elements

        Returns a field's page arrays concatenated todgether.
        """
        array_library_string = uproot._util.get_array_library(arrays[0])
        library = numpy if array_library_string == "numpy" else uproot.extras.cupy()

        # Check if column stores offset values
        if field_metadata.dtype_byte in uproot.const.rntuple_index_types:
            # Extract the last offset values:
            last_elements = [
                (arr[-1] if len(arr) > 0 else library.zeros((), dtype=arr.dtype))
                for arr in arrays[:-1]
            ]  # First value always zero, therefore skip first arr.
            last_offsets = library.cumsum(library.array(last_elements))
            for i in range(1, len(arrays)):
                arrays[i] += last_offsets[i - 1]

        res = library.concatenate(arrays, axis=0)

        # No longer needed; free memory
        del arrays

        if field_metadata.dtype_byte in uproot.const.rntuple_index_types:
            # for offsets
            res = numpy.insert(res, 0, 0) if library == numpy else _cupy_insert0(res)

        if pad_missing_element:
            first_element_index = self.column_records[
                field_metadata.ncol
            ].first_element_index
            res = numpy.pad(res, (first_element_index, 0))

        return res


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
        self._is_anonymous = None
        self._is_ignored = None

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
        # We rename subfields of tuples to match Awkward
        name = self._ntuple.field_records[self._fid].field_name
        if (
            not self.top_level
            and self.parent.record.struct_role == uproot.const.RNTupleFieldRole.RECORD
            and re.fullmatch(r"_[0-9]+", name) is not None
        ):
            name = name[1:]
        return name

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
    def record(self):
        """
        The field record of the ``RField``.
        """
        return self._ntuple.field_records[self._fid]

    @property
    def is_anonymous(self):
        """
        There are some anonymous fields in the RNTuple specification that we hide from the user
        to simplify the interface. These are fields named `_0` that are children of a collection
        or variant field.
        """
        if self._is_anonymous is None:
            self._is_anonymous = not self.top_level and (
                self.parent.record.struct_role
                in (
                    uproot.const.RNTupleFieldRole.COLLECTION,
                    uproot.const.RNTupleFieldRole.VARIANT,
                )
                or self.parent.record.flags & uproot.const.RNTupleFieldFlags.REPETITIVE
            )
        return self._is_anonymous

    @property
    def is_ignored(self):
        """
        There are some fields in the RNTuple specification named `:_i` (for `i=0,1,2,...`)
        that encode class hierarchy. These are not useful in Uproot, so they are ignored.
        """
        if self._is_ignored is None:
            self._is_ignored = (
                not self.top_level
                and self.parent.record.struct_role
                == uproot.const.RNTupleFieldRole.RECORD
                and re.fullmatch(r":_[0-9]+", self.name) is not None
            )

        return self._is_ignored

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
        # TODO: This needs to be optimized for performance
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
        interpreter="cpu",
        backend="cpu",
        ak_add_doc=False,
        # For compatibility reasons we also accepts kwargs meant for TTrees
        interpretation=None,
        interpretation_executor=None,
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
        arrays = self.arrays(
            entry_start=entry_start,
            entry_stop=entry_stop,
            library=library,
            interpreter=interpreter,
            backend=backend,
            ak_add_doc=ak_add_doc,
        )
        if self.name in arrays.fields:
            arrays = arrays[self.name]
        # tuples are a trickier since indices no longer match
        else:
            if self.name.isdigit() and arrays.fields == ["0"]:
                arrays = arrays["0"]
            else:
                raise AssertionError(
                    "The array was not constructed correctly. Please report this issue."
                )
        return arrays


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
class FieldClusterMetadata:
    """
    A FieldClusterMetadata stores metadata for a given field within a cluster
    needed to read, decompress, and deserialize the data.
    """

    ncol: int
    dtype_byte: type
    dtype_str: str
    dtype: numpy.dtype
    dtype_toread: numpy.dtype
    split: bool
    zigzag: bool
    delta: bool
    isbit: bool
    nbits: int


@dataclasses.dataclass
class FieldPayload:
    """
    A FieldPayload contains the compressed and decompression target output
    buffers for a particular column in a particular cluster of all pages. It
    contains pointers to portions of the cluster data which correspond to the
    different pages of that cluster.
    """

    key: str
    data: CupyArray  # Type: ignore
    page_is_compressed: bool
    algorithm: str
    compression_level: int
    pages: list[CupyArray] = dataclasses.field(default_factory=list)
    output: list[CupyArray] = dataclasses.field(default_factory=list)

    def _add_page(self, page: CupyArray):
        self.pages.append(page)

    def _add_output(self, buffer: CupyArray):
        self.output.append(buffer)

    def _decompress(self):
        if self.page_is_compressed and self.algorithm is not None:
            kvikio_nvcomp_codec = uproot.extras.kvikio_nvcomp_codec()
            codec = kvikio_nvcomp_codec.NvCompBatchCodec(self.algorithm)
            codec.decode_batch(self.pages, self.output)


@dataclasses.dataclass
class FieldRefsCluster:
    """
    A FieldRefsCluster contains the FieldPayload for all requested fields
    in a given cluster.
    """

    cluster_i: int
    fieldpayloads: dict[str, FieldPayload] = dataclasses.field(default_factory=dict)

    def _add_field(self, FieldPayload):
        self.fieldpayloads[FieldPayload.key] = FieldPayload


@dataclasses.dataclass
class ClusterRefs:
    """
    A ClusterRefs contains the FieldRefsCluster for multiple clusters. It also
    contains routines for steering and executing parallel decompression of
    payload datas and for accessing field payload datas across multiple clusters.
    """

    clusters: [int] = dataclasses.field(default_factory=list)
    columns: list[str] = dataclasses.field(default_factory=list)
    refs: dict[int:FieldRefsCluster] = dataclasses.field(default_factory=dict)

    def _add_cluster(self, Cluster):
        for nCol in Cluster.fieldpayloads.keys():
            if nCol not in self.columns:
                self.columns.append(nCol)
        self.refs[Cluster.cluster_i] = Cluster

    def _grab_field_output(self, nCol):
        output_list = []
        for cluster in self.refs.values():
            colbuffer = cluster.fieldpayloads[nCol].data
            output_list.append(colbuffer)

        return output_list

    def _decompress(self):
        to_decompress = {}
        target = {}
        # organize data by compression algorithm
        for cluster in self.refs.values():
            for fieldpayload in cluster.fieldpayloads.values():
                if fieldpayload.algorithm is not None:
                    if fieldpayload.algorithm not in to_decompress.keys():
                        to_decompress[fieldpayload.algorithm] = []
                        target[fieldpayload.algorithm] = []
                    if fieldpayload.page_is_compressed:
                        to_decompress[fieldpayload.algorithm].extend(fieldpayload.pages)
                        target[fieldpayload.algorithm].extend(fieldpayload.output)

        # Batch decompress
        kvikio_nvcomp_codec = uproot.extras.kvikio_nvcomp_codec()
        for algorithm, batch in to_decompress.items():
            codec = kvikio_nvcomp_codec.NvCompBatchCodec(algorithm)
            codec.decode_batch(batch, target[algorithm])

        # Clean up compressed buffers from memory after decompression
        for cluster in self.refs.values():
            for fieldpayload in cluster.fieldpayloads.values():
                # Clear python references to GPU memory
                del fieldpayload.pages
                fieldpayload.pages = []
                # Tell GPU to free unused memory blocks
                cupy = uproot.extras.cupy()
                mempool = cupy.get_default_memory_pool()
                mempool.free_all_blocks()


uproot.classes["ROOT::RNTuple"] = Model_ROOT_3a3a_RNTuple

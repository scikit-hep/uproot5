# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
This module defines typed views of the metadata of an ``RNTuple``.

The rest of Uproot reads RNTuple metadata only through these classes, which do
not depend on the parser that builds them. They hold what data reading and the
update path need, with Python ``str`` names and integer column types.
"""

from __future__ import annotations

import dataclasses

import uproot.const


@dataclasses.dataclass(frozen=True)
class FieldRecord:
    """
    A field of the ``RNTuple`` schema, from the header or from a schema
    extension in the footer.
    """

    field_id: int
    field_version: int
    type_version: int
    parent_field_id: int
    struct_role: uproot.const.RNTupleFieldRole
    flags: uproot.const.RNTupleFieldFlags
    field_name: str
    type_name: str
    type_alias: str
    field_desc: str
    repetition: int
    """The array size of a repetitive field, or 0 if the field is not repetitive."""
    source_field_id: int | None
    """The field that a projected field points to, or None."""
    checksum: int | None
    """The ROOT type checksum of the field, or None."""


@dataclasses.dataclass(frozen=True)
class ColumnRecord:
    """
    A physical column of the ``RNTuple`` schema, from the header or from a
    schema extension in the footer.
    """

    column_id: int
    type: int
    nbits: int
    field_id: int
    flags: uproot.const.RNTupleColumnFlags
    repr_idx: int
    first_element_index: int
    """The index of the first element of a deferred column, or 0. Negative if
    the column is deferred and suppressed."""
    min_value: float | None
    max_value: float | None


@dataclasses.dataclass(frozen=True)
class AliasColumnRecord:
    """
    An alias column of the ``RNTuple`` schema.
    """

    physical_id: int
    field_id: int


@dataclasses.dataclass(frozen=True)
class ClusterGroup:
    """
    A cluster group, as the footer records it, with the location of its page
    list envelope.
    """

    min_entry_num: int
    entry_span: int
    num_clusters: int
    page_list_length: int
    """The uncompressed size of the page list envelope."""
    page_list_num_bytes: int
    """The stored size of the page list envelope."""
    page_list_offset: int


@dataclasses.dataclass(frozen=True)
class Schema:
    """
    Everything the header and the footer of an ``RNTuple`` say about it.
    """

    name: str
    description: str
    writer: str
    header_checksum: int
    fields: tuple[FieldRecord, ...]
    """The fields of the header, then those of the footer's schema extension."""
    columns: tuple[ColumnRecord, ...]
    """The columns of the header, then those of the footer's schema extension."""
    alias_columns: tuple[AliasColumnRecord, ...]
    num_header_fields: int
    num_header_columns: int
    cluster_groups: tuple[ClusterGroup, ...]


@dataclasses.dataclass(frozen=True)
class Page:
    """
    A page of a column in a cluster.
    """

    num_elements: int
    offset: int
    num_bytes: int
    """The stored size of the page, without its checksum."""
    has_checksum: bool


@dataclasses.dataclass(frozen=True)
class ColumnRange:
    """
    A physical column in a cluster.
    """

    column_id: int
    suppressed: bool
    """Whether the column is a representation that is inactive in this cluster."""
    element_offset: int | None
    """The element offset as stored in the page list, or None if the page list
    does not list the column (a cluster written before a schema extension)."""
    compression_settings: int | None
    """The compression settings as stored, or None if the column is suppressed
    or not listed."""
    pages: tuple[Page, ...]


@dataclasses.dataclass(frozen=True)
class Cluster:
    """
    A cluster: its entries and, for every physical column, its pages.
    """

    cluster_id: int
    cluster_group_id: int
    num_first_entry: int
    num_entries: int
    flags: uproot.const.RNTupleClusterFlags
    columns: tuple[ColumnRange, ...]

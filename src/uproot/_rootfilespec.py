# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
This module is the only one in Uproot that uses rootfilespec, which parses the
metadata of ROOT files.

rootfilespec does no I/O: its objects describe where the data they point to is
(as locators, with an ``offset``, a ``size`` and a ``read_from`` method), and
this module fetches it through a :doc:`uproot.source.chunk.Source`. What
rootfilespec returns is turned into the views of
:doc:`uproot._rntuple_metadata`, so that the rest of Uproot does not depend on
rootfilespec's classes and names.

It is imported lazily, when an ``RNTuple`` is read.
"""

from __future__ import annotations

import contextlib
import queue
import struct
import types

from rootfilespec.bootstrap import BOOTSTRAP_CONTEXT, ROOT3a3aRNTuple
from rootfilespec.rntuple.RLocator import LargeLocator, StandardLocator
from rootfilespec.rntuple.RNTuple import RNTuple
from rootfilespec.serializable import BufferContext, ReadBuffer

import uproot._rntuple_metadata
import uproot.const
import uproot.exceptions


def decode(name):
    """
    Returns rootfilespec's ``bytes`` as a ``str``, as Uproot decodes the strings
    of a ROOT file: bytes that are not UTF-8 are kept as surrogates.
    """
    return name.decode(errors="surrogateescape")


@contextlib.contextmanager
def translate_errors(file_path, object_path):
    """
    Raises the errors of rootfilespec as :doc:`uproot.exceptions.FormatError`
    (or ``NotImplementedError`` for what it does not support), with the paths
    of the file and the object.
    """
    try:
        yield
    except NotImplementedError as err:
        raise NotImplementedError(
            f"{err}\nin file {file_path}\nin object {object_path}"
        ) from err
    except (ValueError, KeyError, IndexError, TypeError, struct.error) as err:
        raise uproot.exceptions.FormatError(str(err), file_path, object_path) from err


def _buffer(data, offset):
    return ReadBuffer(
        memoryview(data), 0, BOOTSTRAP_CONTEXT, BufferContext(abspos=offset)
    )


def fetch(source, locators):
    """
    Args:
        source (:doc:`uproot.source.chunk.Source`): Where to fetch from; a source
            with only a ``chunk`` method is read one locator at a time.
        locators (list of rootfilespec locators): What to fetch.

    Returns what each locator points to, fetched together if the source allows.
    """
    ranges = [(loc.offset, loc.offset + loc.size) for loc in locators]
    if hasattr(source, "chunks"):
        chunks = source.chunks(ranges, notifications=queue.Queue())
    else:
        chunks = [source.chunk(start, stop) for start, stop in ranges]
    return [
        loc.read_from(_buffer(chunk.raw_data, loc.offset))
        for loc, chunk in zip(locators, chunks, strict=True)
    ]


def read_anchor(payload):
    """
    Returns the ``ROOT::RNTuple`` anchor in ``payload``, the uncompressed object
    of its ``TKey``, once its checksum, byte count, class version and format
    epoch are checked.
    """
    anchor, _ = ROOT3a3aRNTuple.read(_buffer(payload, None))
    return anchor


def anchor_members(anchor):
    """
    Returns the members of the anchor, by their names in ROOT.
    """
    return {
        name: getattr(anchor, name)
        for name in (
            "fVersionEpoch",
            "fVersionMajor",
            "fVersionMinor",
            "fVersionPatch",
            "fSeekHeader",
            "fNBytesHeader",
            "fLenHeader",
            "fSeekFooter",
            "fNBytesFooter",
            "fLenFooter",
            "fMaxKeySize",
        )
    }


def fetch_schema_envelopes(source, anchor):
    """
    Returns the header and the footer envelopes that the anchor points to,
    fetched together.
    """
    return fetch(source, [anchor.header_locator, anchor.footer_locator])


def fetch_rntuple(source, header, footer):
    """
    Returns the rootfilespec ``RNTuple`` of the envelopes, with the page lists
    that the footer points to, fetched together.
    """
    page_lists = fetch(source, footer.pagelist_locators)
    return RNTuple.from_envelopes(header, footer, page_lists)


def _field_record(field_id, record):
    return uproot._rntuple_metadata.FieldRecord(
        field_id=field_id,
        field_version=record.fFieldVersion,
        type_version=record.fTypeVersion,
        parent_field_id=record.fParentFieldID,
        struct_role=uproot.const.RNTupleFieldRole(record.fStructuralRole),
        flags=uproot.const.RNTupleFieldFlags(record.fFlags),
        field_name=decode(record.fFieldName),
        type_name=decode(record.fTypeName),
        type_alias=decode(record.fTypeAlias),
        field_desc=decode(record.fFieldDescription),
        repetition=record.fArraySize or 0,
        source_field_id=record.fSourceFieldID,
        checksum=record.fTypeChecksum,
    )


def _column_record(column_id, record):
    return uproot._rntuple_metadata.ColumnRecord(
        column_id=column_id,
        type=int(record.fColumnType),
        nbits=record.fBitsOnStorage,
        field_id=record.fFieldID,
        flags=uproot.const.RNTupleColumnFlags(record.fFlags),
        repr_idx=record.fRepresentationIndex,
        first_element_index=record.fFirstElementIndex or 0,
        min_value=record.fMinValue,
        max_value=record.fMaxValue,
    )


def schema(header, footer):
    """
    Returns the :doc:`uproot._rntuple_metadata.Schema` of the header and the
    footer envelopes.
    """
    extension = footer.schemaExtension
    field_records = [*header.fieldDescriptions, *extension.fieldDescriptions]
    column_records = [*header.columnDescriptions, *extension.columnDescriptions]
    alias_records = [
        *header.aliasColumnDescriptions,
        *extension.aliasColumnDescriptions,
    ]
    cluster_groups = []
    for group in footer.clusterGroups:
        link = group.pagelistLink
        cluster_groups.append(
            uproot._rntuple_metadata.ClusterGroup(
                min_entry_num=group.fMinEntryNumber,
                entry_span=group.fEntrySpan,
                num_clusters=group.fNClusters,
                page_list_length=link.length,
                page_list_num_bytes=getattr(link.locator, "size", None),
                page_list_offset=getattr(link.locator, "offset", None),
            )
        )
    return uproot._rntuple_metadata.Schema(
        name=decode(header.fName),
        description=decode(header.fDescription),
        writer=decode(header.fLibrary),
        header_checksum=header.checksum,
        fields=tuple(_field_record(i, r) for i, r in enumerate(field_records)),
        columns=tuple(_column_record(i, r) for i, r in enumerate(column_records)),
        alias_columns=tuple(
            uproot._rntuple_metadata.AliasColumnRecord(r.fPhysicalColumnID, r.fFieldID)
            for r in alias_records
        ),
        num_header_fields=len(header.fieldDescriptions),
        num_header_columns=len(header.columnDescriptions),
        cluster_groups=tuple(cluster_groups),
    )


def _page(description):
    locator = description.page_locator.locator
    return uproot._rntuple_metadata.Page(
        num_elements=description.n_elements,
        offset=locator.offset,
        num_bytes=locator.size,
        has_checksum=description.has_checksum,
    )


def _column_range(column_range):
    locations = column_range.pageLocations
    compression = None
    if locations is not None and locations.compressionsettings is not None:
        compression = locations.compressionsettings.compressionsettings
    return uproot._rntuple_metadata.ColumnRange(
        column_id=column_range.column.columnID,
        suppressed=column_range.suppressed,
        element_offset=None if locations is None else locations.elementoffset,
        compression_settings=compression,
        pages=tuple(_page(page.pageDescription) for page in column_range.pages),
    )


def clusters(rntuple):
    """
    Returns the :doc:`uproot._rntuple_metadata.Cluster` of a rootfilespec
    ``RNTuple``, in cluster ID order.
    """
    return [
        uproot._rntuple_metadata.Cluster(
            cluster_id=cluster.clusterID,
            cluster_group_id=cluster.clusterGroupID,
            num_first_entry=cluster.summary.fFirstEntryNumber,
            num_entries=cluster.summary.fNEntries,
            flags=uproot.const.RNTupleClusterFlags(cluster.summary.fFeatureFlag),
            columns=tuple(_column_range(r) for r in cluster.columnRanges),
        )
        for cluster in rntuple.clusters()
    ]


# What the deprecated RNTuple.header, .footer and .page_list_envelopes return:
# the metadata with the names that Uproot's own reader gave it.


def _legacy_env_header(envelope):
    return {
        "env_type_id": uproot.const.RNTupleEnvelopeType(envelope.typeID),
        "env_length": envelope.length,
    }


def _legacy_locator(locator):
    if isinstance(locator, StandardLocator):
        locator_type = uproot.const.RNTupleLocatorType.STANDARD
    elif isinstance(locator, LargeLocator):
        locator_type = uproot.const.RNTupleLocatorType.LARGE
    else:
        return types.SimpleNamespace(type=locator.locatorType)
    return types.SimpleNamespace(
        num_bytes=locator.size, offset=locator.offset, type=locator_type
    )


def _legacy_extra_type_info(record):
    return types.SimpleNamespace(
        content_id=uproot.const.RNTupleExtraTypeIdentifier(record.fContentIdentifier),
        type_ver=record.fTypeVersion,
        type_name=decode(record.fTypeName),
    )


def legacy_header(header, schema):
    num_fields, num_columns = schema.num_header_fields, schema.num_header_columns
    num_aliases = len(header.aliasColumnDescriptions)
    return types.SimpleNamespace(
        env_header=_legacy_env_header(header),
        feature_flag=header.featureFlags.flags,
        ntuple_name=schema.name,
        ntuple_description=schema.description,
        writer_identifier=schema.writer,
        field_records=list(schema.fields[:num_fields]),
        column_records=list(schema.columns[:num_columns]),
        alias_column_records=list(schema.alias_columns[:num_aliases]),
        extra_type_infos=[
            _legacy_extra_type_info(r) for r in header.extraTypeInformations
        ],
        checksum=header.checksum,
    )


def legacy_footer(header, footer, schema):
    num_aliases = len(header.aliasColumnDescriptions)
    extension = footer.schemaExtension
    attribute_sets = footer.attributeSets.items if footer.attributeSets else []
    return types.SimpleNamespace(
        env_header=_legacy_env_header(footer),
        feature_flag=footer.featureFlags.flags,
        header_checksum=footer.headerChecksum,
        extension_links=types.SimpleNamespace(
            size=extension.fSize,
            field_records=list(schema.fields[schema.num_header_fields :]),
            column_records=list(schema.columns[schema.num_header_columns :]),
            alias_column_records=list(schema.alias_columns[num_aliases:]),
            extra_type_info=[
                _legacy_extra_type_info(r) for r in extension.extraTypeInformations
            ],
        ),
        cluster_group_records=[
            types.SimpleNamespace(
                min_entry_num=group.fMinEntryNumber,
                entry_span=group.fEntrySpan,
                num_clusters=group.fNClusters,
                page_list_link=types.SimpleNamespace(
                    env_uncomp_size=group.pagelistLink.length,
                    locator=_legacy_locator(group.pagelistLink.locator),
                ),
            )
            for group in footer.clusterGroups
        ],
        linked_attribute_sets=[
            types.SimpleNamespace(
                schema_version_major=record.fSchemaVersionMajor,
                schema_version_minor=record.fSchemaVersionMinor,
                anchor_uncompressed_size=record.fAnchorLength,
                locator=_legacy_locator(record.locator),
                name=decode(record.fName),
            )
            for record in attribute_sets
        ],
        checksum=footer.checksum,
    )


def _legacy_column_pages(locations):
    compression = locations.compressionsettings
    return types.SimpleNamespace(
        pages=[
            types.SimpleNamespace(
                num_elements=page.n_elements,
                has_checksum=page.has_checksum,
                locator=_legacy_locator(page.locator),
            )
            for page in locations.items
        ],
        element_offset=locations.elementoffset,
        suppressed=locations.elementoffset < 0,
        compression_settings=(
            None if compression is None else compression.compressionsettings
        ),
    )


def legacy_page_list_envelopes(rntuple):
    return [
        types.SimpleNamespace(
            env_header=_legacy_env_header(page_list),
            header_checksum=page_list.headerChecksum,
            cluster_summaries=[
                types.SimpleNamespace(
                    num_first_entry=summary.fFirstEntryNumber,
                    num_entries=summary.fNEntries,
                    flags=uproot.const.RNTupleClusterFlags(summary.fFeatureFlag),
                )
                for summary in page_list.clusterSummaries.items
            ],
            pagelinklist=[
                [_legacy_column_pages(locations) for locations in cluster.items]
                for cluster in page_list.pageLocations.items
            ],
            checksum=page_list.checksum,
        )
        for page_list in rntuple.pagelistEnvelopes
    ]

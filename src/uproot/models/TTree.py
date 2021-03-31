# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

"""
This module defines versioned models for ``TTree``.

See :doc:`uproot.behaviors.TBranch` for definitions of ``TTree``-reading
functions.
"""

from __future__ import absolute_import

import struct

import numpy

import uproot
import uproot.behaviors.TTree

_ttree16_format1 = struct.Struct(">qqqqdiiiqqqqq")


class Model_TTree_v16(uproot.behaviors.TTree.TTree, uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TTree`` version 16.
    """

    def read_members(self, chunk, cursor, context, file):
        if self.is_memberwise:
            raise NotImplementedError(
                """memberwise serialization of {0}
in file {1}""".format(
                    type(self).__name__, self.file.file_path
                )
            )
        self._bases.append(
            file.class_named("TNamed", 1).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._bases.append(
            file.class_named("TAttLine", 1).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._bases.append(
            file.class_named("TAttFill", 1).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._bases.append(
            file.class_named("TAttMarker", 2).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        (
            self._members["fEntries"],
            self._members["fTotBytes"],
            self._members["fZipBytes"],
            self._members["fSavedBytes"],
            self._members["fWeight"],
            self._members["fTimerInterval"],
            self._members["fScanField"],
            self._members["fUpdate"],
            self._members["fMaxEntries"],
            self._members["fMaxEntryLoop"],
            self._members["fMaxVirtualSize"],
            self._members["fAutoSave"],
            self._members["fEstimate"],
        ) = cursor.fields(chunk, _ttree16_format1, context)
        self._members["fBranches"] = file.class_named("TObjArray").read(
            chunk, cursor, context, file, self._file, self.concrete
        )
        self._members["fLeaves"] = file.class_named("TObjArray").read(
            chunk, cursor, context, file, self._file, self.concrete
        )
        self._members["fAliases"] = uproot.deserialization.read_object_any(
            chunk, cursor, context, file, self._file, self.concrete
        )

        if file.options["minimal_ttree_metadata"]:
            cursor.skip_after(self)
        else:
            self._members["fIndexValues"] = file.class_named("TArrayD").read(
                chunk, cursor, context, file, self._file, self.concrete
            )
            self._members["fIndex"] = file.class_named("TArrayI").read(
                chunk, cursor, context, file, self._file, self.concrete
            )
            self._members["fTreeIndex"] = uproot.deserialization.read_object_any(
                chunk, cursor, context, file, self._file, self.concrete
            )
            self._members["fFriends"] = uproot.deserialization.read_object_any(
                chunk, cursor, context, file, self._file, self.concrete
            )
            self._members["fUserInfo"] = uproot.deserialization.read_object_any(
                chunk, cursor, context, file, self._file, self.concrete
            )
            self._members["fBranchRef"] = uproot.deserialization.read_object_any(
                chunk, cursor, context, file, self._file, self.concrete
            )

    @property
    def member_names(self):
        minimal = [
            "fEntries",
            "fTotBytes",
            "fZipBytes",
            "fSavedBytes",
            "fWeight",
            "fTimerInterval",
            "fScanField",
            "fUpdate",
            "fMaxEntries",
            "fMaxEntryLoop",
            "fMaxVirtualSize",
            "fAutoSave",
            "fEstimate",
            "fBranches",
            "fLeaves",
            "fAliases",
        ]
        extra = [
            "fIndexValues",
            "fIndex",
            "fTreeIndex",
            "fFriends",
            "fUserInfo",
            "fBranchRef",
        ]
        if self._file.options["minimal_ttree_metadata"]:
            return minimal
        else:
            return minimal + extra

    base_names_versions = [
        ("TNamed", 1),
        ("TAttLine", 1),
        ("TAttFill", 1),
        ("TAttMarker", 2),
    ]
    class_flags = {"has_read_object_any": True}
    class_code = None


_ttree17_format1 = struct.Struct(">qqqqdiiiiqqqqq")


class Model_TTree_v17(uproot.behaviors.TTree.TTree, uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TTree`` version 17.
    """

    def read_members(self, chunk, cursor, context, file):
        if self.is_memberwise:
            raise NotImplementedError(
                """memberwise serialization of {0}
in file {1}""".format(
                    type(self).__name__, self.file.file_path
                )
            )
        self._bases.append(
            file.class_named("TNamed", 1).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._bases.append(
            file.class_named("TAttLine", 1).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._bases.append(
            file.class_named("TAttFill", 1).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._bases.append(
            file.class_named("TAttMarker", 2).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        (
            self._members["fEntries"],
            self._members["fTotBytes"],
            self._members["fZipBytes"],
            self._members["fSavedBytes"],
            self._members["fWeight"],
            self._members["fTimerInterval"],
            self._members["fScanField"],
            self._members["fUpdate"],
            self._members["fDefaultEntryOffsetLen"],
            self._members["fMaxEntries"],
            self._members["fMaxEntryLoop"],
            self._members["fMaxVirtualSize"],
            self._members["fAutoSave"],
            self._members["fEstimate"],
        ) = cursor.fields(chunk, _ttree17_format1, context)
        self._members["fBranches"] = file.class_named("TObjArray").read(
            chunk, cursor, context, file, self._file, self.concrete
        )
        self._members["fLeaves"] = file.class_named("TObjArray").read(
            chunk, cursor, context, file, self._file, self.concrete
        )
        self._members["fAliases"] = uproot.deserialization.read_object_any(
            chunk, cursor, context, file, self._file, self.concrete
        )
        if file.options["minimal_ttree_metadata"]:
            cursor.skip_after(self)
        else:
            self._members["fIndexValues"] = file.class_named("TArrayD").read(
                chunk, cursor, context, file, self._file, self.concrete
            )
            self._members["fIndex"] = file.class_named("TArrayI").read(
                chunk, cursor, context, file, self._file, self.concrete
            )
            self._members["fTreeIndex"] = uproot.deserialization.read_object_any(
                chunk, cursor, context, file, self._file, self.concrete
            )
            self._members["fFriends"] = uproot.deserialization.read_object_any(
                chunk, cursor, context, file, self._file, self.concrete
            )
            self._members["fUserInfo"] = uproot.deserialization.read_object_any(
                chunk, cursor, context, file, self._file, self.concrete
            )
            self._members["fBranchRef"] = uproot.deserialization.read_object_any(
                chunk, cursor, context, file, self._file, self.concrete
            )

    @property
    def member_names(self):
        minimal = [
            "fEntries",
            "fTotBytes",
            "fZipBytes",
            "fSavedBytes",
            "fWeight",
            "fTimerInterval",
            "fScanField",
            "fUpdate",
            "fDefaultEntryOffsetLen",
            "fMaxEntries",
            "fMaxEntryLoop",
            "fMaxVirtualSize",
            "fAutoSave",
            "fEstimate",
            "fBranches",
            "fLeaves",
            "fAliases",
        ]
        extra = [
            "fIndexValues",
            "fIndex",
            "fTreeIndex",
            "fFriends",
            "fUserInfo",
            "fBranchRef",
        ]
        if self._file.options["minimal_ttree_metadata"]:
            return minimal
        else:
            return minimal + extra

    base_names_versions = [
        ("TNamed", 1),
        ("TAttLine", 1),
        ("TAttFill", 1),
        ("TAttMarker", 2),
    ]
    class_flags = {"has_read_object_any": True}
    class_code = None


_ttree18_format1 = struct.Struct(">qqqqqdiiiiqqqqqq")


class Model_TTree_v18(uproot.behaviors.TTree.TTree, uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TTree`` version 18.
    """

    def read_members(self, chunk, cursor, context, file):
        if self.is_memberwise:
            raise NotImplementedError(
                """memberwise serialization of {0}
in file {1}""".format(
                    type(self).__name__, self.file.file_path
                )
            )
        self._bases.append(
            file.class_named("TNamed", 1).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._bases.append(
            file.class_named("TAttLine", 1).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._bases.append(
            file.class_named("TAttFill", 1).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._bases.append(
            file.class_named("TAttMarker", 2).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        (
            self._members["fEntries"],
            self._members["fTotBytes"],
            self._members["fZipBytes"],
            self._members["fSavedBytes"],
            self._members["fFlushedBytes"],
            self._members["fWeight"],
            self._members["fTimerInterval"],
            self._members["fScanField"],
            self._members["fUpdate"],
            self._members["fDefaultEntryOffsetLen"],
            self._members["fMaxEntries"],
            self._members["fMaxEntryLoop"],
            self._members["fMaxVirtualSize"],
            self._members["fAutoSave"],
            self._members["fAutoFlush"],
            self._members["fEstimate"],
        ) = cursor.fields(chunk, _ttree18_format1, context)
        self._members["fBranches"] = file.class_named("TObjArray").read(
            chunk, cursor, context, file, self._file, self.concrete
        )
        self._members["fLeaves"] = file.class_named("TObjArray").read(
            chunk, cursor, context, file, self._file, self.concrete
        )
        self._members["fAliases"] = uproot.deserialization.read_object_any(
            chunk, cursor, context, file, self._file, self.concrete
        )
        if file.options["minimal_ttree_metadata"]:
            cursor.skip_after(self)
        else:
            self._members["fIndexValues"] = file.class_named("TArrayD").read(
                chunk, cursor, context, file, self._file, self.concrete
            )
            self._members["fIndex"] = file.class_named("TArrayI").read(
                chunk, cursor, context, file, self._file, self.concrete
            )
            self._members["fTreeIndex"] = uproot.deserialization.read_object_any(
                chunk, cursor, context, file, self._file, self.concrete
            )
            self._members["fFriends"] = uproot.deserialization.read_object_any(
                chunk, cursor, context, file, self._file, self.concrete
            )
            self._members["fUserInfo"] = uproot.deserialization.read_object_any(
                chunk, cursor, context, file, self._file, self.concrete
            )
            self._members["fBranchRef"] = uproot.deserialization.read_object_any(
                chunk, cursor, context, file, self._file, self.concrete
            )

    @property
    def member_values(self):
        minimal = [
            "fEntries",
            "fTotBytes",
            "fZipBytes",
            "fSavedBytes",
            "fFlushedBytes",
            "fWeight",
            "fTimerInterval",
            "fScanField",
            "fUpdate",
            "fDefaultEntryOffsetLen",
            "fMaxEntries",
            "fMaxEntryLoop",
            "fMaxVirtualSize",
            "fAutoSave",
            "fAutoFlush",
            "fEstimate",
            "fBranches",
            "fLeaves",
            "fAliases",
        ]
        extra = [
            "fIndexValues",
            "fIndex",
            "fTreeIndex",
            "fFriends",
            "fUserInfo",
            "fBranchRef",
        ]
        if self._file.options["minimal_ttree_metadata"]:
            return minimal
        else:
            return minimal + extra

    base_names_versions = [
        ("TNamed", 1),
        ("TAttLine", 1),
        ("TAttFill", 1),
        ("TAttMarker", 2),
    ]
    class_flags = {"has_read_object_any": True}
    class_code = None


_ttree19_format1 = struct.Struct(">qqqqqdiiiiIqqqqqq")
_ttree19_dtype1 = numpy.dtype(">i8")
_ttree19_dtype2 = numpy.dtype(">i8")


class Model_TTree_v19(uproot.behaviors.TTree.TTree, uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TTree`` version 19.
    """

    def read_members(self, chunk, cursor, context, file):
        if self.is_memberwise:
            raise NotImplementedError(
                """memberwise serialization of {0}
in file {1}""".format(
                    type(self).__name__, self.file.file_path
                )
            )
        self._bases.append(
            file.class_named("TNamed", 1).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._bases.append(
            file.class_named("TAttLine", 1).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._bases.append(
            file.class_named("TAttFill", 1).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._bases.append(
            file.class_named("TAttMarker", 2).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        (
            self._members["fEntries"],
            self._members["fTotBytes"],
            self._members["fZipBytes"],
            self._members["fSavedBytes"],
            self._members["fFlushedBytes"],
            self._members["fWeight"],
            self._members["fTimerInterval"],
            self._members["fScanField"],
            self._members["fUpdate"],
            self._members["fDefaultEntryOffsetLen"],
            self._members["fNClusterRange"],
            self._members["fMaxEntries"],
            self._members["fMaxEntryLoop"],
            self._members["fMaxVirtualSize"],
            self._members["fAutoSave"],
            self._members["fAutoFlush"],
            self._members["fEstimate"],
        ) = cursor.fields(chunk, _ttree19_format1, context)
        tmp = _ttree19_dtype1
        if context.get("speedbump", True):
            cursor.skip(1)
        self._members["fClusterRangeEnd"] = cursor.array(
            chunk, self.member("fNClusterRange"), tmp, context
        )
        tmp = _ttree19_dtype2
        if context.get("speedbump", True):
            cursor.skip(1)
        self._members["fClusterSize"] = cursor.array(
            chunk, self.member("fNClusterRange"), tmp, context
        )
        self._members["fBranches"] = file.class_named("TObjArray").read(
            chunk, cursor, context, file, self._file, self.concrete
        )
        self._members["fLeaves"] = file.class_named("TObjArray").read(
            chunk, cursor, context, file, self._file, self.concrete
        )
        self._members["fAliases"] = uproot.deserialization.read_object_any(
            chunk, cursor, context, file, self._file, self.concrete
        )
        if file.options["minimal_ttree_metadata"]:
            cursor.skip_after(self)
        else:
            self._members["fIndexValues"] = file.class_named("TArrayD").read(
                chunk, cursor, context, file, self._file, self.concrete
            )
            self._members["fIndex"] = file.class_named("TArrayI").read(
                chunk, cursor, context, file, self._file, self.concrete
            )
            self._members["fTreeIndex"] = uproot.deserialization.read_object_any(
                chunk, cursor, context, file, self._file, self.concrete
            )
            self._members["fFriends"] = uproot.deserialization.read_object_any(
                chunk, cursor, context, file, self._file, self.concrete
            )
            self._members["fUserInfo"] = uproot.deserialization.read_object_any(
                chunk, cursor, context, file, self._file, self.concrete
            )
            self._members["fBranchRef"] = uproot.deserialization.read_object_any(
                chunk, cursor, context, file, self._file, self.concrete
            )

    @property
    def member_names(self):
        minimal = [
            "fEntries",
            "fTotBytes",
            "fZipBytes",
            "fSavedBytes",
            "fFlushedBytes",
            "fWeight",
            "fTimerInterval",
            "fScanField",
            "fUpdate",
            "fDefaultEntryOffsetLen",
            "fNClusterRange",
            "fMaxEntries",
            "fMaxEntryLoop",
            "fMaxVirtualSize",
            "fAutoSave",
            "fAutoFlush",
            "fEstimate",
            "fClusterRangeEnd",
            "fClusterSize",
            "fBranches",
            "fLeaves",
            "fAliases",
        ]
        extra = [
            "fIndexValues",
            "fIndex",
            "fTreeIndex",
            "fFriends",
            "fUserInfo",
            "fBranchRef",
        ]
        if self._file.options["minimal_ttree_metadata"]:
            return minimal
        else:
            return minimal + extra

    base_names_versions = [
        ("TNamed", 1),
        ("TAttLine", 1),
        ("TAttFill", 1),
        ("TAttMarker", 2),
    ]
    class_flags = {"has_read_object_any": True}
    class_code = None


_ttree20_format1 = struct.Struct(">qqqqqdiiiiIqqqqqq")
_ttree20_dtype1 = numpy.dtype(">i8")
_ttree20_dtype2 = numpy.dtype(">i8")


class Model_TTree_v20(uproot.behaviors.TTree.TTree, uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TTree`` version 20.
    """

    def read_members(self, chunk, cursor, context, file):
        if self.is_memberwise:
            raise NotImplementedError(
                """memberwise serialization of {0}
in file {1}""".format(
                    type(self).__name__, self.file.file_path
                )
            )
        self._bases.append(
            file.class_named("TNamed", 1).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._bases.append(
            file.class_named("TAttLine", 2).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._bases.append(
            file.class_named("TAttFill", 2).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._bases.append(
            file.class_named("TAttMarker", 2).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        (
            self._members["fEntries"],
            self._members["fTotBytes"],
            self._members["fZipBytes"],
            self._members["fSavedBytes"],
            self._members["fFlushedBytes"],
            self._members["fWeight"],
            self._members["fTimerInterval"],
            self._members["fScanField"],
            self._members["fUpdate"],
            self._members["fDefaultEntryOffsetLen"],
            self._members["fNClusterRange"],
            self._members["fMaxEntries"],
            self._members["fMaxEntryLoop"],
            self._members["fMaxVirtualSize"],
            self._members["fAutoSave"],
            self._members["fAutoFlush"],
            self._members["fEstimate"],
        ) = cursor.fields(chunk, _ttree20_format1, context)
        tmp = _ttree20_dtype1
        if context.get("speedbump", True):
            cursor.skip(1)
        self._members["fClusterRangeEnd"] = cursor.array(
            chunk, self.member("fNClusterRange"), tmp, context
        )
        tmp = _ttree20_dtype2
        if context.get("speedbump", True):
            cursor.skip(1)
        self._members["fClusterSize"] = cursor.array(
            chunk, self.member("fNClusterRange"), tmp, context
        )
        self._members["fIOFeatures"] = file.class_named("ROOT::TIOFeatures").read(
            chunk, cursor, context, file, self._file, self.concrete
        )
        self._members["fBranches"] = file.class_named("TObjArray").read(
            chunk, cursor, context, file, self._file, self.concrete
        )
        self._members["fLeaves"] = file.class_named("TObjArray").read(
            chunk, cursor, context, file, self._file, self.concrete
        )
        self._members["fAliases"] = uproot.deserialization.read_object_any(
            chunk, cursor, context, file, self._file, self.concrete
        )
        if file.options["minimal_ttree_metadata"]:
            cursor.skip_after(self)
        else:
            self._members["fIndexValues"] = file.class_named("TArrayD").read(
                chunk, cursor, context, file, self._file, self.concrete
            )
            self._members["fIndex"] = file.class_named("TArrayI").read(
                chunk, cursor, context, file, self._file, self.concrete
            )
            self._members["fTreeIndex"] = uproot.deserialization.read_object_any(
                chunk, cursor, context, file, self._file, self.concrete
            )
            self._members["fFriends"] = uproot.deserialization.read_object_any(
                chunk, cursor, context, file, self._file, self.concrete
            )
            self._members["fUserInfo"] = uproot.deserialization.read_object_any(
                chunk, cursor, context, file, self._file, self.concrete
            )
            self._members["fBranchRef"] = uproot.deserialization.read_object_any(
                chunk, cursor, context, file, self._file, self.concrete
            )

    @property
    def member_names(self):
        minimal = [
            "fEntries",
            "fTotBytes",
            "fZipBytes",
            "fSavedBytes",
            "fFlushedBytes",
            "fWeight",
            "fTimerInterval",
            "fScanField",
            "fUpdate",
            "fDefaultEntryOffsetLen",
            "fNClusterRange",
            "fMaxEntries",
            "fMaxEntryLoop",
            "fMaxVirtualSize",
            "fAutoSave",
            "fAutoFlush",
            "fEstimate",
            "fClusterRangeEnd",
            "fClusterSize",
            "fIOFeatures",
            "fBranches",
            "fLeaves",
            "fAliases",
        ]
        extra = [
            "fIndexValues",
            "fIndex",
            "fTreeIndex",
            "fFriends",
            "fUserInfo",
            "fBranchRef",
        ]
        if self._file.options["minimal_ttree_metadata"]:
            return minimal
        else:
            return minimal + extra

    base_names_versions = [
        ("TNamed", 1),
        ("TAttLine", 2),
        ("TAttFill", 2),
        ("TAttMarker", 2),
    ]
    class_flags = {"has_read_object_any": True}
    class_code = None


class Model_TTree(uproot.model.DispatchByVersion):
    """
    A :doc:`uproot.model.DispatchByVersion` for ``TTree``.
    """

    known_versions = {
        16: Model_TTree_v16,
        17: Model_TTree_v17,
        18: Model_TTree_v18,
        19: Model_TTree_v19,
        20: Model_TTree_v20,
    }


_tiofeatures_format1 = struct.Struct(">B")


class Model_ROOT_3a3a_TIOFeatures(uproot.model.Model):
    """
    A versionless :doc:`uproot.model.Model` for ``ROOT::TIOFeatures``.
    """

    def read_members(self, chunk, cursor, context, file):
        if self.is_memberwise:
            raise NotImplementedError(
                """memberwise serialization of {0}
in file {1}""".format(
                    type(self).__name__, self.file.file_path
                )
            )
        cursor.skip(4)
        self._members["fIOBits"] = cursor.field(chunk, _tiofeatures_format1, context)


uproot.classes["TTree"] = Model_TTree
uproot.classes["ROOT::TIOFeatures"] = Model_ROOT_3a3a_TIOFeatures

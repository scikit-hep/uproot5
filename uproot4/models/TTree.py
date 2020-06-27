# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import struct

import numpy

import uproot4.model
import uproot4.deserialization
import uproot4.behaviors.TTree


_ttree16_format1 = struct.Struct(">qqqqdiiiqqqqq")


class Model_TTree_v16(uproot4.behaviors.TTree.TTree, uproot4.model.VersionedModel):
    def read_members(self, chunk, cursor, context):
        self._bases.append(
            self.class_named("TNamed", 1).read(
                chunk, cursor, context, self._file, self._parent
            )
        )
        self._bases.append(
            self.class_named("TAttLine", 1).read(
                chunk, cursor, context, self._file, self._parent
            )
        )
        self._bases.append(
            self.class_named("TAttFill", 1).read(
                chunk, cursor, context, self._file, self._parent
            )
        )
        self._bases.append(
            self.class_named("TAttMarker", 2).read(
                chunk, cursor, context, self._file, self._parent
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
        self._members["fBranches"] = self.class_named("TObjArray").read(
            chunk, cursor, context, self._file, self
        )
        self._members["fLeaves"] = self.class_named("TObjArray").read(
            chunk, cursor, context, self._file, self
        )
        self._members["fAliases"] = uproot4.deserialization.read_object_any(
            chunk, cursor, context, self._file, self._parent
        )

        if self._file.options["minimal_ttree_metadata"]:
            cursor.skip_after(self)
        else:
            self._members["fIndexValues"] = self.class_named("TArrayD").read(
                chunk, cursor, context, self._file, self
            )
            self._members["fIndex"] = self.class_named("TArrayI").read(
                chunk, cursor, context, self._file, self
            )
            self._members["fTreeIndex"] = uproot4.deserialization.read_object_any(
                chunk, cursor, context, self._file, self._parent
            )
            self._members["fFriends"] = uproot4.deserialization.read_object_any(
                chunk, cursor, context, self._file, self._parent
            )
            self._members["fUserInfo"] = uproot4.deserialization.read_object_any(
                chunk, cursor, context, self._file, self._parent
            )
            self._members["fBranchRef"] = uproot4.deserialization.read_object_any(
                chunk, cursor, context, self._file, self._parent
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


class Model_TTree_v17(uproot4.behaviors.TTree.TTree, uproot4.model.VersionedModel):
    def read_members(self, chunk, cursor, context):
        self._bases.append(
            self.class_named("TNamed", 1).read(
                chunk, cursor, context, self._file, self._parent
            )
        )
        self._bases.append(
            self.class_named("TAttLine", 1).read(
                chunk, cursor, context, self._file, self._parent
            )
        )
        self._bases.append(
            self.class_named("TAttFill", 1).read(
                chunk, cursor, context, self._file, self._parent
            )
        )
        self._bases.append(
            self.class_named("TAttMarker", 2).read(
                chunk, cursor, context, self._file, self._parent
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
        self._members["fBranches"] = self.class_named("TObjArray").read(
            chunk, cursor, context, self._file, self
        )
        self._members["fLeaves"] = self.class_named("TObjArray").read(
            chunk, cursor, context, self._file, self
        )
        self._members["fAliases"] = uproot4.deserialization.read_object_any(
            chunk, cursor, context, self._file, self._parent
        )
        if self._file.options["minimal_ttree_metadata"]:
            cursor.skip_after(self)
        else:
            self._members["fIndexValues"] = self.class_named("TArrayD").read(
                chunk, cursor, context, self._file, self
            )
            self._members["fIndex"] = self.class_named("TArrayI").read(
                chunk, cursor, context, self._file, self
            )
            self._members["fTreeIndex"] = uproot4.deserialization.read_object_any(
                chunk, cursor, context, self._file, self._parent
            )
            self._members["fFriends"] = uproot4.deserialization.read_object_any(
                chunk, cursor, context, self._file, self._parent
            )
            self._members["fUserInfo"] = uproot4.deserialization.read_object_any(
                chunk, cursor, context, self._file, self._parent
            )
            self._members["fBranchRef"] = uproot4.deserialization.read_object_any(
                chunk, cursor, context, self._file, self._parent
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


class Model_TTree_v18(uproot4.behaviors.TTree.TTree, uproot4.model.VersionedModel):
    def read_members(self, chunk, cursor, context):
        self._bases.append(
            self.class_named("TNamed", 1).read(
                chunk, cursor, context, self._file, self._parent
            )
        )
        self._bases.append(
            self.class_named("TAttLine", 1).read(
                chunk, cursor, context, self._file, self._parent
            )
        )
        self._bases.append(
            self.class_named("TAttFill", 1).read(
                chunk, cursor, context, self._file, self._parent
            )
        )
        self._bases.append(
            self.class_named("TAttMarker", 2).read(
                chunk, cursor, context, self._file, self._parent
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
        self._members["fBranches"] = self.class_named("TObjArray").read(
            chunk, cursor, context, self._file, self
        )
        self._members["fLeaves"] = self.class_named("TObjArray").read(
            chunk, cursor, context, self._file, self
        )
        self._members["fAliases"] = uproot4.deserialization.read_object_any(
            chunk, cursor, context, self._file, self._parent
        )
        if self._file.options["minimal_ttree_metadata"]:
            cursor.skip_after(self)
        else:
            self._members["fIndexValues"] = self.class_named("TArrayD").read(
                chunk, cursor, context, self._file, self
            )
            self._members["fIndex"] = self.class_named("TArrayI").read(
                chunk, cursor, context, self._file, self
            )
            self._members["fTreeIndex"] = uproot4.deserialization.read_object_any(
                chunk, cursor, context, self._file, self._parent
            )
            self._members["fFriends"] = uproot4.deserialization.read_object_any(
                chunk, cursor, context, self._file, self._parent
            )
            self._members["fUserInfo"] = uproot4.deserialization.read_object_any(
                chunk, cursor, context, self._file, self._parent
            )
            self._members["fBranchRef"] = uproot4.deserialization.read_object_any(
                chunk, cursor, context, self._file, self._parent
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


class Model_TTree_v19(uproot4.behaviors.TTree.TTree, uproot4.model.VersionedModel):
    def read_members(self, chunk, cursor, context):
        self._bases.append(
            self.class_named("TNamed", 1).read(
                chunk, cursor, context, self._file, self._parent
            )
        )
        self._bases.append(
            self.class_named("TAttLine", 1).read(
                chunk, cursor, context, self._file, self._parent
            )
        )
        self._bases.append(
            self.class_named("TAttFill", 1).read(
                chunk, cursor, context, self._file, self._parent
            )
        )
        self._bases.append(
            self.class_named("TAttMarker", 2).read(
                chunk, cursor, context, self._file, self._parent
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
        self._members["fBranches"] = self.class_named("TObjArray").read(
            chunk, cursor, context, self._file, self
        )
        self._members["fLeaves"] = self.class_named("TObjArray").read(
            chunk, cursor, context, self._file, self
        )
        self._members["fAliases"] = uproot4.deserialization.read_object_any(
            chunk, cursor, context, self._file, self._parent
        )
        if self._file.options["minimal_ttree_metadata"]:
            cursor.skip_after(self)
        else:
            self._members["fIndexValues"] = self.class_named("TArrayD").read(
                chunk, cursor, context, self._file, self
            )
            self._members["fIndex"] = self.class_named("TArrayI").read(
                chunk, cursor, context, self._file, self
            )
            self._members["fTreeIndex"] = uproot4.deserialization.read_object_any(
                chunk, cursor, context, self._file, self._parent
            )
            self._members["fFriends"] = uproot4.deserialization.read_object_any(
                chunk, cursor, context, self._file, self._parent
            )
            self._members["fUserInfo"] = uproot4.deserialization.read_object_any(
                chunk, cursor, context, self._file, self._parent
            )
            self._members["fBranchRef"] = uproot4.deserialization.read_object_any(
                chunk, cursor, context, self._file, self._parent
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


class Model_TTree_v20(uproot4.behaviors.TTree.TTree, uproot4.model.VersionedModel):
    def read_members(self, chunk, cursor, context):
        self._bases.append(
            self.class_named("TNamed", 1).read(
                chunk, cursor, context, self._file, self._parent
            )
        )
        self._bases.append(
            self.class_named("TAttLine", 2).read(
                chunk, cursor, context, self._file, self._parent
            )
        )
        self._bases.append(
            self.class_named("TAttFill", 2).read(
                chunk, cursor, context, self._file, self._parent
            )
        )
        self._bases.append(
            self.class_named("TAttMarker", 2).read(
                chunk, cursor, context, self._file, self._parent
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
        self._members["fIOFeatures"] = self.class_named("ROOT::TIOFeatures").read(
            chunk, cursor, context, self._file, self
        )
        self._members["fBranches"] = self.class_named("TObjArray").read(
            chunk, cursor, context, self._file, self
        )
        self._members["fLeaves"] = self.class_named("TObjArray").read(
            chunk, cursor, context, self._file, self
        )
        self._members["fAliases"] = uproot4.deserialization.read_object_any(
            chunk, cursor, context, self._file, self._parent
        )
        if self._file.options["minimal_ttree_metadata"]:
            cursor.skip_after(self)
        else:
            self._members["fIndexValues"] = self.class_named("TArrayD").read(
                chunk, cursor, context, self._file, self
            )
            self._members["fIndex"] = self.class_named("TArrayI").read(
                chunk, cursor, context, self._file, self
            )
            self._members["fTreeIndex"] = uproot4.deserialization.read_object_any(
                chunk, cursor, context, self._file, self._parent
            )
            self._members["fFriends"] = uproot4.deserialization.read_object_any(
                chunk, cursor, context, self._file, self._parent
            )
            self._members["fUserInfo"] = uproot4.deserialization.read_object_any(
                chunk, cursor, context, self._file, self._parent
            )
            self._members["fBranchRef"] = uproot4.deserialization.read_object_any(
                chunk, cursor, context, self._file, self._parent
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


class Model_TTree(uproot4.model.DispatchByVersion):
    known_versions = {
        16: Model_TTree_v16,
        17: Model_TTree_v17,
        18: Model_TTree_v18,
        19: Model_TTree_v19,
        20: Model_TTree_v20,
    }


_tiofeatures_format1 = struct.Struct(">B")


class Model_ROOT_3a3a_TIOFeatures(uproot4.model.Model):
    def read_members(self, chunk, cursor, context):
        cursor.skip(4)
        self._members["fIOBits"] = cursor.field(chunk, _tiofeatures_format1, context)


uproot4.classes["TTree"] = Model_TTree
uproot4.classes["ROOT::TIOFeatures"] = Model_ROOT_3a3a_TIOFeatures

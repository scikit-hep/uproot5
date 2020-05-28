# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import struct

# try:
#     from collections.abc import Mapping
# except ImportError:
#     from collections import Mapping

import uproot4.model
import uproot4.deserialization
import uproot4.models.TObject


_ttree16_format1 = struct.Struct(">qqqqdiiiqqqqq")


class Model_TTree_v16(uproot4.model.VersionedModel):
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
        ) = cursor.fields(chunk, _ttree16_format1)
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

    base_names_versions = []
    member_names = [
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
        "fIndexValues",
        "fIndex",
        "fTreeIndex",
        "fFriends",
        "fUserInfo",
        "fBranchRef",
    ]
    class_flags = {"has_read_object_any": True}
    hooks = {}
    class_code = None


class Model_TTree(uproot4.model.DispatchByVersion):
    known_versions = {16: Model_TTree_v16}


_tiofeatures_format1 = struct.Struct(">B")


class Model_ROOT_3a3a_TIOFeatures(uproot4.model.Model):
    def read_members(self, chunk, cursor, context):
        cursor.skip(4)
        self._members["fIOBits"] = cursor.field(chunk, _tiofeatures_format1)


uproot4.classes["TTree"] = Model_TTree
uproot4.classes["ROOT::TIOFeatures"] = Model_ROOT_3a3a_TIOFeatures

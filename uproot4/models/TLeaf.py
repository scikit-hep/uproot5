# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import struct

import uproot4.model
import uproot4.deserialization


_tleaf2_format0 = struct.Struct(">iii??")


class Model_TLeaf_v2(uproot4.model.VersionedModel):
    def read_members(self, chunk, cursor, context):
        self._bases.append(
            self.class_named("TNamed", 1).read(
                chunk, cursor, context, self._file, self._parent
            )
        )
        (
            self._members["fLen"],
            self._members["fLenType"],
            self._members["fOffset"],
            self._members["fIsRange"],
            self._members["fIsUnsigned"],
        ) = cursor.fields(chunk, _tleaf2_format0)
        self._members["fLeafCount"] = uproot4.deserialization.read_object_any(
            chunk, cursor, context, self._file, self._parent
        )

    base_names_versions = [("TNamed", 1)]
    member_names = [
        "fLen",
        "fLenType",
        "fOffset",
        "fIsRange",
        "fIsUnsigned",
        "fLeafCount",
    ]
    class_flags = {"has_read_object_any": True}
    hooks = None


class Model_TLeaf(uproot4.model.DispatchByVersion):
    known_versions = {2: Model_TLeaf_v2}


uproot4.classes["TLeaf"] = Model_TLeaf

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
        ) = cursor.fields(chunk, _tleaf2_format0, context)
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
    class_code = None


class Model_TLeaf(uproot4.model.DispatchByVersion):
    known_versions = {2: Model_TLeaf_v2}


_tleafb1_format1 = struct.Struct(">bb")


class Model_TLeafB_v1(uproot4.model.VersionedModel):
    def read_members(self, chunk, cursor, context):
        self._bases.append(
            self.class_named("TLeaf", 2).read(
                chunk, cursor, context, self._file, self._parent
            )
        )
        self._members["fMinimum"], self._members["fMaximum"] = cursor.fields(
            chunk, _tleafb1_format1, context
        )

    base_names_versions = [("TLeaf", 2)]
    member_names = ["fMinimum", "fMaximum"]
    class_flags = {}
    class_code = None


class Model_TLeafB(uproot4.model.DispatchByVersion):
    known_versions = {1: Model_TLeafB_v1}


_tleafc1_format1 = struct.Struct(">ii")


class Model_TLeafC_v1(uproot4.model.VersionedModel):
    def read_members(self, chunk, cursor, context):
        self._bases.append(
            self.class_named("TLeaf", 2).read(
                chunk, cursor, context, self._file, self._parent
            )
        )
        self._members["fMinimum"], self._members["fMaximum"] = cursor.fields(
            chunk, _tleafc1_format1, context
        )

    base_names_versions = [("TLeaf", 2)]
    member_names = ["fMinimum", "fMaximum"]
    class_flags = {}
    class_code = None


class Model_TLeafC(uproot4.model.DispatchByVersion):
    known_versions = {1: Model_TLeafC_v1}


_tleafd1_format1 = struct.Struct(">dd")


class Model_TLeafD_v1(uproot4.model.VersionedModel):
    def read_members(self, chunk, cursor, context):
        self._bases.append(
            self.class_named("TLeaf", 2).read(
                chunk, cursor, context, self._file, self._parent
            )
        )
        self._members["fMinimum"], self._members["fMaximum"] = cursor.fields(
            chunk, _tleafd1_format1, context
        )

    base_names_versions = [("TLeaf", 2)]
    member_names = ["fMinimum", "fMaximum"]
    class_flags = {}
    class_code = None


class Model_TLeafD(uproot4.model.DispatchByVersion):
    known_versions = {1: Model_TLeafD_v1}


_tleaff1_format1 = struct.Struct(">ff")


class Model_TLeafF_v1(uproot4.model.VersionedModel):
    def read_members(self, chunk, cursor, context):
        self._bases.append(
            self.class_named("TLeaf", 2).read(
                chunk, cursor, context, self._file, self._parent
            )
        )
        self._members["fMinimum"], self._members["fMaximum"] = cursor.fields(
            chunk, _tleaff1_format1, context
        )

    base_names_versions = [("TLeaf", 2)]
    member_names = ["fMinimum", "fMaximum"]
    class_flags = {}
    class_code = None


class Model_TLeafF(uproot4.model.DispatchByVersion):
    known_versions = {1: Model_TLeafF_v1}


_tleafi1_format1 = struct.Struct(">ii")


class Model_TLeafI_v1(uproot4.model.VersionedModel):
    def read_members(self, chunk, cursor, context):
        self._bases.append(
            self.class_named("TLeaf", 2).read(
                chunk, cursor, context, self._file, self._parent
            )
        )
        self._members["fMinimum"], self._members["fMaximum"] = cursor.fields(
            chunk, _tleafi1_format1, context
        )

    base_names_versions = [("TLeaf", 2)]
    member_names = ["fMinimum", "fMaximum"]
    class_flags = {}
    class_code = None


class Model_TLeafI(uproot4.model.DispatchByVersion):
    known_versions = {1: Model_TLeafI_v1}


_tleafl1_format0 = struct.Struct(">qq")


class Model_TLeafL_v1(uproot4.model.VersionedModel):
    def read_members(self, chunk, cursor, context):
        self._bases.append(
            self.class_named("TLeaf", 2).read(
                chunk, cursor, context, self._file, self._parent
            )
        )
        self._members["fMinimum"], self._members["fMaximum"] = cursor.fields(
            chunk, _tleafl1_format0, context
        )

    base_names_versions = [("TLeaf", 2)]
    member_names = ["fMinimum", "fMaximum"]
    class_flags = {}
    class_code = None


class Model_TLeafL(uproot4.model.DispatchByVersion):
    known_versions = {1: Model_TLeafL_v1}


_tleafO1_format1 = struct.Struct(">??")


class Model_TLeafO_v1(uproot4.model.VersionedModel):
    def read_members(self, chunk, cursor, context):
        self._bases.append(
            self.class_named("TLeaf", 2).read(
                chunk, cursor, context, self._file, self._parent
            )
        )
        self._members["fMinimum"], self._members["fMaximum"] = cursor.fields(
            chunk, _tleafO1_format1, context
        )

    base_names_versions = [("TLeaf", 2)]
    member_names = ["fMinimum", "fMaximum"]
    class_flags = {}
    class_code = None


class Model_TLeafO(uproot4.model.DispatchByVersion):
    known_versions = {1: Model_TLeafO_v1}


_tleafs1_format1 = struct.Struct(">hh")


class Model_TLeafS_v1(uproot4.model.VersionedModel):
    def read_members(self, chunk, cursor, context):
        self._bases.append(
            self.class_named("TLeaf", 2).read(
                chunk, cursor, context, self._file, self._parent
            )
        )
        self._members["fMinimum"], self._members["fMaximum"] = cursor.fields(
            chunk, _tleafs1_format1, context
        )

    base_names_versions = [("TLeaf", 2)]
    member_names = ["fMinimum", "fMaximum"]
    class_flags = {}
    class_code = None


class Model_TLeafS(uproot4.model.DispatchByVersion):
    known_versions = {1: Model_TLeafS_v1}


_tleafelement1_format1 = struct.Struct(">ii")


class Model_TLeafElement_v1(uproot4.model.VersionedModel):
    def read_members(self, chunk, cursor, context):
        self._bases.append(
            self.class_named("TLeaf", 2).read(
                chunk, cursor, context, self._file, self._parent
            )
        )
        self._members["fID"], self._members["fType"] = cursor.fields(
            chunk, _tleafelement1_format1, context
        )

    base_names_versions = [("TLeaf", 2)]
    member_names = ["fID", "fType"]
    class_flags = {}
    class_code = None


class Model_TLeafElement(uproot4.model.DispatchByVersion):
    known_versions = {1: Model_TLeafElement_v1}


class Model_TLeafD32_v1(uproot4.model.VersionedModel):
    def read_members(self, chunk, cursor, context):
        self._bases.append(
            self.class_named("TLeaf", 2).read(
                chunk, cursor, context, self._file, self._parent
            )
        )
        self._members["fMinimum"] = cursor.double32(chunk, context)
        self._members["fMaximum"] = cursor.double32(chunk, context)

    base_names_versions = [("TLeaf", 2)]
    member_names = ["fMinimum", "fMaximum"]
    class_flags = {}


class Model_TLeafD32(uproot4.model.DispatchByVersion):
    known_versions = {1: Model_TLeafD32_v1}


class Model_TLeafF16_v1(uproot4.model.VersionedModel):
    def read_members(self, chunk, cursor, context):
        self._bases.append(
            self.class_named("TLeaf", 2).read(
                chunk, cursor, context, self._file, self._parent
            )
        )
        self._members["fMinimum"] = cursor.float16(chunk, 12, context)
        self._members["fMaximum"] = cursor.float16(chunk, 12, context)

    base_names_versions = [("TLeaf", 2)]
    member_names = ["fMinimum", "fMaximum"]
    class_flags = {}


class Model_TLeafF16(uproot4.model.DispatchByVersion):
    known_versions = {1: Model_TLeafF16_v1}


uproot4.classes["TLeaf"] = Model_TLeaf
uproot4.classes["TLeafB"] = Model_TLeafB
uproot4.classes["TLeafC"] = Model_TLeafC
uproot4.classes["TLeafD"] = Model_TLeafD
uproot4.classes["TLeafF"] = Model_TLeafF
uproot4.classes["TLeafI"] = Model_TLeafI
uproot4.classes["TLeafL"] = Model_TLeafL
uproot4.classes["TLeafO"] = Model_TLeafO
uproot4.classes["TLeafS"] = Model_TLeafS
uproot4.classes["TLeafElement"] = Model_TLeafElement
uproot4.classes["TLeafD32"] = Model_TLeafD32
uproot4.classes["TLeafF16"] = Model_TLeafF16

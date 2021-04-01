# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

"""
This module defines versioned models for ``TLeaf`` and its subclasses.
"""

from __future__ import absolute_import

import struct

import uproot

_tleaf2_format0 = struct.Struct(">iii??")


class Model_TLeaf_v2(uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TLeaf`` version 2.
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
        (
            self._members["fLen"],
            self._members["fLenType"],
            self._members["fOffset"],
            self._members["fIsRange"],
            self._members["fIsUnsigned"],
        ) = cursor.fields(chunk, _tleaf2_format0, context)
        self._members["fLeafCount"] = uproot.deserialization.read_object_any(
            chunk, cursor, context, file, self._file, self.concrete
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


class Model_TLeaf(uproot.model.DispatchByVersion):
    """
    A :doc:`uproot.model.DispatchByVersion` for ``TLeaf``.
    """

    known_versions = {2: Model_TLeaf_v2}


_tleafO1_format1 = struct.Struct(">??")


class Model_TLeafO_v1(uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TLeafO`` version 1
    (``numpy.bool_``).
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
            file.class_named("TLeaf", 2).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._members["fMinimum"], self._members["fMaximum"] = cursor.fields(
            chunk, _tleafO1_format1, context
        )

    base_names_versions = [("TLeaf", 2)]
    member_names = ["fMinimum", "fMaximum"]
    class_flags = {}
    class_code = None


class Model_TLeafO(uproot.model.DispatchByVersion):
    """
    A :doc:`uproot.model.DispatchByVersion` for ``TLeafO`` (``numpy.bool_``).
    """

    known_versions = {1: Model_TLeafO_v1}


_tleafb1_format1 = struct.Struct(">bb")


class Model_TLeafB_v1(uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TLeafB`` version 1
    (``numpy.int8``).
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
            file.class_named("TLeaf", 2).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._members["fMinimum"], self._members["fMaximum"] = cursor.fields(
            chunk, _tleafb1_format1, context
        )

    base_names_versions = [("TLeaf", 2)]
    member_names = ["fMinimum", "fMaximum"]
    class_flags = {}
    class_code = None


class Model_TLeafB(uproot.model.DispatchByVersion):
    """
    A :doc:`uproot.model.DispatchByVersion` for ``TLeafB`` (``numpy.int8``).
    """

    known_versions = {1: Model_TLeafB_v1}


_tleafs1_format1 = struct.Struct(">hh")


class Model_TLeafS_v1(uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TLeafS`` version 1
    (``numpy.int16``).
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
            file.class_named("TLeaf", 2).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._members["fMinimum"], self._members["fMaximum"] = cursor.fields(
            chunk, _tleafs1_format1, context
        )

    base_names_versions = [("TLeaf", 2)]
    member_names = ["fMinimum", "fMaximum"]
    class_flags = {}
    class_code = None


class Model_TLeafS(uproot.model.DispatchByVersion):
    """
    A :doc:`uproot.model.DispatchByVersion` for ``TLeafS`` (``numpy.int16``).
    """

    known_versions = {1: Model_TLeafS_v1}


_tleafi1_format1 = struct.Struct(">ii")


class Model_TLeafI_v1(uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TLeafI`` version 1
    (``numpy.int32``).
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
            file.class_named("TLeaf", 2).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._members["fMinimum"], self._members["fMaximum"] = cursor.fields(
            chunk, _tleafi1_format1, context
        )

    base_names_versions = [("TLeaf", 2)]
    member_names = ["fMinimum", "fMaximum"]
    class_flags = {}
    class_code = None


class Model_TLeafI(uproot.model.DispatchByVersion):
    """
    A :doc:`uproot.model.DispatchByVersion` for ``TLeafI`` (``numpy.int32``).
    """

    known_versions = {1: Model_TLeafI_v1}


_tleafl1_format0 = struct.Struct(">qq")


class Model_TLeafL_v1(uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TLeafL`` version 1
    (``numpy.int64``).
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
            file.class_named("TLeaf", 2).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._members["fMinimum"], self._members["fMaximum"] = cursor.fields(
            chunk, _tleafl1_format0, context
        )

    base_names_versions = [("TLeaf", 2)]
    member_names = ["fMinimum", "fMaximum"]
    class_flags = {}
    class_code = None


class Model_TLeafL(uproot.model.DispatchByVersion):
    """
    A :doc:`uproot.model.DispatchByVersion` for ``TLeafL`` (``numpy.int64``).
    """

    known_versions = {1: Model_TLeafL_v1}


_tleaff1_format1 = struct.Struct(">ff")


class Model_TLeafF_v1(uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TLeafF`` version 1
    (``numpy.float32``).
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
            file.class_named("TLeaf", 2).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._members["fMinimum"], self._members["fMaximum"] = cursor.fields(
            chunk, _tleaff1_format1, context
        )

    base_names_versions = [("TLeaf", 2)]
    member_names = ["fMinimum", "fMaximum"]
    class_flags = {}
    class_code = None


class Model_TLeafF(uproot.model.DispatchByVersion):
    """
    A :doc:`uproot.model.DispatchByVersion` for ``TLeafF`` (``numpy.float32``).
    """

    known_versions = {1: Model_TLeafF_v1}


_tleafd1_format1 = struct.Struct(">dd")


class Model_TLeafD_v1(uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TLeafD`` version 1
    (``numpy.float64``).
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
            file.class_named("TLeaf", 2).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._members["fMinimum"], self._members["fMaximum"] = cursor.fields(
            chunk, _tleafd1_format1, context
        )

    base_names_versions = [("TLeaf", 2)]
    member_names = ["fMinimum", "fMaximum"]
    class_flags = {}
    class_code = None


class Model_TLeafD(uproot.model.DispatchByVersion):
    """
    A :doc:`uproot.model.DispatchByVersion` for ``TLeafD`` (``numpy.float64``).
    """

    known_versions = {1: Model_TLeafD_v1}


_tleafc1_format1 = struct.Struct(">ii")


class Model_TLeafC_v1(uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TLeafC`` version 1
    (variable-length strings).
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
            file.class_named("TLeaf", 2).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._members["fMinimum"], self._members["fMaximum"] = cursor.fields(
            chunk, _tleafc1_format1, context
        )

    base_names_versions = [("TLeaf", 2)]
    member_names = ["fMinimum", "fMaximum"]
    class_flags = {}
    class_code = None


class Model_TLeafC(uproot.model.DispatchByVersion):
    """
    A :doc:`uproot.model.DispatchByVersion` for ``TLeafC`` (variable-length
    strings).
    """

    known_versions = {1: Model_TLeafC_v1}


class Model_TLeafF16_v1(uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TLeafF16`` version 1
    (ROOT's ``Float16_t``).
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
            file.class_named("TLeaf", 2).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._members["fMinimum"] = cursor.float16(chunk, 12, context)
        self._members["fMaximum"] = cursor.float16(chunk, 12, context)

    base_names_versions = [("TLeaf", 2)]
    member_names = ["fMinimum", "fMaximum"]
    class_flags = {}


class Model_TLeafF16(uproot.model.DispatchByVersion):
    """
    A :doc:`uproot.model.DispatchByVersion` for ``TLeafF16`` (ROOT's
    ``Float16_t``).
    """

    known_versions = {1: Model_TLeafF16_v1}


class Model_TLeafD32_v1(uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TLeafD32`` version 1
    (ROOT's ``Double32_t``).
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
            file.class_named("TLeaf", 2).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._members["fMinimum"] = cursor.double32(chunk, context)
        self._members["fMaximum"] = cursor.double32(chunk, context)

    base_names_versions = [("TLeaf", 2)]
    member_names = ["fMinimum", "fMaximum"]
    class_flags = {}


class Model_TLeafD32(uproot.model.DispatchByVersion):
    """
    A :doc:`uproot.model.DispatchByVersion` for ``TLeafD32`` (ROOT's
    ``Double32_t``).
    """

    known_versions = {1: Model_TLeafD32_v1}


_tleafelement1_format1 = struct.Struct(">ii")


class Model_TLeafElement_v1(uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TLeafElement`` version 1
    (arbitrary objects, associated with ``TBranchElement``).
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
            file.class_named("TLeaf", 2).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._members["fID"], self._members["fType"] = cursor.fields(
            chunk, _tleafelement1_format1, context
        )

    base_names_versions = [("TLeaf", 2)]
    member_names = ["fID", "fType"]
    class_flags = {}
    class_code = None


class Model_TLeafElement(uproot.model.DispatchByVersion):
    """
    A :doc:`uproot.model.DispatchByVersion` for ``TLeafElement``
    (arbitrary objects, associated with ``TBranchElement``).
    """

    known_versions = {1: Model_TLeafElement_v1}


uproot.classes["TLeaf"] = Model_TLeaf
uproot.classes["TLeafB"] = Model_TLeafB
uproot.classes["TLeafS"] = Model_TLeafS
uproot.classes["TLeafI"] = Model_TLeafI
uproot.classes["TLeafL"] = Model_TLeafL
uproot.classes["TLeafF"] = Model_TLeafF
uproot.classes["TLeafD"] = Model_TLeafD
uproot.classes["TLeafC"] = Model_TLeafC
uproot.classes["TLeafO"] = Model_TLeafO
uproot.classes["TLeafF16"] = Model_TLeafF16
uproot.classes["TLeafD32"] = Model_TLeafD32
uproot.classes["TLeafElement"] = Model_TLeafElement

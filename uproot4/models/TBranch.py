# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import struct

import numpy

import uproot4.model
import uproot4.deserialization
import uproot4.behaviors.TBranch


_tbranch10_format1 = struct.Struct(">iiiiqiIiqqq")
_tbranch10_dtype1 = numpy.dtype(">i4")
_tbranch10_dtype2 = numpy.dtype(">i8")
_tbranch10_dtype3 = numpy.dtype(">i8")


class Model_TBranch_v10(
    uproot4.behaviors.TBranch.TBranch, uproot4.model.VersionedModel
):
    def read_members(self, chunk, cursor, context):
        self._bases.append(
            self.class_named("TNamed", 1).read(
                chunk, cursor, context, self._file, self._parent
            )
        )
        self._bases.append(
            self.class_named("TAttFill", 1).read(
                chunk, cursor, context, self._file, self._parent
            )
        )
        (
            self._members["fCompress"],
            self._members["fBasketSize"],
            self._members["fEntryOffsetLen"],
            self._members["fWriteBasket"],
            self._members["fEntryNumber"],
            self._members["fOffset"],
            self._members["fMaxBaskets"],
            self._members["fSplitLevel"],
            self._members["fEntries"],
            self._members["fTotBytes"],
            self._members["fZipBytes"],
        ) = cursor.fields(chunk, _tbranch10_format1, context)
        self._members["fBranches"] = self.class_named("TObjArray").read(
            chunk, cursor, context, self._file, self
        )
        self._members["fLeaves"] = self.class_named("TObjArray").read(
            chunk, cursor, context, self._file, self
        )
        self._cursor_baskets = cursor.copy()
        if self._file.options["minimal_ttree_metadata"]:
            cursor.skip_over(chunk, context)
        else:
            self._members["fBaskets"] = self.class_named("TObjArray").read(
                chunk, cursor, context, self._file, self
            )
        tmp = _tbranch10_dtype1
        if context.get("speedbump", True):
            cursor.skip(1)
        self._members["fBasketBytes"] = cursor.array(
            chunk, self.member("fMaxBaskets"), tmp, context
        )
        tmp = _tbranch10_dtype2
        if context.get("speedbump", True):
            cursor.skip(1)
        self._members["fBasketEntry"] = cursor.array(
            chunk, self.member("fMaxBaskets"), tmp, context
        )
        tmp = _tbranch10_dtype3
        if context.get("speedbump", True):
            if cursor.bytes(chunk, 1, context)[0] == 2:
                tmp = numpy.dtype(">i8")
        self._members["fBasketSeek"] = cursor.array(
            chunk, self.member("fMaxBaskets"), tmp, context
        )
        if self._file.options["minimal_ttree_metadata"]:
            cursor.skip_after(self)
        else:
            self._members["fFileName"] = self.class_named("TString").read(
                chunk, cursor, context, self._file, self
            )

    @property
    def member_names(self):
        out = [
            "fCompress",
            "fBasketSize",
            "fEntryOffsetLen",
            "fWriteBasket",
            "fEntryNumber",
            "fOffset",
            "fMaxBaskets",
            "fSplitLevel",
            "fEntries",
            "fTotBytes",
            "fZipBytes",
            "fBranches",
            "fLeaves",
            "fBaskets",
            "fBasketBytes",
            "fBasketEntry",
            "fBasketSeek",
            "fFileName",
        ]
        if self._file.options["minimal_ttree_metadata"]:
            out.remove("fBaskets")
            out.remove("fFileName")
        return out

    base_names_versions = [("TNamed", 1), ("TAttFill", 1)]
    class_flags = {}
    class_code = None


_tbranch11_format1 = struct.Struct(">iiiiqiIiqqqq")
_tbranch11_dtype1 = numpy.dtype(">i4")
_tbranch11_dtype2 = numpy.dtype(">i8")
_tbranch11_dtype3 = numpy.dtype(">i8")


class Model_TBranch_v11(
    uproot4.behaviors.TBranch.TBranch, uproot4.model.VersionedModel
):
    def read_members(self, chunk, cursor, context):
        self._bases.append(
            self.class_named("TNamed", 1).read(
                chunk, cursor, context, self._file, self._parent
            )
        )
        self._bases.append(
            self.class_named("TAttFill", 1).read(
                chunk, cursor, context, self._file, self._parent
            )
        )
        (
            self._members["fCompress"],
            self._members["fBasketSize"],
            self._members["fEntryOffsetLen"],
            self._members["fWriteBasket"],
            self._members["fEntryNumber"],
            self._members["fOffset"],
            self._members["fMaxBaskets"],
            self._members["fSplitLevel"],
            self._members["fEntries"],
            self._members["fFirstEntry"],
            self._members["fTotBytes"],
            self._members["fZipBytes"],
        ) = cursor.fields(chunk, _tbranch11_format1, context)
        self._members["fBranches"] = self.class_named("TObjArray").read(
            chunk, cursor, context, self._file, self
        )
        self._members["fLeaves"] = self.class_named("TObjArray").read(
            chunk, cursor, context, self._file, self
        )
        self._cursor_baskets = cursor.copy()
        if self._file.options["minimal_ttree_metadata"]:
            cursor.skip_over(chunk, context)
        else:
            self._members["fBaskets"] = self.class_named("TObjArray").read(
                chunk, cursor, context, self._file, self
            )
        tmp = _tbranch11_dtype1
        if context.get("speedbump", True):
            cursor.skip(1)
        self._members["fBasketBytes"] = cursor.array(
            chunk, self.member("fMaxBaskets"), tmp, context
        )
        tmp = _tbranch11_dtype2
        if context.get("speedbump", True):
            cursor.skip(1)
        self._members["fBasketEntry"] = cursor.array(
            chunk, self.member("fMaxBaskets"), tmp, context
        )
        tmp = _tbranch11_dtype3
        if context.get("speedbump", True):
            if cursor.bytes(chunk, 1, context)[0] == 2:
                tmp = numpy.dtype(">i8")
        self._members["fBasketSeek"] = cursor.array(
            chunk, self.member("fMaxBaskets"), tmp, context
        )
        if self._file.options["minimal_ttree_metadata"]:
            cursor.skip_after(self)
        else:
            self._members["fFileName"] = self.class_named("TString").read(
                chunk, cursor, context, self._file, self
            )

    @property
    def member_names(self):
        out = [
            "fCompress",
            "fBasketSize",
            "fEntryOffsetLen",
            "fWriteBasket",
            "fEntryNumber",
            "fOffset",
            "fMaxBaskets",
            "fSplitLevel",
            "fEntries",
            "fFirstEntry",
            "fTotBytes",
            "fZipBytes",
            "fBranches",
            "fLeaves",
            "fBaskets",
            "fBasketBytes",
            "fBasketEntry",
            "fBasketSeek",
            "fFileName",
        ]
        if self._file.options["minimal_ttree_metadata"]:
            out.remove("fBaskets")
            out.remove("fFileName")
        return out

    base_names_versions = [("TNamed", 1), ("TAttFill", 1)]
    class_flags = {}
    class_code = None


_tbranch12_format1 = struct.Struct(">iiiiqiIiqqqq")
_tbranch12_dtype1 = numpy.dtype(">i4")
_tbranch12_dtype2 = numpy.dtype(">i8")
_tbranch12_dtype3 = numpy.dtype(">i8")


class Model_TBranch_v12(
    uproot4.behaviors.TBranch.TBranch, uproot4.model.VersionedModel
):
    def read_members(self, chunk, cursor, context):
        self._bases.append(
            self.class_named("TNamed", 1).read(
                chunk, cursor, context, self._file, self._parent
            )
        )
        self._bases.append(
            self.class_named("TAttFill", 1).read(
                chunk, cursor, context, self._file, self._parent
            )
        )
        (
            self._members["fCompress"],
            self._members["fBasketSize"],
            self._members["fEntryOffsetLen"],
            self._members["fWriteBasket"],
            self._members["fEntryNumber"],
            self._members["fOffset"],
            self._members["fMaxBaskets"],
            self._members["fSplitLevel"],
            self._members["fEntries"],
            self._members["fFirstEntry"],
            self._members["fTotBytes"],
            self._members["fZipBytes"],
        ) = cursor.fields(chunk, _tbranch12_format1, context)
        self._members["fBranches"] = self.class_named("TObjArray").read(
            chunk, cursor, context, self._file, self
        )
        self._members["fLeaves"] = self.class_named("TObjArray").read(
            chunk, cursor, context, self._file, self
        )
        self._cursor_baskets = cursor.copy()
        if self._file.options["minimal_ttree_metadata"]:
            cursor.skip_over(chunk, context)
        else:
            self._members["fBaskets"] = self.class_named("TObjArray").read(
                chunk, cursor, context, self._file, self
            )
        tmp = _tbranch12_dtype1
        if context.get("speedbump", True):
            cursor.skip(1)
        self._members["fBasketBytes"] = cursor.array(
            chunk, self.member("fMaxBaskets"), tmp, context
        )
        tmp = _tbranch12_dtype2
        if context.get("speedbump", True):
            cursor.skip(1)
        self._members["fBasketEntry"] = cursor.array(
            chunk, self.member("fMaxBaskets"), tmp, context
        )
        tmp = _tbranch12_dtype3
        if context.get("speedbump", True):
            if cursor.bytes(chunk, 1, context)[0] == 2:
                tmp = numpy.dtype(">i8")
        self._members["fBasketSeek"] = cursor.array(
            chunk, self.member("fMaxBaskets"), tmp, context
        )
        if self._file.options["minimal_ttree_metadata"]:
            cursor.skip_after(self)
        else:
            self._members["fFileName"] = self.class_named("TString").read(
                chunk, cursor, context, self._file, self
            )

    @property
    def member_names(self):
        out = [
            "fCompress",
            "fBasketSize",
            "fEntryOffsetLen",
            "fWriteBasket",
            "fEntryNumber",
            "fOffset",
            "fMaxBaskets",
            "fSplitLevel",
            "fEntries",
            "fFirstEntry",
            "fTotBytes",
            "fZipBytes",
            "fBranches",
            "fLeaves",
            "fBaskets",
            "fBasketBytes",
            "fBasketEntry",
            "fBasketSeek",
            "fFileName",
        ]
        if self._file.options["minimal_ttree_metadata"]:
            out.remove("fBaskets")
            out.remove("fFileName")
        return out

    base_names_versions = [("TNamed", 1), ("TAttFill", 1)]
    class_flags = {}
    class_code = None


_tbranch13_format1 = struct.Struct(">iiiiq")
_tbranch13_format2 = struct.Struct(">iIiqqqq")
_tbranch13_dtype1 = numpy.dtype(">i4")
_tbranch13_dtype2 = numpy.dtype(">i8")
_tbranch13_dtype3 = numpy.dtype(">i8")


class Model_TBranch_v13(
    uproot4.behaviors.TBranch.TBranch, uproot4.model.VersionedModel
):
    def read_members(self, chunk, cursor, context):
        self._bases.append(
            self.class_named("TNamed", 1).read(
                chunk, cursor, context, self._file, self._parent
            )
        )
        self._bases.append(
            self.class_named("TAttFill", 2).read(
                chunk, cursor, context, self._file, self._parent
            )
        )
        (
            self._members["fCompress"],
            self._members["fBasketSize"],
            self._members["fEntryOffsetLen"],
            self._members["fWriteBasket"],
            self._members["fEntryNumber"],
        ) = cursor.fields(chunk, _tbranch13_format1, context)
        self._members["fIOFeatures"] = self.class_named("ROOT::TIOFeatures").read(
            chunk, cursor, context, self._file, self
        )
        (
            self._members["fOffset"],
            self._members["fMaxBaskets"],
            self._members["fSplitLevel"],
            self._members["fEntries"],
            self._members["fFirstEntry"],
            self._members["fTotBytes"],
            self._members["fZipBytes"],
        ) = cursor.fields(chunk, _tbranch13_format2, context)
        self._members["fBranches"] = self.class_named("TObjArray").read(
            chunk, cursor, context, self._file, self
        )
        self._members["fLeaves"] = self.class_named("TObjArray").read(
            chunk, cursor, context, self._file, self
        )
        self._cursor_baskets = cursor.copy()
        if self._file.options["minimal_ttree_metadata"]:
            cursor.skip_over(chunk, context)
        else:
            self._members["fBaskets"] = self.class_named("TObjArray").read(
                chunk, cursor, context, self._file, self
            )
        tmp = _tbranch13_dtype1
        if context.get("speedbump", True):
            cursor.skip(1)
        self._members["fBasketBytes"] = cursor.array(
            chunk, self.member("fMaxBaskets"), tmp, context
        )
        tmp = _tbranch13_dtype2
        if context.get("speedbump", True):
            cursor.skip(1)
        self._members["fBasketEntry"] = cursor.array(
            chunk, self.member("fMaxBaskets"), tmp, context
        )
        tmp = _tbranch13_dtype3
        if context.get("speedbump", True):
            if cursor.bytes(chunk, 1, context)[0] == 2:
                tmp = numpy.dtype(">i8")
        self._members["fBasketSeek"] = cursor.array(
            chunk, self.member("fMaxBaskets"), tmp, context
        )
        if self._file.options["minimal_ttree_metadata"]:
            cursor.skip_after(self)
        else:
            self._members["fFileName"] = self.class_named("TString").read(
                chunk, cursor, context, self._file, self
            )

    @property
    def member_names(self):
        out = [
            "fCompress",
            "fBasketSize",
            "fEntryOffsetLen",
            "fWriteBasket",
            "fEntryNumber",
            "fIOFeatures",
            "fOffset",
            "fMaxBaskets",
            "fSplitLevel",
            "fEntries",
            "fFirstEntry",
            "fTotBytes",
            "fZipBytes",
            "fBranches",
            "fLeaves",
            "fBaskets",
            "fBasketBytes",
            "fBasketEntry",
            "fBasketSeek",
            "fFileName",
        ]
        if self._file.options["minimal_ttree_metadata"]:
            out.remove("fBaskets")
            out.remove("fFileName")
        return out

    base_names_versions = [("TNamed", 1), ("TAttFill", 2)]
    class_flags = {}
    class_code = None


class Model_TBranch(uproot4.model.DispatchByVersion):
    known_versions = {
        10: Model_TBranch_v10,
        11: Model_TBranch_v11,
        12: Model_TBranch_v12,
        13: Model_TBranch_v13,
    }


_tbranchelement8_format1 = struct.Struct(">Iiiiii")


class Model_TBranchElement_v8(
    uproot4.behaviors.TBranch.TBranch, uproot4.model.VersionedModel
):
    def read_members(self, chunk, cursor, context):
        self._bases.append(
            self.class_named("TBranch", 10).read(
                chunk, cursor, context, self._file, self._parent
            )
        )
        self._cursor_baskets = self._bases[0]._cursor_baskets
        self._members["fClassName"] = self.class_named("TString").read(
            chunk, cursor, context, self._file, self
        )
        self._members["fParentName"] = self.class_named("TString").read(
            chunk, cursor, context, self._file, self
        )
        self._members["fClonesName"] = self.class_named("TString").read(
            chunk, cursor, context, self._file, self
        )
        (
            self._members["fCheckSum"],
            self._members["fClassVersion"],
            self._members["fID"],
            self._members["fType"],
            self._members["fStreamerType"],
            self._members["fMaximum"],
        ) = cursor.fields(chunk, _tbranchelement8_format1, context)
        self._members["fBranchCount"] = uproot4.deserialization.read_object_any(
            chunk, cursor, context, self._file, self._parent
        )
        self._members["fBranchCount2"] = uproot4.deserialization.read_object_any(
            chunk, cursor, context, self._file, self._parent
        )

    base_names_versions = [("TBranch", 10)]
    member_names = [
        "fClassName",
        "fParentName",
        "fClonesName",
        "fCheckSum",
        "fClassVersion",
        "fID",
        "fType",
        "fStreamerType",
        "fMaximum",
        "fBranchCount",
        "fBranchCount2",
    ]
    class_flags = {"has_read_object_any": True}
    class_code = None


_tbranchelement9_format1 = struct.Struct(">Iiiiii")


class Model_TBranchElement_v9(
    uproot4.behaviors.TBranch.TBranch, uproot4.model.VersionedModel
):
    def read_members(self, chunk, cursor, context):
        self._bases.append(
            self.class_named("TBranch", 12).read(
                chunk, cursor, context, self._file, self._parent
            )
        )
        self._cursor_baskets = self._bases[0]._cursor_baskets
        self._members["fClassName"] = self.class_named("TString").read(
            chunk, cursor, context, self._file, self
        )
        self._members["fParentName"] = self.class_named("TString").read(
            chunk, cursor, context, self._file, self
        )
        self._members["fClonesName"] = self.class_named("TString").read(
            chunk, cursor, context, self._file, self
        )
        (
            self._members["fCheckSum"],
            self._members["fClassVersion"],
            self._members["fID"],
            self._members["fType"],
            self._members["fStreamerType"],
            self._members["fMaximum"],
        ) = cursor.fields(chunk, _tbranchelement9_format1, context)
        self._members["fBranchCount"] = uproot4.deserialization.read_object_any(
            chunk, cursor, context, self._file, self._parent
        )
        self._members["fBranchCount2"] = uproot4.deserialization.read_object_any(
            chunk, cursor, context, self._file, self._parent
        )

    base_names_versions = [("TBranch", 12)]
    member_names = [
        "fClassName",
        "fParentName",
        "fClonesName",
        "fCheckSum",
        "fClassVersion",
        "fID",
        "fType",
        "fStreamerType",
        "fMaximum",
        "fBranchCount",
        "fBranchCount2",
    ]
    class_flags = {"has_read_object_any": True}
    class_code = None


_tbranchelement10_format1 = struct.Struct(">Ihiiii")


class Model_TBranchElement_v10(
    uproot4.behaviors.TBranch.TBranch, uproot4.model.VersionedModel
):
    def read_members(self, chunk, cursor, context):
        self._bases.append(
            self.class_named("TBranch", 12).read(
                chunk, cursor, context, self._file, self._parent
            )
        )
        self._cursor_baskets = self._bases[0]._cursor_baskets
        self._members["fClassName"] = self.class_named("TString").read(
            chunk, cursor, context, self._file, self
        )
        self._members["fParentName"] = self.class_named("TString").read(
            chunk, cursor, context, self._file, self
        )
        self._members["fClonesName"] = self.class_named("TString").read(
            chunk, cursor, context, self._file, self
        )
        (
            self._members["fCheckSum"],
            self._members["fClassVersion"],
            self._members["fID"],
            self._members["fType"],
            self._members["fStreamerType"],
            self._members["fMaximum"],
        ) = cursor.fields(chunk, _tbranchelement10_format1, context)
        self._members["fBranchCount"] = uproot4.deserialization.read_object_any(
            chunk, cursor, context, self._file, self._parent
        )
        self._members["fBranchCount2"] = uproot4.deserialization.read_object_any(
            chunk, cursor, context, self._file, self._parent
        )

    base_names_versions = [("TBranch", 12)]
    member_names = [
        "fClassName",
        "fParentName",
        "fClonesName",
        "fCheckSum",
        "fClassVersion",
        "fID",
        "fType",
        "fStreamerType",
        "fMaximum",
        "fBranchCount",
        "fBranchCount2",
    ]
    class_flags = {"has_read_object_any": True}
    class_code = None


class Model_TBranchElement(uproot4.model.DispatchByVersion):
    known_versions = {
        8: Model_TBranchElement_v8,
        9: Model_TBranchElement_v9,
        10: Model_TBranchElement_v10,
    }


uproot4.classes["TBranch"] = Model_TBranch
uproot4.classes["TBranchElement"] = Model_TBranchElement

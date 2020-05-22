# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import struct
import re

import numpy

import uproot4.model
import uproot4.const
import uproot4.deserialization


_canonical_typename_patterns = [
    (re.compile(r"\bChar_t\b"), "char"),
    (re.compile(r"\bUChar_t\b"), "unsigned char"),
    (re.compile(r"\bShort_t\b"), "short"),
    (re.compile(r"\bUShort_t\b"), "unsigned short"),
    (re.compile(r"\bInt_t\b"), "int"),
    (re.compile(r"\bUInt_t\b"), "unsigned int"),
    (re.compile(r"\bSeek_t\b"), "int"),  # file pointer
    (re.compile(r"\bLong_t\b"), "long"),
    (re.compile(r"\bULong_t\b"), "unsigned long"),
    (re.compile(r"\bFloat_t\b"), "float"),
    (re.compile(r"\bFloat16_t\b"), "float"),  # 32-bit, written as 16, trunc mantissa
    (re.compile(r"\bDouble_t\b"), "double"),
    (re.compile(r"\bDouble32_t\b"), "double"),  # 64-bit, written as 32
    (re.compile(r"\bLongDouble_t\b"), "long double"),
    (re.compile(r"\bText_t\b"), "char"),
    (re.compile(r"\bBool_t\b"), "bool"),
    (re.compile(r"\bByte_t\b"), "unsigned char"),
    (re.compile(r"\bVersion_t\b"), "short"),  # class version id
    (re.compile(r"\bOption_t\b"), "const char"),  # option string
    (re.compile(r"\bSsiz_t\b"), "int"),  # string size
    (re.compile(r"\bReal_t\b"), "float"),  # TVector/TMatrix element
    (re.compile(r"\bLong64_t\b"), "long long"),  # portable int64
    (re.compile(r"\bULong64_t\b"), "unsigned long long"),  # portable uint64
    (re.compile(r"\bAxis_t\b"), "double"),  # axis values type
    (re.compile(r"\bStat_t\b"), "double"),  # statistics type
    (re.compile(r"\bFont_t\b"), "short"),  # font number
    (re.compile(r"\bStyle_t\b"), "short"),  # style number
    (re.compile(r"\bMarker_t\b"), "short"),  # marker number
    (re.compile(r"\bWidth_t\b"), "short"),  # line width
    (re.compile(r"\bColor_t\b"), "short"),  # color number
    (re.compile(r"\bSCoord_t\b"), "short"),  # screen coordinates
    (re.compile(r"\bCoord_t\b"), "double"),  # pad world coordinates
    (re.compile(r"\bAngle_t\b"), "float"),  # graphics angle
    (re.compile(r"\bSize_t\b"), "float"),  # attribute size
]


def _canonical_typename(name):
    for pattern, replacement in _canonical_typename_patterns:
        name = pattern.sub(replacement, name)
    return name


_tstreamerinfo_format1 = struct.Struct(">Ii")


class Class_TStreamerInfo(uproot4.model.Model):
    def read_members(self, chunk, cursor):
        name, self._members["fTitle"] = uproot4.deserialization.name_title(
            chunk, cursor, self._file.file_path
        )
        self._members["fUniqueID"], self._members["fBits"] = 0, 0
        self._members["fName"] = _canonical_typename(name)

        self._members["fCheckSum"], self._members["fClassVersion"] = cursor.fields(
            chunk, _tstreamerinfo_format1
        )

        self._members["fElements"] = uproot4.deserialization.read_object_any(
            chunk, cursor, self._file, self._parent
        )

    def postprocess(self):
        # no circular dependencies
        self._file = None
        self._parent = None
        return self

    @property
    def name(self):
        return self._members["fName"]

    @property
    def class_version(self):
        return self._members["fClassVersion"]

    @property
    def elements(self):
        return self._members["fElements"]

    def dependencies(self, classes, streamers, file_path):
        out = []
        for element in self.elements:
            out.extend(element.dependencies(classes, streamers, file_path, self.name))
        return out

    def new_class(self, classes, streamers, file_path):
        dependencies = self.dependencies(classes, streamers, file_path)

        print(self.name)
        print(dependencies)




        raise Exception



_tstreamerelement_format1 = struct.Struct(">iiii")
_tstreamerelement_format2 = struct.Struct(">i")
_tstreamerelement_format3 = struct.Struct(">ddd")
_tstreamerelement_dtype1 = numpy.dtype(">i4")


class Class_TStreamerElement(uproot4.model.Model):
    def read_members(self, chunk, cursor):
        # https://github.com/root-project/root/blob/master/core/meta/src/TStreamerElement.cxx#L505

        self._members["fUniqueID"], self._members["fBits"] = 0, 0
        (
            self._members["fName"],
            self._members["fTitle"],
        ) = uproot4.deserialization.name_title(chunk, cursor, self._file.file_path)

        (
            self._members["fType"],
            self._members["fSize"],
            self._members["fArrayLength"],
            self._members["fArrayDim"],
        ) = cursor.fields(chunk, _tstreamerelement_format1)

        if self._instance_version == 1:
            n = cursor.field(chunk, _tstreamerelement_format2)
            self._members["fMaxIndex"] = cursor.array(
                chunk, n, _tstreamerelement_dtype1
            )
        else:
            self._members["fMaxIndex"] = cursor.array(
                chunk, 5, _tstreamerelement_dtype1
            )

        self._members["fTypeName"] = _canonical_typename(cursor.string(chunk))

        if self._members["fType"] == 11 and self._members["fTypeName"] in (
            "Bool_t" or "bool"
        ):
            self._members["fType"] = 18

        if self._instance_version <= 2:
            # FIXME
            # self._fSize = self._fArrayLength * gROOT->GetType(GetTypeName())->Size()
            pass

        if self._instance_version > 3:
            # FIXME
            # if (TestBit(kHasRange)) GetRange(GetTitle(),fXmin,fXmax,fFactor)
            pass

    def postprocess(self):
        # no circular dependencies
        self._file = None
        self._parent = None
        return self

    @property
    def name(self):
        return self.member("fName")

    @property
    def title(self):
        return self.member("fTitle")

    @property
    def type_name(self):
        return self.member("fTypeName")

    def dependencies(self, classes, streamers, file_path, to_satisfy):
        return []


_tstreamerbase_format1 = struct.Struct(">i")


class Class_TStreamerArtificial(Class_TStreamerElement):
    def read_members(self, chunk, cursor):
        self._bases.append(
            Class_TStreamerElement.read(chunk, cursor, self._file, self._parent)
        )


class Class_TStreamerBase(Class_TStreamerElement):
    def read_members(self, chunk, cursor):
        self._bases.append(
            Class_TStreamerElement.read(chunk, cursor, self._file, self._parent)
        )
        if self._instance_version >= 2:
            self._members["fBaseVersion"] = cursor.field(chunk, _tstreamerbase_format1)

    @property
    def base_version(self):
        return self._members["fBaseVersion"]

    def dependencies(self, classes, streamers, file_path, to_satisfy):
        out = []

        if self.name not in classes:
            out.append(self.name)

            streamer_versions = streamers.get(self.name)
            if streamer_versions is None:
                raise ValueError(
                    """cannot find {0} to satisfy {1}
in file {2}""".format(self.name, to_satisfy, file_path))

            elif self.base_version not in streamer_versions:
                raise ValueError(
                    """cannot find {0} version {1} to satisfy {2}
in file {3}""".format(self.name, self.base_version, to_satisfy, file_path))

            else:
                streamer = streamer_versions[self.base_version]
                out.extend(streamer.dependencies(classes, streamers, file_path))

        return out


_tstreamerbasicpointer_format1 = struct.Struct(">i")


class Class_TStreamerBasicPointer(Class_TStreamerElement):
    def read_members(self, chunk, cursor):
        self._bases.append(
            Class_TStreamerElement.read(chunk, cursor, self._file, self._parent)
        )
        self._members["fCountVersion"] = cursor.field(
            chunk, _tstreamerbasicpointer_format1
        )
        self._members["fCountName"] = cursor.string(chunk)
        self._members["fCountClass"] = cursor.string(chunk)


class Class_TStreamerBasicType(Class_TStreamerElement):
    def read_members(self, chunk, cursor):
        self._bases.append(
            Class_TStreamerElement.read(chunk, cursor, self._file, self._parent)
        )
        if (
            uproot4.const.kOffsetL
            < self._bases[0]._members["fType"]
            < uproot4.const.kOffsetP
        ):
            self._bases[0]._members["fType"] -= uproot4.const.kOffsetL

        basic = True

        if self._bases[0]._members["fType"] in (
            uproot4.const.kBool,
            uproot4.const.kUChar,
            uproot4.const.kChar,
        ):
            self._bases[0]._members["fSize"] = 1

        elif self._bases[0]._members["fType"] in (
            uproot4.const.kUShort,
            uproot4.const.kShort,
        ):
            self._bases[0]._members["fSize"] = 2

        elif self._bases[0]._members["fType"] in (
            uproot4.const.kBits,
            uproot4.const.kUInt,
            uproot4.const.kInt,
            uproot4.const.kCounter,
        ):
            self._bases[0]._members["fSize"] = 4

        elif self._bases[0]._members["fType"] in (
            uproot4.const.kULong,
            uproot4.const.kULong64,
            uproot4.const.kLong,
            uproot4.const.kLong64,
        ):
            self._bases[0]._members["fSize"] = 8

        elif self._bases[0]._members["fType"] in (
            uproot4.const.kFloat,
            uproot4.const.kFloat16,
        ):
            self._bases[0]._members["fSize"] = 4

        elif self._bases[0]._members["fType"] in (
            uproot4.const.kDouble,
            uproot4.const.kDouble32,
        ):
            self._bases[0]._members["fSize"] = 8

        elif self._bases[0]._members["fType"] == uproot4.const.kCharStar:
            self._bases[0]._members["fSize"] = numpy.dtype(numpy.intp).itemsize

        else:
            basic = False

        if basic and self._bases[0]._members["fArrayLength"] > 0:
            self._bases[0]._members["fSize"] *= self._bases[0]._members["fArrayLength"]


_tstreamerloop_format1 = struct.Struct(">i")


class Class_TStreamerLoop(Class_TStreamerElement):
    def read_members(self, chunk, cursor):
        self._bases.append(
            Class_TStreamerElement.read(chunk, cursor, self._file, self._parent)
        )
        self._members["fCountVersion"] = cursor.field(chunk, _tstreamerloop_format1)
        self._members["fCountName"] = cursor.string(chunk)
        self._members["fCountClass"] = cursor.string(chunk)

    @property
    def count_version(self):
        return self._members["fCountVersion"]

    @property
    def count_name(self):
        return self._members["fCountName"]

    @property
    def count_class(self):
        return self._members["fCountClass"]

    def dependencies(self, classes, streamers, file_path, to_satisfy):
        out = []

        type_name = self.type_name.rstrip("*")
        if type_name not in classes:
            out.append(type_name)

            streamer_versions = streamers.get(type_name)
            if streamer_versions is None:
                raise ValueError(
                    """cannot find {0} to satisfy {1}
in file {2}""".format(type_name, to_satisfy, file_path))

            elif self.count_version not in streamer_versions:
                raise ValueError(
                    """cannot find {0} version {1} to satisfy {2}
in file {3}""".format(type_name, self.count_version, to_satisfy, file_path))

            else:
                streamer = stramer_versions[self.count_version]
                out.extend(streamer.dependencies(classes, streamers, file_path))

        return out


class Class_TStreamerObject(Class_TStreamerElement):
    def read_members(self, chunk, cursor):
        self._bases.append(
            Class_TStreamerElement.read(chunk, cursor, self._file, self._parent)
        )


class Class_TStreamerObjectAny(Class_TStreamerElement):
    def read_members(self, chunk, cursor):
        self._bases.append(
            Class_TStreamerElement.read(chunk, cursor, self._file, self._parent)
        )


class Class_TStreamerObjectAnyPointer(Class_TStreamerElement):
    def read_members(self, chunk, cursor):
        self._bases.append(
            Class_TStreamerElement.read(chunk, cursor, self._file, self._parent)
        )


class Class_TStreamerObjectPointer(Class_TStreamerElement):
    def read_members(self, chunk, cursor):
        self._bases.append(
            Class_TStreamerElement.read(chunk, cursor, self._file, self._parent)
        )


_tstreamerstl_format1 = struct.Struct(">ii")


class Class_TStreamerSTL(Class_TStreamerElement):
    def read_members(self, chunk, cursor):
        self._bases.append(
            Class_TStreamerElement.read(chunk, cursor, self._file, self._parent)
        )
        self._members["fSTLtype"], self._members["fCtype"] = cursor.fields(
            chunk, _tstreamerstl_format1
        )

        if self._members["fSTLtype"] in (
            uproot4.const.kSTLmultimap,
            uproot4.const.kSTLset,
        ):
            if self._bases[0]._members["fTypeName"].startswith(
                "std::set"
            ) or self._bases[0]._members["fTypeName"].startswith("set"):
                self._members["fSTLtype"] = uproot4.const.kSTLset

            elif self._bases[0]._members["fTypeName"].startswith(
                "std::multimap"
            ) or self._bases[0]._members["fTypeName"].startswith("multimap"):
                self._members["fSTLtype"] = uproot4.const.kSTLmultimap


class Class_TStreamerSTLstring(Class_TStreamerElement):
    def read_members(self, chunk, cursor):
        self._bases.append(
            Class_TStreamerElement.read(chunk, cursor, self._file, self._parent)
        )


class Class_TStreamerString(Class_TStreamerElement):
    def read_members(self, chunk, cursor):
        self._bases.append(
            Class_TStreamerElement.read(chunk, cursor, self._file, self._parent)
        )


uproot4.classes["TStreamerInfo"] = Class_TStreamerInfo
uproot4.classes["TStreamerElement"] = Class_TStreamerElement
uproot4.classes["TStreamerArtificial"] = Class_TStreamerArtificial
uproot4.classes["TStreamerBase"] = Class_TStreamerBase
uproot4.classes["TStreamerBasicPointer"] = Class_TStreamerBasicPointer
uproot4.classes["TStreamerBasicType"] = Class_TStreamerBasicType
uproot4.classes["TStreamerLoop"] = Class_TStreamerLoop
uproot4.classes["TStreamerObject"] = Class_TStreamerObject
uproot4.classes["TStreamerObjectAny"] = Class_TStreamerObjectAny
uproot4.classes["TStreamerObjectAnyPointer"] = Class_TStreamerObjectAnyPointer
uproot4.classes["TStreamerObjectPointer"] = Class_TStreamerObjectPointer
uproot4.classes["TStreamerSTL"] = Class_TStreamerSTL
uproot4.classes["TStreamerSTLstring"] = Class_TStreamerSTLstring
uproot4.classes["TStreamerString"] = Class_TStreamerString

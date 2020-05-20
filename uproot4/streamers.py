# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import struct
import re

import numpy

import uproot4.model


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


class ROOT_TStreamerInfo(uproot4.model.Model):
    def read_members(self, chunk, cursor):
        name, self._members["fTitle"] = uproot4.model._name_title(
            chunk, cursor, self._file.file_path
        )
        self._members["fName"] = _canonical_typename(name)

        self._members["fCheckSum"], self._members["fClassVersion"] = cursor.fields(
            chunk, _tstreamerinfo_format1
        )

        self._members["fElements"] = uproot4.model._read_object_any(
            chunk, cursor, self._file, self._parent
        )


uproot4.classes["TStreamerInfo"] = ROOT_TStreamerInfo


_tstreamerelement_format1 = struct.Struct(">iiii")
_tstreamerelement_format2 = struct.Struct(">i")
_tstreamerelement_format3 = struct.Struct(">ddd")
_tstreamerelement_dtype1 = numpy.dtype(">i4")


class ROOT_TStreamerElement(uproot4.model.Model):
    def read_members(self, chunk, cursor):
        # https://github.com/root-project/root/blob/master/core/meta/src/TStreamerElement.cxx#L505

        self._members["fOffset"] = 0
        self._members["fName"], self._members["fTitle"] = uproot4.model._name_title(
            chunk, cursor, self._file.file_path
        )

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

        self._members["fXmin"], self._members["fXmax"], self._members["fFactor"] = (
            0.0,
            0.0,
            0.0,
        )

        if self._instance_version == 3:
            (
                self._members["fXmin"],
                self._members["fXmax"],
                self._members["fFactor"],
            ) = cursor.fields(chunk, _tstreamerelement_format3)

        if self._instance_version > 3:
            # FIXME
            # if (TestBit(kHasRange)) GetRange(GetTitle(),fXmin,fXmax,fFactor)
            pass


uproot4.classes["TStreamerElement"] = ROOT_TStreamerElement


_tstreamerbase_format1 = struct.Struct(">i")


class ROOT_TStreamerBase(uproot4.model.Model):
    def read_members(self, chunk, cursor):
        self._bases.append(
            ROOT_TStreamerElement.read(chunk, cursor, self._file, self._parent, None)
        )

        if self._instance_version >= 2:
            self._members["fBaseVersion"] = cursor.field(chunk, _tstreamerbase_format1)


uproot4.classes["TStreamerBase"] = ROOT_TStreamerBase

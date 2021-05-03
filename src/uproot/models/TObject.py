# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

"""
This module defines a versionless model for ``TObject``.
"""

from __future__ import absolute_import

import struct

import numpy

import uproot

_tobject_format1 = struct.Struct(">h")
_tobject_format2 = struct.Struct(">II")


class Model_TObject(uproot.model.Model):
    """
    A versionless :doc:`uproot.model.Model` for ``TObject``.
    """

    def read_numbytes_version(self, chunk, cursor, context):
        pass

    def read_members(self, chunk, cursor, context, file):
        if self.is_memberwise:
            raise NotImplementedError(
                """memberwise serialization of {0}
in file {1}""".format(
                    type(self).__name__, self.file.file_path
                )
            )
        self._instance_version = cursor.field(chunk, _tobject_format1, context)
        if numpy.int64(self._instance_version) & uproot.const.kByteCountVMask:
            cursor.skip(4)
        self._members["@fUniqueID"], self._members["@fBits"] = cursor.fields(
            chunk, _tobject_format2, context
        )
        self._members["@fBits"] = (
            numpy.uint32(self._members["@fBits"]) | uproot.const.kIsOnHeap
        )
        if self._members["@fBits"] & uproot.const.kIsReferenced:
            cursor.skip(2)
        self._members["@fBits"] = int(self._members["@fBits"])

    def _serialize(self, out, header, name):
        # struct.pack(">hII", 1, 0, uproot.const.kNotDeleted)
        out.append(b"\x00\x01\x00\x00\x00\x00\x02\x00\x00\x00")

    @classmethod
    def strided_interpretation(
        cls, file, header=False, tobject_header=True, breadcrumbs=(), original=None
    ):
        members = []
        if tobject_header:
            members.append(("@instance_version", numpy.dtype(">u2")))
            members.append(("@num_bytes", numpy.dtype(">u4")))
            members.append(("@fUniqueID", numpy.dtype(">u4")))
            members.append(("@fBits", numpy.dtype(">u4")))
            members.append(("@pidf", numpy.dtype(">u2")))
        return uproot.interpretation.objects.AsStridedObjects(
            cls, members, original=original
        )

    @classmethod
    def awkward_form(
        cls, file, index_format="i64", header=False, tobject_header=True, breadcrumbs=()
    ):
        awkward = uproot.extras.awkward()
        contents = {}
        if tobject_header:
            contents["@instance_version"] = uproot._util.awkward_form(
                numpy.dtype("u2"),
                file,
                index_format,
                header,
                tobject_header,
                breadcrumbs,
            )
            contents["@num_bytes"] = uproot._util.awkward_form(
                numpy.dtype("u4"),
                file,
                index_format,
                header,
                tobject_header,
                breadcrumbs,
            )
            contents["@fUniqueID"] = uproot._util.awkward_form(
                numpy.dtype("u4"),
                file,
                index_format,
                header,
                tobject_header,
                breadcrumbs,
            )
            contents["@fBits"] = uproot._util.awkward_form(
                numpy.dtype("u4"),
                file,
                index_format,
                header,
                tobject_header,
                breadcrumbs,
            )
            contents["@pidf"] = uproot._util.awkward_form(
                numpy.dtype("u2"),
                file,
                index_format,
                header,
                tobject_header,
                breadcrumbs,
            )
        return awkward.forms.RecordForm(
            contents,
            parameters={"__record__": "TObject"},
        )

    def __repr__(self):
        return "<TObject {0} {1} at 0x{2:012x}>".format(
            self._members.get("fUniqueID"), self._members.get("fBits"), id(self)
        )

    def tojson(self):
        return {
            "_typename": self.classname,
            "fUniqueID": self.member("@fUniqueID"),
            "fBits": self.member("@fBits"),
        }


uproot.classes["TObject"] = Model_TObject

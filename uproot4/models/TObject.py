# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import struct

import numpy

import uproot4.model
import uproot4.deserialization


_tobject_format1 = struct.Struct(">h")
_tobject_format2 = struct.Struct(">II")


class Model_TObject(uproot4.model.Model):
    def read_numbytes_version(self, chunk, cursor, context):
        pass

    def read_members(self, chunk, cursor, context):
        self._instance_version = cursor.field(chunk, _tobject_format1, context)
        if numpy.int64(self._instance_version) & uproot4.const.kByteCountVMask:
            cursor.skip(4)
        self._members["@fUniqueID"], self._members["@fBits"] = cursor.fields(
            chunk, _tobject_format2, context
        )
        self._members["@fBits"] = (
            numpy.uint32(self._members["@fBits"]) | uproot4.const.kIsOnHeap
        )
        if self._members["@fBits"] & uproot4.const.kIsReferenced:
            cursor.skip(2)
        self._members["@fBits"] = int(self._members["@fBits"])

    @classmethod
    def strided_interpretation(
        cls, file, header=False, tobject_header=True, original=None
    ):
        members = []
        if tobject_header:
            members.append(("@instance_version", numpy.dtype(">u2")))
            members.append(("@num_bytes", numpy.dtype(">u4")))
            members.append(("@fUniqueID", numpy.dtype(">u4")))
            members.append(("@fBits", numpy.dtype(">u4")))
            members.append(("@pidf", numpy.dtype(">u2")))
        return uproot4.interpretation.objects.AsStridedObjects(
            cls, members, original=original
        )

    @classmethod
    def awkward_form(cls, file, header=False, tobject_header=True):
        import awkward1

        contents = {}
        if tobject_header:
            contents["@instance_version"] = uproot4._util.awkward_form(
                numpy.dtype("u2"), file, header, tobject_header
            )
            contents["@num_bytes"] = uproot4._util.awkward_form(
                numpy.dtype("u4"), file, header, tobject_header
            )
            contents["@fUniqueID"] = uproot4._util.awkward_form(
                numpy.dtype("u4"), file, header, tobject_header
            )
            contents["@fBits"] = uproot4._util.awkward_form(
                numpy.dtype("u4"), file, header, tobject_header
            )
            contents["@pidf"] = uproot4._util.awkward_form(
                numpy.dtype("u2"), file, header, tobject_header
            )
        return awkward1.forms.RecordForm(
            contents, parameters={"__record__": "TObject"},
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


uproot4.classes["TObject"] = Model_TObject

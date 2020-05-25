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
        self._instance_version = cursor.field(chunk, _tobject_format1)
        if numpy.int64(self._instance_version) & uproot4.const.kByteCountVMask:
            cursor.skip(4)
        self._members["fUniqueID"], self._members["fBits"] = cursor.fields(
            chunk, _tobject_format2
        )
        self._members["fBits"] = (
            numpy.uint32(self._members["fBits"]) | uproot4.const.kIsOnHeap
        )
        if self._members["fBits"] & uproot4.const.kIsReferenced:
            cursor.skip(2)
        self._members["fBits"] = int(self._members["fBits"])


uproot4.classes["TObject"] = Model_TObject

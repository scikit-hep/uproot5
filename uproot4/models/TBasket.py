# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import struct

import uproot4.model
import uproot4.deserialization


_tbasket_format1 = struct.Struct(">iIhh")
_tbasket_format2 = struct.Struct(">Hiiii")


class Model_TBasket(uproot4.model.Model):
    def read_numbytes_version(self, chunk, cursor, context):
        pass

    def read_members(self, chunk, cursor, context):
        cursor.skip(6)
        (
            self._members["fObjlen"],
            self._members["fDatime"],
            self._members["fKeylen"],
            self._members["fCycle"],
        ) = cursor.fields(chunk, _tbasket_format1)

        # skip the class name, name, and title
        cursor.move_to(
            self._cursor.index + self._members["fKeylen"] - _tbasket_format2.size - 1
        )

        (
            self._members["fVersion"],
            self._members["fBufferSize"],
            self._members["fNevBufSize"],
            self._members["fNevBuf"],
            self._members["fLast"],
        ) = cursor.fields(chunk, _tbasket_format2)

        cursor.skip(1)

        if self._members["fNevBufSize"] > 8:
            self._byte_offsets = cursor.bytes(chunk, self._members["fNevBuf"] * 4 + 8)
            cursor.skip(-4)
        else:
            self._byte_offsets = None

        # there's a second TKey here
        cursor.skip(self._members["fKeylen"])

        self._data = cursor.bytes(chunk, self.border)

    @property
    def data(self):
        return self._data

    @property
    def byte_offsets(self):
        return self._byte_offsets

    @property
    def border(self):
        return self._members["fLast"] - self._members["fKeylen"]


uproot4.classes["TBasket"] = Model_TBasket

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import struct

import numpy

import uproot4.model
import uproot4.deserialization
import uproot4.compression


_tbasket_format1 = struct.Struct(">ihiIhh")
_tbasket_format2 = struct.Struct(">Hiiii")
_tbasket_offsets_dtype = numpy.dtype(">i4")


class Model_TBasket(uproot4.model.Model):
    def read_numbytes_version(self, chunk, cursor, context):
        pass

    def read_members(self, chunk, cursor, context):
        (
            self._members["fNbytes"],
            self._key_version,
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

        # if self._members["fNevBufSize"] > 8:
        #     self._byte_offsets = cursor.bytes(chunk, self._members["fNevBuf"] * 4 + 8)
        #     cursor.skip(-4)
        # else:
        #     self._byte_offsets = None

        # if context.get("second_key", True):
        #     cursor.skip(self._members["fKeylen"])

        if self.compressed_bytes != self.uncompressed_bytes:
            uncompressed = uproot4.compression.decompress(
                chunk, cursor, {}, self.compressed_bytes, self.uncompressed_bytes,
            )
            self._raw_data = uncompressed.get(0, self.uncompressed_bytes)
        else:
            self._raw_data = cursor.bytes(chunk, self.uncompressed_bytes)

        if self.border == self.uncompressed_bytes:
            self._data = self._raw_data
            self._byte_offsets = None
        else:
            self._data = self._raw_data[: self.border]
            raw_byte_offsets = self._raw_data[self.border :].view(
                _tbasket_offsets_dtype
            )
            # subtracting fKeylen makes a new buffer and converts to native endian
            self._byte_offsets = raw_byte_offsets[1:] - self._members["fKeylen"]
            # so modifying it in place doesn't have non-local consequences
            self._byte_offsets[-1] = self.border

    @property
    def uncompressed_bytes(self):
        return self._members["fObjlen"]

    @property
    def compressed_bytes(self):
        return self._members["fNbytes"] - self._members["fKeylen"]

    @property
    def border(self):
        return self._members["fLast"] - self._members["fKeylen"]

    @property
    def raw_data(self):
        return self._raw_data

    @property
    def data(self):
        return self._data

    @property
    def byte_offsets(self):
        return self._byte_offsets


uproot4.classes["TBasket"] = Model_TBasket

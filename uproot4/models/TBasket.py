# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import struct

import numpy

import uproot4.model
import uproot4.deserialization
import uproot4.compression
import uproot4.behaviors.TBranch
import uproot4.const


_tbasket_format1 = struct.Struct(">ihiIhh")
_tbasket_format2 = struct.Struct(">Hiiii")
_tbasket_offsets_dtype = numpy.dtype(">i4")


class Model_TBasket(uproot4.model.Model):
    def __repr__(self):
        basket_num = self._basket_num if self._basket_num is not None else "(unknown)"
        return "<TBasket {0} of {1} at 0x{2:012x}>".format(
            basket_num, repr(self._parent.name), id(self)
        )

    def read_numbytes_version(self, chunk, cursor, context):
        pass

    def read_members(self, chunk, cursor, context):
        assert isinstance(self._parent, uproot4.behaviors.TBranch.TBranch)
        self._basket_num = context.get("basket_num")

        (
            self._members["fNbytes"],
            self._key_version,
            self._members["fObjlen"],
            self._members["fDatime"],
            self._members["fKeylen"],
            self._members["fCycle"],
        ) = cursor.fields(chunk, _tbasket_format1, context)

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
        ) = cursor.fields(chunk, _tbasket_format2, context)

        cursor.skip(1)

        if self.is_embedded:
            if self._members["fNevBufSize"] > 8:
                raw_byte_offsets = cursor.bytes(
                    chunk, 8 + self.num_entries * 4, context
                ).view(_tbasket_offsets_dtype)
                cursor.skip(-4)

                # subtracting fKeylen makes a new buffer and converts to native endian
                self._byte_offsets = raw_byte_offsets[1:] - self._members["fKeylen"]
                # so modifying it in place doesn't have non-local consequences
                self._byte_offsets[-1] = self.border

            else:
                self._byte_offsets = None

            # second key has no new information
            cursor.skip(self._members["fKeylen"])

            self._raw_data = None
            self._data = cursor.bytes(chunk, self.border, context, copy_if_memmap=True)

        else:
            if self.compressed_bytes != self.uncompressed_bytes:
                uncompressed = uproot4.compression.decompress(
                    chunk, cursor, {}, self.compressed_bytes, self.uncompressed_bytes,
                )
                self._raw_data = uncompressed.get(
                    0,
                    self.uncompressed_bytes,
                    uproot4.source.cursor.Cursor(0),
                    context,
                )
            else:
                self._raw_data = cursor.bytes(
                    chunk, self.uncompressed_bytes, context, copy_if_memmap=True
                )

            if self.border != self.uncompressed_bytes:
                self._data = self._raw_data[: self.border]
                raw_byte_offsets = self._raw_data[self.border :].view(
                    _tbasket_offsets_dtype
                )

                # subtracting fKeylen makes a new buffer and converts to native endian
                self._byte_offsets = raw_byte_offsets[1:] - self._members["fKeylen"]
                # so modifying it in place doesn't have non-local consequences
                self._byte_offsets[-1] = self.border

            else:
                self._data = self._raw_data
                self._byte_offsets = None

    @property
    def basket_num(self):
        return self._basket_num

    @property
    def key_version(self):
        return self._key_version

    @property
    def num_entries(self):
        return self._members["fNevBuf"]

    @property
    def is_embedded(self):
        return self._members["fNbytes"] <= self._members["fKeylen"]

    @property
    def uncompressed_bytes(self):
        if self.is_embedded:
            if self._byte_offsets is None:
                return self._data.nbytes
            else:
                return self._data.nbytes + 4 + self.num_entries * 4
        else:
            return self._members["fObjlen"]

    @property
    def compressed_bytes(self):
        if self.is_embedded:
            if self._byte_offsets is None:
                return self._data.nbytes
            else:
                return self._data.nbytes + 4 + self.num_entries * 4
        else:
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

    @property
    def counts(self):
        count_branch = self._parent.count_branch
        if count_branch is not None:
            entry_offsets = count_branch.entry_offsets
            entry_start = entry_offsets[self._basket_num]
            entry_stop = entry_offsets[self._basket_num + 1]
            return count_branch.array(
                entry_start=entry_start, entry_stop=entry_stop, library="np"
            )

    def array(self, interpretation=None):
        if interpretation is None:
            interpretation = self._parent.interpretation
        return interpretation.basket_array(
            self.data,
            self.byte_offsets,
            self,
            self.parent,
            self.parent.context,
            self._members["fKeylen"],
        )


uproot4.classes["TBasket"] = Model_TBasket

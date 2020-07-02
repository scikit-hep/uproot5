# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import struct

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence

import uproot4.model
import uproot4.deserialization
import uproot4.models.TObject
import uproot4.models.TBasket


_tobjarray_format1 = struct.Struct(">ii")


class Model_TObjArray(uproot4.model.Model, Sequence):
    def read_members(self, chunk, cursor, context):
        self._bases.append(
            uproot4.models.TObject.Model_TObject.read(
                chunk, cursor, context, self._file, self._parent
            )
        )

        self._members["fName"] = cursor.string(chunk, context)
        self._members["fSize"], self._members["fLowerBound"] = cursor.fields(
            chunk, _tobjarray_format1, context
        )

        self._data = []
        for i in range(self._members["fSize"]):
            item = uproot4.deserialization.read_object_any(
                chunk, cursor, context, self._file, self._parent
            )
            self._data.append(item)

    def __repr__(self):
        return "<{0} of {1} items at 0x{2:012x}>".format(
            uproot4.model.classname_pretty(self.classname, self.class_version),
            len(self),
            id(self),
        )

    def __getitem__(self, where):
        return self._data[where]

    def __len__(self):
        return len(self._data)

    def tojson(self):
        return {
            "_typename": "TObjArray",
            "name": "TObjArray",
            "arr": [x.tojson() for x in self._data],
        }


uproot4.classes["TObjArray"] = Model_TObjArray


class Model_TObjArrayOfTBaskets(Model_TObjArray):
    def read_members(self, chunk, cursor, context):
        self._bases.append(
            uproot4.models.TObject.Model_TObject.read(
                chunk, cursor, context, self._file, self._parent
            )
        )

        self._members["fName"] = cursor.string(chunk, context)
        self._members["fSize"], self._members["fLowerBound"] = cursor.fields(
            chunk, _tobjarray_format1, context
        )

        self._data = []
        for i in range(self._members["fSize"]):
            item = uproot4.deserialization.read_object_any(
                chunk,
                cursor,
                context,
                self._file,
                self._parent,
                as_class=uproot4.models.TBasket.Model_TBasket,
            )
            self._data.append(item)

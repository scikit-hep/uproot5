# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import struct

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence

import uproot4.model
import uproot4.deserialization


_tobjarray_format1 = struct.Struct(">ii")


class ROOT_TObjArray(uproot4.model.Model, Sequence):
    def read_members(self, chunk, cursor):
        uproot4.deserialization.skip_tobject(chunk, cursor)

        self._members["fName"] = cursor.string(chunk)
        self._members["fSize"], self._members["fLowerBound"] = cursor.fields(
            chunk, _tobjarray_format1
        )

        self._data = []
        for i in range(self._members["fSize"]):
            item = uproot4.deserialization.read_object_any(
                chunk, cursor, self._file, self._parent
            )
            self._data.append(item)

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


uproot4.classes["TObjArray"] = ROOT_TObjArray

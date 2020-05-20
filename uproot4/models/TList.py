# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import struct
try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence

import uproot4.model


_tlist_format1 = struct.Struct(">i")
_tlist_format2 = struct.Struct(">B")


class ROOT_TList(uproot4.model.Model, Sequence):
    def read_members(self, chunk, cursor):
        cursor.debug(chunk, 200)

        uproot4.model._skip_tobject(chunk, cursor)

        self._members["fName"] = cursor.string(chunk)
        self._members["fSize"] = cursor.field(chunk, _tlist_format1)

        print(self._members)

        self._data = []
        for i in range(self._members["fSize"]):
            item = uproot4.classes._read_object_any(chunk, cursor, self._file, self)
            self._data.append(item)

            # ignore "option"
            n = cursor.field(chunk, _tlist_format2)
            cursor.skip(n)

    def __getitem__(self, where):
        return self._data[where]

    def __len__(self):
        return len(self._data)


uproot4.classes["TList"] = ROOT_TList

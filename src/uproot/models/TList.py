# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

"""
This module defines a versionless model for ``TList``.
"""

from __future__ import absolute_import

import struct

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence

import uproot

_tlist_format1 = struct.Struct(">i")


class Model_TList(uproot.model.Model, Sequence):
    """
    A versionless :doc:`uproot.model.Model` for ``TList``.
    """

    def read_members(self, chunk, cursor, context, file):
        if self.is_memberwise:
            raise NotImplementedError(
                """memberwise serialization of {0}
in file {1}""".format(
                    type(self).__name__, self.file.file_path
                )
            )
        self._bases.append(
            uproot.models.TObject.Model_TObject.read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )

        self._members["fName"] = cursor.string(chunk, context)
        self._members["fSize"] = cursor.field(chunk, _tlist_format1, context)

        self._data = []
        self._starts = []
        self._stops = []
        for _ in uproot._util.range(self._members["fSize"]):
            self._starts.append(cursor.index)

            item = uproot.deserialization.read_object_any(
                chunk, cursor, context, file, self._file, self._parent
            )
            self._data.append(item)

            cursor.bytestring(chunk, context)  # read past and ignore "option"

            self._stops.append(cursor.index)

    def __repr__(self):
        if self.class_version is None:
            version = ""
        else:
            version = " (version {0})".format(self.class_version)
        return "<{0}{1} of {2} items at 0x{3:012x}>".format(
            self.classname,
            version,
            len(self),
            id(self),
        )

    def __getitem__(self, where):
        return self._data[where]

    def __len__(self):
        return len(self._data)

    @property
    def byte_ranges(self):
        return zip(self._starts, self._stops)

    def tojson(self):
        return {
            "_typename": "TList",
            "name": "TList",
            "arr": [x.tojson() for x in self._data],
            "opt": [],
        }


uproot.classes["TList"] = Model_TList

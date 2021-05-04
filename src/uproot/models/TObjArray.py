# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

"""
This module defines a versionless model for ``TObjArray``.
"""

from __future__ import absolute_import

import struct

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence

import uproot

_tobjarray_format1 = struct.Struct(">ii")


class Model_TObjArray(uproot.model.Model, Sequence):
    """
    A versionless :doc:`uproot.model.Model` for ``TObjArray``.

    This also satisfies Python's abstract ``Sequence`` protocol.
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
        self._members["fSize"], self._members["fLowerBound"] = cursor.fields(
            chunk, _tobjarray_format1, context
        )

        self._data = []
        for _ in uproot._util.range(self._members["fSize"]):
            item = uproot.deserialization.read_object_any(
                chunk, cursor, context, file, self._file, self._parent
            )
            self._data.append(item)

    def _serialize(self, out, header, name):
        where = len(out)
        for x in self._bases:
            x._serialize(out, True, name)
        out.append(uproot.serialization.string(self._members["fName"]))
        out.append(
            _tobjarray_format1.pack(
                self._members["fSize"], self._members["fLowerBound"]
            )
        )
        for item in self._data:
            uproot.serialization._serialize_object_any(out, item, name)
        if header:
            out.insert(
                where,
                uproot.serialization.numbytes_version(
                    sum(len(x) for x in out[where:]), self._instance_version
                ),
            )

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

    def tojson(self):
        return {
            "_typename": "TObjArray",
            "name": "TObjArray",
            "arr": [x.tojson() for x in self._data],
        }


uproot.classes["TObjArray"] = Model_TObjArray


class Model_TObjArrayOfTBaskets(Model_TObjArray):
    """
    A specialized :doc:`uproot.model.Model` for a ``TObjArray`` of ``TBaskets``.
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
        self._members["fSize"], self._members["fLowerBound"] = cursor.fields(
            chunk, _tobjarray_format1, context
        )

        self._data = []
        for _ in uproot._util.range(self._members["fSize"]):
            item = uproot.deserialization.read_object_any(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                as_class=uproot.models.TBasket.Model_TBasket,
            )
            self._data.append(item)

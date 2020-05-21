# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import uproot4.model


class Class_TString(uproot4.model.Model, str):
    def read_numbytes_version(self, chunk, cursor):
        pass

    def read_members(self, chunk, cursor):
        self._data = cursor.string(chunk)

    def postprocess(self):
        return Class_TString(
            self._data,
            self._cursor,
            self._file,
            self._parent,
            self._encoded_classname,
            self._members,
            self._bases,
        )

    def __new__(cls, data, cursor, file, parent, encoded_classname, members, bases):
        self = str.__new__(cls, data)
        self._cursor = cursor
        self._file = file
        self._parent = parent
        self._encoded_classname = encoded_classname
        self._members = members
        self._bases = bases
        return self


uproot4.classes["TString"] = Class_TString

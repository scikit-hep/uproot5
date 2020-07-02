# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import uproot4.model


class Model_TString(uproot4.model.Model, str):
    def read_numbytes_version(self, chunk, cursor, context):
        pass

    def read_members(self, chunk, cursor, context):
        self._data = cursor.string(chunk, context)

    def postprocess(self, chunk, cursor, context):
        out = Model_TString(self._data)
        out._cursor = self._cursor
        out._file = self._file
        out._parent = self._parent
        out._members = self._members
        out._bases = self._bases
        out._num_bytes = self._num_bytes
        out._instance_version = self._instance_version
        return out

    def __repr__(self):
        return "<{0} {1} at 0x{2:012x}>".format(
            uproot4.model.classname_pretty(self.classname, self.class_version),
            str.__repr__(self),
            id(self),
        )

    def tojson(self):
        return str(self)

    @classmethod
    def awkward_form(cls, file, header=False, tobject_header=True):
        return uproot4.containers.AsString(False, typename="TString").awkward_form(
            file, header, tobject_header
        )


uproot4.classes["TString"] = Model_TString

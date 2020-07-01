# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import uproot4.model
import uproot4.models.TObject


class Model_TObjString(uproot4.model.Model, str):
    def read_members(self, chunk, cursor, context):
        self._bases.append(
            uproot4.models.TObject.Model_TObject.read(
                chunk, cursor, context, self._file, self._parent
            )
        )
        self._data = cursor.string(chunk, context)

    def postprocess(self, chunk, cursor, context):
        out = Model_TObjString(self._data)
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

    @classmethod
    def awkward_form(cls, file, header=False, tobject_header=True):
        import awkward1

        return awkward1.forms.ListOffsetForm(
            "i32",
            awkward1.forms.NumpyForm((), 1, "B", parameters={"__array__": "char"}),
            parameters={
                "__array__": "string",
                "uproot": {"as": "TObjString", "header": True, "length_bytes": "1-5"},
            },
        )


uproot4.classes["TObjString"] = Model_TObjString

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import numpy

import uproot4.model
import uproot4.models.TObject
import uproot4.containers


class Model_TNamed(uproot4.model.Model):
    def read_members(self, chunk, cursor, context):
        self._bases.append(
            uproot4.models.TObject.Model_TObject.read(
                chunk, cursor, context, self._file, self._parent
            )
        )

        self._members["fName"] = cursor.string(chunk, context)
        self._members["fTitle"] = cursor.string(chunk, context)

    def __repr__(self):
        title = ""
        if self._members["fTitle"] != "":
            title = " title=" + repr(self._members["fTitle"])
        return "<TNamed {0}{1} at 0x{2:012x}>".format(
            repr(self._members["fName"]), title, id(self)
        )

    @classmethod
    def awkward_form(cls, file, header=False, tobject_header=True):
        import awkward1

        contents = {}
        if header:
            contents["@num_bytes"] = uproot4._util.awkward_form(
                numpy.dtype("u4"), file, header, tobject_header
            )
            contents["@instance_version"] = uproot4._util.awkward_form(
                numpy.dtype("u2"), file, header, tobject_header
            )
        contents["fName"] = uproot4.containers.AsString(
            False, typename="TString"
        ).awkward_form(file, header, tobject_header)
        contents["fTitle"] = uproot4.containers.AsString(
            False, typename="TString"
        ).awkward_form(file, header, tobject_header)
        return awkward1.forms.RecordForm(contents, parameters={"__record__": "TNamed"},)


uproot4.classes["TNamed"] = Model_TNamed

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

"""
This module defines a versionless model for ``TNamed``.
"""

from __future__ import absolute_import

import numpy

import uproot


class Model_TNamed(uproot.model.Model):
    """
    A versionless :doc:`uproot.model.Model` for ``TNamed``.
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
                concrete=self._concrete,
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
    def awkward_form(
        cls, file, index_format="i64", header=False, tobject_header=True, breadcrumbs=()
    ):
        awkward = uproot.extras.awkward()
        contents = {}
        if header:
            contents["@num_bytes"] = uproot._util.awkward_form(
                numpy.dtype("u4"),
                file,
                index_format,
                header,
                tobject_header,
                breadcrumbs,
            )
            contents["@instance_version"] = uproot._util.awkward_form(
                numpy.dtype("u2"),
                file,
                index_format,
                header,
                tobject_header,
                breadcrumbs,
            )
        contents["fName"] = uproot.containers.AsString(
            False, typename="TString"
        ).awkward_form(file, index_format, header, tobject_header, breadcrumbs)
        contents["fTitle"] = uproot.containers.AsString(
            False, typename="TString"
        ).awkward_form(file, index_format, header, tobject_header, breadcrumbs)
        return awkward.forms.RecordForm(contents, parameters={"__record__": "TNamed"},)


uproot.classes["TNamed"] = Model_TNamed

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

"""
This module defines versionless models of ``TRef`` and ``TRefArray``.
"""


import json
import struct
from collections.abc import Sequence

import numpy

import uproot

_tref_format1 = struct.Struct(">xxIxxxxxx")

# Note: https://github.com/root-project/root/blob/master/io/doc/TFile/TRef.txt
#       https://github.com/root-project/root/blob/master/io/doc/TFile/TRefArray.txt


class Model_TRef(uproot.model.Model):
    """
    A versionless :doc:`uproot.model.Model` for ``TRef``.

    This model does not deserialize all fields, only the reference number.
    """

    @property
    def ref(self):
        """
        The reference number as an integer.
        """
        return self._ref

    def read_numbytes_version(self, chunk, cursor, context):
        pass

    def read_members(self, chunk, cursor, context, file):
        if self.is_memberwise:
            raise NotImplementedError(
                """memberwise serialization of {}
in file {}""".format(
                    type(self).__name__, self.file.file_path
                )
            )
        self._ref = cursor.field(chunk, _tref_format1, context)

    def __repr__(self):
        return f"<TRef {self._ref}>"

    @classmethod
    def strided_interpretation(
        cls, file, header=False, tobject_header=True, breadcrumbs=(), original=None
    ):
        members = []
        members.append(("@pidf", numpy.dtype(">u2")))
        members.append(("ref", numpy.dtype(">u4")))
        members.append(("@other1", numpy.dtype(">u2")))
        members.append(("@other2", numpy.dtype(">u4")))

        return uproot.interpretation.objects.AsStridedObjects(
            cls, members, original=original
        )

    @classmethod
    def awkward_form(cls, file, context):
        awkward = uproot.extras.awkward()
        contents = {}
        if context["tobject_header"]:
            contents["@pidf"] = uproot._util.awkward_form(
                numpy.dtype("u2"), file, context
            )
            contents["ref"] = uproot._util.awkward_form(
                numpy.dtype("u4"), file, context
            )
            contents["@other1"] = uproot._util.awkward_form(
                numpy.dtype("u2"), file, context
            )
            contents["@other2"] = uproot._util.awkward_form(
                numpy.dtype("u4"), file, context
            )
        return awkward._v2.forms.RecordForm(
            list(contents.values()),
            list(contents.keys()),
            parameters={"__record__": "TRef"},
        )


_trefarray_format1 = struct.Struct(">i")
_trefarray_dtype = numpy.dtype(">i4")


class Model_TRefArray(uproot.model.Model, Sequence):
    """
    A versionless :doc:`uproot.model.Model` for ``TRefArray``.

    This also satisfies Python's abstract ``Sequence`` protocol.
    """

    @property
    def refs(self):
        """
        The reference number as a ``numpy.ndarray`` of ``dtype(">i4")``.
        """
        return self._data

    @property
    def nbytes(self):
        """
        The number of bytes in :ref:`uproot.models.TRef.Model_TRefArray.refs`.
        """
        return self._data.nbytes

    @property
    def name(self):
        """
        The name of this TRefArray.
        """
        return self._members["fName"]

    def read_members(self, chunk, cursor, context, file):
        helper_obj = uproot._awkward_forth.GenHelper(context)
        if helper_obj.is_forth():
            awkward = uproot.extras.awkward()  # noqa:F841
            forth_obj = helper_obj.get_gen_obj()
        if self.is_memberwise:
            raise NotImplementedError(
                """memberwise serialization of {}
in file {}""".format(
                    type(self).__name__, self.file.file_path
                )
            )
        if helper_obj.is_forth():
            form_keys = forth_obj.get_keys(6)

            helper_obj.add_to_pre("10 stream skip\n")
            helper_obj.add_to_pre(
                f"stream !B-> stack dup 255 = if drop stream !I-> stack then dup node{form_keys[1]}-offsets +<- stack stream #!B-> node{form_keys[2]}-data\n"
            )
            helper_obj.add_to_pre(
                f"stream !I-> stack dup node{form_keys[3]}-data <- stack\n"
            )
            helper_obj.add_to_pre("6 stream skip\n")
            helper_obj.add_to_pre(
                f"dup node{form_keys[4]}-offsets +<- stack stream #!I-> node{form_keys[5]}-data\n"
            )
            keys = [
                f"node{form_keys[1]}-offsets",
                f"node{form_keys[3]}-data",
                f"node{form_keys[4]}-offsets",
                f"node{form_keys[2]}-data",
                f"node{form_keys[5]}-data",
            ]
            if forth_obj.should_add_form():
                for elem in keys:
                    forth_obj.add_form_key(elem)
                temp_bool = "false"
                temp_aform = f'{{"class": "RecordArray", "contents": {{"fname": {{"class": "ListOffsetArray", "offsets": "i64", "content": {{"class": "NumpyArray", "primitive": "uint8", "inner_shape": [], "has_identifier": false, "parameters": {{"__array__": "char"}}, "form_key": "node{form_keys[2]}"}}, "has_identifier": false, "parameters": {{"uproot": {{"as": "vector", "header": {temp_bool}}}}}, "form_key": "node{form_keys[1]}"}}, "fSize": {{"class": "NumpyArray", "primitive": "int64", "inner_shape": [], "has_identifier": false, "parameters": {{}}, "form_key": "node{form_keys[3]}"}}, "refs": {{"class": "ListOffsetArray", "offsets": "i64", "content": {{"class": "NumpyArray", "primitive": "int64", "inner_shape": [], "has_identifier": false, "parameters": {{}}, "form_key": "node{form_keys[5]}"}}, "has_identifier": false, "parameters": {{}}, "form_key": "node{form_keys[4]}"}}}}, "has_identifier": false, "parameters": {{}}, "form_key": "node{form_keys[0]}"}}'
                forth_obj.add_form(json.loads(temp_aform))
            helper_obj.add_to_header(
                f"output node{form_keys[1]}-offsets int64\noutput node{form_keys[2]}-data uint8\noutput node{form_keys[3]}-data int64\noutput node{form_keys[4]}-offsets int64\noutput node{form_keys[5]}-data int64\n"
            )
            helper_obj.add_to_init(
                f"0 node{form_keys[1]}-offsets <- stack\n0 node{form_keys[4]}-offsets <- stack\n"
            )
            temp = forth_obj.add_node(
                f"node{form_keys[0]}",
                helper_obj.get_pre(),
                helper_obj.get_post(),
                helper_obj.get_init(),
                helper_obj.get_header(),
                "i64",
                1,
                None,
            )
            forth_obj.go_to(temp)

        cursor.skip(10)
        self._members["fName"] = cursor.string(chunk, context)
        self._members["fSize"] = cursor.field(chunk, _trefarray_format1, context)
        cursor.skip(6)
        self._data = cursor.array(
            chunk, self._members["fSize"], _trefarray_dtype, context
        )

    def __getitem__(self, where):
        return self._data[where]

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        if self.class_version is None:
            version = ""
        else:
            version = f" (version {self.class_version})"
        return "<{}{} {} at 0x{:012x}>".format(
            self.classname,
            version,
            numpy.array2string(
                self._data,
                max_line_width=numpy.inf,
                separator=", ",
                formatter={"float": lambda x: "%g" % x},
                threshold=6,
            ),
            id(self),
        )

    @classmethod
    def awkward_form(cls, file, context):
        awkward = uproot.extras.awkward()
        contents = {}
        contents["fName"] = uproot.containers.AsString(
            False, typename="TString"
        ).awkward_form(file, context)
        contents["fSize"] = uproot._util.awkward_form(numpy.dtype("i4"), file, context)
        contents["refs"] = awkward._v2.forms.ListOffsetForm(
            context["index_format"],
            uproot._util.awkward_form(numpy.dtype("i4"), file, context),
        )
        return awkward._v2.forms.RecordForm(
            list(contents.values()),
            list(contents.keys()),
            parameters={"__record__": "TRefArray"},
        )


uproot.classes["TRef"] = Model_TRef
uproot.classes["TRefArray"] = Model_TRefArray

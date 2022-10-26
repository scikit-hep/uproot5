# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

"""
This module defines a versionless model for ``TObject``.
"""


import json
import struct

import numpy

import uproot

_tobject_format1 = struct.Struct(">h")
_tobject_format2 = struct.Struct(">II")


class Model_TObject(uproot.model.Model):
    """
    A versionless :doc:`uproot.model.Model` for ``TObject``.
    """

    def read_numbytes_version(self, chunk, cursor, context):
        pass

    def read_members(self, chunk, cursor, context, file):
        helper_obj = uproot._awkward_forth.GenHelper(context)
        start_index = cursor._index
        if helper_obj.is_forth():
            forth_obj = helper_obj.get_gen_obj()
            # raise NotImplementedError
        if self.is_memberwise:
            raise NotImplementedError(
                """memberwise serialization of {}
in file {}""".format(
                    type(self).__name__, self.file.file_path
                )
            )
        self._instance_version = cursor.field(chunk, _tobject_format1, context)
        if numpy.int64(self._instance_version) & uproot.const.kByteCountVMask:
            cursor.skip(4)
        self._members["@fUniqueID"], self._members["@fBits"] = cursor.fields(
            chunk, _tobject_format2, context
        )
        self._members["@fBits"] = (
            numpy.uint32(self._members["@fBits"]) | uproot.const.kIsOnHeap
        )
        if self._members["@fBits"] & uproot.const.kIsReferenced:
            cursor.skip(2)
        self._members["@fBits"] = int(self._members["@fBits"])
        if helper_obj.is_forth():
            skip_length = cursor._index - start_index
            helper_obj.add_to_pre(f"{skip_length} stream skip \n")
            if forth_obj.should_add_form():
                temp_aform = '{"class": "RecordArray", "contents":[], "parameters": {"__record__": "TObject"}}'
                forth_obj.add_form(json.loads(temp_aform))
            forth_obj.add_node(
                "TObjext",
                helper_obj.get_pre(),
                helper_obj.get_post(),
                helper_obj.get_init(),
                helper_obj.get_header(),
                "i64",
                0,
                {},
            )

    writable = True

    def _serialize(self, out, header, name, tobject_flags):
        out.append(b"\x00\x01" + _tobject_format2.pack(0, tobject_flags))

    @classmethod
    def strided_interpretation(
        cls, file, header=False, tobject_header=True, breadcrumbs=(), original=None
    ):
        members = []
        if tobject_header:
            members.append(("@instance_version", numpy.dtype(">u2")))
            members.append(("@num_bytes", numpy.dtype(">u4")))
            members.append(("@fUniqueID", numpy.dtype(">u4")))
            members.append(("@fBits", numpy.dtype(">u4")))
            members.append(("@pidf", numpy.dtype(">u2")))
        return uproot.interpretation.objects.AsStridedObjects(
            cls, members, original=original
        )

    @classmethod
    def awkward_form(cls, file, context):
        awkward = uproot.extras.awkward()
        contents = {}
        if context["tobject_header"]:
            contents["@instance_version"] = uproot._util.awkward_form(
                numpy.dtype("u2"), file, context
            )
            contents["@num_bytes"] = uproot._util.awkward_form(
                numpy.dtype("u4"), file, context
            )
            contents["@fUniqueID"] = uproot._util.awkward_form(
                numpy.dtype("u4"), file, context
            )
            contents["@fBits"] = uproot._util.awkward_form(
                numpy.dtype("u4"), file, context
            )
            contents["@pidf"] = uproot._util.awkward_form(
                numpy.dtype("u2"), file, context
            )
        return awkward.forms.RecordForm(
            list(contents.values()),
            list(contents.keys()),
            parameters={"__record__": "TObject"},
        )

    def __repr__(self):
        return "<TObject {} {} at 0x{:012x}>".format(
            self._members.get("fUniqueID"), self._members.get("fBits"), id(self)
        )

    def tojson(self):
        return {
            "_typename": self.classname,
            "fUniqueID": self.member("@fUniqueID"),
            "fBits": self.member("@fBits"),
        }


uproot.classes["TObject"] = Model_TObject

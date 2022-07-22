# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

"""
This module defines versioned models for ``TDatime``.
"""


import struct

import numpy

import uproot
import uproot.behaviors.TDatime
import uproot._awkward_forth
import json

_tdatime_format1 = struct.Struct(">I")


class Model_TDatime(uproot.behaviors.TDatime.TDatime, uproot.model.Model):
    """
    A versionless :doc:`uproot.model.Model` for ``TDatime``.
    """

    def read_numbytes_version(self, chunk, cursor, context):
        pass

    def read_members(self, chunk, cursor, context, file):
        helper_obj = uproot._awkward_forth.GenHelper(context)
        if helper_obj.is_forth():
            forth_obj = helper_obj.get_gen_obj()
            key = forth_obj.get_keys(1)
            form_key = f"part0-node{key}-data"
            helper_obj.add_to_header(f"output part0-node{key}-data int32\n")
            helper_obj.add_to_pre(
                f"stream !I-> part0-node{key}-data\n"
            )
            form_key = f"part0-node{key}-data"
            if forth_obj.should_add_form():
                forth_obj.add_form_key(form_key)
                temp_aform = {"class": "RecordArray", "contents": {'fDatime': {"class": "NumpyArray", "primitive": "uint32", "form_key": f"node{key}"}}, "parameters": {"__record__": "TDatime"}}
                forth_obj.add_form(temp_aform)
            temp_form = forth_obj.add_node(
                f"node{key}",
                helper_obj.get_pre(),
                helper_obj.get_post(),
                helper_obj.get_init(),
                helper_obj.get_header(),
                "i64",
                0,
                None,
            )
            forth_obj.go_to(temp_form)
        self._members["fDatime"] = cursor.field(chunk, _tdatime_format1, context)

    @classmethod
    def strided_interpretation(
        cls, file, header=False, tobject_header=True, breadcrumbs=(), original=None
    ):
        members = []
        if header:
            members.append(("@num_bytes", numpy.dtype(">u4")))
            members.append(("@instance_version", numpy.dtype(">u2")))
        members.append(("fDatime", numpy.dtype(">u4")))
        return uproot.interpretation.objects.AsStridedObjects(
            cls, members, original=original
        )

    @classmethod
    def awkward_form(cls, file, context):
        awkward = uproot.extras.awkward()
        contents = {}
        if context["header"]:
            contents["@num_bytes"] = uproot._util.awkward_form(
                numpy.dtype("u4"), file, context
            )
            contents["@instance_version"] = uproot._util.awkward_form(
                numpy.dtype("u2"), file, context
            )
        contents["fDatime"] = uproot._util.awkward_form(
            numpy.dtype(">u4"), file, context
        )
        return awkward.forms.RecordForm(contents, parameters={"__record__": "TDatime"})

    base_names_versions = []
    member_names = ["fDatime"]
    class_flags = {}
    class_code = None


uproot.classes["TDatime"] = Model_TDatime

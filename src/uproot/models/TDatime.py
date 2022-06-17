# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

"""
This module defines versioned models for ``TDatime``.
"""


import struct

import numpy

import uproot
import uproot.behaviors.TDatime

_tdatime_format1 = struct.Struct(">I")


class Model_TDatime(uproot.behaviors.TDatime.TDatime, uproot.model.Model):
    """
    A versionless :doc:`uproot.model.Model` for ``TDatime``.
    """

    def read_numbytes_version(self, chunk, cursor, context):
        pass

    def read_members(self, chunk, cursor, context, file):
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

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

"""
This module defines versioned models for ``TAttLine``, ``TAttFill``, and ``TAttMarker``.
"""

from __future__ import absolute_import

import struct

import numpy

import uproot


_tattline1_format1 = struct.Struct(">hhh")


class Model_TAttLine_v1(uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TAttLine`` version 1.
    """

    def read_members(self, chunk, cursor, context, file):
        if self.is_memberwise:
            raise NotImplementedError(
                """memberwise serialization of {0}
in file {1}""".format(
                    type(self).__name__, self.file.file_path
                )
            )
        (
            self._members["fLineColor"],
            self._members["fLineStyle"],
            self._members["fLineWidth"],
        ) = cursor.fields(chunk, _tattline1_format1, context)

    @classmethod
    def strided_interpretation(
        cls, file, header=False, tobject_header=True, breadcrumbs=(), original=None
    ):
        members = []
        if header:
            members.append(("@num_bytes", numpy.dtype(">u4")))
            members.append(("@instance_version", numpy.dtype(">u2")))
        members.append(("fLineColor", numpy.dtype(">i2")))
        members.append(("fLineStyle", numpy.dtype(">i2")))
        members.append(("fLineWidth", numpy.dtype(">i2")))
        return uproot.interpretation.objects.AsStridedObjects(
            cls, members, original=original
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
        contents["fLineColor"] = uproot._util.awkward_form(
            numpy.dtype("i2"), file, index_format, header, tobject_header, breadcrumbs
        )
        contents["fLineStyle"] = uproot._util.awkward_form(
            numpy.dtype("i2"), file, index_format, header, tobject_header, breadcrumbs
        )
        contents["fLineWidth"] = uproot._util.awkward_form(
            numpy.dtype("i2"), file, index_format, header, tobject_header, breadcrumbs
        )
        return awkward.forms.RecordForm(contents, parameters={"__record__": "TAttLine"})

    base_names_versions = []
    member_names = ["fLineColor", "fLineStyle", "fLineWidth"]
    class_flags = {}
    class_code = None


class Model_TAttLine_v2(uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TAttLine`` version 2.
    """

    def read_members(self, chunk, cursor, context, file):
        if self.is_memberwise:
            raise NotImplementedError(
                """memberwise serialization of {0}
in file {1}""".format(
                    type(self).__name__, self.file.file_path
                )
            )
        (
            self._members["fLineColor"],
            self._members["fLineStyle"],
            self._members["fLineWidth"],
        ) = cursor.fields(chunk, _tattline1_format1, context)

    @classmethod
    def strided_interpretation(
        cls, file, header=False, tobject_header=True, breadcrumbs=(), original=None
    ):
        members = []
        if header:
            members.append(("@num_bytes", numpy.dtype(">u4")))
            members.append(("@instance_version", numpy.dtype(">u2")))
        members.append(("fLineColor", numpy.dtype(">i2")))
        members.append(("fLineStyle", numpy.dtype(">i2")))
        members.append(("fLineWidth", numpy.dtype(">i2")))
        return uproot.interpretation.objects.AsStridedObjects(
            cls, members, original=original
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
        contents["fLineColor"] = uproot._util.awkward_form(
            numpy.dtype("i2"), file, index_format, header, tobject_header, breadcrumbs
        )
        contents["fLineStyle"] = uproot._util.awkward_form(
            numpy.dtype("i2"), file, index_format, header, tobject_header, breadcrumbs
        )
        contents["fLineWidth"] = uproot._util.awkward_form(
            numpy.dtype("i2"), file, index_format, header, tobject_header, breadcrumbs
        )
        return awkward.forms.RecordForm(contents, parameters={"__record__": "TAttLine"})

    base_names_versions = []
    member_names = ["fLineColor", "fLineStyle", "fLineWidth"]
    class_flags = {}
    class_code = None


class Model_TAttLine(uproot.model.DispatchByVersion):
    """
    A :doc:`uproot.model.DispatchByVersion` for ``TAttLine``.
    """

    known_versions = {1: Model_TAttLine_v1, 2: Model_TAttLine_v2}


_tattfill1_format1 = struct.Struct(">hh")
_tattfill2_format1 = struct.Struct(">hh")


class Model_TAttFill_v1(uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TAttFill`` version 1.
    """

    def read_members(self, chunk, cursor, context, file):
        if self.is_memberwise:
            raise NotImplementedError(
                """memberwise serialization of {0}
in file {1}""".format(
                    type(self).__name__, self.file.file_path
                )
            )
        self._members["fFillColor"], self._members["fFillStyle"] = cursor.fields(
            chunk, _tattfill1_format1, context
        )

    @classmethod
    def strided_interpretation(
        cls, file, header=False, tobject_header=True, breadcrumbs=(), original=None
    ):
        members = []
        if header:
            members.append(("@num_bytes", numpy.dtype(">u4")))
            members.append(("@instance_version", numpy.dtype(">u2")))
        members.append(("fFillColor", numpy.dtype(">i2")))
        members.append(("fFillStyle", numpy.dtype(">i2")))
        return uproot.interpretation.objects.AsStridedObjects(
            cls, members, original=original
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
        contents["fFillColor"] = uproot._util.awkward_form(
            numpy.dtype("i2"), file, index_format, header, tobject_header, breadcrumbs
        )
        contents["fFillStyle"] = uproot._util.awkward_form(
            numpy.dtype("i2"), file, index_format, header, tobject_header, breadcrumbs
        )
        return awkward.forms.RecordForm(contents, parameters={"__record__": "TAttFill"})

    base_names_versions = []
    member_names = ["fFillColor", "fFillStyle"]
    class_flags = {}
    class_code = None


class Model_TAttFill_v2(uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TAttFill`` version 2.
    """

    def read_members(self, chunk, cursor, context, file):
        if self.is_memberwise:
            raise NotImplementedError(
                """memberwise serialization of {0}
in file {1}""".format(
                    type(self).__name__, self.file.file_path
                )
            )
        self._members["fFillColor"], self._members["fFillStyle"] = cursor.fields(
            chunk, _tattfill2_format1, context
        )

    @classmethod
    def strided_interpretation(
        cls, file, header=False, tobject_header=True, breadcrumbs=(), original=None
    ):
        members = []
        if header:
            members.append(("@num_bytes", numpy.dtype(">u4")))
            members.append(("@instance_version", numpy.dtype(">u2")))
        members.append(("fFillColor", numpy.dtype(">i2")))
        members.append(("fFillStyle", numpy.dtype(">i2")))
        return uproot.interpretation.objects.AsStridedObjects(
            cls, members, original=original
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
        contents["fFillColor"] = uproot._util.awkward_form(
            numpy.dtype("i2"), file, index_format, header, tobject_header, breadcrumbs
        )
        contents["fFillStyle"] = uproot._util.awkward_form(
            numpy.dtype("i2"), file, index_format, header, tobject_header, breadcrumbs
        )
        return awkward.forms.RecordForm(contents, parameters={"__record__": "TAttFill"})

    base_names_versions = []
    member_names = ["fFillColor", "fFillStyle"]
    class_flags = {}
    class_code = None


class Model_TAttFill(uproot.model.DispatchByVersion):
    """
    A :doc:`uproot.model.DispatchByVersion` for ``TAttFill``.
    """

    known_versions = {1: Model_TAttFill_v1, 2: Model_TAttFill_v2}


_tattmarker2_format1 = struct.Struct(">hhf")


class Model_TAttMarker_v2(uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TAttMarker`` version 2.
    """

    def read_members(self, chunk, cursor, context, file):
        if self.is_memberwise:
            raise NotImplementedError(
                """memberwise serialization of {0}
in file {1}""".format(
                    type(self).__name__, self.file.file_path
                )
            )
        (
            self._members["fMarkerColor"],
            self._members["fMarkerStyle"],
            self._members["fMarkerSize"],
        ) = cursor.fields(chunk, _tattmarker2_format1, context)

    @classmethod
    def strided_interpretation(
        cls, file, header=False, tobject_header=True, breadcrumbs=(), original=None
    ):
        members = []
        if header:
            members.append(("@num_bytes", numpy.dtype(">u4")))
            members.append(("@instance_version", numpy.dtype(">u2")))
        members.append(("fMarkerColor", numpy.dtype(">i2")))
        members.append(("fMarkerStyle", numpy.dtype(">i2")))
        members.append(("fMarkerSize", numpy.dtype(">f4")))
        return uproot.interpretation.objects.AsStridedObjects(
            cls, members, original=original
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
        contents["fMarkerColor"] = uproot._util.awkward_form(
            numpy.dtype("i2"), file, index_format, header, tobject_header, breadcrumbs
        )
        contents["fMarkerStyle"] = uproot._util.awkward_form(
            numpy.dtype("i2"), file, index_format, header, tobject_header, breadcrumbs
        )
        contents["fMarkerSize"] = uproot._util.awkward_form(
            numpy.dtype("f4"), file, index_format, header, tobject_header, breadcrumbs
        )
        return awkward.forms.RecordForm(
            contents, parameters={"__record__": "TAttMarker"}
        )

    base_names_versions = []
    member_names = ["fMarkerColor", "fMarkerStyle", "fMarkserSize"]
    class_flags = {}
    class_code = None


class Model_TAttMarker(uproot.model.DispatchByVersion):
    """
    A :doc:`uproot.model.DispatchByVersion` for ``TAttMarker``.
    """

    known_versions = {2: Model_TAttMarker_v2}


uproot.classes["TAttLine"] = Model_TAttLine
uproot.classes["TAttFill"] = Model_TAttFill
uproot.classes["TAttMarker"] = Model_TAttMarker

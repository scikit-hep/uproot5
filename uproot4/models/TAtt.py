# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import struct

import numpy

import uproot4.model


_tattline1_format1 = struct.Struct(">hhh")


class Model_TAttLine_v1(uproot4.model.VersionedModel):
    def read_members(self, chunk, cursor, context):
        (
            self._members["fLineColor"],
            self._members["fLineStyle"],
            self._members["fLineWidth"],
        ) = cursor.fields(chunk, _tattline1_format1, context)

    @classmethod
    def strided_interpretation(cls, file, header=False, tobject_header=True):
        members = []
        if header:
            members.append(("@num_bytes", numpy.dtype(">u4")))
            members.append(("@instance_version", numpy.dtype(">u2")))
        members.append(("fLineColor", numpy.dtype(">i2")))
        members.append(("fLineStyle", numpy.dtype(">i2")))
        members.append(("fLineWidth", numpy.dtype(">i2")))
        return uproot4.interpretation.objects.AsStridedObjects(cls, members)

    base_names_versions = []
    member_names = ["fLineColor", "fLineStyle", "fLineWidth"]
    class_flags = {}
    class_code = None


class Model_TAttLine_v2(uproot4.model.VersionedModel):
    def read_members(self, chunk, cursor, context):
        (
            self._members["fLineColor"],
            self._members["fLineStyle"],
            self._members["fLineWidth"],
        ) = cursor.fields(chunk, _tattline1_format1, context)

    @classmethod
    def strided_interpretation(cls, file, header=False, tobject_header=True):
        members = []
        if header:
            members.append(("@num_bytes", numpy.dtype(">u4")))
            members.append(("@instance_version", numpy.dtype(">u2")))
        members.append(("fLineColor", numpy.dtype(">i2")))
        members.append(("fLineStyle", numpy.dtype(">i2")))
        members.append(("fLineWidth", numpy.dtype(">i2")))
        return uproot4.interpretation.objects.AsStridedObjects(cls, members)

    base_names_versions = []
    member_names = ["fLineColor", "fLineStyle", "fLineWidth"]
    class_flags = {}
    class_code = None


_tattfill1_format1 = struct.Struct(">hh")
_tattfill2_format1 = struct.Struct(">hh")


class Model_TAttFill_v1(uproot4.model.VersionedModel):
    def read_members(self, chunk, cursor, context):
        self._members["fFillColor"], self._members["fFillStyle"] = cursor.fields(
            chunk, _tattfill1_format1, context
        )

    @classmethod
    def strided_interpretation(cls, file, header=False, tobject_header=True):
        members = []
        if header:
            members.append(("@num_bytes", numpy.dtype(">u4")))
            members.append(("@instance_version", numpy.dtype(">u2")))
        members.append(("fFillColor", numpy.dtype(">i2")))
        members.append(("fFillStyle", numpy.dtype(">i2")))
        return uproot4.interpretation.objects.AsStridedObjects(cls, members)

    base_names_versions = []
    member_names = ["fFillColor", "fFillStyle"]
    class_flags = {}
    class_code = None


class Model_TAttFill_v2(uproot4.model.VersionedModel):
    def read_members(self, chunk, cursor, context):
        self._members["fFillColor"], self._members["fFillStyle"] = cursor.fields(
            chunk, _tattfill2_format1, context
        )

    @classmethod
    def strided_interpretation(cls, file, header=False, tobject_header=True):
        members = []
        if header:
            members.append(("@num_bytes", numpy.dtype(">u4")))
            members.append(("@instance_version", numpy.dtype(">u2")))
        members.append(("fFillColor", numpy.dtype(">i2")))
        members.append(("fFillStyle", numpy.dtype(">i2")))
        return uproot4.interpretation.objects.AsStridedObjects(cls, members)

    base_names_versions = []
    member_names = ["fFillColor", "fFillStyle"]
    class_flags = {}
    class_code = None


_tattmarker2_format1 = struct.Struct(">hhf")


class Model_TAttMarker_v2(uproot4.model.VersionedModel):
    def read_members(self, chunk, cursor, context):
        (
            self._members["fMarkerColor"],
            self._members["fMarkerStyle"],
            self._members["fMarkerSize"],
        ) = cursor.fields(chunk, _tattmarker2_format1, context)

    @classmethod
    def strided_interpretation(cls, file, header=False, tobject_header=True):
        members = []
        if header:
            members.append(("@num_bytes", numpy.dtype(">u4")))
            members.append(("@instance_version", numpy.dtype(">u2")))
        members.append(("fMarkerColor", numpy.dtype(">i2")))
        members.append(("fMarkerStyle", numpy.dtype(">i2")))
        members.append(("fMarkerSize", numpy.dtype(">f4")))
        return uproot4.interpretation.objects.AsStridedObjects(cls, members)

    base_names_versions = []
    member_names = ["fMarkerColor", "fMarkerStyle", "fMarkserSize"]
    class_flags = {}
    class_code = None


class Model_TAttLine(uproot4.model.DispatchByVersion):
    known_versions = {1: Model_TAttLine_v1, 2: Model_TAttLine_v2}


class Model_TAttFill(uproot4.model.DispatchByVersion):
    known_versions = {1: Model_TAttFill_v1, 2: Model_TAttFill_v2}


class Model_TAttMarker(uproot4.model.DispatchByVersion):
    known_versions = {2: Model_TAttMarker_v2}


uproot4.classes["TAttLine"] = Model_TAttLine
uproot4.classes["TAttFill"] = Model_TAttFill
uproot4.classes["TAttMarker"] = Model_TAttMarker

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import struct

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence

import numpy

import uproot4.model


_tref_format1 = struct.Struct(">xxIxxxxxx")


class Model_TRef(uproot4.model.Model):
    def read_numbytes_version(self, chunk, cursor, context):
        pass

    def read_members(self, chunk, cursor, context):
        self._id = cursor.field(chunk, _tref_format1, context)

    @property
    def id(self):
        return self._id

    def __repr__(self):
        return "<TRef {0}>".format(self._id)

    @classmethod
    def strided_interpretation(
        cls, file, header=False, tobject_header=True, original=None
    ):
        members = []
        members.append(("@pidf", numpy.dtype(">u2")))
        members.append(("id", numpy.dtype(">u4")))
        members.append(("@other1", numpy.dtype(">u2")))
        members.append(("@other2", numpy.dtype(">u4")))

        return uproot4.interpretation.objects.AsStridedObjects(
            cls, members, original=original
        )

    @classmethod
    def awkward_form(cls, file, header=False, tobject_header=True):
        import awkward1

        contents = {}
        if tobject_header:
            contents["@pidf"] = uproot4._util.awkward_form(
                numpy.dtype("u2"), file, header, tobject_header
            )
            contents["id"] = uproot4._util.awkward_form(
                numpy.dtype("u4"), file, header, tobject_header
            )
            contents["@other1"] = uproot4._util.awkward_form(
                numpy.dtype("u2"), file, header, tobject_header
            )
            contents["@other2"] = uproot4._util.awkward_form(
                numpy.dtype("u4"), file, header, tobject_header
            )
        return awkward1.forms.RecordForm(contents, parameters={"__record__": "TRef"},)


_trefarray_format1 = struct.Struct(">i")
_trefarray_dtype = numpy.dtype(">i4")


class Model_TRefArray(uproot4.model.Model, Sequence):
    def read_members(self, chunk, cursor, context):
        cursor.skip(10)
        self._members["fName"] = cursor.string(chunk, context)
        self._members["fSize"] = cursor.field(chunk, _trefarray_format1, context)
        cursor.skip(6)
        self._data = cursor.array(
            chunk, self._members["fSize"], _trefarray_dtype, context
        )

    @property
    def name(self):
        return self._members["fName"]

    @property
    def nbytes(self):
        return self._data.nbytes

    def __getitem__(self, where):
        return self._data[where]

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return "<{0} {1} at 0x{2:012x}>".format(
            uproot4.model.classname_pretty(self.classname, self.class_version),
            str(self._data),
            id(self),
        )


uproot4.classes["TRef"] = Model_TRef
uproot4.classes["TRefArray"] = Model_TRefArray

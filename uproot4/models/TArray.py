# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import struct

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence

import numpy

import uproot4.model
import uproot4.deserialization


_tarray_format1 = struct.Struct(">i")


class Class_TArray(uproot4.model.Model, Sequence):
    def read_members(self, chunk, cursor):
        self._members["fN"] = cursor.field(chunk, _tarray_format1)
        self._data = cursor.array(chunk, self._members["fN"], self.dtype)

    def __array__(self):
        return self._data

    def __getitem__(self, where):
        return self._data[where]

    def __len__(self):
        return len(self._data)


class Class_TArrayC(Class_TArray):
    dtype = numpy.dtype(">i1")


class Class_TArrayS(Class_TArray):
    dtype = numpy.dtype(">i2")


class Class_TArrayI(Class_TArray):
    dtype = numpy.dtype(">i4")


class Class_TArrayL(Class_TArray):
    dtype = numpy.dtype(numpy.int_).newbyteorder(">")


class Class_TArrayL64(Class_TArray):
    dtype = numpy.dtype(">i8")


class Class_TArrayF(Class_TArray):
    dtype = numpy.dtype(">f4")


class Class_TArrayD(Class_TArray):
    dtype = numpy.dtype(">f8")


uproot4.classes["TArray"] = Class_TArray
uproot4.classes["TArrayC"] = Class_TArrayC
uproot4.classes["TArrayS"] = Class_TArrayS
uproot4.classes["TArrayI"] = Class_TArrayI
uproot4.classes["TArrayL"] = Class_TArrayL
uproot4.classes["TArrayL64"] = Class_TArrayL64
uproot4.classes["TArrayF"] = Class_TArrayF
uproot4.classes["TArrayD"] = Class_TArrayD

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

"""
This module defines versioned models for ``TLeaf`` and its subclasses.
"""


import struct

import numpy

import uproot
import uproot._util
import uproot.model


class Model_TMatrixTSym_3c_double_3e__v5(uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TMatrixTSym<double>`` version 2,
    which shows up as version 5 because it's reading the ``TMatrixTBase<double>``
    header.
    """

    def read_members(self, chunk, cursor, context, file):
        context["cancel_forth"] = True
        if self.is_memberwise:
            raise NotImplementedError(
                "memberwise serialization of {}\nin file {}".format(
                    type(self).__name__, self.file.file_path
                )
            )
        self._bases.append(
            file.class_named("TObject", 1).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        (
            self._members["fNrows"],
            self._members["fNcols"],
            self._members["fRowLwb"],
            self._members["fColLwb"],
            self._members["fNelems"],
            self._members["fNrowIndex"],
            self._members["fTol"],
        ) = cursor.fields(chunk, self._format0, context)

        num_elements = self.member("fNrows") * (self.member("fNcols") + 1) // 2
        self._members["fElements"] = cursor.array(
            chunk, num_elements, self._dtype0, context
        )
        self._num_bytes += self._members["fElements"].nbytes

    _format0 = struct.Struct(">iiiiiid")
    _format_memberwise0 = struct.Struct(">i")
    _format_memberwise1 = struct.Struct(">i")
    _format_memberwise2 = struct.Struct(">i")
    _format_memberwise3 = struct.Struct(">i")
    _format_memberwise4 = struct.Struct(">i")
    _format_memberwise5 = struct.Struct(">i")
    _format_memberwise6 = struct.Struct(">d")
    _dtype0 = numpy.dtype(">f8")
    base_names_versions = [("TObject", 1)]
    member_names = [
        "fNrows",
        "fNcols",
        "fRowLwb",
        "fColLwb",
        "fNelems",
        "fNrowIndex",
        "fTol",
    ]
    class_flags = {}


class Model_TMatrixTSym_3c_double_3e_(uproot.model.DispatchByVersion):
    """
    A :doc:`uproot.model.DispatchByVersion` for ``TMatrixTSym<double>``.
    """

    known_versions = {5: Model_TMatrixTSym_3c_double_3e__v5}


uproot.classes["TMatrixTSym<double>"] = Model_TMatrixTSym_3c_double_3e_

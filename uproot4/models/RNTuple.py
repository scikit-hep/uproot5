# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import struct

import uproot4.model
import uproot4.deserialization


_rntuple_format1 = struct.Struct(">IIQIIQIIQ")


class Model_ROOT_3a3a_Experimental_3a3a_RNTuple(uproot4.model.Model):
    def read_members(self, chunk, cursor, context):
        cursor.skip(4)
        (
            self._members["fVersion"],
            self._members["fSize"],
            self._members["fSeekHeader"],
            self._members["fNBytesHeader"],
            self._members["fLenHeader"],
            self._members["fSeekFooter"],
            self._members["fNBytesFooter"],
            self._members["fLenFooter"],
            self._members["fReserved"],
        ) = cursor.fields(chunk, _rntuple_format1, context)


uproot4.classes[
    "ROOT::Experimental::RNTuple"
] = Model_ROOT_3a3a_Experimental_3a3a_RNTuple

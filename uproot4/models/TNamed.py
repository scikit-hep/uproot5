# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import uproot4.model
import uproot4.deserialization


class Class_TNamed(uproot4.model.Model):
    def read_members(self, chunk, cursor, context):
        uproot4.deserialization.skip_tobject(chunk, cursor)
        self._members["fName"] = cursor.string(chunk)
        self._members["fTitle"] = cursor.string(chunk)


uproot4.classes["TNamed"] = Class_TNamed

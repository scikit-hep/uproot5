# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import uproot4.deserialization
import uproot4.models.TList


class Class_THashList(uproot4.models.TList.Class_TList):
    def read_members(self, chunk, cursor, context):
        self._bases.append(
            uproot4.models.TList.read(chunk, cursor, context, self._file, self._parent)
        )


uproot4.classes["THashList"] = Class_THashList

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import uproot4.models.TList


class Model_THashList(uproot4.model.Model):
    def read_members(self, chunk, cursor, context):
        self._bases.append(
            uproot4.models.TList.Model_TList.read(
                chunk, cursor, context, self._file, self._parent
            )
        )


uproot4.classes["THashList"] = Model_THashList

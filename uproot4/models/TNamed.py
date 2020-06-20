# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import uproot4.model
import uproot4.models.TObject


class Model_TNamed(uproot4.model.Model):
    def read_members(self, chunk, cursor, context):
        self._bases.append(
            uproot4.models.TObject.Model_TObject.read(
                chunk, cursor, context, self._file, self._parent
            )
        )

        self._members["fName"] = cursor.string(chunk, context)
        self._members["fTitle"] = cursor.string(chunk, context)

    def __repr__(self):
        title = ""
        if self._members["fTitle"] != "":
            title = " title=" + repr(self._members["fTitle"])
        return "<TNamed {0}{1} at 0x{2:012x}>".format(
            repr(self._members["fName"]), title, id(self)
        )


uproot4.classes["TNamed"] = Model_TNamed

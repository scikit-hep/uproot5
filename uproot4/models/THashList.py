# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import uproot4.models.TList


class Model_THashList(uproot4.model.Model):
    def read_numbytes_version(self, chunk, cursor, context):
        pass

    def read_members(self, chunk, cursor, context):
        self._bases.append(
            uproot4.models.TList.Model_TList.read(
                chunk, cursor, context, self._file, self._parent
            )
        )

    def __repr__(self):
        return "<{0} of {1} items at 0x{2:012x}>".format(
            uproot4.model.classname_pretty(self.classname, self.class_version),
            len(self),
            id(self),
        )

    def __getitem__(self, where):
        return self._bases[0][where]

    def __len__(self):
        return len(self._bases[0])


uproot4.classes["THashList"] = Model_THashList

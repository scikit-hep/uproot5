# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

"""
This module defines a versionless model for ``THashList``.
"""

from __future__ import absolute_import

import uproot


class Model_THashList(uproot.model.Model):
    """
    A versionless :doc:`uproot.model.Model` for ``THashList``.
    """

    def read_numbytes_version(self, chunk, cursor, context):
        pass

    def read_members(self, chunk, cursor, context, file):
        if self.is_memberwise:
            raise NotImplementedError(
                """memberwise serialization of {0}
in file {1}""".format(
                    type(self).__name__, self.file.file_path
                )
            )
        self._bases.append(
            uproot.models.TList.Model_TList.read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self._concrete,
            )
        )

    def __repr__(self):
        if self.class_version is None:
            version = ""
        else:
            version = " (version {0})".format(self.class_version)
        return "<{0}{1} of {2} items at 0x{3:012x}>".format(
            self.classname, version, len(self), id(self),
        )

    def __getitem__(self, where):
        return self._bases[0][where]

    def __len__(self):
        return len(self._bases[0])


uproot.classes["THashList"] = Model_THashList

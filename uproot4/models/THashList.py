# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

"""
Defines a versionless model for ``THashList``.
"""

from __future__ import absolute_import

import uproot4.models.TList


class Model_THashList(uproot4.model.Model):
    """
    A versionless :py:class:`~uproot4.model.Model` for ``THashList``.
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
            uproot4.models.TList.Model_TList.read(
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


uproot4.classes["THashList"] = Model_THashList

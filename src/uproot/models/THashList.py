# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

"""
This module defines a versionless model for ``THashList``.
"""


import uproot


class Model_THashList(uproot.model.Model):
    """
    A versionless :doc:`uproot.model.Model` for ``THashList``.
    """

    def read_numbytes_version(self, chunk, cursor, context):
        pass

    def read_members(self, chunk, cursor, context, file):
        context["cancel_forth"] = True
        if self.is_memberwise:
            raise NotImplementedError(
                """memberwise serialization of {}
in file {}""".format(
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
                concrete=self.concrete,
            )
        )

    def __repr__(self):
        if self.class_version is None:
            version = ""
        else:
            version = f" (version {self.class_version})"
        return "<{}{} of {} items at 0x{:012x}>".format(
            self.classname,
            version,
            len(self),
            id(self),
        )

    def __getitem__(self, where):
        return self._bases[0][where]

    def __len__(self):
        return len(self._bases[0])

    writable = True

    def _serialize(self, out, header, name, tobject_flags):
        assert len(self._bases) == 1, "Fatal error on THashList serialization."
        for x in self._bases:
            x._serialize(out, True, None, tobject_flags)


uproot.classes["THashList"] = Model_THashList

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

"""
This module defines a versionless model of ``TString``.
"""


import uproot


class Model_TString(uproot.model.Model, str):
    """
    A versionless :doc:`uproot.model.Model` for ``TString``.

    This is also a Python ``str`` (string).
    """

    def read_numbytes_version(self, chunk, cursor, context):
        pass

    def read_members(self, chunk, cursor, context, file):
        if self.is_memberwise:
            raise NotImplementedError(
                """memberwise serialization of {}
in file {}""".format(
                    type(self).__name__, self.file.file_path
                )
            )
        self._data = cursor.string(chunk, context)

    def postprocess(self, chunk, cursor, context, file):
        out = Model_TString(self._data)
        out._cursor = self._cursor
        out._file = self._file
        out._parent = self._parent
        out._members = self._members
        out._bases = self._bases
        out._num_bytes = self._num_bytes
        out._instance_version = self._instance_version
        return out

    def __repr__(self):
        if self.class_version is None:
            version = ""
        else:
            version = f" (version {self.class_version})"
        return "<{}{} {} at 0x{:012x}>".format(
            self.classname, version, str.__repr__(self), id(self)
        )

    def tojson(self):
        return str(self)

    @classmethod
    def awkward_form(cls, file, context):
        return uproot.containers.AsString(False, typename="TString").awkward_form(
            file, context
        )

    writable = True
    _is_memberwise = False

    def _serialize(self, out, header, name, tobject_flags):
        import uproot.writing._cascade

        where = len(out)
        for x in self._bases:
            x._serialize(out, True, None, tobject_flags)

        out.append(uproot.serialization.string(self))

        if header:
            num_bytes = sum(len(x) for x in out[where:])
            version = 2
            out.insert(where, uproot.serialization.numbytes_version(num_bytes, version))


uproot.classes["TString"] = Model_TString

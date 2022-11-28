# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

"""
This module defines a versionless model of ``TString``.
"""


import uproot
import uproot._awkward_forth


class Model_TString(uproot.model.Model, str):
    """
    A versionless :doc:`uproot.model.Model` for ``TString``.

    This is also a Python ``str`` (string).
    """

    def read_numbytes_version(self, chunk, cursor, context):
        pass

    def read_members(self, chunk, cursor, context, file):
        forth_stash = uproot._awkward_forth.forth_stash(context)
        if forth_stash is not None:
            forth_obj = forth_stash.get_gen_obj()
            keys = forth_obj.get_keys(2)
            offsets_num = keys[0]
            data_num = keys[1]
        if self.is_memberwise:
            raise NotImplementedError(
                """memberwise serialization of {}
in file {}""".format(
                    type(self).__name__, self.file.file_path
                )
            )
        if forth_stash is not None:
            forth_stash.add_to_pre(
                f" stream !B-> stack dup 255 = if drop stream !I-> stack then dup node{offsets_num}-offsets +<- stack stream #!B-> node{data_num}-data\n"
            )
            if forth_obj.should_add_form():
                temp_aform = {
                    "class": "ListOffsetArray",
                    "offsets": "i64",
                    "content": {
                        "class": "NumpyArray",
                        "primitive": "uint8",
                        "inner_shape": [],
                        "parameters": {"__array__": "char"},
                        "form_key": f"node{data_num}",
                    },
                    "parameters": {"__array__": "string"},
                    "form_key": f"node{offsets_num}",
                }
                forth_obj.add_form(temp_aform)

                form_keys = [
                    f"node{data_num}-data",
                    f"node{offsets_num}-offsets",
                ]
                for elem in form_keys:
                    forth_obj.add_form_key(elem)
            forth_stash.add_to_header(
                f"output node{offsets_num}-offsets int64\noutput node{data_num}-data uint8\n"
            )
            forth_stash.add_to_init(f"0 node{offsets_num}-offsets <- stack\n")
            temp_form = forth_obj.add_node(
                f"node{offsets_num}",
                forth_stash.get_attrs(),
                "i64",
                0,
                None,
            )
            forth_obj.go_to(temp_form)
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

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import uproot4.interpretation
import uproot4.extras


class Group(object):
    pass


class AsGrouped(uproot4.interpretation.Interpretation):
    def __init__(self, branch, subbranches, typename=None):
        self._branch = branch
        self._subbranches = subbranches
        self._typename = typename

    @property
    def branch(self):
        return self._branch

    @property
    def subbranches(self):
        return self._subbranches

    def __repr__(self):
        return "AsGroup({0}, {1})".format(self._branch, self._subbranches)

    def __eq__(self, other):
        return (
            isinstance(other, AsGrouped)
            and self._branch == other._branch
            and self._subbranches == other._subbranches
        )

    @property
    def cache_key(self):
        return "{0}({1},[{2}])".format(
            type(self).__name__,
            self._branch.name,
            ",".join(
                "{0}:{1}".format(repr(x), y.cache_key)
                for x, y in self._subbranches.items()
            ),
        )

    @property
    def typename(self):
        if self._typename is not None:
            return self._typename
        else:
            return "(group of {0})".format(
                ", ".join(
                    "{0}:{1}".format(x, y.typename)
                    for x, y in self._subbranches.items()
                )
            )

    def awkward_form(self, file, index_format="i64", header=False, tobject_header=True):
        awkward1 = uproot4.extras.awkward1()

        record = {}
        for x, y in self._subbranches.items():
            record[x] = y.awkward_form(file, index_format, header, tobject_header)

        return awkward1.forms.RecordForm(record)

    def basket_array(self, data, byte_offsets, basket, branch, context, cursor_offset):
        raise ValueError(
            """grouping branches like {0} should not be read directly; instead read the subbranches:

    {1}

in file {2}
in object {3}""".format(
                repr(self._branch.name),
                ", ".join(repr(x) for x in self._subbranches),
                self._branch.file.file_path,
                self._branch.object_path,
            )
        )

    def final_array(
        self, basket_arrays, entry_start, entry_stop, entry_offsets, library, branch
    ):
        raise ValueError(
            """grouping branches like {0} should not be read directly; instead read the subbranches:

    {1}

in file {2}
in object {3}""".format(
                repr(self._branch.name),
                ", ".join(repr(x) for x in self._subbranches),
                self._branch.file.file_path,
                self._branch.object_path,
            )
        )

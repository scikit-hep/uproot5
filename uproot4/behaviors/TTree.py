# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import uproot4.behaviors.TBranch


class TTree(uproot4.behaviors.TBranch.HasBranches):
    @property
    def name(self):
        return self.member("fName")

    @property
    def title(self):
        return self.member("fTitle")

    @property
    def num_entries(self):
        return self.member("fEntries")

    @property
    def aliases(self):
        aliases = self.member("fAliases")
        if aliases is None:
            return {}
        else:
            return dict(
                (alias.member("fName"), alias.member("fTitle")) for alias in aliases
            )

    @property
    def tree(self):
        return self

    def __repr__(self):
        if len(self) == 0:
            return "<TTree {0} at 0x{1:012x}>".format(repr(self.name), id(self))
        else:
            return "<TTree {0} ({1} branches) at 0x{2:012x}>".format(
                repr(self.name), len(self), id(self)
            )

    def postprocess(self, chunk, cursor, context):
        self._chunk = chunk

        self._lookup = {}

        return self

    @property
    def cache_key(self):
        return self.parent.parent.cache_key + self.name

    @property
    def object_path(self):
        return self.parent.parent.object_path + self.name

    @property
    def chunk(self):
        return self._chunk

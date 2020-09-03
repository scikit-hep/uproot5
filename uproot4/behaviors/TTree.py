# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

"""
Defines the behaviors of ``TTree``, which is almost entirely inherited from
functions in :doc:`uproot4.behaviors.TBranch`.
"""

from __future__ import absolute_import

import uproot4.behaviors.TBranch


class TTree(uproot4.behaviors.TBranch.HasBranches):
    """
    Behaviors for a ``TTree``, which mostly consist of array-reading methods.

    Since a :doc:`uproot4.behavior.TTree.TTree` is a
    :doc:`uproot4.behavior.TBranch.HasBranches`, it is also a Python
    ``Mapping``, which uses square bracket syntax to extract subbranches:

    .. code-block:: python

        my_tree["branch"]
        my_tree["branch"]["subbranch"]
        my_tree["branch/subbranch"]
        my_tree["branch/subbranch/subsubbranch"]
    """

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
        aliases = self.member("fAliases", none_if_missing=True)
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

    def postprocess(self, chunk, cursor, context, file):
        self._chunk = chunk
        self._lookup = {}
        return self

    @property
    def cache_key(self):
        return "{0}{1};{2}".format(
            self.parent.parent.cache_key, self.name, self.parent.fCycle
        )

    @property
    def object_path(self):
        return self.parent.object_path

    @property
    def chunk(self):
        return self._chunk

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

"""
This module defines the behaviors of ``TTree``, which is almost entirely inherited from
functions in :doc:`uproot.behaviors.TBranch`.
"""

from __future__ import absolute_import

import uproot


class TTree(uproot.behaviors.TBranch.HasBranches):
    """
    Behaviors for a ``TTree``, which mostly consist of array-reading methods.

    Since a :doc:`uproot.behaviors.TTree.TTree` is a
    :doc:`uproot.behaviors.TBranch.HasBranches`, it is also a Python
    ``Mapping``, which uses square bracket syntax to extract subbranches:

    .. code-block:: python

        my_tree["branch"]
        my_tree["branch"]["subbranch"]
        my_tree["branch/subbranch"]
        my_tree["branch/subbranch/subsubbranch"]
    """

    def __repr__(self):
        if len(self) == 0:
            return "<TTree {0} at 0x{1:012x}>".format(repr(self.name), id(self))
        else:
            return "<TTree {0} ({1} branches) at 0x{2:012x}>".format(
                repr(self.name), len(self), id(self)
            )

    @property
    def name(self):
        """
        Name of the ``TTree``.
        """
        return self.member("fName")

    @property
    def title(self):
        """
        Title of the ``TTree``.
        """
        return self.member("fTitle")

    @property
    def object_path(self):
        """
        Object path of the ``TTree``.
        """
        return self.parent.object_path

    @property
    def cache_key(self):
        """
        String that uniquely specifies this ``TTree`` in its path, to use as
        part of object and array cache keys.
        """
        return "{0}{1};{2}".format(
            self.parent.parent.cache_key, self.name, self.parent.fCycle
        )

    @property
    def num_entries(self):
        """
        The number of entries in the ``TTree``, as reported by ``fEntries``.

        In principle, this could disagree with the
        :ref:`uproot.behaviors.TBranch.TBranch.num_entries`, which is from the
        ``TBranch``'s ``fEntries``. See that method's documentation for yet more
        ways the number of entries could, in principle, be inconsistent.
        """
        return self.member("fEntries")

    @property
    def tree(self):
        """
        Returns ``self`` because this is a ``TTree``.
        """
        return self

    @property
    def aliases(self):
        u"""
        The ``TTree``'s ``fAliases``, which are used as the ``aliases``
        argument to :ref:`uproot.behaviors.TBranch.HasBranches.arrays`,
        :ref:`uproot.behaviors.TBranch.HasBranches.iterate`,
        :doc:`uproot.behaviors.TBranch.iterate`, and
        :doc:`uproot.behaviors.TBranch.concatenate` if one is not given.

        The return type is always a dict of str \u2192 str, even if there
        are no aliases (an empty dict).
        """
        aliases = self.member("fAliases", none_if_missing=True)
        if aliases is None:
            return {}
        else:
            return dict(
                (alias.member("fName"), alias.member("fTitle")) for alias in aliases
            )

    @property
    def chunk(self):
        """
        The :doc:`uproot.source.chunk.Chunk` from which this ``TTree`` was
        read (as a strong, not weak, reference).

        The reason the chunk is retained is to read
        :ref:`uproot.behaviors.TBranch.TBranch.embedded_baskets` only if
        necessary (if the file was opened with
        ``options["minimal_ttree_metadata"]=True``, the reading of these
        ``TBaskets`` is deferred until they are accessed).

        Holding a strong reference to a chunk holds a strong reference to
        its :doc:`uproot.source.chunk.Source`, preventing open file handles
        from going out of scope, but so does the
        :doc:`uproot.reading.ReadOnlyFile` that ``TTree`` needs to read data
        on demand.
        """
        return self._chunk

    def postprocess(self, chunk, cursor, context, file):
        self._chunk = chunk
        self._lookup = {}
        return self

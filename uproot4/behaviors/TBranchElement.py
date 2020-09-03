# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

"""
Defines the behaviors of ``TBranchElement``, which is entirely inherited from
functions in :doc:`uproot4.behaviors.TBranch.TBranch`.
"""

from __future__ import absolute_import

import uproot4.behaviors.TBranch


class TBranchElement(uproot4.behaviors.TBranch.TBranch):
    """
    Behaviors for a ``TBranchElement``, which mostly consist of array-reading
    methods.

    Since a :doc:`uproot4.behavior.TBranchElement.TBranchElement` is a
    :doc:`uproot4.behavior.TBranch.HasBranches`, it is also a Python
    ``Mapping``, which uses square bracket syntax to extract subbranches:

    .. code-block:: python

        my_branchelement["subbranch"]
        my_branchelement["subbranch"]["subsubbranch"]
        my_branchelement["subbranch/subsubbranch"]
    """

    pass

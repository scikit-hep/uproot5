# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

"""
Defines compute engines for expressions passed to
:doc:`uproot4.behavior.TBranch.HasBranches.arrays`.

The default is :doc:`uproot4.compute.python.ComputePython`.

All compute engines must be subclasses of :doc:`uproot4.compute.Compute`.
"""

from __future__ import absolute_import


class Compute(object):
    """
    Superclass of all computation backends for expressions passed to
    :doc:`uproot4.behavior.TBranch.HasBranches.arrays`.

    The default is :doc:`uproot4.compute.python.ComputePython`.
    """

    pass

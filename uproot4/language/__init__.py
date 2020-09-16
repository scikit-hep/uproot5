# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

"""
Defines languages for expressions passed to
:py:meth:`~uproot4.behavior.TBranch.HasBranches.arrays` (and similar).

The default is :py:class:`~uproot4.language.python.PythonLanguage`.

All languages must be subclasses of :py:class:`~uproot4.language.Language`.
"""

from __future__ import absolute_import


class Language(object):
    """
    Abstract class for all languages, which are used to compute the expressions
    that are passed to :py:meth:`~uproot4.behavior.TBranch.HasBranches.arrays` (and
    similar).

    The default is :py:class:`~uproot4.language.python.PythonLanguage`.
    """

    pass

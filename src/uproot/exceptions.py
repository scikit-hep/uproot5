# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

"""
This module defines Uproot-specific exceptions, such as
:doc:`uproot.exceptions.KeyInFileError`.
"""

from __future__ import absolute_import

import uproot


class KeyInFileError(KeyError):
    """
    Exception raised by attempts to find ROOT objects in ``TDirectories``
    or ``TBranches`` in :doc:`uproot.behaviors.TBranch.HasBranches`, which
    both have a Python ``Mapping`` interface (square bracket syntax to extract
    items).

    This exception descends from Python's ``KeyError``, so it can be used in
    the normal way by interfaces that expect a missing item in a ``Mapping``
    to raise ``KeyError``, but it provides more information, depending on
    availability:

    * ``because``: an explanatory message
    * ``cycle``: the ROOT cycle number requested, if any
    * ``keys``: a list or partial list of keys that *are* in the object, in case
      of misspelling
    * ``file_path``: a path (or URL) to the file
    * ``object_path``: a path to the object within the ROOT file.
    """

    def __init__(
        self, key, because="", cycle=None, keys=None, file_path=None, object_path=None
    ):
        super(KeyInFileError, self).__init__(key)
        self.key = key
        self.because = because
        self.cycle = cycle
        self.keys = keys
        self.file_path = file_path
        self.object_path = object_path

    def __str__(self):
        if self.because == "":
            because = ""
        else:
            because = " because " + self.because

        with_keys = ""
        if self.keys is not None:
            to_show = None
            sorted_keys = sorted(
                self.keys, key=lambda x: uproot._util.damerau_levenshtein(self.key, x)
            )
            for key in sorted_keys:
                if to_show is None:
                    to_show = repr(key)
                else:
                    to_show += ", " + repr(key)
                if len(to_show) > 200:
                    to_show += "..."
                    break
            if to_show is None:
                to_show = "(none!)"
            with_keys = "\n\n    Available keys: {0}\n".format(to_show)

        in_file = ""
        if self.file_path is not None:
            in_file = "\nin file {0}".format(self.file_path)

        in_object = ""
        if self.object_path is not None:
            in_object = "\nin object {0}".format(self.object_path)

        if self.cycle == "any":
            return """not found: {0} (with any cycle number){1}{2}{3}{4}""".format(
                repr(self.key), because, with_keys, in_file, in_object
            )
        elif self.cycle is None:
            return """not found: {0}{1}{2}{3}{4}""".format(
                repr(self.key), because, with_keys, in_file, in_object
            )
        else:
            return """not found: {0} with cycle {1}{2}{3}{4}{5}""".format(
                repr(self.key), self.cycle, because, with_keys, in_file, in_object
            )

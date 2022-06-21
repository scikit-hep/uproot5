# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

"""
This module defines the behaviors of ``TDatime``.
"""


import uproot


class TDatime:
    """
    Behaviors for TDatime: return values as py:class:`datetime.datetime`
    """

    def to_datetime(self):
        return uproot._util.code_to_datetime(self._members["fDatime"])

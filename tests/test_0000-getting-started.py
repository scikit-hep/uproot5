# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import uproot4

def test():
    assert uproot4.stuff() == "This is a test."

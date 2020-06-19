# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

try:
    from collections.abc import Sequence
    # from collections.abc import Set
    # from collections.abc import Mapping
except ImportError:
    from collections import Sequence
    # from collections import Set
    # from collections import Mapping

# import numpy

import uproot4._util


class STLContainer(object):
    pass


def _tostring(value):
    if uproot4._util.isstr(value):
        return repr(value)
    else:
        return str(value)


def _str_with_ellipsis(tostring, length, lbracket, rbracket, limit_value):
    leftlen = len(lbracket)
    rightlen = len(rbracket)
    left, right, i, j, done = [], [], 0, length - 1, False

    while True:
        if i > j:
            done = True
            break
        x = tostring(i) + ("" if i == length - 1 else ", ")
        i += 1
        dotslen = 0 if i > j else 5
        if leftlen + rightlen + len(x) + dotslen > limit_value:
            break
        left.append(x)
        leftlen += len(x)

        if i > j:
            done = True
            break
        y = tostring(j) + ("" if j == length - 1 else ", ")
        j -= 1
        dotslen = 0 if i > j else 5
        if leftlen + rightlen + len(y) + dotslen > limit_value:
            break
        right.insert(0, y)
        rightlen += len(y)

    if length == 0:
        return lbracket + rbracket
    elif done:
        return lbracket + "".join(left) + "".join(right) + rbracket
    elif len(left) == 0 and len(right) == 0:
        return lbracket + "{0}, ...".format(tostring(0)) + rbracket
    elif len(right) == 0:
        return lbracket + "".join(left) + "..." + rbracket
    else:
        return lbracket + "".join(left) + "..., " + "".join(right) + rbracket


class STLVector(STLContainer, Sequence):
    def __init__(self, content):
        self._content = content

    def __str__(self, limit_value=85):
        def tostring(i):
            return _tostring(self[i])

        return _str_with_ellipsis(tostring, len(self), "[", "]", limit_value)

    def __repr__(self, limit_value=85):
        return "<STLVector {0} at 0x{1:012x}>".format(
            self.__str__(limit_value=limit_value - 30), id(self)
        )

    def __getitem__(self, where):
        return self._content[where]

    def __len__(self):
        return len(self._content)

    def __contains__(self, what):
        return what in self._content

    def __iter__(self):
        return iter(self._content)

    def __reversed__(self):
        return STLVector(self._content[::-1])

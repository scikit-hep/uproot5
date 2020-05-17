# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import uproot4._const
import uproot4._util


class Compression(object):
    @classmethod
    def from_code_pair(cls, algorithm, level):
        if algorithm == 0 or level == 0:
            return None
        elif algorithm in algorithm_codes:
            return algorithm_codes[algorithm](level)
        else:
            raise ValueError(
                "unrecognized compression algorithm code: {0}".format(algorithm)
            )

    @classmethod
    def from_code(cls, code):
        return cls.from_code_pair(code // 100, code % 100)

    def __init__(self, level):
        self.level = level

    @property
    def level(self):
        return self._level

    @level.setter
    def level(self, value):
        if not uproot4._util.isint(value):
            raise TypeError("Compression level must be an integer")
        if not 0 <= value <= 9:
            raise ValueError("Compression level must be between 0 and 9 (inclusive)")
        self._level = int(value)

    def __repr__(self):
        return "{0}({1})".format(type(self).__name__, self._level)

    @property
    def code_pair(self):
        for const, cls in algorithm_codes.items():
            if type(self) is cls:
                return const, self._level
        else:
            raise ValueError("unrecognized compression type: {0}".format(type(self)))

    @property
    def code(self):
        algorithm, level = self.code_pair
        return algorithm * 100 + level


class ZLIB(Compression):
    pass


class LZMA(Compression):
    pass


class LZ4(Compression):
    pass


class ZSTD(Compression):
    pass


algorithm_codes = {
    uproot4._const.kZLIB: ZLIB,
    uproot4._const.kLZMA: LZMA,
    uproot4._const.kLZ4: LZ4,
    uproot4._const.kZSTD: ZSTD,
}

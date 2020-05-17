# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import re

import uproot4._util


_classname_encode_pattern = re.compile(br"[^a-zA-Z0-9]+")
_classname_decode_version = re.compile(br".*_v([0-9]+)")
_classname_decode_pattern = re.compile(br"_(([0-9a-f][0-9a-f])+)_")

if uproot4._util.py2:

    def _classname_encode_convert(bad_characters):
        g = bad_characters.group(0)
        return b"_" + b"".join("{0:02x}".format(ord(x)).encode() for x in g) + b"_"

    def _classname_decode_convert(hex_characters):
        g = hex_characters.group(1)
        return b"".join(chr(int(g[i : i + 2], 16)) for i in range(0, len(g), 2))


else:

    def _classname_encode_convert(bad_characters):
        g = bad_characters.group(0)
        return b"_" + b"".join("{0:02x}".format(x).encode() for x in g) + b"_"

    def _classname_decode_convert(hex_characters):
        g = hex_characters.group(1)
        return bytes(int(g[i : i + 2], 16) for i in range(0, len(g), 2))


class Model(object):
    @staticmethod
    def classname_encode(classname, version=None):
        if version is None:
            v = ""
        else:
            v = "_v" + str(version)

        raw = classname.encode()
        out = _classname_encode_pattern.sub(_classname_encode_convert, raw)
        return "ROOT_" + out.decode() + v

    @staticmethod
    def classname_decode(classname):
        if not classname.startswith("ROOT_"):
            raise ValueError("not an encoded classname: {0}".format(classname))

        raw = classname[5:].encode()

        m = _classname_decode_version.match(raw)
        if m is None:
            version = None
        else:
            version = int(m.group(1))
            raw = raw[: -len(m.group(1)) - 2]

        out = _classname_decode_pattern.sub(_classname_decode_convert, raw)
        return out.decode(), version

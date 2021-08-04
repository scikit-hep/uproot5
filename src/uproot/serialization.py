# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

"""
FIXME: docstring
"""

from __future__ import absolute_import

import struct

import numpy

import uproot.const
import uproot.deserialization


def string(data):
    """
    FIXME: docstring
    """
    bytestring = data.encode(errors="surrogateescape")
    length = len(bytestring)
    if length < 255:
        return struct.pack(">B%ds" % length, length, bytestring)
    else:
        return struct.pack(">BI%ds" % length, 255, length, bytestring)


def numbytes_version(num_bytes, version):
    """
    FIXME: docstring
    """
    return uproot.deserialization._numbytes_version_1.pack(
        numpy.uint32(num_bytes + 2) | uproot.const.kByteCountMask, version
    )


_serialize_object_any_format1 = struct.Struct(">II")


def _serialize_object_any(out, model, name):
    if model is None:
        out.append(b"\x00\x00\x00\x00")

    else:
        where = len(out)
        model._serialize(out, True, name, numpy.uint32(0x00000000))

        classname = model.classname.encode(errors="surrogateescape") + b"\x00"
        num_bytes = sum(len(x) for x in out[where:]) + len(classname) + 4
        bcnt = numpy.uint32(num_bytes) | uproot.const.kByteCountMask
        tag = uproot.const.kNewClassTag

        out.insert(where, _serialize_object_any_format1.pack(bcnt, tag) + classname)


def serialize_object_any(model, name=None):
    """
    FIXME: docstring
    """
    out = []
    _serialize_object_any(out, model, name)
    return b"".join(out)

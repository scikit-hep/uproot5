# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import struct

import numpy

import uproot4.const
import uproot4.model
import uproot4._util


_numbytes_version_1 = struct.Struct(">IH")
_numbytes_version_2 = struct.Struct(">H")


def numbytes_version(chunk, cursor, move=True):
    num_bytes, version = cursor.fields(chunk, _numbytes_version_1, move=False)
    num_bytes = numpy.int64(num_bytes)

    if num_bytes & uproot4.const.kByteCountMask:
        num_bytes = int(num_bytes & ~uproot4.const.kByteCountMask) + 4
        if move:
            cursor.skip(_numbytes_version_1.size)

    else:
        num_bytes = None
        version = cursor.field(chunk, _numbytes_version_2, move=move)

    return num_bytes, version


def numbytes_check(start_cursor, stop_cursor, num_bytes, classname, file_path):
    if num_bytes is not None:
        observed = stop_cursor.displacement(start_cursor)
        if observed != num_bytes:
            raise ValueError(
                """instance of ROOT class {0} has {1} bytes; expected {2}
in file {3}""".format(
                    classname, observed, num_bytes, file_path
                )
            )


_skip_tobject_format1 = struct.Struct(">h")
_skip_tobject_format2 = struct.Struct(">II")


def skip_tobject(chunk, cursor):
    version = cursor.field(chunk, _skip_tobject_format1)
    if numpy.int64(version) & uproot4.const.kByteCountVMask:
        cursor.skip(4)
    fUniqueID, fBits = cursor.fields(chunk, _skip_tobject_format2)
    fBits = numpy.uint32(fBits) | uproot4.const.kIsOnHeap
    if fBits & uproot4.const.kIsReferenced:
        cursor.skip(2)


def name_title(chunk, cursor, file_path):
    start_cursor = cursor.copy()
    num_bytes, version = numbytes_version(chunk, cursor)

    skip_tobject(chunk, cursor)
    name = cursor.string(chunk)
    title = cursor.string(chunk)

    numbytes_check(start_cursor, cursor, num_bytes, "TNamed", file_path)
    return name, title


_map_string_string_format1 = struct.Struct(">I")


def map_string_string(chunk, cursor):
    cursor.skip(12)
    size = cursor.field(chunk, _map_string_string_format1)
    cursor.skip(6)
    keys = [cursor.string(chunk) for i in range(size)]
    cursor.skip(6)
    values = [cursor.string(chunk) for i in range(size)]
    return dict(zip(keys, values))


_read_object_any_format1 = struct.Struct(">I")


def read_object_any(chunk, cursor, context, file, parent, as_class=None):
    # TBufferFile::ReadObjectAny()
    # https://github.com/root-project/root/blob/c4aa801d24d0b1eeb6c1623fd18160ef2397ee54/io/io/src/TBufferFile.cxx#L2684
    # https://github.com/root-project/root/blob/c4aa801d24d0b1eeb6c1623fd18160ef2397ee54/io/io/src/TBufferFile.cxx#L2404

    beg = cursor.displacement()
    bcnt = numpy.int64(cursor.field(chunk, _read_object_any_format1))

    if (bcnt & uproot4.const.kByteCountMask) == 0 or (
        bcnt == uproot4.const.kNewClassTag
    ):
        vers = 0
        start = 0
        tag = bcnt
        bcnt = 0
    else:
        vers = 1
        start = cursor.displacement()
        tag = numpy.int64(cursor.field(chunk, _read_object_any_format1))
        bcnt = int(bcnt)

    if tag & uproot4.const.kClassMask == 0:
        # reference object

        if tag == 0:
            return None  # return null

        elif tag == 1:
            return parent  # return parent

        elif tag not in cursor.refs:
            # jump past this object
            cursor.move_to(cursor.origin + beg + bcnt + 4)
            return None  # return null

        else:
            return cursor.refs[int(tag)]  # return object

    elif tag == uproot4.const.kNewClassTag:
        # new class and object

        classname = cursor.classname(chunk)

        cls = file.class_named(classname)

        if vers > 0:
            cursor.refs[start + uproot4.const.kMapOffset] = cls
        else:
            cursor.refs[len(cursor.refs) + 1] = cls

        if as_class is None:
            obj = cls.read(chunk, cursor, context, file, parent)
        else:
            obj = as_class.read(chunk, cursor, context, file, parent)

        if vers > 0:
            cursor.refs[beg + uproot4.const.kMapOffset] = obj
        else:
            cursor.refs[len(cursor.refs) + 1] = obj

        return obj  # return object

    else:
        # reference class, new object

        ref = int(tag & ~uproot4.const.kClassMask)

        if as_class is None:
            if ref not in cursor.refs:
                raise OSError(
                    """invalid class-tag reference
in file: {0}""".format(
                        file.file_path
                    )
                )

            cls = cursor.refs[ref]  # reference class
            obj = cls.read(chunk, cursor, context, file, parent)

        else:
            obj = as_class.read(chunk, cursor, context, file, parent)

        if vers > 0:
            cursor.refs[beg + uproot4.const.kMapOffset] = obj
        else:
            cursor.refs[len(cursor.refs) + 1] = obj

        return obj  # return object

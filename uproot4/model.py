# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import re
import struct

import numpy

import uproot4._const
import uproot4._util


class Model(object):
    @classmethod
    def read(cls, chunk, cursor, file, parent, encoded_classname):
        self = cls.__new__(cls)
        self._cursor = cursor.copy()
        self._file = file
        self._parent = parent
        if encoded_classname is not None:
            assert encoded_classname.startswith("ROOT_")
            self._encoded_classname = encoded_classname
        elif type(cls).__name__.startswith("ROOT_"):
            self._encoded_classname = type(cls).__name__
        else:
            self._encoded_classname = "ROOT__3f3f3f_"

        self.hook_before_read(
            chunk=chunk, cursor=cursor, encoded_classname=encoded_classname,
        )

        self._members = {}
        self._bases = []
        self.read_numbytes_version(chunk, cursor)

        self.hook_before_read_members(
            chunk=chunk, cursor=cursor, encoded_classname=encoded_classname,
        )

        self.read_members(chunk, cursor)

        self.hook_after_read_members(
            chunk=chunk, cursor=cursor, encoded_classname=encoded_classname,
        )

        self.check_numbytes(cursor)

        self.hook_before_postprocess(encoded_classname=encoded_classname,)

        self.postprocess()

        return self

    def __repr__(self):
        return "<{0} at 0x{1:012x}>".format(classname_pretty(self), id(self))

    def read_numbytes_version(self, chunk, cursor):
        self._num_bytes, self._instance_version = _numbytes_version(chunk, cursor)

    def read_members(self, chunk, cursor):
        pass

    def check_numbytes(self, cursor):
        _numbytes_check(
            self._cursor,
            cursor,
            self._num_bytes,
            classname_pretty(self),
            self._file.file_path,
        )

    def postprocess(self):
        pass

    def hook_before_read(self, **kwargs):
        pass

    def hook_before_read_members(self, **kwargs):
        pass

    def hook_after_read_members(self, **kwargs):
        pass

    def hook_before_postprocess(self, **kwargs):
        pass

    @property
    def cursor(self):
        return self._cursor

    @property
    def file(self):
        return self._file

    @property
    def parent(self):
        return self._parent

    @property
    def encoded_classname(self):
        return self._encoded_classname

    @property
    def classname(self):
        return classname_decode(self._encoded_classname)[0]

    @property
    def class_version(self):
        return classname_decode(self._encoded_classname)[1]

    @property
    def num_bytes(self):
        return self._num_bytes

    @property
    def instance_version(self):
        return self._instance_version

    @property
    def members(self):
        return self._members

    @property
    def bases(self):
        return self._bases

    def member(self, name):
        if name in self._members:
            return self._members[name]
        else:
            for base in reversed(self._bases):
                if name in base._members:
                    return base._members[name]
            else:
                raise KeyError(
                    """C++ member {0} not found
in file {1}""".format(
                        repr(name), self._file.file_path
                    )
                )

    def has_version(self, version):
        return True

    def class_of_version(self, version):
        return type(self)


class UnknownClass(Model):
    def read_members(self, chunk, cursor):
        if self._num_bytes is not None:
            cursor.skip(self._num_bytes)

        else:
            raise ValueError(
                """Unknown class {0} that cannot be skipped because its """
                """number of bytes is unknown.
""".format(
                    classname_pretty(self._encoded_classname)
                )
            )

    def __repr__(self):
        return "<Unknown {0}>".format(self.classname)


class UnknownClassVersion(Model):
    def read_members(self, chunk, cursor):
        if self._num_bytes is not None:
            cursor.skip(self._num_bytes)

        else:
            raise ValueError(
                """Class {0} with unknown version {1} (known versions: {2}) """
                """ cannot be skipped because its number of bytes is unknown.
""".format(
                    self.classname,
                    self._instance_version,
                    ", ".join(str(x) for x in self._known_versions),
                )
            )

    def __repr__(self):
        return "<{0} with unknown version {1}>".format(
            self.classname, self._instance_version
        )

    @property
    def known_versions(self):
        return self._known_versions

    def has_version(self, version):
        return version in self._known_versions

    def class_of_version(self, version):
        return self._known_versions[version]


class VersionedModel(Model):
    @property
    def known_versions(self):
        return self._known_versions


class ModelVersions(object):
    @classmethod
    def read(cls, chunk, cursor, file, parent, encoded_classname):
        num_bytes, instance_version = _numbytes_version(chunk, cursor, move=False)

        if instance_version in cls._known_versions:
            versioned_cls = cls._known_versions[instance_version]
            self = versioned_cls.read(chunk, cursor, file, parent, encoded_classname,)
            self._known_versions = cls._known_versions
            return self

        elif num_bytes is not None:
            self = UnknownClassVersion.read(
                chunk, cursor, file, parent, encoded_classname,
            )
            self._known_versions = cls._known_versions
            return self

        else:
            raise ValueError(
                """Unknown version {0} for class {1} (known versions: {2})"""
                """ that cannot be skipped because its number of bytes is unknown.
""".format(
                    instance_version,
                    classname_pretty(cls),
                    ", ".join(str(x) for x in self._known_versions),
                )
            )

    @property
    def known_versions(self):
        return self._known_versions

    def has_version(self, version):
        return version in self._known_versions

    def class_of_version(self, version):
        return self._known_versions[version]


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


def classname_encode(classname, version=None):
    if classname.startswith("ROOT_"):
        raise ValueError("classname is already encoded: {0}".format(classname))

    if version is None:
        v = ""
    else:
        v = "_v" + str(version)

    raw = classname.encode()
    out = _classname_encode_pattern.sub(_classname_encode_convert, raw)
    return "ROOT_" + out.decode() + v


def classname_decode(encoded_classname):
    if not encoded_classname.startswith("ROOT_"):
        raise ValueError("not an encoded classname: {0}".format(encoded_classname))

    raw = encoded_classname[5:].encode()

    m = _classname_decode_version.match(raw)
    if m is None:
        version = None
    else:
        version = int(m.group(1))
        raw = raw[: -len(m.group(1)) - 2]

    out = _classname_decode_pattern.sub(_classname_decode_convert, raw)
    return out.decode(), version


def classname_version(encoded_classname):
    m = _classname_decode_version.match(encoded_classname.encode())
    if m is None:
        return None
    else:
        return int(m.group(1))


def classname_pretty(obj):
    if type(obj).__name__.startswith("ROOT_"):
        encoded_classname = type(obj).__name__
    elif getattr(obj, "__name__", "").startswith("ROOT_"):
        encoded_classname = obj.__name__
    else:
        encoded_classname = obj

    classname, version = classname_decode(encoded_classname)
    if version is None:
        return classname
    else:
        return "{0} (version {1})".format(classname, version)


_numbytes_version_1 = struct.Struct(">IH")
_numbytes_version_2 = struct.Struct(">H")


def _numbytes_version(chunk, cursor, move=True):
    num_bytes, version = cursor.fields(chunk, _numbytes_version_1, move=False)
    num_bytes = numpy.int64(num_bytes)

    if num_bytes & uproot4._const.kByteCountMask:
        num_bytes = int(num_bytes & ~uproot4._const.kByteCountMask) + 4
        if move:
            cursor.skip(_numbytes_version_1.size)

    else:
        num_bytes = None
        version = cursor.field(chunk, _numbytes_version_2, move=move)

    return num_bytes, version


def _numbytes_check(start_cursor, stop_cursor, num_bytes, classname, file_path):
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


def _skip_tobject(chunk, cursor):
    version = cursor.field(chunk, _skip_tobject_format1)
    if numpy.int64(version) & uproot4._const.kByteCountVMask:
        cursor.skip(4)
    fUniqueID, fBits = cursor.fields(chunk, _skip_tobject_format2)
    fBits = numpy.uint32(fBits) | uproot4._const.kIsOnHeap
    if fBits & uproot4._const.kIsReferenced:
        cursor.skip(2)


def _name_title(chunk, cursor, file_path):
    start_cursor = cursor.copy()
    num_bytes, version = _numbytes_version(chunk, cursor)

    _skip_tobject(chunk, cursor)
    name = cursor.string(chunk)
    title = cursor.string(chunk)

    _numbytes_check(start_cursor, cursor, num_bytes, "TNamed", file_path)
    return name, title


_map_string_string_format1 = struct.Struct(">I")


def _map_string_string(chunk, cursor):
    cursor.skip(12)
    size = cursor.field(chunk, _map_string_string_format1)
    cursor.skip(6)
    keys = [cursor.string(chunk) for i in range(size)]
    cursor.skip(6)
    values = [cursor.string(chunk) for i in range(size)]
    return dict(zip(keys, values))


_read_object_any_format1 = struct.Struct(">I")


def _read_object_any(chunk, cursor, file, parent, as_class=None):
    # TBufferFile::ReadObjectAny()
    # https://github.com/root-project/root/blob/c4aa801d24d0b1eeb6c1623fd18160ef2397ee54/io/io/src/TBufferFile.cxx#L2684
    # https://github.com/root-project/root/blob/c4aa801d24d0b1eeb6c1623fd18160ef2397ee54/io/io/src/TBufferFile.cxx#L2404

    beg = cursor.displacement()
    bcnt = numpy.int64(cursor.field(chunk, _read_object_any_format1))

    if (bcnt & uproot4._const.kByteCountMask) == 0 or (
        bcnt == uproot4._const.kNewClassTag
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

    if tag & uproot4._const.kClassMask == 0:
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

    elif tag == uproot4._const.kNewClassTag:
        # new class and object

        classname = cursor.classname(chunk)
        encoded_classname = classname_encode(classname)

        cls = file.class_named(classname)

        if vers > 0:
            cursor.refs[start + uproot4._const.kMapOffset] = cls
        else:
            cursor.refs[len(cursor.refs) + 1] = cls

        if as_class is None:
            obj = cls.read(chunk, cursor, file, parent, encoded_classname)
        else:
            obj = as_class.read(chunk, cursor, file, parent, encoded_classname)

        if vers > 0:
            cursor.refs[beg + uproot4._const.kMapOffset] = obj
        else:
            cursor.refs[len(cursor.refs) + 1] = obj

        return obj  # return object

    else:
        # reference class, new object

        ref = int(tag & ~uproot4._const.kClassMask)

        if as_class is None:
            if ref not in cursor.refs:
                raise OSError(
                    """invalid class-tag reference
in file: {0}""".format(
                        file.file_path
                    )
                )

            cls = cursor.refs[ref]  # reference class
            obj = cls.read(chunk, cursor, file, parent, None)

        else:
            obj = as_class.read(chunk, cursor, file, parent, None)

        if vers > 0:
            cursor.refs[beg + uproot4._const.kMapOffset] = obj
        else:
            cursor.refs[len(cursor.refs) + 1] = obj

        return obj  # return object

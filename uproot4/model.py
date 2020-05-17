# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import re
import struct

import numpy

import uproot4._const
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


class Model(object):
    @classmethod
    def read(cls, cursor, chunk, file, parent, options, encoded_classname=None):
        cls.hook_before_read(
            cursor=cursor,
            chunk=chunk,
            file=file,
            parent=parent,
            options=options,
            encoded_classname=encoded_classname,
        )

        self = cls.__new__(cls)
        self._cursor = cursor.copy(link_refs=True)
        self._file = file
        self._parent = parent
        if type(cls).__name__.startswith("ROOT_"):
            self._encoded_classname = type(cls).__name__
        elif encoded_classname is not None:
            self._encoded_classname = encoded_classname
        else:
            self._encoded_classname = "ROOT__3f3f3f_"
        self._num_bytes = None
        self._members = {}
        self._bases = []

        self.hook_before_read_members(
            cursor=cursor,
            chunk=chunk,
            file=file,
            parent=parent,
            options=options,
            encoded_classname=encoded_classname,
        )
        return self

    @classmethod
    def hook_before_read(cls, **kwargs):
        pass

    def hook_before_read_members(self, **kwargs):
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


_numbytes_version_1 = struct.Struct(">IH")
_numbytes_version_2 = struct.Struct(">H")


def _numbytes_version(cursor, chunk):
    num_bytes, version = cursor.fields(chunk, _numbytes_version_1, move=False)
    num_bytes = numpy.int64(num_bytes)

    if num_bytes & uproot4._const.kByteCountMask:
        num_bytes = int(num_bytes & ~uproot4._const.kByteCountMask)
        cursor.skip(_numbytes_version_1.size)

    else:
        num_bytes = None
        version = cursor.field(chunk, _numbytes_version_2)

    return num_bytes, version


def _numbytes_check(start_cursor, stop_cursor, num_bytes, classname, file_path):
    if num_bytes is not None:
        observed = stop_cursor.displacement(start_cursor)
        if observed != num_bytes:
            raise ValueError(
                """instance of ROOT class {0} has {1} bytes; expected {2}
in file {4}""".format(
                    classname, observed, num_bytes, file_path
                )
            )


class UnknownClass(Model):
    @classmethod
    def read(cls, cursor, chunk, file, parent, options, encoded_classname=None):
        self = Model.read(
            cursor, chunk, file, parent, options, encoded_classname=encoded_classname
        )

        num_bytes, version = _numbytes_version(cursor, chunk)

        if num_bytes is not None:
            self._num_bytes = num_bytes
            cursor.skip(num_bytes)
            return self

        else:
            raise ValueError(
                """Unknown class {1} that cannot be skipped because its """
                """number of bytes is unknown.
""".format(
                    classname_pretty(self._encoded_classname), list(cls._class_versions)
                )
            )

    def __repr__(self):
        return "<Unknown {0}>".format(self.classname)


class UnknownClassVersion(Model):
    def read_members(self, cursor, options):
        cursor.skip(self._num_bytes)

    def __repr__(self):
        return "<Unknown {0} version {1}>".format(
            self.classname, self._instance_version
        )

    @property
    def instance_version(self):
        return self._instance_version

    @property
    def known_versions(self):
        return self._known_versions


class ModelVersions(object):
    @classmethod
    def read(cls, cursor, chunk, file, parent, options, encoded_classname=None):
        start_cursor = cursor.copy(link_refs=True)
        num_bytes, version = _numbytes_version(cursor, chunk)

        if version in cls._class_versions:
            versioned_cls = cls._class_versions[version]
            self = versioned_cls.read(
                start_cursor,
                chunk,
                file,
                parent,
                options,
                encoded_classname=encoded_classname,
            )

        elif num_bytes is not None:
            self = UnknownClassVersion.read(
                start_cursor,
                chunk,
                file,
                parent,
                options,
                encoded_classname=encoded_classname,
            )
            self._instance_version = version
            self._known_versions = cls._class_versions

        else:
            raise ValueError(
                """Unknown version {0} for class {1} (known versions: {2})"""
                """ that cannot be skipped because its number of bytes is unknown.
""".format(
                    version, classname_pretty(cls), list(cls._class_versions)
                )
            )

        self._num_bytes = num_bytes

        self.read_members(cursor, options)
        self.hook_after_read_members(
            cursor=cursor,
            chunk=chunk,
            file=file,
            parent=parent,
            options=options,
            encoded_classname=encoded_classname,
            start_cursor=start_cursor,
            num_bytes=num_bytes,
            version=version,
            versioned_cls=versioned_cls,
        )

        _numbytes_check(
            start_cursor, cursor, num_bytes, classname_pretty(self), file.file_path
        )

        return self

    def hook_after_read_members(self, **kwargs):
        pass

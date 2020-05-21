# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import re

import numpy

import uproot4.const
import uproot4.deserialization
import uproot4._util


class Model(object):
    @classmethod
    def read(cls, chunk, cursor, file, parent):
        self = cls.__new__(cls)
        self._cursor = cursor.copy()
        self._file = file
        self._parent = parent

        self.hook_before_read(chunk=chunk, cursor=cursor)

        self._members = {}
        self._bases = []
        self.read_numbytes_version(chunk, cursor)

        self.hook_before_read_members(chunk=chunk, cursor=cursor)

        self.read_members(chunk, cursor)

        self.hook_after_read_members(chunk=chunk, cursor=cursor)

        self.check_numbytes(cursor)

        self.hook_before_postprocess()

        return self.postprocess()

    def __repr__(self):
        return "<{0} at 0x{1:012x}>".format(classname_pretty(self), id(self))

    def read_numbytes_version(self, chunk, cursor):
        (
            self._num_bytes,
            self._instance_version,
        ) = uproot4.deserialization.numbytes_version(chunk, cursor)

    def read_members(self, chunk, cursor):
        pass

    def check_numbytes(self, cursor):
        uproot4.deserialization.numbytes_check(
            self._cursor,
            cursor,
            self._num_bytes,
            classname_pretty(self),
            self._file.file_path,
        )

    def postprocess(self):
        return self

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
        return type(self).__name__

    @property
    def classname(self):
        return classname_decode(self.encoded_classname)[0]

    @property
    def class_version(self):
        return classname_decode(self.encoded_classname)[1]

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
    def all_members(self):
        out = {}
        for base in self._bases:
            out.update(base.all_members)
        out.update(self._members)
        return out

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

    def tojson(self):
        out = {"_typename": self.classname}
        for k, v in self.all_members.items():
            if isinstance(v, Model):
                out[k] = v.tojson()
            elif isinstance(v, (numpy.number, numpy.ndarray)):
                out[k] = v.tolist()
            else:
                out[k] = v
        return out

    def has_version(self, version):
        return True

    def class_of_version(self, version):
        return type(self)


class UnknownClass(Model):
    def read_members(self, chunk, cursor):
        if self._num_bytes is not None:
            cursor.skip(self._num_bytes - cursor.displacement(self._cursor))

        else:
            raise ValueError(
                """Unknown class {0} that cannot be skipped because its """
                """number of bytes is unknown.
""".format(
                    classname_pretty(self._encoded_classname)
                )
            )

    def __repr__(self):
        return "<Unknown {0} at 0x{1:012x}>".format(self.classname, id(self))


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
        return "<{0} with unknown version {1} at 0x{2:012x}>".format(
            self.classname, self._instance_version, id(self)
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


class DispatchByVersion(object):
    @classmethod
    def read(cls, chunk, cursor, file, parent):
        num_bytes, instance_version = uproot4.deserialization.numbytes_version(
            chunk, cursor, move=False
        )

        if instance_version in cls._known_versions:
            versioned_cls = cls._known_versions[instance_version]
            self = versioned_cls.read(chunk, cursor, file, parent)
            self._known_versions = cls._known_versions
            return self

        elif num_bytes is not None:
            self = UnknownClassVersion.read(chunk, cursor, file, parent)
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


def classname_encode(classname, version=None, unknown=False):
    if unknown:
        prefix = "Unknown_"
    else:
        prefix = "ROOT_"
    if classname.startswith(prefix):
        raise ValueError("classname is already encoded: {0}".format(classname))

    if version is None:
        v = ""
    else:
        v = "_v" + str(version)

    raw = classname.encode()
    out = _classname_encode_pattern.sub(_classname_encode_convert, raw)
    return prefix + out.decode() + v


def classname_decode(encoded_classname):
    if encoded_classname.startswith("Unknown_"):
        raw = encoded_classname[8:].encode()
    elif encoded_classname.startswith("ROOT_"):
        raw = encoded_classname[5:].encode()
    else:
        raise ValueError("not an encoded classname: {0}".format(encoded_classname))

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
    typename = type(obj).__name__
    objname = getattr(obj, "__name__", "")

    if typename.startswith("ROOT_") or typename.startswith("Unknown_"):
        encoded_classname = typename
    elif objname.startswith("ROOT_") or objname.startswith("Unknown_"):
        encoded_classname = objname
    else:
        encoded_classname = obj

    classname, version = classname_decode(encoded_classname)
    if version is None:
        return classname
    else:
        return "{0} (version {1})".format(classname, version)

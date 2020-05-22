# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import re

import numpy

import uproot4.const
import uproot4.deserialization
import uproot4._util


class Model(object):
    @classmethod
    def read(cls, chunk, cursor, context, file, parent):
        self = cls.__new__(cls)
        self._cursor = cursor.copy()
        self._file = file
        self._parent = parent

        self.hook_before_read(chunk=chunk, cursor=cursor, context=context)

        self._members = {}
        self._bases = []
        self.read_numbytes_version(chunk, cursor, context)

        self.hook_before_read_members(chunk=chunk, cursor=cursor, context=context)

        self.read_members(chunk, cursor, context)

        self.hook_after_read_members(chunk=chunk, cursor=cursor, context=context)

        self.check_numbytes(cursor, context)

        self.hook_before_postprocess()

        return self.postprocess()

    def __repr__(self):
        return "<{0} at 0x{1:012x}>".format(
            classname_pretty(self.classname, self.class_version), id(self)
        )

    def read_numbytes_version(self, chunk, cursor, context):
        (
            self._num_bytes,
            self._instance_version,
        ) = uproot4.deserialization.numbytes_version(chunk, cursor)

    def read_members(self, chunk, cursor, context):
        pass

    def check_numbytes(self, cursor, context):
        uproot4.deserialization.numbytes_check(
            self._cursor,
            cursor,
            self._num_bytes,
            classname_pretty(self.classname, self.class_version),
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


class UnknownClass(Model):
    def read_members(self, chunk, cursor, context):
        if self._num_bytes is not None:
            cursor.skip(self._num_bytes - cursor.displacement(self._cursor))

        else:
            raise ValueError(
                """Unknown class {0} that cannot be skipped because its """
                """number of bytes is unknown.
""".format(
                    self.classname
                )
            )

    def __repr__(self):
        return "<Unknown {0} at 0x{1:012x}>".format(self.classname, id(self))


class VersionedModel(Model):
    @property
    def known_versions(self):
        return self._known_versions


class UnknownClassVersion(VersionedModel):
    def read_members(self, chunk, cursor, context):
        if self._num_bytes is not None:
            cursor.skip(self._num_bytes - cursor.displacement(self._cursor))

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


class DispatchByVersion(object):
    @classmethod
    def read(cls, chunk, cursor, context, file, parent):
        num_bytes, version = uproot4.deserialization.numbytes_version(
            chunk, cursor, move=False
        )

        versioned_cls = cls._known_versions.get(version)

        if versioned_cls is not None:
            self = versioned_cls.read(chunk, cursor, context, file, parent)
            self._known_versions = cls._known_versions
            return self

        elif num_bytes is not None:
            classname, _ = classname_decode(cls.__name__)
            streamer = file.streamer_named(classname, version)

            if streamer is not None:
                versioned_cls = streamer.new_class(
                    file.classes, file.streamers, file.file_path
                )
                self = versioned_cls.read(chunk, cursor, context, file, parent)
                self._known_versions = cls._known_versions
                return self

            else:
                unknown_cls = uproot4.unknown_classes.get(classname)
                if unknown_cls is None:
                    unknown_cls = uproot4._util.new_class(
                        classname_encode(classname, version, unknown=True),
                        (UnknownClassVersion,),
                        {},
                    )
                    uproot4.unknown_classes[classname] = unknown_cls

                self = unknown_cls.read(chunk, cursor, context, file, parent)
                self._known_versions = cls._known_versions
                return self

        else:
            raise ValueError(
                """Unknown version {0} for class {1} (known versions: {2})"""
                """ that cannot be skipped because its number of bytes is unknown.
""".format(
                    version,
                    classname_decode(type(cls).__name__)[0],
                    ", ".join(str(x) for x in self._known_versions),
                )
            )

    @property
    def known_versions(self):
        return self._known_versions

    @classmethod
    def has_version(cls, version):
        return version in cls._known_versions

    @classmethod
    def class_of_version(cls, version):
        return cls._known_versions[version]


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
        prefix = "Class_"
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
    elif encoded_classname.startswith("Class_"):
        raw = encoded_classname[6:].encode()
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


def classname_pretty(classname, version):
    if version is None:
        return classname
    else:
        return "{0} (version {1})".format(classname, version)


def has_class_named(classname, version=None, classes=None):
    if classes is None:
        classes = uproot4.classes

    cls = classes.get(classname)
    if cls is None:
        return False

    if version is not None and isinstance(cls, DispatchByVersion):
        return cls.has_version(version)
    else:
        return True


def class_named(classname, version=None, classes=None):
    if classes is None:
        classes = uproot4.classes
        where = "the given 'classes' dict"
    else:
        where = "uproot4.classes"

    cls = classes.get(classname)
    if cls is None:
        raise ValueError("no class named {0} in {1}".format(classname, where))

    if version is not None and isinstance(cls, DispatchByVersion):
        if cls.has_version(version):
            return cls.class_of_version(version)
        else:
            raise ValueError(
                "no class named {0} with version {1} in {2}".format(
                    classname, version, where
                )
            )

    else:
        return cls

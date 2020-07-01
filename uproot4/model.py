# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import re
import sys

import numpy

import uproot4.const
import uproot4._util
import uproot4.interpretation.objects


bootstrap_classnames = [
    "TStreamerInfo",
    "TStreamerElement",
    "TStreamerArtificial",
    "TStreamerBase",
    "TStreamerBasicPointer",
    "TStreamerBasicType",
    "TStreamerLoop",
    "TStreamerObject",
    "TStreamerObjectAny",
    "TStreamerObjectAnyPointer",
    "TStreamerObjectPointer",
    "TStreamerSTL",
    "TStreamerSTLstring",
    "TStreamerString",
    "TList",
    "TObjArray",
    "TObjString",
]


def bootstrap_classes():
    import uproot4.streamers
    import uproot4.models.TList
    import uproot4.models.TObjArray
    import uproot4.models.TObjString

    custom_classes = {}
    for classname in bootstrap_classnames:
        custom_classes[classname] = uproot4.classes[classname]

    return custom_classes


class Model(object):
    class_streamer = None

    @classmethod
    def empty(cls):
        self = cls.__new__(cls)
        self._cursor = None
        self._file = None
        self._parent = None
        self._members = {}
        self._bases = []
        self._num_bytes = None
        self._instance_version = None
        return self

    @classmethod
    def read(cls, chunk, cursor, context, file, parent):
        self = cls.__new__(cls)
        self._cursor = cursor.copy()
        self._file = file
        self._parent = parent
        self._members = {}
        self._bases = []
        self._num_bytes = None
        self._instance_version = None

        old_breadcrumbs = context.get("breadcrumbs", ())
        context["breadcrumbs"] = old_breadcrumbs + (self,)

        self.hook_before_read(chunk=chunk, cursor=cursor, context=context)

        self.read_numbytes_version(chunk, cursor, context)

        if context.get("in_TBranch", False):
            if self._num_bytes is None and self._instance_version != self.class_version:
                self._instance_version = None
                cursor = self._cursor

            elif self._instance_version == 0:
                cursor.skip(4)

        self.hook_before_read_members(chunk=chunk, cursor=cursor, context=context)

        self.read_members(chunk, cursor, context)

        self.hook_after_read_members(chunk=chunk, cursor=cursor, context=context)

        self.check_numbytes(chunk, cursor, context)

        self.hook_before_postprocess(chunk=chunk, cursor=cursor, context=context)

        out = self.postprocess(chunk, cursor, context)

        context["breadcrumbs"] = old_breadcrumbs

        return out

    def __repr__(self):
        return "<{0} at 0x{1:012x}>".format(
            classname_pretty(self.classname, self.class_version), id(self)
        )

    def read_numbytes_version(self, chunk, cursor, context):
        import uproot4.deserialization

        (
            self._num_bytes,
            self._instance_version,
        ) = uproot4.deserialization.numbytes_version(chunk, cursor, context)

    def read_members(self, chunk, cursor, context):
        pass

    @classmethod
    def strided_interpretation(
        cls, file, header=False, tobject_header=True, original=None
    ):
        raise uproot4.interpretation.objects.CannotBeStrided(
            classname_decode(cls.__name__)[0]
        )

    @classmethod
    def awkward_form(cls, file, header=False, tobject_header=True):
        raise uproot4.interpretation.objects.CannotBeAwkward(
            classname_decode(cls.__name__)[0]
        )

    def check_numbytes(self, chunk, cursor, context):
        import uproot4.deserialization

        uproot4.deserialization.numbytes_check(
            chunk,
            self._cursor,
            cursor,
            self._num_bytes,
            self.classname,
            context,
            getattr(self._file, "file_path"),
        )

    def postprocess(self, chunk, cursor, context):
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

    def has_member(self, name, bases=True, recursive_bases=True):
        if name in self._members:
            return True
        if bases:
            for base in reversed(self._bases):
                if recursive_bases:
                    if base.has_member(
                        name, bases=bases, recursive_bases=recursive_bases
                    ):
                        return True
                else:
                    if name in base._members:
                        return True
        return False

    def member(self, name, bases=True, recursive_bases=True, none_if_missing=False):
        if name in self._members:
            return self._members[name]
        if bases:
            for base in reversed(self._bases):
                if recursive_bases:
                    if base.has_member(
                        name, bases=bases, recursive_bases=recursive_bases
                    ):
                        return base.member(
                            name, bases=bases, recursive_bases=recursive_bases
                        )
                else:
                    if name in base._members:
                        return base._members[name]

        if none_if_missing:
            return None
        else:
            raise uproot4.KeyInFileError(
                name,
                """{0}.{1} has only the following members:

    {2}
""".format(
                    type(self).__module__,
                    type(self).__name__,
                    ", ".join(repr(x) for x in self.all_members),
                ),
                file_path=getattr(self._file, "file_path"),
            )

    def tojson(self):
        out = {}
        for base in self._bases:
            tmp = base.tojson()
            if isinstance(tmp, dict):
                out.update(tmp)
        for k, v in self.members.items():
            if isinstance(v, Model):
                out[k] = v.tojson()
            elif isinstance(v, (numpy.number, numpy.ndarray)):
                out[k] = v.tolist()
            else:
                out[k] = v
        out["_typename"] = self.classname
        return out

    def __enter__(self):
        """
        Passes __enter__ to the file and returns self.
        """
        if self._file is not None:
            self._file.source.__enter__()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """
        Passes __exit__ to the file, which closes physical files and shuts down
        any other resources, such as thread pools for parallel reading.
        """
        if self._file is not None:
            self._file.source.__exit__(exception_type, exception_value, traceback)

    def close(self):
        """
        Closes the file from which this object is derived.
        """
        if self._file is not None:
            self._file.close()

    @property
    def closed(self):
        """
        True if the associated file is closed; False otherwise.
        """
        if self._file is not None:
            return self._file.closed
        else:
            return None


class UnknownClass(Model):
    def read_members(self, chunk, cursor, context):
        self._chunk = chunk
        self._context = context

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

    @property
    def chunk(self):
        return self._chunk

    @property
    def context(self):
        return self._context

    def __repr__(self):
        return "<Unknown {0} at 0x{1:012x}>".format(self.classname, id(self))

    def debug(
        self, skip_bytes=0, limit_bytes=None, dtype=None, offset=0, stream=sys.stdout
    ):
        cursor = self._cursor.copy()
        cursor.skip(skip_bytes)
        cursor.debug(
            self._chunk,
            context=self._context,
            limit_bytes=limit_bytes,
            dtype=dtype,
            offset=offset,
            stream=stream,
        )


class VersionedModel(Model):
    def class_named(self, classname, version=None):
        return self._file.class_named(classname, version)


class UnknownClassVersion(VersionedModel):
    def read_members(self, chunk, cursor, context):
        self._chunk = chunk
        self._context = context

        if self._num_bytes is not None:
            cursor.skip(self._num_bytes - cursor.displacement(self._cursor))

        else:
            raise ValueError(
                """Class {0} with unknown version {1} cannot be skipped """
                """because its number of bytes is unknown.
""".format(
                    self.classname, self._instance_version,
                )
            )

    @property
    def chunk(self):
        return self._chunk

    @property
    def context(self):
        return self._context

    def __repr__(self):
        return "<{0} with unknown version {1} at 0x{2:012x}>".format(
            self.classname, self._instance_version, id(self)
        )

    def debug(
        self, skip_bytes=0, limit_bytes=None, dtype=None, offset=0, stream=sys.stdout
    ):
        cursor = self._cursor.copy()
        cursor.skip(skip_bytes)
        cursor.debug(
            self._chunk,
            context=self._context,
            limit_bytes=limit_bytes,
            dtype=dtype,
            offset=offset,
            stream=stream,
        )


class DispatchByVersion(object):
    @classmethod
    def read(cls, chunk, cursor, context, file, parent):
        import uproot4.deserialization

        start_cursor = cursor.copy()
        num_bytes, version = uproot4.deserialization.numbytes_version(
            chunk, cursor, context, move=False
        )

        versioned_cls = cls.known_versions.get(version)

        if versioned_cls is not None:
            pass

        elif num_bytes is not None:
            versioned_cls = cls.new_class(file, version)

        elif context.get("in_TBranch", False):
            versioned_cls = cls.new_class(file, "max")
            cursor = start_cursor

        else:
            raise ValueError(
                """Unknown version {0} for class {1} that cannot be skipped """
                """because its number of bytes is unknown.
""".format(
                    version, classname_decode(cls.__name__)[0],
                )
            )

        return cls.postprocess(
            versioned_cls.read(chunk, cursor, context, file, parent),
            chunk,
            cursor,
            context,
        )

    @classmethod
    def new_class(cls, file, version):
        classname, _ = classname_decode(cls.__name__)
        streamer = file.streamer_named(classname, version)

        if streamer is None:
            streamer = file.streamer_named(classname, "max")

        if streamer is not None:
            versioned_cls = streamer.new_class(file)
            versioned_cls.class_streamer = streamer
            cls.known_versions[streamer.class_version] = versioned_cls
            return versioned_cls

        else:
            unknown_cls = uproot4.unknown_classes.get(classname)
            if unknown_cls is None:
                unknown_cls = uproot4._util.new_class(
                    classname_encode(classname, version, unknown=True),
                    (UnknownClassVersion,),
                    {},
                )
                uproot4.unknown_classes[classname] = unknown_cls
            return unknown_cls

    @classmethod
    def postprocess(cls, self, chunk, cursor, context):
        return self

    @classmethod
    def has_version(cls, version):
        return version in cls.known_versions

    @classmethod
    def class_of_version(cls, version):
        return cls.known_versions.get(version)

    @classmethod
    def strided_interpretation(
        cls, file, header=False, tobject_header=True, original=None
    ):
        versioned_cls = file.class_named(classname_decode(cls.__name__)[0], "max")
        return versioned_cls.strided_interpretation(
            file, header=header, tobject_header=tobject_header
        )

    @classmethod
    def awkward_form(cls, file, header=False, tobject_header=True):
        versioned_cls = file.class_named(classname_decode(cls.__name__)[0], "max")
        return versioned_cls.awkward_form(
            file, header=header, tobject_header=tobject_header
        )


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
        prefix = "Model_"
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
    elif encoded_classname.startswith("Model_"):
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


def maybe_custom_classes(custom_classes):
    if custom_classes is None:
        return uproot4.classes
    else:
        return custom_classes


def has_class_named(classname, version=None, custom_classes=None):
    cls = maybe_custom_classes(custom_classes).get(classname)
    if cls is None:
        return False

    if version is not None and isinstance(cls, DispatchByVersion):
        return cls.has_version(version)
    else:
        return True


def class_named(classname, version=None, custom_classes=None):
    if custom_classes is None:
        classes = uproot4.classes
        where = "the 'custom_classes' dict"
    else:
        where = "uproot4.classes"

    cls = classes.get(classname)
    if cls is None:
        raise ValueError("no class named {0} in {1}".format(classname, where))

    if version is not None and isinstance(cls, DispatchByVersion):
        versioned_cls = cls.class_of_version(version)
        if versioned_cls is not None:
            return versioned_cls
        else:
            raise ValueError(
                "no class named {0} with version {1} in {2}".format(
                    classname, version, where
                )
            )

    else:
        return cls

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

"""
Defines utilities for modeling C++ objects as Python objects and the
:doc:`uproot4.model.Model` class, which is the superclass of all objects that
are read from ROOT files.

The :doc:`uproot4.model.VersionedModel` class is the superclass of all models
whose deserialization routines are specialized by ROOT class version.

A :doc:`uproot4.model.DispatchByVersion` subclass selects a versioned model
after reading its version bytes.

The :doc:`uproot4.model.UnknownClass` and
:doc:`uproot4.model.UnknownClassVersion` are placeholders for data that could
not be modeled, either because the class has no streamer or no streamer for its
version.
"""

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
    """
    Returns the basic classes that are needed to load other classes (streamers,
    TList, TObjArray, TObjString).
    """
    import uproot4.streamers
    import uproot4.models.TList
    import uproot4.models.TObjArray
    import uproot4.models.TObjString

    custom_classes = {}
    for classname in bootstrap_classnames:
        custom_classes[classname] = uproot4.classes[classname]

    return custom_classes


def reset_classes():
    """
    Removes all classes from ``uproot4.classes`` and ``uproot4.unknown_classes``
    and refills ``uproot4.classes`` with original versions of these classes.
    """
    if uproot4._util.py2:
        reload = __builtins__["reload"]
    else:
        from importlib import reload

    uproot4.classes = {}
    uproot4.unknown_classes = {}

    reload(uproot4.streamers)
    reload(uproot4.models.TObject)
    reload(uproot4.models.TString)
    reload(uproot4.models.TArray)
    reload(uproot4.models.TNamed)
    reload(uproot4.models.TList)
    reload(uproot4.models.THashList)
    reload(uproot4.models.TObjArray)
    reload(uproot4.models.TObjString)
    reload(uproot4.models.TAtt)
    reload(uproot4.models.TRef)
    reload(uproot4.models.TTree)
    reload(uproot4.models.TBranch)
    reload(uproot4.models.TLeaf)
    reload(uproot4.models.TBasket)
    reload(uproot4.models.RNTuple)


_classname_encode_pattern = re.compile(br"[^a-zA-Z0-9]+")
_classname_decode_version = re.compile(br".*_v([0-9]+)")
_classname_decode_pattern = re.compile(br"_(([0-9a-f][0-9a-f])+)_")

if uproot4._util.py2:

    def _classname_decode_convert(hex_characters):
        g = hex_characters.group(1)
        return b"".join(
            chr(int(g[i : i + 2], 16)) for i in uproot4._util.range(0, len(g), 2)
        )

    def _classname_encode_convert(bad_characters):
        g = bad_characters.group(0)
        return b"_" + b"".join("{0:02x}".format(ord(x)).encode() for x in g) + b"_"


else:

    def _classname_decode_convert(hex_characters):
        g = hex_characters.group(1)
        return bytes(int(g[i : i + 2], 16) for i in uproot4._util.range(0, len(g), 2))

    def _classname_encode_convert(bad_characters):
        g = bad_characters.group(0)
        return b"_" + b"".join("{0:02x}".format(x).encode() for x in g) + b"_"


def classname_decode(encoded_classname):
    """
    Converts a Python (encoded) classname, such as ``Model_Some_3a3a_Thing``
    into a C++ (decoded) classname, such as ``Some::Thing``.

    C++ classnames can include namespace delimiters (``::``) and template
    arguments (``<`` and ``>``), which have to be translated into
    ``[A-Za-z_][A-Za-z0-9_]*`` for Python. Non-conforming characters and also
    underscores are translated to their hexadecimal equivalents and surrounded
    by underscores. Additionally, Python models of C++ classes are prepended
    with ``Model_`` (or ``Unknown_`` if a streamer isn't found).
    """
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


def classname_encode(classname, version=None, unknown=False):
    """
    Converts a C++ (decoded) classname, such as ``Some::Thing`` into a Python
    classname (encoded), such as ``Model_Some_3a3a_Thing``.

    If ``version`` is a number such as ``2``, the Python name is suffixed by
    version, such as ``Model_Some_3a3a_Thing_v2``.

    If ``unknown`` is True, the ``Model_`` prefix becomes ``Unknown_``.

    C++ classnames can include namespace delimiters (``::``) and template
    arguments (``<`` and ``>``), which have to be translated into
    ``[A-Za-z_][A-Za-z0-9_]*`` for Python. Non-conforming characters and also
    underscores are translated to their hexadecimal equivalents and surrounded
    by underscores. Additionally, Python models of C++ classes are prepended
    with ``Model_`` (or ``Unknown_`` if a streamer isn't found).
    """
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


def classname_version(encoded_classname):
    """
    Extracts a version number from a Python (encoded) classname, if it has one.

    For example, ``Model_Some_3a3a_Thing_v2`` returns ``2``.

    A name without a version number, such as ``Model_Some_3a3a_Thing``, returns
    None.
    """
    m = _classname_decode_version.match(encoded_classname.encode())
    if m is None:
        return None
    else:
        return int(m.group(1))


def class_named(classname, version=None, custom_classes=None):
    """
    Returns a class with a given C++ (decoded) classname.

    If ``version`` is None, no attempt is made to find a specific version.

    * If the class is a :doc:`uproot4.model.DispatchByVersion`, then this is
      object returned.
    * If the class is a versionless model, then this is the object returned.

    If ``version`` is an integer, an attempt is made to find the specific
    version.

    * If the class is a :doc:`uproot4.model.DispatchByVersion`, then it is
      queried for a versioned model.
    * If the class is a versionless model, then this is the object returned.

    If ``custom_classes`` are provided, then these are searched (exclusively)
    for the class. If ``custom_classes`` is None, then ``uproot4.classes`` is
    used.

    No classes are created if a class is not found (an error is raised).
    """
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


def has_class_named(classname, version=None, custom_classes=None):
    """
    Returns True if :doc:`uproot4.model.class_named` would find a class,
    False if it would raise an exception.
    """
    cls = maybe_custom_classes(custom_classes).get(classname)
    if cls is None:
        return False

    if version is not None and isinstance(cls, DispatchByVersion):
        return cls.has_version(version)
    else:
        return True


def maybe_custom_classes(custom_classes):
    """
    Passes through ``custom_classes`` if it is not None; returns
    ``uproot4.classes`` otherwise.
    """
    if custom_classes is None:
        return uproot4.classes
    else:
        return custom_classes


class Model(object):
    """
    Abstract class for all objects extracted from ROOT files (except for
    :doc:`uproot4.reading.ReadOnlyFile`, :doc:`uproot4.reading.ReadOnlyDirectory`,
    and :doc:`uproot4.reading.ReadOnlyKey`).

    A model is instantiated from a file using the :doc:`uproot4.model.Model.read`
    classmethod or synthetically using the :doc:`uproot4.model.Model.empty`
    classmethod, not through a normal constructor.

    Models point back to the file from which they were created, though only a
    few classes (named in ``uproot4.reading.must_be_attached``) have an open,
    readable file attached; the rest have a :doc:`uproot4.reading.DetachedFile`
    with information about the file, while not holding the file open.

    Uproot recognizes *some* of ROOT's thousands of classes, by way of methods
    and properties defined in :doc:`uproot4.behaviors`. Examples include

    * :doc:`uproot4.behaviors.TTree.TTree`
    * :doc:`uproot4.behaviors.TH1.TH1`

    These classes are the most convenient to work with and have specialized
    documentation.

    Classes that don't have any predefined behaviors are still usable through
    their member data.

    * :doc:`uproot4.model.Model.members`: a dict of C++ member names and values
      directly in this class.
    * :doc:`uproot4.model.Model.all_members`: a dict of C++ member names and
      values in this class or any superclasses.
    * :doc:`uproot4.model.Model.member`: method that takes a C++ member name
      and returns its value (from this or any superclass).
    * :doc:`uproot4.model.Model.has_member`: method that takes a C++ member
      name and returns True if it exists (in this or any superclass), False
      otherwise.

    Accessing a data structure through its C++ members may be a prelude to
    adding custom behaviors for it. Before we know what conveniences to add, we
    need to know how they'll be used: this information comes from the user
    community.

    Pythonic models don't follow the same class inheritance tree as their C++
    counterparts: most of them are direct subclasses of
    :doc:`uproot4.model.Model`, :doc:`uproot4.model.DispatchByVersion`, or
    :doc:`uproot4.model.VersionedModel`. To separate an object's members
    from its superclass members, a model instance is created for each and
    the superclass parts are included in a list called
    :doc:`uproot4.model.Model.bases`.
    """
    class_streamer = None
    behaviors = ()

    def __repr__(self):
        if self.class_version is None:
            version = ""
        else:
            version = " (version {0})".format(self.class_version)
        return "<{0}{1} at 0x{2:012x}>".format(self.classname, version, id(self))

    def __enter__(self):
        if isinstance(self._file, uproot4.reading.ReadOnlyFile):
            self._file.source.__enter__()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if isinstance(self._file, uproot4.reading.ReadOnlyFile):
            self._file.source.__exit__(exception_type, exception_value, traceback)

    @property
    def classname(self):
        """
        The C++ (decoded) classname of the modeled class.

        See :doc:`uproot4.model.classname_decode`,
        :doc:`uproot4.model.classname_encode`, and
        :doc:`uproot4.model.classname_version`.
        """
        return classname_decode(self.encoded_classname)[0]

    @property
    def encoded_classname(self):
        """
        The Python (encoded) classname of the modeled class. May or may not
        include version.

        See :doc:`uproot4.model.classname_decode`,
        :doc:`uproot4.model.classname_encode`, and
        :doc:`uproot4.model.classname_version`.
        """
        return type(self).__name__

    @property
    def class_version(self):
        """
        The version number of the modeled class (int) if any; None otherwise.

        See :doc:`uproot4.model.classname_decode`,
        :doc:`uproot4.model.classname_encode`, and
        :doc:`uproot4.model.classname_version`.
        """
        return classname_decode(self.encoded_classname)[1]

    @property
    def cursor(self):
        """
        A cursor pointing to the start of this instance in the byte stream
        (before :doc:`uproot4.model.Model.read_numbytes_version`).
        """
        return self._cursor

    @property
    def file(self):
        """
        A :doc:`uproot4.reading.ReadOnlyFile`, which may be open and readable,
        or a :doc:`uproot4.reading.DetachedFile`, which only contains
        information about the original file (not an open file handle).
        """
        return self._file

    def close(self):
        """
        Closes the file from which this object is derived, if such a file is
        still attached (i.e. not :doc:`uproot4.reading.DetachedFile`).
        """
        if isinstance(self._file, uproot4.reading.ReadOnlyFile):
            self._file.close()

    @property
    def closed(self):
        """
        True if the associated file is known to be closed; False if it is known
        to be open. If the associated file is detached
        (:doc:`uproot4.reading.DetachedFile`), then the value is None.
        """
        if isinstance(self._file, uproot4.reading.ReadOnlyFile):
            return self._file.closed
        else:
            return None

    @property
    def parent(self):
        """
        The object that was deserialized before this one in recursive descent,
        usually the containing object (or the container's container).
        """
        return self._parent

    @property
    def concrete(self):
        """
        The Python instance corresponding to the concrete (instantiated) class
        in C++, which is ``self`` if this is the concrete class or another
        object if this is actually a holder of superclass members for that other
        object (i.e. if this object is in the other's
        :doc:`uproot4.model.Model.bases`).
        """
        return self._concrete

    @property
    def members(self):
        """
        A dict of C++ member data directly associated with this class (i.e. not
        its superclasses). For all members, see
        :doc:`uproot4.model.Model.all_members`.
        """
        return self._members

    @property
    def all_members(self):
        """
        A dict of C++ member data for this class and its superclasses. For only
        direct members, see :doc:`uproot4.model.Model.members`.
        """
        out = {}
        for base in self._bases:
            out.update(base.all_members)
        out.update(self._members)
        return out

    def has_member(self, name, all=True):
        """
        Returns True if calling :doc:`uproot4.model.Model.member` with the same
        arguments would return a value; False if the member is missing.
        """
        if name in self._members:
            return True
        if all:
            for base in reversed(self._bases):
                if base.has_member(name, all=True):
                    return True
        return False

    def member(self, name, all=True, none_if_missing=False):
        """
        Args:
            name (str): The name of the member datum to retrieve.
            all (bool): If True, recursively search all superclasses in
                :doc:`uproot4.model.Model.bases`. Otherwise, search the
                direct class only.
            none_if_missing (bool): If a member datum doesn't exist in the
                search path, ``none_if_missing=True`` has this function return
                None, but ``none_if_missing=False`` would have it raise an
                exception. Note that None is a possible value for some member
                data.

        Returns a C++ member datum by name.
        """
        if name in self._members:
            return self._members[name]
        if all:
            for base in reversed(self._bases):
                if base.has_member(name, all=True):
                    return base.member(name, all=True)

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

    @property
    def bases(self):
        """
        List of :doc:`uproot4.model.Model` objects representing superclass data
        for this object in the order given in C++ (opposite method resolution
        order).

        * If this object has no superclasses, ``bases`` is empty.
        * If it has one superclass, which itself might have superclasses,
          ``bases`` has length 1.
        * Only if this object *multiply inherits* from more than one superclass
          at the same level does ``bases`` have length greater than 1.

        Since multiple inheritance is usually avoided, ``bases`` rarely has
        length greater than 1. A linear chain of superclasses deriving from
        super-superclasses is represented by ``bases`` containing an object
        whose ``bases`` contains objects.
        """
        return self._bases

    def base(self, *cls):
        """
        Extracts instances from :doc:`uproot4.model.Model.bases` by Python class
        type. The ``cls`` may be a single class or a (varargs) list of classes
        to match.
        """
        out = []
        for x in getattr(self, "_bases", []):
            if isinstance(x, cls):
                out.append(x)
            if isinstance(x, Model):
                out.extend(x.base(*cls))
        return out

    @property
    def num_bytes(self):
        """
        Number of bytes expected in the (uncompressed) serialization of this
        instance.

        This value may be None (unknown before reading) or an integer.

        If the value is an integer and the object exists (no exceptions in
        :doc:`uproot4.model.Model.read`), then the expected number of bytes
        agreed with the actual number of bytes, and this numer is reliable.

        If this object is re-serialized, it won't necessarily occupy the same
        number of bytes.
        """
        return self._num_bytes

    @property
    def instance_version(self):
        """
        Version of this instance as read from the byte stream.

        If this model is versioned (:doc:`uproot4.model.VersionedModel`), the
        ``instance_version`` ought to be equal to the
        :doc:`uproot4.model.class_version`.

        If this model is versionless, the ``instance_version`` contains new
        information about the actual version deserialized.
        """
        return self._instance_version

    @property
    def is_memberwise(self):
        """
        True if the object was serialized in ROOT's memberwise format; False
        otherwise.
        """
        return self._is_memberwise

    @classmethod
    def awkward_form(cls, file, index_format="i64", header=False, tobject_header=True):
        """
        Args:
            file (:doc:`uproot4.reading.ReadOnlyFile`): File to use to generate
                :doc:`uproot4.model.Model` classes from its
                :doc:`uproot4.reading.ReadOnlyFile.streamers` and ``file_path``
                for error messages.
            index_format (str): Format to use for indexes of the
                ``awkward1.forms.Form``; may be ``"i32"``, ``"u32"``, or
                ``"i64"``.
            header (bool): If True, include headers in the Form's ``"uproot"``
                parameters.
            tobject_header (bool): If True, include headers for ``TObject``
                classes in the Form's ``"uproot"`` parameters.

        The ``awkward1.forms.Form`` to use to put objects of type type in an
        Awkward Array.
        """
        raise uproot4.interpretation.objects.CannotBeAwkward(
            classname_decode(cls.__name__)[0]
        )

    @classmethod
    def strided_interpretation(
        cls, file, header=False, tobject_header=True, original=None
    ):
        """
        Args:
            file (:doc:`uproot4.reading.ReadOnlyFile`): File to use to generate
                :doc:`uproot4.model.Model` classes from its
                :doc:`uproot4.reading.ReadOnlyFile.streamers` and ``file_path``
                for error messages.
            header (bool): If True, assume the outermost object has a header.
            tobject_header (bool): If True, assume that ``TObjects`` have headers.
            original (None, :doc:`uproot4.model.Model`, or :doc:`uproot4.containers.Container`): The
                original, non-strided model or container.

        Returns a list of (str, ``numpy.dtype``) pairs to build a
        :doc:`uproot4.interpretation.objects.AsStridedObjects` interpretation.
        """
        raise uproot4.interpretation.objects.CannotBeStrided(
            classname_decode(cls.__name__)[0]
        )

    def tojson(self):
        """
        Serializes this object in its ROOT JSON form (as Python lists and dicts,
        which can be passed to ``json.dump`` or ``json.dumps``).
        """
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

    @classmethod
    def empty(cls):
        """
        Creates a model instance (of subclass ``cls``) with no data; all
        required attributes are None or empty.
        """
        self = cls.__new__(cls)
        self._cursor = None
        self._file = None
        self._parent = None
        self._members = {}
        self._bases = []
        self._num_bytes = None
        self._instance_version = None
        self._is_memberwise = False
        return self

    @classmethod
    def read(cls, chunk, cursor, context, file, selffile, parent, concrete=None):
        """
        Args:
            cls (subclass of :doc:`uproot4.model.Model`): Class to instantiate.
            chunk (:doc:`uproot4.source.chunk.Chunk`): Buffer of contiguous data
                from the file :doc:`uproot4.source.chunk.Source`.
            cursor (:doc:`uproot4.source.cursor.Cursor`): Current position in
                that ``chunk``.
            context (dict): Auxiliary data used in deserialization.
            file (:doc:`uproot4.reading.ReadOnlyFile`): An open file object,
                capable of generating new :doc:`uproot4.model.Model` classes
                from its :doc:`uproot4.reading.ReadOnlyFile.streamers`.
            selffile (:doc:`uproot4.reading.CommonFileMethods`): A possibly
                :doc:`uproot4.reading.DetachedFile` associated with this object.
            parent (None or calling object): The previous ``read`` in the
                recursive descent.
            concrete (None or :doc:`uproot4.model.Model` instance): If None,
                this model corresponds to the concrete (instantiated) class in
                C++. Otherwise, this model represents a superclass part of the
                object, and ``concrete`` points to the concrete instance.

        Creates a model instance by reading data from a file.
        """
        self = cls.__new__(cls)
        self._cursor = cursor.copy()
        self._file = selffile
        self._parent = parent
        if concrete is None:
            self._concrete = self
        else:
            self._concrete = concrete

        self._members = {}
        self._bases = []
        self._num_bytes = None
        self._instance_version = None
        self._is_memberwise = False

        old_breadcrumbs = context.get("breadcrumbs", ())
        context["breadcrumbs"] = old_breadcrumbs + (self,)

        self.hook_before_read(chunk=chunk, cursor=cursor, context=context, file=file)

        self.read_numbytes_version(chunk, cursor, context)

        if context.get("in_TBranch", False):
            if self._num_bytes is None and self._instance_version != self.class_version:
                self._instance_version = None
                cursor = self._cursor

            elif self._instance_version == 0:
                cursor.skip(4)

        self.hook_before_read_members(
            chunk=chunk, cursor=cursor, context=context, file=file
        )

        self.read_members(chunk, cursor, context, file)

        self.hook_after_read_members(
            chunk=chunk, cursor=cursor, context=context, file=file
        )

        self.check_numbytes(chunk, cursor, context)

        self.hook_before_postprocess(
            chunk=chunk, cursor=cursor, context=context, file=file
        )

        out = self.postprocess(chunk, cursor, context, file)

        context["breadcrumbs"] = old_breadcrumbs

        return out

    def read_numbytes_version(self, chunk, cursor, context):
        """
        Args:
            chunk (:doc:`uproot4.source.chunk.Chunk`): Buffer of contiguous data
                from the file :doc:`uproot4.source.chunk.Source`.
            cursor (:doc:`uproot4.source.cursor.Cursor`): Current position in
                that ``chunk``.
            context (dict): Auxiliary data used in deserialization.

        Reads the number of bytes and instance version from the byte stream,
        which is usually 6 bytes (4 + 2). Bits with special meanings are
        appropriately masked out.

        Some types don't have a 6-byte header or handle it differently; in
        those cases, this method should be overridden.

        Returns a 3-tuple of

        * number of bytes (int or None if unknown)
        * instance version (int)
        * is memberwise (bool): True if the memberwise bit is set in the version
          number; False otherwise.
        """
        import uproot4.deserialization

        (
            self._num_bytes,
            self._instance_version,
            self._is_memberwise,
        ) = uproot4.deserialization.numbytes_version(chunk, cursor, context)

    def read_members(self, chunk, cursor, context, file):
        """
        Args:
            chunk (:doc:`uproot4.source.chunk.Chunk`): Buffer of contiguous data
                from the file :doc:`uproot4.source.chunk.Source`.
            cursor (:doc:`uproot4.source.cursor.Cursor`): Current position in
                that ``chunk``.
            context (dict): Auxiliary data used in deserialization.
            file (:doc:`uproot4.reading.ReadOnlyFile`): An open file object,
                capable of generating new :doc:`uproot4.model.Model` classes
                from its :doc:`uproot4.reading.ReadOnlyFile.streamers`.

        Reads the member data for this class. The abstract class
        :doc:`uproot4.model.Model` has an empty ``read_members`` method; this
        *must* be overridden by subclasses.
        """
        pass

    def check_numbytes(self, chunk, cursor, context):
        """
        Args:
            chunk (:doc:`uproot4.source.chunk.Chunk`): Buffer of contiguous data
                from the file :doc:`uproot4.source.chunk.Source`.
            cursor (:doc:`uproot4.source.cursor.Cursor`): Current position in
                that ``chunk``.
            context (dict): Auxiliary data used in deserialization.

        Reads nothing; checks the expected number of bytes against the actual
        movement of the ``cursor`` at the end of the object, possibly raising
        a :doc:`uproot4.deserialization.DeserializationError` exception.

        If :doc:`uproot4.model.Model.num_bytes` is None, this method does
        nothing.

        It is *possible* that a subclass would override this method, but not
        likely.
        """
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

    def postprocess(self, chunk, cursor, context, file):
        """
        Args:
            chunk (:doc:`uproot4.source.chunk.Chunk`): Buffer of contiguous data
                from the file :doc:`uproot4.source.chunk.Source`.
            cursor (:doc:`uproot4.source.cursor.Cursor`): Current position in
                that ``chunk``.
            context (dict): Auxiliary data used in deserialization.
            file (:doc:`uproot4.reading.ReadOnlyFile`): An open file object,
                capable of generating new :doc:`uproot4.model.Model` classes
                from its :doc:`uproot4.reading.ReadOnlyFile.streamers`.

        Called for any additional processing after the object has been fully
        read.

        The return value from this method is the object that actually represents
        the ROOT data, which might be a different instance or even a different
        type from this class. The default in :doc:`uproot4.model.Model` is to
        return ``self``.
        """
        return self

    def hook_before_read(self, **kwargs):
        """
        Called in :doc:`uproot4.model.Model.read`, before any data have been
        read.
        """
        pass

    def hook_before_read_members(self, **kwargs):
        """
        Called in :doc:`uproot4.model.Model.read`, after
        :doc:`uproot4.model.Model.read_numbytes_version` and before
        :doc:`uproot4.model.Model.read_members`.
        """
        pass

    def hook_after_read_members(self, **kwargs):
        """
        Called in :doc:`uproot4.model.Model.read`, after
        :doc:`uproot4.model.Model.read_members` and before
        :doc:`uproot4.model.Model.check_numbytes`.
        """
        pass

    def hook_before_postprocess(self, **kwargs):
        """
        Called in :doc:`uproot4.model.Model.read`, after
        :doc:`uproot4.model.Model.check_numbytes` and before
        :doc:`uproot4.model.Model.postprocess`.
        """
        pass


class UnknownClass(Model):
    def read_members(self, chunk, cursor, context, file):
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

    def __getstate__(self):
        return (
            {
                "base_names_versions": self.base_names_versions,
                "member_names": self.member_names,
                "class_flags": self.class_flags,
                "class_code": self.class_code,
                "class_streamer": self.class_streamer,
                "behaviors": self.behaviors,
            },
            dict(self.__dict__),
        )

    def __setstate__(self, state):
        class_data, instance_data = state
        self.__dict__.update(instance_data)


class DynamicModel(VersionedModel):
    def __setstate__(self, state):
        cls = type(self)
        class_data, instance_data = state
        for k, v in class_data.items():
            if not hasattr(cls, k):
                setattr(cls, k, v)
        cls.__bases__ = (
            tuple(x for x in class_data["behaviors"] if x not in cls.__bases__)
            + cls.__bases__
        )
        self.__dict__.update(instance_data)


class UnknownClassVersion(VersionedModel):
    def read_members(self, chunk, cursor, context, file):
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
    def read(cls, chunk, cursor, context, file, selffile, parent, concrete=None):
        import uproot4.deserialization

        start_cursor = cursor.copy()
        (num_bytes, version, is_memberwise,) = uproot4.deserialization.numbytes_version(
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
            versioned_cls.read(
                chunk, cursor, context, file, selffile, parent, concrete=concrete
            ),
            chunk,
            cursor,
            context,
            file,
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
    def postprocess(cls, self, chunk, cursor, context, file):
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
    def awkward_form(cls, file, index_format="i64", header=False, tobject_header=True):
        versioned_cls = file.class_named(classname_decode(cls.__name__)[0], "max")
        return versioned_cls.awkward_form(file, index_format, header, tobject_header)

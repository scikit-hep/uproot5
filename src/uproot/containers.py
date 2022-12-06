# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

"""
This module interpretations and models for standard containers, such as
``std::vector`` and simple arrays.

See :doc:`uproot.interpretation` and :doc:`uproot.model`.
"""


import json
import struct
import types
from collections.abc import KeysView, Mapping, Sequence, Set, ValuesView

import numpy

import uproot
import uproot._awkward_forth

_stl_container_size = struct.Struct(">I")
_stl_object_type = numpy.dtype(object)


def _content_typename(content):
    if isinstance(content, numpy.dtype):
        return uproot.interpretation.numerical._dtype_kind_itemsize_to_typename[
            content.kind, content.itemsize
        ]
    elif isinstance(content, type):
        return uproot.model.classname_decode(content.__name__)[0]
    else:
        return content.typename


def _content_cache_key(content):
    if isinstance(content, numpy.dtype):
        bo = uproot.interpretation.numerical._numpy_byteorder_to_cache_key[
            content.byteorder
        ]
        return f"{bo}{content.kind}{content.itemsize}"
    elif isinstance(content, type):
        return content.__name__
    else:
        return content.cache_key


def _read_nested(
    model, length, chunk, cursor, context, file, selffile, parent, header=True
):
    forth_stash = uproot._awkward_forth.forth_stash(context)

    if forth_stash is not None:
        forth_obj = forth_stash.get_gen_obj()

    if isinstance(model, numpy.dtype):
        symbol = uproot._awkward_forth.symbol_dict.get(model)

        if symbol is None and forth_stash is not None:
            raise TypeError("Cannot be awkward")
        if forth_stash is not None:
            key = forth_obj.get_keys(1)
            form_key = f"node{key}-data"
            forth_stash.add_to_header(
                f"output node{key}-data {uproot._awkward_forth.convert_dtype(symbol)}\n"
            )
            forth_stash.add_to_pre(f"stream #!{symbol}-> node{key}-data\n")
            if forth_obj.should_add_form():
                forth_obj.add_form_key(form_key)
                forth_obj.add_form(
                    {
                        "class": "NumpyArray",
                        "primitive": f"{uproot._awkward_forth.convert_dtype(symbol)}",
                        "form_key": f"node{key}",
                    }
                )
            temp = forth_obj.add_node(
                f"node{key}",
                forth_stash.get_attrs(),
                "i64",
                1,
                None,
            )
            forth_obj.go_to(temp)
        return cursor.array(chunk, length, model, context)

    else:
        values = numpy.empty(length, dtype=_stl_object_type)
        if isinstance(model, AsContainer):
            if forth_stash is not None:
                if length == 0:
                    forth_obj.var_set = True
                temp_count = context["forth"].gen.count_obj
            for i in range(length):
                if forth_stash is not None:
                    context["forth"].gen.count_obj = temp_count
                values[i] = model.read(
                    chunk, cursor, context, file, selffile, parent, header=header
                )
        else:
            if forth_stash is not None:
                temp_count = context["forth"].gen.count_obj
            for i in range(length):
                if forth_stash is not None:
                    if "temp_ref" in context.keys():
                        context["forth"].gen.go_to(context["temp_ref"])
                    context["forth"].gen.count_obj = temp_count
                values[i] = model.read(chunk, cursor, context, file, selffile, parent)
        if forth_stash is not None:
            if "temp_ref" in context.keys():
                del context["temp_ref"]
        return values


def _tostring(value):
    if uproot._util.isstr(value):
        return repr(value)
    else:
        return str(value)


def _str_with_ellipsis(tostring, length, lbracket, rbracket, limit):
    leftlen = len(lbracket)
    rightlen = len(rbracket)
    left, right, i, j, done = [], [], 0, length - 1, False

    while True:
        if i > j:
            done = True
            break
        x = tostring(i) + ("" if i == length - 1 else ", ")
        i += 1
        dotslen = 0 if i > j else 5
        if leftlen + rightlen + len(x) + dotslen > limit:
            break
        left.append(x)
        leftlen += len(x)

        if i > j:
            done = True
            break
        y = tostring(j) + ("" if j == length - 1 else ", ")
        j -= 1
        dotslen = 0 if i > j else 5
        if leftlen + rightlen + len(y) + dotslen > limit:
            break
        right.insert(0, y)
        rightlen += len(y)

    if length == 0:
        return lbracket + rbracket
    elif done:
        return lbracket + "".join(left) + "".join(right) + rbracket
    elif len(left) == len(right) == 0:
        return lbracket + f"{tostring(0)}, ..." + rbracket
    elif len(right) == 0:
        return lbracket + "".join(left) + "..." + rbracket
    else:
        return lbracket + "".join(left) + "..., " + "".join(right) + rbracket


class AsContainer:
    """
    Abstract class for all descriptions of data as containers, such as
    ``std::vector``.

    Note that these are not :doc:`uproot.interpretation.Interpretation`
    objects, since they are recursively nestable and have a ``read``
    instance method like :doc:`uproot.model.Model`'s ``read`` classmethod.

    A nested tree of :doc:`uproot.containers.AsContainer` instances and
    :doc:`uproot.model.Model` class objects may be the ``model`` argument
    of a :doc:`uproot.interpretation.objects.AsObjects`.
    """

    @property
    def cache_key(self):
        """
        String that uniquely specifies this container, to use as part of
        an array's cache key.
        """
        raise AssertionError

    @property
    def typename(self):
        """
        String that describes this container as a C++ type.

        This type might not exactly correspond to the type in C++, but it would
        have equivalent meaning.
        """
        raise AssertionError

    def awkward_form(self, file, context):
        """
        Args:
            file (:doc:`uproot.reading.CommonFileMethods`): The file associated
                with this interpretation's ``TBranch``.
            context (dict): Context for the Form-generation; defaults are
                the remaining arguments below.
            index_format (str): Format to use for indexes of the
                ``awkward.forms.Form``; may be ``"i32"``, ``"u32"``, or
                ``"i64"``.
            header (bool): If True, include header fields of each C++ class.
            tobject_header (bool): If True, include header fields of each ``TObject``
                base class.
            breadcrumbs (tuple of class objects): Used to check for recursion.
                Types that contain themselves cannot be Awkward Arrays because the
                depth of instances is unknown.

        The ``awkward.forms.Form`` to use to put objects of type type in an
        Awkward Array.
        """
        raise AssertionError

    def strided_interpretation(
        self, file, header=False, tobject_header=True, breadcrumbs=(), original=None
    ):
        """
        Args:
            file (:doc:`uproot.reading.ReadOnlyFile`): File to use to generate
                :doc:`uproot.model.Model` classes from its
                :ref:`uproot.reading.ReadOnlyFile.streamers` and ``file_path``
                for error messages.
            header (bool): If True, assume the outermost object has a header.
            tobject_header (bool): If True, assume that ``TObjects`` have headers.
            breadcrumbs (tuple of class objects): Used to check for recursion.
                Types that contain themselves cannot be strided because the
                depth of instances is unknown.
            original (None, :doc:`uproot.model.Model`, or :doc:`uproot.containers.Container`): The
                original, non-strided model or container.

        Returns a list of (str, ``numpy.dtype``) pairs to build a
        :doc:`uproot.interpretation.objects.AsStridedObjects` interpretation.
        """
        raise uproot.interpretation.objects.CannotBeStrided(self.typename)

    @property
    def header(self):
        """
        If True, assume this container has a header.
        """
        return self._header

    @header.setter
    def header(self, value):
        if value is True or value is False:
            self._header = value
        else:
            raise TypeError(f"{type(self).__name__}.header must be True or False")

    def read(self, chunk, cursor, context, file, selffile, parent, header=True):
        """
        Args:
            chunk (:doc:`uproot.source.chunk.Chunk`): Buffer of contiguous data
                from the file :doc:`uproot.source.chunk.Source`.
            cursor (:doc:`uproot.source.cursor.Cursor`): Current position in
                that ``chunk``.
            context (dict): Auxiliary data used in deserialization.
            file (:doc:`uproot.reading.ReadOnlyFile`): An open file object,
                capable of generating new :doc:`uproot.model.Model` classes
                from its :ref:`uproot.reading.ReadOnlyFile.streamers`.
            selffile (:doc:`uproot.reading.CommonFileMethods`): A possibly
                :doc:`uproot.reading.DetachedFile` associated with this object.
            parent (None or calling object): The previous ``read`` in the
                recursive descent.
            header (bool): If True, enable the container's
                :ref:`uproot.containers.AsContainer.header`.

        Read one object as part of a recursive descent.
        """
        raise AssertionError

    def __eq__(self, other):
        raise AssertionError

    def __ne__(self, other):
        return not self == other


class AsDynamic(AsContainer):
    """
    Args:
        model (None, :doc:`uproot.model.Model`, or :doc:`uproot.containers.Container`): Optional
            description of the data, used in
            :ref:`uproot.containers.AsDynamic.awkward_form` but ignored in
            :ref:`uproot.containers.AsDynamic.read`.

    A :doc:`uproot.containers.AsContainer` for one object whose class may
    not be known before reading.

    The byte-stream consists of a class name followed by instance data. Only
    known use: in ``TBranchObject`` branches.
    """

    def __init__(self, model=None):
        self._model = model

    @property
    def model(self):
        """
        Optional description of the data, used in
        :ref:`uproot.containers.AsDynamic.awkward_form` but ignored in
        :ref:`uproot.containers.AsDynamic.read`.
        """
        return self._model

    def __repr__(self):
        if self._model is None:
            model = ""
        elif isinstance(self._model, type):
            model = "model=" + self._model.__name__
        else:
            model = "model=" + repr(self._model)
        return f"AsDynamic({model})"

    @property
    def cache_key(self):
        if self._model is None:
            return "AsDynamic(None)"
        else:
            return f"AsDynamic({_content_cache_key(self._model)})"

    @property
    def typename(self):
        if self._model is None:
            return "void*"
        else:
            return uproot.model.classname_decode(self._model.__name__)[0]

    def awkward_form(self, file, context):
        awkward = uproot.extras.awkward()
        if self._model is None:
            raise uproot.interpretation.objects.CannotBeAwkward("dynamic type")
        else:
            return awkward.forms.ListOffsetForm(
                context["index_format"],
                uproot._util.awkward_form(self._model, file, context),
            )

    def read(self, chunk, cursor, context, file, selffile, parent, header=True):
        # AwkwardForth testing: test_0637's tests aren't expected to enter here
        context["cancel_forth"] = True
        classname = cursor.string(chunk, context)
        cursor.skip(1)
        cls = file.class_named(classname)
        return cls.read(chunk, cursor, context, file, selffile, parent)


class AsFIXME(AsContainer):
    """
    Args:
        message (str): Required string, prefixes the error message.

    A :doc:`uproot.containers.AsContainer` for types that are known to be
    unimplemented. The name is intended to be conspicuous, so that such cases
    may be more easily fixed.

    :ref:`uproot.containers.AsFIXME.read` raises a
    :doc:`uproot.deserialization.DeserializationError` asking for a bug-report.
    """

    def __init__(self, message):
        self.message = message

    def __hash__(self):
        return hash((AsFIXME, self.message))

    def __repr__(self):
        return f"AsFIXME({self.message!r})"

    @property
    def cache_key(self):
        return f"AsFIXME({self.message!r})"

    @property
    def typename(self):
        return "unknown"

    def awkward_form(self, file, context):
        raise uproot.interpretation.objects.CannotBeAwkward(self.message)

    def read(self, chunk, cursor, context, file, selffile, parent, header=True):
        raise uproot.deserialization.DeserializationError(
            self.message + "; please file a bug report!", None, None, None, None
        )

    def __eq__(self, other):
        if isinstance(other, AsFIXME):
            return self.message == other.message
        else:
            return False


class AsString(AsContainer):
    """
    Args:
        header (bool): Sets the :ref:`uproot.containers.AsContainer.header`.
        length_bytes ("1-5" or "4"): Method used to determine the length of
            a string: "1-5" means one byte if the length is less than 256,
            otherwise the true length is in the next four bytes; "4" means
            always four bytes.
        typename (None or str): If None, construct a plausible C++ typename.
            Otherwise, take the suggestion as given.

    A :doc:`uproot.containers.AsContainer` for strings nested withing other
    objects.

    This is not an :doc:`uproot.interpretation.Interpretation`; it *must* be
    nested, at least within :doc:`uproot.interpretation.objects.AsObjects`.

    Note that the :doc:`uproot.interpretation.strings.AsStrings` class is
    for a ``TBranch`` that contains only strings.

    (:ref:`uproot.interpretation.objects.AsObjects.simplify` converts an
    :doc:`uproot.interpretation.objects.AsObjects` of
    :doc:`uproot.containers.AsString` into a
    :doc:`uproot.interpretation.strings.AsStrings`.)
    """

    def __init__(self, header, length_bytes="1-5", typename=None):
        self.header = header
        if length_bytes in ("1-5", "4"):
            self._length_bytes = length_bytes
        else:
            raise ValueError("length_bytes must be '1-5' or '4'")
        self._typename = typename

    @property
    def length_bytes(self):
        """
        Method used to determine the length of a string: "1-5" means one byte
        if the length is less than 256, otherwise the true length is in the
        next four bytes; "4" means always four bytes.
        """
        return self._length_bytes

    def __hash__(self):
        return hash((AsString, self._header, self._length_bytes))

    def __repr__(self):
        args = [repr(self._header)]
        if self._length_bytes != "1-5":
            args.append(f"length_bytes={self._length_bytes!r}")
        return "AsString({})".format(", ".join(args))

    @property
    def cache_key(self):
        return f"AsString({self._header},{self._length_bytes!r})"

    @property
    def typename(self):
        if self._typename is None:
            return "std::string"
        else:
            return self._typename

    def awkward_form(self, file, context):
        awkward = uproot.extras.awkward()
        return awkward.forms.ListOffsetForm(
            context["index_format"],
            awkward.forms.NumpyForm("uint8", parameters={"__array__": "char"}),
            parameters={"__array__": "string"},
        )

    def read(self, chunk, cursor, context, file, selffile, parent, header=True):
        # AwkwardForth testing: test_0637's 00,03,25,27,30,33,35,36,38,39,45,47,51,56,57,58,60,61,63,65,68,70,71,72,73,74,75,78,79

        forth_stash = uproot._awkward_forth.forth_stash(context)

        if forth_stash is not None:
            # raise NotImplementedError
            forth_obj = forth_stash.get_gen_obj()
            keys = forth_obj.get_keys(2)
            offsets_num = keys[0]
            data_num = keys[1]
        if self._header and header:
            start_cursor = cursor.copy()
            (
                num_bytes,
                instance_version,
                is_memberwise,
            ) = uproot.deserialization.numbytes_version(chunk, cursor, context)
            if forth_stash is not None:
                temp_jump = cursor._index - start_cursor._index
                if temp_jump != 0:
                    forth_stash.add_to_pre(f"{temp_jump} stream skip\n")

        if self._length_bytes == "1-5":
            out = cursor.string(chunk, context)
            if forth_stash is not None:
                forth_stash.add_to_pre(
                    f"stream !B-> stack dup 255 = if drop stream !I-> stack then dup node{offsets_num}-offsets +<- stack stream #!B-> node{data_num}-data\n"
                )
        elif self._length_bytes == "4":
            length = cursor.field(chunk, _stl_container_size, context)
            out = cursor.string_with_length(chunk, context, length)
            if forth_stash is not None:
                forth_stash.add_to_pre(
                    f"stream !I-> stack dup node{offsets_num}-offsets +<- stack stream #B-> node{data_num}-data\n"
                )
        else:
            raise AssertionError(repr(self._length_bytes))

        if self._header and header:
            uproot.deserialization.numbytes_check(
                chunk,
                start_cursor,
                cursor,
                num_bytes,
                self.typename,
                context,
                file.file_path,
            )

        if forth_stash is not None:
            if forth_obj.should_add_form():
                temp_aform = f'{{"class": "ListOffsetArray", "offsets": "i64", "content": {{"class": "NumpyArray", "primitive": "uint8", "inner_shape": [], "parameters": {{"__array__": "char"}}, "form_key": "node{data_num}"}}, "parameters": {{"__array__": "string"}}, "form_key": "node{offsets_num}"}}'
                forth_obj.add_form(json.loads(temp_aform))

                form_keys = [
                    f"node{data_num}-data",
                    f"node{offsets_num}-offsets",
                ]
                for elem in form_keys:
                    forth_obj.add_form_key(elem)
            forth_stash.add_to_header(
                f"output node{offsets_num}-offsets int64\noutput node{data_num}-data uint8\n"
            )
            forth_stash.add_to_init(f"0 node{offsets_num}-offsets <- stack\n")
            temp_form = forth_obj.add_node(
                f"node{offsets_num}",
                forth_stash.get_attrs(),
                "i64",
                0,
                None,
            )
            forth_obj.go_to(temp_form)
        return out

    def __eq__(self, other):
        return (
            isinstance(other, AsString)
            and self.header == other.header
            and self.length_bytes == other.length_bytes
        )


class AsPointer(AsContainer):
    """
    Args:
        pointee (None, :doc:`uproot.model.Model`, or :doc:`uproot.containers.Container`): Optional
            description of the data, used in
            :ref:`uproot.containers.AsPointer.awkward_form` but ignored in
            :ref:`uproot.containers.AsPointer.read`.

    A :doc:`uproot.containers.AsContainer` for an object referred to by
    pointer, meaning that it could be None (``nullptr``) or identical to
    an already-read object.

    The deserialization procedure calls
    :doc:`uproot.deserialization.read_object_any`.
    """

    def __init__(self, pointee=None):
        self._pointee = pointee

    @property
    def pointee(self):
        """
        Optional description of the data, used in
        :ref:`uproot.containers.AsPointer.awkward_form` but ignored in
        :ref:`uproot.containers.AsPointer.read`.
        """
        return self._pointee

    def __hash__(self):
        return hash((AsPointer, self._pointee))

    def __repr__(self):
        if self._pointee is None:
            pointee = ""
        elif isinstance(self._pointee, type):
            pointee = self._pointee.__name__
        else:
            pointee = repr(self._pointee)
        return f"AsPointer({pointee})"

    @property
    def cache_key(self):
        if self._pointee is None:
            return "AsPointer(None)"
        else:
            return f"AsPointer({_content_cache_key(self._pointee)})"

    @property
    def typename(self):
        if self._pointee is None:
            return "void*"
        else:
            return _content_typename(self._pointee) + "*"

    def awkward_form(self, file, context):
        raise uproot.interpretation.objects.CannotBeAwkward("arbitrary pointer")

    def read(self, chunk, cursor, context, file, selffile, parent, header=True):
        # AwkwardForth testing: test_0637's 29,45,46,49,50 (Awkward Form discovered at read-time)

        return uproot.deserialization.read_object_any(
            chunk, cursor, context, file, selffile, parent
        )

    def __eq__(self, other):
        if isinstance(other, AsPointer):
            return self._pointee == other._pointee
        else:
            return False


class AsArray(AsContainer):
    """
    Args:
        header (bool): Sets the :ref:`uproot.containers.AsContainer.header`.
        speedbump (bool): If True, one byte must be skipped before reading the
            data.
        values (:doc:`uproot.model.Model`, :doc:`uproot.containers.Container`, or ``numpy.dtype``): Data
            type for data nested in the array.
        inner_shape (Optional[Tuple[int]]): Shape not counting first axis, if any

    A :doc:`uproot.containers.AsContainer` for simple arrays (not
    ``std::vector``).
    """

    def __init__(self, header, speedbump, values, inner_shape=()):
        self._header = header
        self._speedbump = speedbump
        self._values = values
        self._inner_shape = inner_shape
        if isinstance(values, numpy.dtype) and inner_shape:
            raise RuntimeError("Logic error: the dtype should have contained the shape")

    @property
    def speedbump(self):
        """
        If True, one byte must be skipped before reading the data.
        """
        return self._speedbump

    @property
    def values(self):
        """
        Data type for data nested in the array. May be a
        :doc:`uproot.model.Model`, :doc:`uproot.containers.Container`, or
        ``numpy.dtype``.
        """
        return self._values

    @property
    def inner_shape(self):
        """
        The shape of possible extra dimensions in this array
        """
        return self._inner_shape

    def __repr__(self):
        if isinstance(self._values, type):
            values = self._values.__name__
        else:
            values = repr(self._values)
        return "AsArray({}, {}, {}, {})".format(
            self.header, self.speedbump, values, self.inner_shape
        )

    @property
    def cache_key(self):
        return "AsArray({},{},{},{})".format(
            self.header,
            self.speedbump,
            _content_cache_key(self._values),
            self.inner_shape,
        )

    @property
    def typename(self):
        shape = "".join(f"[{d}]" for d in self.inner_shape)
        return _content_typename(self._values) + "[]" + shape

    def awkward_form(self, file, context):
        awkward = uproot.extras.awkward()
        values_form = uproot._util.awkward_form(self._values, file, context)
        for dim in reversed(self.inner_shape):
            values_form = awkward.forms.RegularForm(values_form, dim)
        return awkward.forms.ListOffsetForm(context["index_format"], values_form)

    def read(self, chunk, cursor, context, file, selffile, parent, header=True):
        # AwkwardForth testing: test_0637's 01,02,23,24,25,26,27,28,30,51,52

        forth_stash = uproot._awkward_forth.forth_stash(context)

        if forth_stash is not None:
            forth_obj = forth_stash.get_gen_obj()
            offsets_num = forth_obj.get_keys(1)

        if self._header and header:
            start_cursor = cursor.copy()
            (
                num_bytes,
                instance_version,
                is_memberwise,
            ) = uproot.deserialization.numbytes_version(chunk, cursor, context)
            if forth_stash is not None:
                temp_jump = cursor._index - start_cursor._index
                if temp_jump != 0:
                    forth_stash.add_to_pre(f"{temp_jump} stream skip\n")
            if is_memberwise:
                raise NotImplementedError(
                    """memberwise serialization of {}
in file {}""".format(
                        type(self).__name__, selffile.file_path
                    )
                )

            if isinstance(self._values, numpy.dtype):
                remainder = chunk.get(
                    cursor.index, cursor.index + num_bytes, cursor, context
                )
                return remainder.view(self._values).reshape(-1, *self.inner_shape)

            else:
                if forth_stash is not None:
                    forth_stash.add_to_header(
                        f"output node{offsets_num}-offsets int64\n"
                    )
                    form_key = f"node{offsets_num}-offsets"
                    forth_stash.add_to_init(f"0 node{offsets_num}-offsets <- stack\n")
                    forth_stash.add_to_pre(
                        "0 bytestops I-> stack \nbegin\ndup stream pos <>\nwhile\nswap 1 + swap\n"
                    )
                    if len(self.inner_shape) > 0:
                        forth_stash.add_to_post(
                            f"repeat\nswap {self.inner_shape[0]} / node{offsets_num}-offsets +<- stack drop\n"
                        )
                    else:
                        forth_stash.add_to_post(
                            f"repeat\nswap node{offsets_num}-offsets +<- stack drop\n"
                        )
                    if forth_obj.should_add_form():
                        forth_obj.add_form_key(form_key)
                        if len(self.inner_shape) > 0:
                            temp_aform = f'{{ "class":"ListOffsetArray", "offsets":"i64", "content": {{"class": "RegularArray", "content": "NULL", "size": {self.inner_shape[0]}}}, "parameters": {{}}, "form_key": "node{offsets_num}"}}'
                            forth_obj.add_form(json.loads(temp_aform), traverse=2)
                        else:
                            temp_aform = f'{{ "class":"ListOffsetArray", "offsets":"i64", "content": "NULL", "parameters": {{}}, "form_key": "node{offsets_num}"}}'
                            forth_obj.add_form(json.loads(temp_aform))
                    temp = forth_obj.add_node(
                        f"node{offsets_num}",
                        forth_stash.get_attrs(),
                        "i64",
                        1,
                        {},
                    )

                if cursor.index >= chunk.stop and forth_stash is not None:
                    forth_obj.var_set = True
                out = []
                if forth_stash is not None:
                    temp_count = forth_obj.count_obj
                while cursor.displacement(start_cursor) < num_bytes:
                    if forth_stash is not None:
                        forth_obj.count_obj = temp_count
                    out.append(
                        self._values.read(
                            chunk, cursor, context, file, selffile, parent
                        )
                    )
                    if forth_stash is not None:
                        forth_obj.go_to(temp["content"])

                if self._header and header:
                    uproot.deserialization.numbytes_check(
                        chunk,
                        start_cursor,
                        cursor,
                        num_bytes,
                        self.typename,
                        context,
                        file.file_path,
                    )
                return uproot._util.objectarray1d(out).reshape(-1, *self.inner_shape)

        else:

            if self._speedbump:
                if forth_stash is not None:
                    forth_stash.add_to_pre("1 stream skip\n")
                cursor.skip(1)

            if isinstance(self._values, numpy.dtype):
                remainder = chunk.remainder(cursor.index, cursor, context)
                return remainder.view(self._values).reshape(-1, *self.inner_shape)

            else:

                if forth_stash is not None:
                    forth_stash.add_to_header(
                        f"output node{offsets_num}-offsets int64\n"
                    )
                    form_key = f"node{offsets_num}-offsets"
                    forth_stash.add_to_init(f"0 node{offsets_num}-offsets <- stack\n")
                    forth_stash.add_to_pre(
                        "0 bytestops I-> stack \nbegin\ndup stream pos <>\nwhile\nswap 1 + swap\n"
                    )
                    if len(self.inner_shape) > 0:
                        forth_stash.add_to_post(
                            f"repeat\nswap {self.inner_shape[0]} / node{offsets_num}-offsets +<- stack drop\n"
                        )
                    else:
                        forth_stash.add_to_post(
                            f"repeat\nswap node{offsets_num}-offsets +<- stack drop\n"
                        )
                    if forth_obj.should_add_form():
                        forth_obj.add_form_key(form_key)
                        if len(self.inner_shape) > 0:
                            temp_aform = f'{{ "class":"ListOffsetArray", "offsets":"i64", "content": {{"class": "RegularArray", "content": "NULL", "size": {self.inner_shape[0]}}}, "parameters": {{}}, "form_key": "node{offsets_num}"}}'
                            forth_obj.add_form(json.loads(temp_aform), traverse=2)
                        else:
                            temp_aform = f'{{ "class":"ListOffsetArray", "offsets":"i64", "content": "NULL", "parameters": {{}}, "form_key": "node{offsets_num}"}}'
                            forth_obj.add_form(json.loads(temp_aform))
                    temp = forth_obj.add_node(
                        f"node{offsets_num}",
                        forth_stash.get_attrs(),
                        "i64",
                        1,
                        {},
                    )

                if cursor.index >= chunk.stop and forth_stash is not None:
                    forth_obj.var_set = True
                out = []
                if forth_stash is not None:
                    temp_count = forth_obj.count_obj
                while cursor.index < chunk.stop:
                    if forth_stash is not None:
                        forth_obj.count_obj = temp_count
                    out.append(
                        self._values.read(
                            chunk, cursor, context, file, selffile, parent
                        )
                    )
                    if forth_stash is not None:
                        forth_obj.go_to(temp)

                # if forth_stash is not None:
                #    forth_obj.go_to(temp)
                return uproot._util.objectarray1d(out).reshape(-1, *self.inner_shape)


class AsRVec(AsContainer):
    """
    Args:
        header (bool): Sets the :ref:`uproot.containers.AsContainer.header`.
        values (:doc:`uproot.model.Model` or :doc:`uproot.containers.Container`): Data
            type for data nested in the container.

    A :doc:`uproot.containers.AsContainer` for ``ROOT::VecOps::RVec``.
    """

    def __init__(self, header, values):
        self.header = header
        if isinstance(values, AsContainer):
            self._values = values
        elif isinstance(values, type) and issubclass(
            values, (uproot.model.Model, uproot.model.DispatchByVersion)
        ):
            self._values = values
        else:
            self._values = numpy.dtype(values)

    def __hash__(self):
        return hash((AsRVec, self._header, self._values))

    @property
    def values(self):
        """
        Data type for data nested in the container.
        """
        return self._values

    def __repr__(self):
        if isinstance(self._values, type):
            values = self._values.__name__
        else:
            values = repr(self._values)
        return f"AsRVec({self._header}, {values})"

    @property
    def cache_key(self):
        return f"AsRVec({self._header},{_content_cache_key(self._values)})"

    @property
    def typename(self):
        return f"RVec<{_content_typename(self._values)}>"

    def awkward_form(self, file, context):
        awkward = uproot.extras.awkward()
        return awkward.forms.ListOffsetForm(
            context["index_format"],
            uproot._util.awkward_form(self._values, file, context),
        )

    def read(self, chunk, cursor, context, file, selffile, parent, header=True):
        # AwkwardForth testing: test_0637's (none! untested! but it's just like AsVector)
        context["cancel_forth"] = True
        if self._header and header:
            start_cursor = cursor.copy()
            (
                num_bytes,
                instance_version,
                is_memberwise,
            ) = uproot.deserialization.numbytes_version(chunk, cursor, context)
        else:
            is_memberwise = False

        # Note: self._values can also be a NumPy dtype, and not necessarily a class
        # (e.g. type(self._values) == type)
        _value_typename = _content_typename(self._values)
        if is_memberwise:
            if not issubclass(self._values, uproot.model.DispatchByVersion):
                raise NotImplementedError(
                    """streamerless memberwise serialization of class {}({})
    in file {}""".format(
                        type(self).__name__, _value_typename, selffile.file_path
                    )
                )

            # uninterpreted header
            cursor.skip(6)

            length = cursor.field(chunk, _stl_container_size, context)

            # no known class version number (maybe in that header? unclear...)
            model = self._values.new_class(file, "max")

            values = numpy.empty(length, dtype=_stl_object_type)

            # only do anything if we have anything to read...
            if length > 0:
                for i in range(length):
                    values[i] = model.read(
                        chunk,
                        cursor,
                        dict(context, reading=False),
                        file,
                        selffile,
                        parent,
                    )

                # memberwise reading!
                for member_index in range(len(values[0].member_names)):
                    for i in range(length):
                        values[i].read_member_n(
                            chunk, cursor, context, file, member_index
                        )
        else:
            length = cursor.field(chunk, _stl_container_size, context)

            values = _read_nested(
                self._values, length, chunk, cursor, context, file, selffile, parent
            )

        out = ROOTRVec(values)
        if self._header and header:
            uproot.deserialization.numbytes_check(
                chunk,
                start_cursor,
                cursor,
                num_bytes,
                self.typename,
                context,
                file.file_path,
            )

        return out

    def __eq__(self, other):
        if not isinstance(other, AsRVec):
            return False

        if self.header != other.header:
            return False

        if isinstance(self.values, numpy.dtype) and isinstance(
            other.values, numpy.dtype
        ):
            return self.values == other.values
        elif not isinstance(self.values, numpy.dtype) and not isinstance(
            other.values, numpy.dtype
        ):
            return self.values == other.values
        else:
            return False


class AsVector(AsContainer):
    """
    Args:
        header (bool): Sets the :ref:`uproot.containers.AsContainer.header`.
        values (:doc:`uproot.model.Model` or :doc:`uproot.containers.Container`): Data
            type for data nested in the container.

    A :doc:`uproot.containers.AsContainer` for ``std::vector``.
    """

    def __init__(self, header, values):
        self.header = header
        if isinstance(values, AsContainer):
            self._values = values
        elif isinstance(values, type) and issubclass(
            values, (uproot.model.Model, uproot.model.DispatchByVersion)
        ):
            self._values = values
        else:
            self._values = numpy.dtype(values)

    def __hash__(self):
        return hash((AsVector, self._header, self._values))

    @property
    def values(self):
        """
        Data type for data nested in the container.
        """
        return self._values

    def __repr__(self):
        if isinstance(self._values, type):
            values = self._values.__name__
        else:
            values = repr(self._values)
        return f"AsVector({self._header}, {values})"

    @property
    def cache_key(self):
        return f"AsVector({self._header},{_content_cache_key(self._values)})"

    @property
    def typename(self):
        return f"std::vector<{_content_typename(self._values)}>"

    def awkward_form(self, file, context):
        awkward = uproot.extras.awkward()
        return awkward.forms.ListOffsetForm(
            context["index_format"],
            uproot._util.awkward_form(self._values, file, context),
        )

    def read(self, chunk, cursor, context, file, selffile, parent, header=True):
        # AwkwardForth testing: test_0637's 00,03,04,06,07,08,09,10,11,12,13,14,15,16,17,23,24,26,27,28,31,33,36,38,41,42,43,44,45,46,49,50,55,56,57,58,59,60,61,62,63,67,68,72,73,76,77,80
        forth_stash = uproot._awkward_forth.forth_stash(context)
        if forth_stash is not None:
            forth_obj = forth_stash.get_gen_obj()

        if self._header and header:
            start_cursor = cursor.copy()
            (
                num_bytes,
                instance_version,
                is_memberwise,
            ) = uproot.deserialization.numbytes_version(chunk, cursor, context)
            if forth_stash is not None:
                temp_jump = cursor._index - start_cursor._index
                if temp_jump != 0:
                    forth_stash.add_to_pre(f"{temp_jump} stream skip\n")
        else:
            is_memberwise = False

        # Note: self._values can also be a NumPy dtype, and not necessarily a class
        # (e.g. type(self._values) == type)
        _value_typename = _content_typename(self._values)

        if is_memberwise:
            if forth_stash is not None:
                context["cancel_forth"] = True
            # let's hard-code in logic for std::pair<T1,T2> for now
            if not _value_typename.startswith("pair"):
                raise NotImplementedError(
                    """memberwise serialization of {}({})
    in file {}""".format(
                        type(self).__name__, _value_typename, selffile.file_path
                    )
                )

            if not issubclass(self._values, uproot.model.DispatchByVersion):
                raise NotImplementedError(
                    """streamerless memberwise serialization of class {}({})
    in file {}""".format(
                        type(self).__name__, _value_typename, selffile.file_path
                    )
                )

            # uninterpreted header
            cursor.skip(6)

            length = cursor.field(chunk, _stl_container_size, context)

            # no known class version number (maybe in that header? unclear...)
            model = self._values.new_class(file, "max")

            values = numpy.empty(length, dtype=_stl_object_type)

            # only do anything if we have anything to read...
            if length > 0:
                for i in range(length):
                    values[i] = model.read(
                        chunk,
                        cursor,
                        dict(context, reading=False),
                        file,
                        selffile,
                        parent,
                    )

                # memberwise reading!
                for member_index in range(len(values[0].member_names)):
                    for i in range(length):
                        values[i].read_member_n(
                            chunk, cursor, context, file, member_index
                        )
        else:
            length = cursor.field(chunk, _stl_container_size, context)
            if forth_stash is not None:
                key = forth_obj.get_keys(1)
                form_key = f"node{key}-offsets"
                forth_stash.add_to_header(f"output node{key}-offsets int64\n")
                forth_stash.add_to_init(f"0 node{key}-offsets <- stack\n")
                forth_stash.add_to_pre(
                    f"stream !I-> stack\n dup node{key}-offsets +<- stack\n"
                )
                # forth_stash.add_to_post("loop\n")
                if forth_obj.should_add_form():
                    forth_obj.add_form_key(form_key)
                    temp_aform = f'{{ "class":"ListOffsetArray", "offsets":"i64", "content": "NULL", "parameters": {{}}, "form_key": "node{key}"}}'
                    forth_obj.add_form(json.loads(temp_aform))
                if not isinstance(self._values, numpy.dtype):
                    forth_stash.add_to_pre("0 do\n")
                    forth_stash.add_to_post("loop\n")
                temp = forth_obj.add_node(
                    f"node{key}",
                    forth_stash.get_attrs(),
                    "i64",
                    1,
                    {},
                )
                context["temp_ref"] = temp

            if length == 0 and forth_stash is not None:
                forth_obj.var_set = True

            values = _read_nested(
                self._values, length, chunk, cursor, context, file, selffile, parent
            )

        if forth_stash is not None and not context["cancel_forth"]:

            forth_obj.go_to(temp)

        out = STLVector(values)

        if self._header and header:
            uproot.deserialization.numbytes_check(
                chunk,
                start_cursor,
                cursor,
                num_bytes,
                self.typename,
                context,
                file.file_path,
            )

        return out

    def __eq__(self, other):
        if not isinstance(other, AsVector):
            return False

        if self.header != other.header:
            return False

        if isinstance(self.values, numpy.dtype) and isinstance(
            other.values, numpy.dtype
        ):
            return self.values == other.values
        elif not isinstance(self.values, numpy.dtype) and not isinstance(
            other.values, numpy.dtype
        ):
            return self.values == other.values
        else:
            return False


class AsSet(AsContainer):
    """
    Args:
        header (bool): Sets the :ref:`uproot.containers.AsContainer.header`.
        keys (:doc:`uproot.model.Model` or :doc:`uproot.containers.Container`): Data
            type for data nested in the container.

    A :doc:`uproot.containers.AsContainer` for ``std::set``.
    """

    def __init__(self, header, keys):
        self.header = header
        if isinstance(keys, AsContainer):
            self._keys = keys
        elif isinstance(keys, type) and issubclass(
            keys, (uproot.model.Model, uproot.model.DispatchByVersion)
        ):
            self._keys = keys
        else:
            self._keys = numpy.dtype(keys)

    def __hash__(self):
        return hash((AsSet, self._header, self._keys))

    @property
    def keys(self):
        """
        Data type for data nested in the container.
        """
        return self._keys

    def __repr__(self):
        if isinstance(self._keys, type):
            keys = self._keys.__name__
        else:
            keys = repr(self._keys)
        return f"AsSet({self._header}, {keys})"

    @property
    def cache_key(self):
        return f"AsSet({self._header},{_content_cache_key(self._keys)})"

    @property
    def typename(self):
        return f"std::set<{_content_typename(self._keys)}>"

    def awkward_form(self, file, context):
        awkward = uproot.extras.awkward()
        return awkward.forms.ListOffsetForm(
            context["index_format"],
            uproot._util.awkward_form(self._keys, file, context),
            parameters={"__array__": "set"},
        )

    def read(self, chunk, cursor, context, file, selffile, parent, header=True):
        # AwkwardForth testing: test_0637's 62,63,64,65,69,70,74,75,77

        forth_stash = uproot._awkward_forth.forth_stash(context)

        if forth_stash is not None:
            forth_obj = forth_stash.get_gen_obj()

        if self._header and header:
            start_cursor = cursor.copy()
            (
                num_bytes,
                instance_version,
                is_memberwise,
            ) = uproot.deserialization.numbytes_version(chunk, cursor, context)
            if forth_stash is not None:
                temp_jump = cursor._index - start_cursor._index
                if temp_jump != 0:
                    forth_stash.add_to_pre(f"{temp_jump} stream skip\n")
        else:
            is_memberwise = False

        if is_memberwise:
            raise NotImplementedError(
                """memberwise serialization of {}
in file {}""".format(
                    type(self).__name__, selffile.file_path
                )
            )

        length = cursor.field(chunk, _stl_container_size, context)
        if forth_stash is not None:
            key = forth_obj.get_keys(1)

            form_key = f"node{key}-offsets"
            forth_stash.add_to_header(f"output node{key}-offsets int64\n")
            forth_stash.add_to_init(f"0 node{key}-offsets <- stack\n")
            forth_stash.add_to_pre(
                f"stream !I-> stack\ndup node{key}-offsets +<- stack\n"
            )
            # forth_stash.add_to_post("loop\n")
            if forth_obj.should_add_form():
                forth_obj.add_form_key(form_key)
                temp_aform = f'{{ "class":"ListOffsetArray", "offsets":"i64", "content": "NULL", "parameters": {{"__array__": "set"}}, "form_key": "node{key}"}}'
                forth_obj.add_form(json.loads(temp_aform))
            if not isinstance(self._keys, numpy.dtype):
                forth_stash.add_to_pre("0 do\n")
                forth_stash.add_to_post("loop\n")
            temp = forth_obj.add_node(
                f"node{key}",
                forth_stash.get_attrs(),
                "i64",
                1,
                {},
            )
        if length == 0 and forth_stash is not None:
            forth_obj.var_set = True
        keys = _read_nested(
            self._keys, length, chunk, cursor, context, file, selffile, parent
        )
        if forth_stash is not None:
            forth_obj.go_to(temp)
        out = STLSet(keys)

        if self._header and header:
            uproot.deserialization.numbytes_check(
                chunk,
                start_cursor,
                cursor,
                num_bytes,
                self.typename,
                context,
                file.file_path,
            )

        return out

    def __eq__(self, other):
        if not isinstance(other, AsSet):
            return False

        if self.header != other.header:
            return False

        if isinstance(self.keys, numpy.dtype) and isinstance(other.keys, numpy.dtype):
            return self.keys == other.keys
        elif not isinstance(self.keys, numpy.dtype) and not isinstance(
            other.keys, numpy.dtype
        ):
            return self.keys == other.keys
        else:
            return False


def _has_nested_header(obj):
    if isinstance(obj, AsContainer):
        return obj.header
    else:
        return False


class AsMap(AsContainer):
    """
    Args:
        header (bool): Sets the :ref:`uproot.containers.AsContainer.header`.
        keys (:doc:`uproot.model.Model` or :doc:`uproot.containers.Container`): Data
            type for the map's keys.
        values (:doc:`uproot.model.Model` or :doc:`uproot.containers.Container`): Data
            type for the map's values.

    A :doc:`uproot.containers.AsContainer` for ``std::map``.
    """

    def __init__(self, header, keys, values):
        self.header = header

        if isinstance(keys, AsContainer):
            self._keys = keys
        elif isinstance(keys, type) and issubclass(
            keys, (uproot.model.Model, uproot.model.DispatchByVersion)
        ):
            self._keys = keys
        else:
            self._keys = numpy.dtype(keys)

        if isinstance(values, AsContainer):
            self._values = values
        elif isinstance(values, type) and issubclass(
            values, (uproot.model.Model, uproot.model.DispatchByVersion)
        ):
            self._values = values
        else:
            self._values = numpy.dtype(values)

    def __hash__(self):
        return hash((AsMap, self._header, self._keys, self._values))

    @property
    def keys(self):
        """
        Data type for the map's keys.
        """
        return self._keys

    @property
    def values(self):
        """
        Data type for the map's values.
        """
        return self._values

    def __repr__(self):
        if isinstance(self._keys, type):
            keys = self._keys.__name__
        else:
            keys = repr(self._keys)
        if isinstance(self._values, type):
            values = self._values.__name__
        else:
            values = repr(self._values)
        return f"AsMap({self._header}, {keys}, {values})"

    @property
    def cache_key(self):
        return "AsMap({},{},{})".format(
            self._header,
            _content_cache_key(self._keys),
            _content_cache_key(self._values),
        )

    @property
    def typename(self):
        return "std::map<{}, {}>".format(
            _content_typename(self._keys), _content_typename(self._values)
        )

    def awkward_form(self, file, context):
        awkward = uproot.extras.awkward()
        return awkward.forms.ListOffsetForm(
            context["index_format"],
            awkward.forms.RecordForm(
                (
                    uproot._util.awkward_form(self._keys, file, context),
                    uproot._util.awkward_form(self._values, file, context),
                ),
                None,
                parameters={"__array__": "sorted_map"},
            ),
        )

    def read(self, chunk, cursor, context, file, selffile, parent, header=True):
        # AwkwardForth testing: test_0637's 00,33,35,39,47,48,66,67,68,69,70,71,72,73,74,75,76,77,78,79
        forth_stash = uproot._awkward_forth.forth_stash(context)

        if forth_stash is not None:
            forth_obj = forth_stash.get_gen_obj()
        if self._header and header:
            start_cursor = cursor.copy()
            (
                num_bytes,
                instance_version,
                is_memberwise,
            ) = uproot.deserialization.numbytes_version(chunk, cursor, context)
        else:
            is_memberwise = False

        if is_memberwise:
            if self._header and header:
                cursor.skip(6)
                if forth_stash is not None:
                    temp_jump = cursor._index - start_cursor._index
                    forth_stash.add_to_pre(f"{temp_jump} stream skip\n")
            length = cursor.field(chunk, _stl_container_size, context)
            if forth_stash is not None:
                key = forth_obj.get_keys(1)
                form_key = f"node{key}-offsets"
                forth_stash.add_to_header(f"output node{key}-offsets int64\n")
                forth_stash.add_to_init(f"0 node{key}-offsets <- stack\n")
                forth_stash.add_to_pre(
                    f"stream !I-> stack\n dup node{key}-offsets +<- stack\n"
                )
            if _has_nested_header(self._keys) and header:
                if forth_stash is not None:
                    forth_stash.add_to_pre("6 stream skip\n")
                cursor.skip(6)
            if forth_stash is not None:
                if not isinstance(self._keys, numpy.dtype):
                    forth_stash.add_to_pre("dup 0 do\n")
                else:
                    forth_stash.add_to_pre("dup\n")
            if forth_stash is not None:
                temp = {"name": "TOP", "content": {}}
                (
                    temp_node,
                    temp_node_top,
                    temp_form,
                    temp_form_top,
                    temp_prev_form,
                ) = forth_obj.replace_form_and_model(None, temp)
                context["temp_ref"] = temp
            keys = _read_nested(
                self._keys,
                length,
                chunk,
                cursor,
                context,
                file,
                selffile,
                parent,
                header=False,
            )
            if forth_stash is not None:
                temp = {"name": "TOP", "content": {}}
                keys_form = forth_obj.top_form
                keys_model = forth_obj.top_node
                (
                    temp_node1,
                    temp_node_top1,
                    temp_form1,
                    temp_form_top1,
                    temp_prev_form1,
                ) = forth_obj.replace_form_and_model(None, temp)
                context["temp_ref"] = temp
            if forth_stash is not None:
                if not isinstance(self._keys, numpy.dtype):
                    keys_model["content"]["post_code"].append("loop\n")
            if _has_nested_header(self._values) and header:
                cursor.skip(6)
                if forth_stash is not None:
                    keys_model["content"]["post_code"].append("6 stream skip\n")
            if forth_stash is not None:
                if not isinstance(self._values, numpy.dtype):
                    keys_model["content"]["post_code"].append("0 do\n")
            values = _read_nested(
                self._values,
                length,
                chunk,
                cursor,
                context,
                file,
                selffile,
                parent,
                header=False,
            )
            if forth_stash is not None:
                values_form = forth_obj.top_form
                values_model = forth_obj.top_node
                if not isinstance(self._values, numpy.dtype):
                    values_model["content"]["post_code"].append("loop \n")
                forth_obj.awkward_model = temp_node
                forth_obj.top_node = temp_node_top
                forth_obj.aform = temp_form
                forth_obj.top_form = temp_form_top
                forth_obj.prev_form = temp_prev_form
                aform = {
                    "class": "ListOffsetArray",
                    "offsets": "i64",
                    "content": {
                        "class": "RecordArray",
                        "contents": [keys_form, values_form],
                        "parameters": {"__array__": "sorted_map"},
                    },
                    "form_key": f"node{key}",
                }
                if forth_obj.should_add_form():
                    forth_obj.add_form_key(form_key)
                    forth_obj.add_form(aform)
                temp = forth_obj.add_node(
                    "nodeMap",
                    forth_stash.get_attrs(),
                    "i64",
                    1,
                    [keys_model["content"], values_model["content"]],
                )
            out = STLMap(keys, values)

            if self._header and header:
                uproot.deserialization.numbytes_check(
                    chunk,
                    start_cursor,
                    cursor,
                    num_bytes,
                    self.typename,
                    context,
                    file.file_path,
                )

            return out

        else:
            if forth_stash is not None:
                raise NotImplementedError(
                    """non-memberwise serialization of {}
in file {}""".format(
                        type(self).__name__, selffile.file_path
                    )
                )
            length = cursor.field(chunk, _stl_container_size, context)
            keys, values = [], []
            for _ in range(length):
                keys.append(
                    _read_nested(
                        self._keys, 1, chunk, cursor, context, file, selffile, parent
                    )
                )
                values.append(
                    _read_nested(
                        self._values, 1, chunk, cursor, context, file, selffile, parent
                    )
                )
            out = STLMap(numpy.concatenate(keys), numpy.concatenate(values))
            if self._header and header:
                uproot.deserialization.numbytes_check(
                    chunk,
                    start_cursor,
                    cursor,
                    num_bytes,
                    self.typename,
                    context,
                    file.file_path,
                )
            return out

    def __eq__(self, other):
        if not isinstance(other, AsMap):
            return False

        if self.header != other.header:
            return False

        if isinstance(self.keys, numpy.dtype) and isinstance(other.keys, numpy.dtype):
            if self.keys != other.keys:
                return False
        elif not isinstance(self.keys, numpy.dtype) and not isinstance(
            other.keys, numpy.dtype
        ):
            if self.keys != other.keys:
                return False
        else:
            return False

        if isinstance(self.values, numpy.dtype) and isinstance(
            other.values, numpy.dtype
        ):
            return self.values == other.values
        elif not isinstance(self.values, numpy.dtype) and not isinstance(
            other.values, numpy.dtype
        ):
            return self.values == other.values
        else:
            return False


class Container:
    """
    Abstract class for Python representations of C++ STL collections.
    """

    def __ne__(self, other):
        return not self == other

    def tolist(self):
        """
        Convert the data this collection contains into nested lists, sets,
        and dicts.
        """
        raise AssertionError


class ROOTRVec(Container, Sequence):
    """
    Args:
        values (``numpy.ndarray`` or iterable): Contents of the ``ROOT::VecOps::RVec``.

    Representation of a C++ ``ROOT::VecOps::RVec`` as a Python ``Sequence``.
    """

    def __init__(self, values):
        if isinstance(values, types.GeneratorType):
            values = numpy.asarray(list(values))
        elif isinstance(values, Set):
            values = numpy.asarray(list(values))
        elif isinstance(values, (list, tuple)):
            values = numpy.asarray(values)

        self._values = values

    def __str__(self, limit=85):
        def tostring(i):
            return _tostring(self._values[i])

        return _str_with_ellipsis(tostring, len(self), "[", "]", limit)

    def __repr__(self, limit=85):
        return "<ROOTRVec {} at 0x{:012x}>".format(
            self.__str__(limit=limit - 30), id(self)
        )

    def __getitem__(self, where):
        return self._values[where]

    def __len__(self):
        return len(self._values)

    def __contains__(self, what):
        return what in self._values

    def __iter__(self):
        return iter(self._values)

    def __reversed__(self):
        return ROOTRVec(self._values[::-1])

    def __eq__(self, other):
        if isinstance(other, ROOTRVec):
            return self._values == other._values
        elif isinstance(other, Sequence):
            return self._values == other
        else:
            return False

    def tolist(self):
        return [
            x.tolist() if isinstance(x, (Container, numpy.ndarray)) else x for x in self
        ]


class STLVector(Container, Sequence):
    """
    Args:
        values (``numpy.ndarray`` or iterable): Contents of the ``std::vector``.

    Representation of a C++ ``std::vector`` as a Python ``Sequence``.
    """

    def __init__(self, values):
        if isinstance(values, types.GeneratorType):
            values = numpy.asarray(list(values))
        elif isinstance(values, Set):
            values = numpy.asarray(list(values))
        elif isinstance(values, (list, tuple)):
            values = numpy.asarray(values)

        self._values = values

    def __str__(self, limit=85):
        def tostring(i):
            return _tostring(self._values[i])

        return _str_with_ellipsis(tostring, len(self), "[", "]", limit)

    def __repr__(self, limit=85):
        return "<STLVector {} at 0x{:012x}>".format(
            self.__str__(limit=limit - 30), id(self)
        )

    def __getitem__(self, where):
        return self._values[where]

    def __len__(self):
        return len(self._values)

    def __contains__(self, what):
        return what in self._values

    def __iter__(self):
        return iter(self._values)

    def __reversed__(self):
        return STLVector(self._values[::-1])

    def __eq__(self, other):
        if isinstance(other, STLVector):
            return self._values == other._values
        elif isinstance(other, Sequence):
            return self._values == other
        else:
            return False

    def tolist(self):
        return [
            x.tolist() if isinstance(x, (Container, numpy.ndarray)) else x for x in self
        ]


class STLSet(Container, Set):
    """
    Args:
        keys (``numpy.ndarray`` or iterable): Contents of the ``std::set``.

    Representation of a C++ ``std::set`` as a Python ``Set``.
    """

    def __init__(self, keys):
        if isinstance(keys, types.GeneratorType):
            keys = numpy.asarray(list(keys))
        elif isinstance(keys, Set):
            keys = numpy.asarray(list(keys))
        else:
            keys = numpy.asarray(keys)

        self._keys = numpy.sort(keys)

    def __str__(self, limit=85):
        def tostring(i):
            return _tostring(self._keys[i])

        return _str_with_ellipsis(tostring, len(self), "{", "}", limit)

    def __repr__(self, limit=85):
        return "<STLSet {} at 0x{:012x}>".format(
            self.__str__(limit=limit - 30), id(self)
        )

    def __len__(self):
        return len(self._keys)

    def __iter__(self):
        return iter(self._keys)

    def __contains__(self, where):
        where = numpy.asarray(where)
        index = numpy.searchsorted(self._keys.astype(where.dtype), where, side="left")

        if uproot._util.isint(index):
            if index < len(self._keys) and self._keys[index] == where:
                return True
            else:
                return False

        else:
            return False

    def __eq__(self, other):
        if isinstance(other, Set):
            if not isinstance(other, STLSet):
                other = STLSet(other)
        else:
            return False

        if len(self._keys) != len(other._keys):
            return False

        keys_same = self._keys == other._keys
        if isinstance(keys_same, bool):
            return keys_same
        else:
            return numpy.all(keys_same)

    def tolist(self):
        return {
            x.tolist() if isinstance(x, (Container, numpy.ndarray)) else x for x in self
        }


class STLMap(Container, Mapping):
    """
    Args:
        keys (``numpy.ndarray`` or iterable): Keys of the ``std::map``.
        values (``numpy.ndarray`` or iterable): Values of the ``std::map``.

    Representation of a C++ ``std::map`` as a Python ``Mapping``.

    The ``keys`` and ``values`` must have the same length.
    """

    @classmethod
    def from_mapping(cls, mapping):
        """
        Construct a :doc:`uproot.containers.STLMap` from a Python object with
        ``keys()`` and ``values()``.
        """
        return STLMap(mapping.keys(), mapping.values())

    def __init__(self, keys, values):
        if KeysView is not None and isinstance(keys, KeysView):
            keys = numpy.asarray(list(keys))
        elif isinstance(keys, types.GeneratorType):
            keys = numpy.asarray(list(keys))
        elif isinstance(keys, Set):
            keys = numpy.asarray(list(keys))
        else:
            keys = numpy.asarray(keys)

        if ValuesView is not None and isinstance(values, ValuesView):
            values = numpy.asarray(list(values))
        elif isinstance(values, types.GeneratorType):
            values = numpy.asarray(list(values))

        if len(keys) != len(values):
            raise ValueError("number of keys must be equal to the number of values")

        index = numpy.argsort(keys)

        self._keys = keys[index]
        try:
            self._values = values[index]
        except Exception:
            self._values = numpy.asarray(values)[index]

    def __str__(self, limit=85):
        def tostring(i):
            return _tostring(self._keys[i]) + ": " + _tostring(self._values[i])

        return _str_with_ellipsis(tostring, len(self), "{", "}", limit)

    def __repr__(self, limit=85):
        return "<STLMap {} at 0x{:012x}>".format(
            self.__str__(limit=limit - 30), id(self)
        )

    def keys(self):
        """
        Keys of the ``std::map`` as a ``numpy.ndarray``.
        """
        return self._keys

    def values(self):
        """
        Values of the ``std::map`` as a ``numpy.ndarray``.
        """
        return self._values

    def items(self):
        """
        Key, value pairs of the ``std::map`` as a ``numpy.ndarray``.
        """
        return numpy.transpose(numpy.vstack([self._keys, self._values]))

    def __getitem__(self, where):
        where = numpy.asarray(where)
        index = numpy.searchsorted(self._keys.astype(where.dtype), where, side="left")

        if uproot._util.isint(index):
            if index < len(self._keys) and self._keys[index] == where:
                return self._values[index]
            else:
                raise KeyError(where)

        elif len(self._keys) == 0:
            values = numpy.empty(len(index))
            return numpy.ma.MaskedArray(values, True)

        else:
            index[index >= len(self._keys)] = 0
            mask = self._keys[index] != where
            return numpy.ma.MaskedArray(self._values[index], mask)

    def get(self, where, default=None):
        """
        Get with a default, like dict.get.
        """
        where = numpy.asarray(where)
        index = numpy.searchsorted(self._keys.astype(where.dtype), where, side="left")

        if uproot._util.isint(index):
            if index < len(self._keys) and self._keys[index] == where:
                return self._values[index]
            else:
                return default

        elif len(self._keys) == 0:
            return numpy.array([default])[numpy.zeros(len(index), numpy.int32)]

        else:
            index[index >= len(self._keys)] = 0
            matches = self._keys[index] == where
            values = self._values[index]
            defaults = numpy.array([default])[numpy.zeros(len(index), numpy.int32)]
            return numpy.where(matches, values, defaults)

    def __len__(self):
        return len(self._keys)

    def __iter__(self):
        return iter(self._keys)

    def __contains__(self, where):
        where = numpy.asarray(where)
        index = numpy.searchsorted(self._keys.astype(where.dtype), where, side="left")

        if uproot._util.isint(index):
            if index < len(self._keys) and self._keys[index] == where:
                return True
            else:
                return False

        else:
            return False

    def __eq__(self, other):
        if isinstance(other, Mapping):
            if not isinstance(other, STLMap):
                other = STLMap(other.keys(), other.values())
        else:
            return False

        if len(self._keys) != len(other._keys):
            return False

        keys_same = self._keys == other._keys
        values_same = self._values == other._values
        if isinstance(keys_same, bool) and isinstance(values_same, bool):
            return keys_same and values_same
        else:
            return numpy.logical_and(keys_same, values_same).all()

    def tolist(self):
        out = {}
        for i in range(len(self)):
            x = self._values[i]
            if isinstance(x, (Container, numpy.ndarray)):
                out[self._keys[i]] = x.tolist()
            else:
                out[self._keys[i]] = x
        return out

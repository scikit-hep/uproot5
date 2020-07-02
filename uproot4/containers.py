# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import types
import struct

try:
    from collections.abc import Sequence
    from collections.abc import Set
    from collections.abc import Mapping
    from collections.abc import KeysView
    from collections.abc import ValuesView
except ImportError:
    from collections import Sequence
    from collections import Set
    from collections import Mapping

    KeysView = None
    ValuesView = None

import numpy

import uproot4._util
import uproot4.model
import uproot4.interpretation.numerical
import uproot4.deserialization


_stl_container_size = struct.Struct(">I")
_stl_object_type = numpy.dtype(numpy.object)


def _content_typename(content):
    if isinstance(content, numpy.dtype):
        return uproot4.interpretation.numerical._dtype_kind_itemsize_to_typename[
            content.kind, content.itemsize
        ]
    elif isinstance(content, type):
        return uproot4.model.classname_decode(content.__name__)[0]
    else:
        return content.typename


def _content_cache_key(content):
    if isinstance(content, numpy.dtype):
        bo = uproot4.interpretation.numerical._numpy_byteorder_to_cache_key[
            content.byteorder
        ]
        return "{0}{1}{2}".format(bo, content.kind, content.itemsize)
    elif isinstance(content, type):
        return content.__name__
    else:
        return content.cache_key


def _read_nested(model, length, chunk, cursor, context, file, parent, header=True):
    if isinstance(model, numpy.dtype):
        return cursor.array(chunk, length, model, context)

    else:
        values = numpy.empty(length, dtype=_stl_object_type)
        if isinstance(model, AsContainer):
            for i in range(length):
                values[i] = model.read(
                    chunk, cursor, context, file, parent, header=header
                )
        else:
            for i in range(length):
                values[i] = model.read(chunk, cursor, context, file, parent)
        return values


def _tostring(value):
    if uproot4._util.isstr(value):
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
    elif len(left) == 0 and len(right) == 0:
        return lbracket + "{0}, ...".format(tostring(0)) + rbracket
    elif len(right) == 0:
        return lbracket + "".join(left) + "..." + rbracket
    else:
        return lbracket + "".join(left) + "..., " + "".join(right) + rbracket


class AsContainer(object):
    @property
    def header(self):
        return self._header

    @header.setter
    def header(self, value):
        if value is True or value is False:
            self._header = value
        else:
            raise TypeError(
                "{0}.header must be True or False".format(type(self).__name__)
            )

    def strided_interpretation(
        self, file, header=False, tobject_header=True, original=None
    ):
        raise uproot4.interpretation.objects.CannotBeStrided(self.typename)

    @property
    def cache_key(self):
        raise AssertionError

    @property
    def typename(self):
        raise AssertionError

    def awkward_form(self, file, header=False, tobject_header=True):
        raise AssertionError

    def read(self, chunk, cursor, context, file, parent, header=True):
        raise AssertionError

    def __eq__(self, other):
        raise AssertionError

    def __ne__(self, other):
        return not self == other


class Container(object):
    def __ne__(self, other):
        return not self == other

    def tolist(self):
        raise AssertionError


class AsFIXME(AsContainer):
    def __init__(self, message):
        self.message = message

    def __hash__(self):
        return hash((AsFIXME, self.message))

    def __repr__(self):
        return "AsFIXME({0})".format(repr(self.message))

    @property
    def cache_key(self):
        return "AsFIXME({0})".format(repr(self.message))

    @property
    def typename(self):
        return "unknown"

    def awkward_form(self, file, header=False, tobject_header=True):
        raise uproot4.interpretation.objects.CannotBeAwkward(self.message)

    def read(self, chunk, cursor, context, file, parent, header=True):
        raise uproot4.deserialization.DeserializationError(
            self.message + "; please file a bug report!", None, None, None, None
        )

    def __eq__(self, other):
        if isinstance(other, AsFIXME):
            return self.message == other.message
        else:
            return False


class AsString(AsContainer):
    def __init__(self, header, length_bytes="1-5", typename=None):
        self.header = header
        if length_bytes in ("1-5", "4"):
            self._length_bytes = length_bytes
        else:
            raise ValueError("length_bytes must be '1-5' or '4'")
        self._typename = typename

    @property
    def length_bytes(self):
        return self._length_bytes

    def __hash__(self):
        return hash((AsString, self._header, self._length_bytes))

    def __repr__(self):
        args = [repr(self._header)]
        if self._length_bytes != "1-5":
            args.append("length_bytes={0}".format(repr(self._length_bytes)))
        return "AsString({0})".format(", ".join(args))

    @property
    def cache_key(self):
        return "AsString({0},{1})".format(self._header, repr(self._length_bytes))

    @property
    def typename(self):
        if self._typename is None:
            return "std::string"
        else:
            return self._typename

    def awkward_form(self, file, header=False, tobject_header=True):
        import awkward1

        return awkward1.forms.ListOffsetForm(
            "i32",
            awkward1.forms.NumpyForm((), 1, "B", parameters={"__array__": "char"}),
            parameters={
                "__array__": "string",
                "uproot": {
                    "as": "string",
                    "header": self._header,
                    "length_bytes": self._length_bytes,
                },
            },
        )

    def read(self, chunk, cursor, context, file, parent, header=True):
        if self._header and header:
            start_cursor = cursor.copy()
            num_bytes, instance_version = uproot4.deserialization.numbytes_version(
                chunk, cursor, context
            )

        if self._length_bytes == "1-5":
            out = cursor.string(chunk, context)
        elif self._length_bytes == "4":
            length = cursor.field(chunk, _stl_container_size, context)
            out = cursor.string_with_length(chunk, context, length)
        else:
            raise AssertionError(repr(self._length_bytes))

        if self._header and header:
            uproot4.deserialization.numbytes_check(
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
        return (
            isinstance(other, AsString)
            and self.header == other.header
            and self.length_bytes == other.length_bytes
        )


class AsPointer(AsContainer):
    def __init__(self, pointee):
        self._pointee = pointee

    @property
    def pointee(self):
        return self._pointee

    def __hash__(self):
        return hash((AsPointer, self._pointee))

    def __repr__(self):
        if isinstance(self._pointee, type):
            pointee = self._pointee.__name__
        else:
            pointee = repr(self._pointee)
        return "AsPointer({0})".format(pointee)

    @property
    def cache_key(self):
        return "AsPointer({0})".format(_content_cache_key(self._pointee))

    @property
    def typename(self):
        return _content_typename(self._pointee) + "*"

    def awkward_form(self, file, header=False, tobject_header=True):
        raise uproot4.interpretation.objects.CannotBeAwkward("arbitrary pointer")

    def read(self, chunk, cursor, context, file, parent, header=True):
        return uproot4.deserialization.read_object_any(
            chunk, cursor, context, file, parent
        )

    def __eq__(self, other):
        if isinstance(other, AsPointer):
            return self._pointee == other._pointee
        else:
            return False


class AsArray(AsContainer):
    def __init__(self, header, values):
        self._header = header
        self._values = values

    @property
    def values(self):
        return self._values

    def __repr__(self):
        if isinstance(self._values, type):
            values = self._values.__name__
        else:
            values = repr(self._values)
        return "AsArray({0}, {1})".format(self.header, values)

    @property
    def cache_key(self):
        return "AsArray({0},{1})".format(self.header, _content_cache_key(self._values))

    @property
    def typename(self):
        return _content_typename(self._values) + "*"

    def awkward_form(self, file, header=False, tobject_header=True):
        import awkward1

        return awkward1.forms.ListOffsetForm(
            "i32",
            uproot4._util.awkward_form(self._values, file, header, tobject_header),
            parameters={"uproot": {"as": "array", "header": self._header}},
        )

    def read(self, chunk, cursor, context, file, parent, header=True):
        if self._header and header:
            cursor.skip(1)

        if isinstance(self._values, numpy.dtype):
            remainder = chunk.remainder(cursor.index, cursor, context)
            return remainder.view(self._values)

        else:
            out = []
            while cursor.index < chunk.stop:
                out.append(self._values.read(chunk, cursor, context, file, parent))
            return numpy.array(out, dtype=numpy.dtype(numpy.object))


class AsDynamic(AsContainer):
    def __init__(self, model=None):
        self._model = model

    @property
    def model(self):
        return self._model

    def __repr__(self):
        if self._model is None:
            model = ""
        elif isinstance(self._model, type):
            model = "model=" + self._model.__name__
        else:
            model = "model=" + repr(self._model)
        return "AsDynamic({0})".format(model)

    @property
    def cache_key(self):
        if self._model is None:
            return "AsDynamic(None)"
        else:
            return "AsDynamic({0})".format(_content_cache_key(self._model))

    @property
    def typename(self):
        if self._model is None:
            return "void*"
        else:
            return _content_typename(self._values) + "*"

    def awkward_form(self, file, header=False, tobject_header=True):
        import awkward1

        if self._model is None:
            raise uproot4.interpretation.objects.CannotBeAwkward("dynamic type")
        else:
            return awkward1.forms.ListOffsetForm(
                "i32",
                uproot4._util.awkward_form(self._model, file, header, tobject_header),
                parameters={"uproot": {"as": "array", "header": self._header}},
            )

    def read(self, chunk, cursor, context, file, parent, header=True):
        classname = cursor.string(chunk, context)
        cursor.skip(1)
        cls = file.class_named(classname)
        return cls.read(chunk, cursor, context, file, parent)


class AsVector(AsContainer):
    def __init__(self, header, values):
        self.header = header
        if isinstance(values, AsContainer):
            self._values = values
        elif isinstance(values, type) and issubclass(
            values, (uproot4.model.Model, uproot4.model.DispatchByVersion)
        ):
            self._values = values
        else:
            self._values = numpy.dtype(values)

    def __hash__(self):
        return hash((AsVector, self._header, self._values))

    @property
    def values(self):
        return self._values

    def __repr__(self):
        if isinstance(self._values, type):
            values = self._values.__name__
        else:
            values = repr(self._values)
        return "AsVector({0}, {1})".format(self._header, values)

    @property
    def cache_key(self):
        return "AsVector({0},{1})".format(
            self._header, _content_cache_key(self._values)
        )

    @property
    def typename(self):
        return "std::vector<{0}>".format(_content_typename(self._values))

    def awkward_form(self, file, header=False, tobject_header=True):
        import awkward1

        return awkward1.forms.ListOffsetForm(
            "i32",
            uproot4._util.awkward_form(self._values, file, header, tobject_header),
            parameters={"uproot": {"as": "vector", "header": self._header}},
        )

    def read(self, chunk, cursor, context, file, parent, header=True):
        if self._header and header:
            start_cursor = cursor.copy()
            num_bytes, instance_version = uproot4.deserialization.numbytes_version(
                chunk, cursor, context
            )

        length = cursor.field(chunk, _stl_container_size, context)

        values = _read_nested(
            self._values, length, chunk, cursor, context, file, parent
        )
        out = STLVector(values)

        if self._header and header:
            uproot4.deserialization.numbytes_check(
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


class STLVector(Container, Sequence):
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
        return "<STLVector {0} at 0x{1:012x}>".format(
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


class AsSet(AsContainer):
    def __init__(self, header, keys):
        self.header = header
        if isinstance(keys, AsContainer):
            self._keys = keys
        elif isinstance(keys, type) and issubclass(
            keys, (uproot4.model.Model, uproot4.model.DispatchByVersion)
        ):
            self._keys = keys
        else:
            self._keys = numpy.dtype(keys)

    def __hash__(self):
        return hash((AsSet, self._header, self._keys))

    @property
    def keys(self):
        return self._keys

    def __repr__(self):
        if isinstance(self._keys, type):
            keys = self._keys.__name__
        else:
            keys = repr(self._keys)
        return "AsSet({0}, {1})".format(self._header, keys)

    @property
    def cache_key(self):
        return "AsSet({0},{1})".format(self._header, _content_cache_key(self._keys))

    @property
    def typename(self):
        return "std::set<{0}>".format(_content_typename(self._keys))

    def awkward_form(self, file, header=False, tobject_header=True):
        import awkward1

        return awkward1.forms.ListOffsetForm(
            "i32",
            uproot4._util.awkward_form(self._keys, file, header, tobject_header),
            parameters={
                "__array__": "set",
                "uproot": {"as": "set", "header": self._header},
            },
        )

    def read(self, chunk, cursor, context, file, parent, header=True):
        if self._header and header:
            start_cursor = cursor.copy()
            num_bytes, instance_version = uproot4.deserialization.numbytes_version(
                chunk, cursor, context
            )

        length = cursor.field(chunk, _stl_container_size, context)

        keys = _read_nested(self._keys, length, chunk, cursor, context, file, parent)
        out = STLSet(keys)

        if self._header and header:
            uproot4.deserialization.numbytes_check(
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


class STLSet(Container, Set):
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
        return "<STLSet {0} at 0x{1:012x}>".format(
            self.__str__(limit=limit - 30), id(self)
        )

    def __len__(self):
        return len(self._keys)

    def __iter__(self):
        return iter(self._keys)

    def __contains__(self, where):
        where = numpy.asarray(where)
        index = numpy.searchsorted(self._keys.astype(where.dtype), where, side="left")

        if uproot4._util.isint(index):
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
        return set(
            x.tolist() if isinstance(x, (Container, numpy.ndarray)) else x for x in self
        )


def _has_nested_header(obj):
    if isinstance(obj, AsContainer):
        return obj.header
    else:
        return False


class AsMap(AsContainer):
    def __init__(self, header, keys, values):
        self.header = header

        if isinstance(keys, AsContainer):
            self._keys = keys
        else:
            self._keys = numpy.dtype(keys)

        if isinstance(values, AsContainer):
            self._values = values
        elif isinstance(values, type) and issubclass(
            values, (uproot4.model.Model, uproot4.model.DispatchByVersion)
        ):
            self._values = values
        else:
            self._values = numpy.dtype(values)

    def __hash__(self):
        return hash((AsMap, self._header, self._keys, self._values))

    @property
    def keys(self):
        return self._keys

    @property
    def values(self):
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
        return "AsMap({0}, {1}, {2})".format(self._header, keys, values)

    @property
    def cache_key(self):
        return "AsMap({0},{1},{2})".format(
            self._header,
            _content_cache_key(self._keys),
            _content_cache_key(self._values),
        )

    @property
    def typename(self):
        return "std::map<{0}, {1}>".format(
            _content_typename(self._keys), _content_typename(self._values)
        )

    def awkward_form(self, file, header=False, tobject_header=True):
        import awkward1

        return awkward1.forms.ListOffsetForm(
            "i32",
            awkward1.forms.RecordForm(
                (
                    uproot4._util.awkward_form(
                        self._keys, file, header, tobject_header
                    ),
                    uproot4._util.awkward_form(
                        self._values, file, header, tobject_header
                    ),
                )
            ),
            parameters={
                "__array__": "sorted_map",
                "uproot": {"as": "map", "header": self._header},
            },
        )

    def read(self, chunk, cursor, context, file, parent, header=True):
        if self._header and header:
            start_cursor = cursor.copy()
            num_bytes, instance_version = uproot4.deserialization.numbytes_version(
                chunk, cursor, context
            )
            cursor.skip(6)

        length = cursor.field(chunk, _stl_container_size, context)

        if _has_nested_header(self._keys) and header:
            cursor.skip(6)
        keys = _read_nested(
            self._keys, length, chunk, cursor, context, file, parent, header=False
        )

        if _has_nested_header(self._values) and header:
            cursor.skip(6)
        values = _read_nested(
            self._values, length, chunk, cursor, context, file, parent, header=False
        )

        out = STLMap(keys, values)

        if self._header and header:
            uproot4.deserialization.numbytes_check(
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


class STLMap(Container, Mapping):
    @classmethod
    def from_mapping(cls, mapping):
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
        return "<STLMap {0} at 0x{1:012x}>".format(
            self.__str__(limit=limit - 30), id(self)
        )

    def __getitem__(self, where):
        where = numpy.asarray(where)
        index = numpy.searchsorted(self._keys.astype(where.dtype), where, side="left")

        if uproot4._util.isint(index):
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
        where = numpy.asarray(where)
        index = numpy.searchsorted(self._keys.astype(where.dtype), where, side="left")

        if uproot4._util.isint(index):
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

        if uproot4._util.isint(index):
            if index < len(self._keys) and self._keys[index] == where:
                return True
            else:
                return False

        else:
            return False

    def keys(self):
        return self._keys

    def values(self):
        return self._values

    def items(self):
        return numpy.transpose(numpy.vstack([self._keys, self._values]))

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

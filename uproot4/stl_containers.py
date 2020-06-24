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
_stl_primitive_types = {
    numpy.dtype("?"): "bool",
    numpy.dtype("i1"): "int8_t",
    numpy.dtype("u1"): "uint8_t",
    numpy.dtype("i2"): "int16_t",
    numpy.dtype(">i2"): "int16_t",
    numpy.dtype("u2"): "unt16_t",
    numpy.dtype(">u2"): "unt16_t",
    numpy.dtype("i4"): "int32_t",
    numpy.dtype(">i4"): "int32_t",
    numpy.dtype("u4"): "unt32_t",
    numpy.dtype(">u4"): "unt32_t",
    numpy.dtype("i8"): "int64_t",
    numpy.dtype(">i8"): "int64_t",
    numpy.dtype("u8"): "unt64_t",
    numpy.dtype(">u8"): "unt64_t",
    numpy.dtype("f4"): "float",
    numpy.dtype(">f4"): "float",
    numpy.dtype("f8"): "double",
    numpy.dtype(">f8"): "double",
}
_stl_object_type = numpy.dtype(numpy.object)


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


def _nested_context(context):
    out = dict(context)
    out["read_stl_header"] = False
    return out


def _read_nested(model, length, chunk, cursor, context, file, parent):
    if isinstance(model, numpy.dtype):
        return cursor.array(chunk, length, model, context)

    else:
        values = numpy.empty(length, dtype=_stl_object_type)
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


class AsSTLContainer(object):
    @property
    def header(self):
        return self._header

    @header.setter
    def header(self, value):
        if value is True or value is False:
            self._header = value
        else:
            raise TypeError("STLContainer.header must be True or False")

    @property
    def cache_key(self):
        raise AssertionError

    @property
    def classname(self):
        raise AssertionError

    def read(self, chunk, cursor, context, file, parent):
        raise AssertionError

    def __eq__(self, other):
        raise AssertionError

    def __ne__(self, other):
        return not self == other

    def tolist(self):
        raise AssertionError


class STLContainer(object):
    def __ne__(self, other):
        return not self == other


class AsString(AsSTLContainer):
    def __init__(self, header):
        self.header = header

    def __hash__(self):
        return hash((AsString, self._header))

    def __repr__(self):
        return "AsString({0})".format(self._header)

    @property
    def cache_key(self):
        return "AsString({0})".format(self._header)

    @property
    def classname(self):
        return "std::string"

    def read(self, chunk, cursor, context, file, parent):
        if self._header:
            start_cursor = cursor.copy()
            num_bytes, instance_version = uproot4.deserialization.numbytes_version(
                chunk, cursor, context
            )

        out = cursor.string(chunk, context)

        if self._header:
            uproot4.deserialization.numbytes_check(
                chunk,
                start_cursor,
                cursor,
                num_bytes,
                self.classname,
                context,
                file.file_path,
            )

        return out

    def __eq__(self, other):
        return isinstance(other, AsString) and self.header == other.header


class AsVector(AsSTLContainer):
    def __init__(self, header, values):
        self.header = header
        if isinstance(values, AsSTLContainer):
            self._values = values
        elif isinstance(values, type) and issubclass(values, uproot4.model.Model):
            self._values = values
        else:
            self._values = numpy.dtype(values)

    def __hash__(self):
        return hash((AsVector, self._header, self._values))

    @property
    def values(self):
        return self._values

    def __repr__(self):
        return "AsVector({0}, {1})".format(self._header, repr(self._values))

    @property
    def cache_key(self):
        return "AsVector({0},{1})".format(
            self._header, _content_cache_key(self._values)
        )

    @property
    def classname(self):
        values = _stl_primitive_types.get(self._values)
        if values is None:
            values = self._values.classname
        return "std::vector<{0}>".format(values)

    def read(self, chunk, cursor, context, file, parent):
        if self._header:
            start_cursor = cursor.copy()
            num_bytes, instance_version = uproot4.deserialization.numbytes_version(
                chunk, cursor, context
            )

        length = cursor.field(chunk, _stl_container_size, context)

        values = _read_nested(
            self._values, length, chunk, cursor, context, file, parent
        )
        out = STLVector(values)

        if self._header:
            uproot4.deserialization.numbytes_check(
                chunk,
                start_cursor,
                cursor,
                num_bytes,
                self.classname,
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


class STLVector(STLContainer, Sequence):
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
            x.tolist() if isinstance(x, (STLContainer, numpy.ndarray)) else x
            for x in self
        ]


class AsSet(AsSTLContainer):
    def __init__(self, header, keys):
        self.header = header
        if isinstance(keys, AsSTLContainer):
            self._keys = keys
        elif isinstance(keys, type) and issubclass(keys, uproot4.model.Model):
            self._keys = keys
        else:
            self._keys = numpy.dtype(keys)

    def __hash__(self):
        return hash((AsSet, self._header, self._keys))

    @property
    def keys(self):
        return self._keys

    def __repr__(self):
        return "AsSet({0}, {1})".format(self._header, repr(self._keys))

    @property
    def cache_key(self):
        return "AsSet({0},{1})".format(self._header, _content_cache_key(self._keys))

    @property
    def classname(self):
        keys = _stl_primitive_types.get(self._keys)
        if keys is None:
            keys = self._keys.classname
        return "std::set<{0}>".format(keys)

    def read(self, chunk, cursor, context, file, parent):
        if self._header:
            start_cursor = cursor.copy()
            num_bytes, instance_version = uproot4.deserialization.numbytes_version(
                chunk, cursor, context
            )

        length = cursor.field(chunk, _stl_container_size, context)

        keys = _read_nested(self._keys, length, chunk, cursor, context, file, parent)
        out = STLSet(keys)

        if self._header:
            uproot4.deserialization.numbytes_check(
                chunk,
                start_cursor,
                cursor,
                num_bytes,
                self.classname,
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


class STLSet(STLContainer, Set):
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
            x.tolist() if isinstance(x, (STLContainer, numpy.ndarray)) else x
            for x in self
        )


class AsMap(AsSTLContainer):
    def __init__(self, header, keys, values):
        self.header = header

        if isinstance(keys, AsSTLContainer):
            self._keys = keys
        else:
            self._keys = numpy.dtype(keys)

        if isinstance(values, AsSTLContainer):
            self._values = values
        elif isinstance(values, type) and issubclass(values, uproot4.model.Model):
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
        return "AsMap({0}, {1}, {2})".format(
            self._header, repr(self._keys), repr(self._values)
        )

    @property
    def cache_key(self):
        return "AsMap({0},{1},{2})".format(
            self._header,
            _content_cache_key(self._keys),
            _content_cache_key(self._values),
        )

    @property
    def classname(self):
        keys = _stl_primitive_types.get(self._keys)
        if keys is None:
            keys = self._keys.classname
        values = _stl_primitive_types.get(self._values)
        if values is None:
            values = self._values.classname
        return "std::map<{0}, {1}>".format(keys, values)

    def read(self, chunk, cursor, context, file, parent):
        if self._header:
            start_cursor = cursor.copy()
            num_bytes, instance_version = uproot4.deserialization.numbytes_version(
                chunk, cursor, context
            )

        cursor.skip(6)

        length = cursor.field(chunk, _stl_container_size, context)

        if self._header:
            cursor.skip(6)
        keys = _read_nested(self._keys, length, chunk, cursor, context, file, parent)

        if self._header:
            cursor.skip(6)
        values = _read_nested(
            self._values, length, chunk, cursor, context, file, parent
        )

        out = STLMap(keys, values)

        if self._header:
            uproot4.deserialization.numbytes_check(
                chunk,
                start_cursor,
                cursor,
                num_bytes,
                self.classname,
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


class STLMap(STLContainer, Mapping):
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
        return dict(
            (k, v.tolist() if isinstance(v, (STLContainer, numpy.ndarray)) else v)
            for k, v in self
        )

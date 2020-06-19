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
import uproot4.deserialization


class STLContainer(object):
    pass


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


_stl_container_size = struct.Struct(">I")
_stl_primitive_types = {
    numpy.dtype("bool"): "bool",
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


class AsSTLContainer(object):
    @property
    def classname(self):
        raise AssertionError

    def read_with_header(self, chunk, cursor, context, file, parent):
        raise AssertionError

    def read(self, chunk, cursor, context, file, parent):
        raise AssertionError


class AsString(AsSTLContainer):
    def __init__(self, is_stl=True):
        self._is_stl = is_stl

    @property
    def is_stl(self):
        return self._is_stl

    def __repr__(self):
        is_stl = ""
        if not self._is_stl:
            is_stl = "is_stl=False"
        return "AsString({0})".format(is_stl)

    @property
    def classname(self):
        if self._is_stl:
            return "std::string"
        else:
            return "const char*"

    def read_with_header(self, chunk, cursor, context, file, parent):
        if not self._is_stl:
            return self.read(chunk, cursor, context, file, parent)

        start_cursor = cursor.copy()
        num_bytes, instance_version = uproot4.deserialization.numbytes_version(
            chunk, cursor, context
        )

        out = cursor.string(chunk, context)

        uproot4.deserialization.numbytes_check(
            start_cursor, cursor, num_bytes, self.classname, context, file.file_path,
        )

        return out

    def read(self, chunk, cursor, context, file, parent):
        return cursor.string(chunk, context)


class AsVector(AsSTLContainer):
    def __init__(self, values):
        if isinstance(values, AsSTLContainer):
            self._values = values
        elif isinstance(values, type) and issubclass(values, uproot4.model.Model):
            self._values = values
        else:
            self._values = numpy.dtype(values)

    @property
    def values(self):
        return self._values

    def __repr__(self):
        return "AsVector({0})".format(repr(self._values))

    @property
    def classname(self):
        values = _stl_primitive_types.get(self._values)
        if values is None:
            values = self._values.classname
        return "std::vector<{0}>".format(values)

    def read_with_header(self, chunk, cursor, context, file, parent):
        start_cursor = cursor.copy()
        num_bytes, instance_version = uproot4.deserialization.numbytes_version(
            chunk, cursor, context
        )
        length = cursor.field(chunk, _stl_container_size, context)

        if isinstance(self._values, numpy.dtype):
            values = cursor.array(chunk, length, self._values, context)
        else:
            values = numpy.empty(length, dtype=_stl_object_type)
            for i in range(length):
                values[i] = self._values.read(chunk, cursor, context, file, parent)

        uproot4.deserialization.numbytes_check(
            start_cursor, cursor, num_bytes, self.classname, context, file.file_path,
        )

        return STLVector(values)

    def read(self, chunk, cursor, context, file, parent):
        length = cursor.field(chunk, _stl_container_size, context)

        if isinstance(self._values, numpy.dtype):
            values = cursor.array(chunk, length, self._values, context)
        else:
            values = numpy.empty(length, dtype=_stl_object_type)
            for i in range(length):
                values[i] = self._values.read(chunk, cursor, context, file, parent)

        return STLVector(values)


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

    def __ne__(self, other):
        return not self == other


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

    def __ne__(self, other):
        return not self == other


class AsMap(AsSTLContainer):
    def __init__(self, keys, values):
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

        print("constructed", self)

    @property
    def keys(self):
        return self._keys

    @property
    def values(self):
        return self._values

    def __repr__(self):
        return "AsMap({0}, {1})".format(repr(self._keys), repr(self._values))

    @property
    def classname(self):
        keys = _stl_primitive_types.get(self._keys)
        if keys is None:
            keys = self._keys.classname
        values = _stl_primitive_types.get(self._values)
        if values is None:
            values = self._values.classname
        return "std::map<{0}, {1}>".format(keys, values)

    def read_with_header(self, chunk, cursor, context, file, parent):
        print(cursor)
        cursor.debug(chunk, limit_bytes=80)

        start_cursor = cursor.copy()
        num_bytes, instance_version = uproot4.deserialization.numbytes_version(
            chunk, cursor, context
        )

        cursor.skip(6)

        length = cursor.field(chunk, _stl_container_size, context)

        print("size", length)

        if isinstance(self._keys, numpy.dtype):
            keys = cursor.array(chunk, length, self._keys, context)
        else:
            keys = numpy.empty(length, dtype=_stl_object_type)
            for i in range(length):
                keys[i] = self._keys.read(chunk, cursor, context, file, parent)

        if isinstance(self._values, numpy.dtype):
            values = cursor.array(chunk, length, self._values, context)
        else:
            values = numpy.empty(length, dtype=_stl_object_type)
            for i in range(length):
                values[i] = self._values.read(chunk, cursor, context, file, parent)

        uproot4.deserialization.numbytes_check(
            start_cursor, cursor, num_bytes, self.classname, context, file.file_path,
        )

        return STLMap(keys, values)

    def read(self, chunk, cursor, context, file, parent):
        length = cursor.field(chunk, _stl_container_size, context)

        if isinstance(self._keys, numpy.dtype):
            keys = cursor.array(chunk, length, self._keys, context)
        else:
            keys = numpy.empty(length, dtype=_stl_object_type)
            for i in range(length):
                keys[i] = self._keys.read(chunk, cursor, context, file, parent)

        if isinstance(self._values, numpy.dtype):
            values = cursor.array(chunk, length, self._values, context)
        else:
            values = numpy.empty(length, dtype=_stl_object_type)
            for i in range(length):
                values[i] = self._values.read(chunk, cursor, context, file, parent)

        return STLMap(keys, values)


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

    def __ne__(self, other):
        return not self == other

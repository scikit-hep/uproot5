# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import uproot4.interpretation

import numpy


def _dtype_shape(dtype):
    shape = ()
    while dtype.subdtype is not None:
        dtype, s = dtype.subdtype
        shape = shape + s
    return dtype, shape


class Numerical(uproot4.interpretation.Interpretation):
    def _wrap_almost_finalized(self, array):
        return array

    def final_array(
        self, basket_arrays, entry_start, entry_stop, entry_offsets, library, branch
    ):
        self.hook_before_final_array(
            basket_arrays=basket_arrays,
            entry_start=entry_start,
            entry_stop=entry_stop,
            entry_offsets=entry_offsets,
            library=library,
            branch=branch,
        )

        if entry_start >= entry_stop:
            output = library.empty((0,), self.to_dtype)

        else:
            length = 0
            start = entry_offsets[0]
            for basket_num, stop in enumerate(entry_offsets[1:]):
                if start <= entry_start and entry_stop <= stop:
                    length += entry_stop - entry_start
                elif start <= entry_start < stop:
                    length += stop - entry_start
                elif start <= entry_stop <= stop:
                    length += entry_stop - start
                elif entry_start < stop and start <= entry_stop:
                    length += stop - start
                start = stop

            output = library.empty((length,), self.to_dtype)

            start = entry_offsets[0]
            for basket_num, stop in enumerate(entry_offsets[1:]):
                if start <= entry_start and entry_stop <= stop:
                    local_start = entry_start - start
                    local_stop = entry_stop - start
                    basket_array = basket_arrays[basket_num]
                    output[:] = basket_array[local_start:local_stop]

                elif start <= entry_start < stop:
                    local_start = entry_start - start
                    local_stop = stop - start
                    basket_array = basket_arrays[basket_num]
                    output[: stop - entry_start] = basket_array[local_start:local_stop]

                elif start <= entry_stop <= stop:
                    local_start = 0
                    local_stop = entry_stop - start
                    basket_array = basket_arrays[basket_num]
                    output[start - entry_start :] = basket_array[local_start:local_stop]

                elif entry_start < stop and start <= entry_stop:
                    basket_array = basket_arrays[basket_num]
                    output[start - entry_start : stop - entry_start] = basket_array

                start = stop

        self.hook_before_library_finalize(
            basket_arrays=basket_arrays,
            entry_start=entry_start,
            entry_stop=entry_stop,
            entry_offsets=entry_offsets,
            library=library,
            branch=branch,
            output=output,
        )

        output = self._wrap_almost_finalized(output)

        output = library.finalize(output, branch, self, entry_start, entry_stop)

        self.hook_after_final_array(
            basket_arrays=basket_arrays,
            entry_start=entry_start,
            entry_stop=entry_stop,
            entry_offsets=entry_offsets,
            library=library,
            branch=branch,
            output=output,
        )

        return output


_numpy_byteorder_to_cache_key = {
    "!": "B",
    ">": "B",
    "<": "L",
    "|": "L",
    "=": "B" if numpy.dtype(">f8").isnative else "L",
}

_dtype_kind_itemsize_to_typename = {
    ("b", 1): "bool",
    ("i", 1): "int8_t",
    ("u", 1): "uint8_t",
    ("i", 2): "int16_t",
    ("u", 2): "uint16_t",
    ("i", 4): "int32_t",
    ("u", 4): "uint32_t",
    ("i", 8): "int64_t",
    ("u", 8): "uint64_t",
    ("f", 4): "float",
    ("f", 8): "double",
}


class AsDtype(Numerical):
    def __init__(self, from_dtype, to_dtype=None):
        self._from_dtype = numpy.dtype(from_dtype)
        if to_dtype is None:
            self._to_dtype = self._from_dtype.newbyteorder("=")
        else:
            self._to_dtype = numpy.dtype(to_dtype)

    def __repr__(self):
        if self._to_dtype == self._from_dtype.newbyteorder("="):
            return "AsDtype({0})".format(repr(str(self._from_dtype)))
        else:
            return "AsDtype({0}, {1})".format(
                repr(str(self._from_dtype)), repr(str(self._to_dtype))
            )

    def __eq__(self, other):
        return (
            type(other) is AsDtype
            and self._from_dtype == other._from_dtype
            and self._to_dtype == other._to_dtype
        )

    @property
    def from_dtype(self):
        return self._from_dtype

    @property
    def to_dtype(self):
        return self._to_dtype

    @property
    def itemsize(self):
        return self._from_dtype.itemsize

    @property
    def numpy_dtype(self):
        return self._to_dtype

    def awkward_form(self, file, index_format="i64", header=False, tobject_header=True):
        import awkward1

        d, s = _dtype_shape(self._to_dtype)
        out = uproot4._util.awkward_form(d, file, index_format, header, tobject_header)
        for size in s[::-1]:
            out = awkward1.forms.RegularForm(out, size)
        return out

    @property
    def cache_key(self):
        def form(dtype, name):
            d, s = _dtype_shape(dtype)
            return "{0}{1}{2}({3}{4})".format(
                _numpy_byteorder_to_cache_key[d.byteorder],
                d.kind,
                d.itemsize,
                ",".join(repr(x) for x in s),
                name,
            )

        if self.from_dtype.names is None:
            from_dtype = form(self.from_dtype, "")
        else:
            from_dtype = (
                "["
                + ",".join(
                    form(self.from_dtype[n], "," + repr(n))
                    for n in self.from_dtype.names
                )
                + "]"
            )

        if self.to_dtype.names is None:
            to_dtype = form(self.to_dtype, "")
        else:
            to_dtype = (
                "["
                + ",".join(
                    form(self.to_dtype[n], "," + repr(n)) for n in self.to_dtype.names
                )
                + "]"
            )

        return "{0}({1},{2})".format(type(self).__name__, from_dtype, to_dtype)

    @property
    def typename(self):
        def form(dtype):
            d, s = _dtype_shape(dtype)
            return _dtype_kind_itemsize_to_typename[d.kind, d.itemsize] + "".join(
                "[" + str(dim) + "]" for dim in s
            )

        if self.from_dtype.names is None:
            return form(self.from_dtype)
        else:
            return (
                "struct {"
                + " ".join(
                    "{0} {1};".format(form(self.from_dtype[n]), n)
                    for n in self.from_dtype.names
                )
                + "}"
            )

    def basket_array(self, data, byte_offsets, basket, branch, context, cursor_offset):
        self.hook_before_basket_array(
            data=data,
            byte_offsets=byte_offsets,
            basket=basket,
            branch=branch,
            context=context,
            cursor_offset=cursor_offset,
        )

        dtype, shape = _dtype_shape(self._from_dtype)
        try:
            output = data.view(dtype).reshape((-1,) + shape)
        except ValueError:
            raise ValueError(
                """basket {0} in tree/branch {1} has the wrong number of bytes ({2}) """
                """for interpretation {3}
in file {4}""".format(
                    basket.basket_num,
                    branch.object_path,
                    len(data),
                    self,
                    branch.file.file_path,
                )
            )

        self.hook_after_basket_array(
            data=data,
            byte_offsets=byte_offsets,
            basket=basket,
            branch=branch,
            context=context,
            output=output,
            cursor_offset=cursor_offset,
        )

        return output


class AsArray(AsDtype):
    def __init__(self):
        raise NotImplementedError


class TruncatedNumerical(Numerical):
    @property
    def low(self):
        return self._low

    @property
    def high(self):
        return self._high

    @property
    def num_bits(self):
        return self._num_bits

    @property
    def truncated(self):
        return self._low == 0.0 and self._high == 0.0

    @property
    def to_dims(self):
        return self._to_dims

    @property
    def from_dtype(self):
        if self.truncated:
            return numpy.dtype(({"exponent": (">u1", 0), "mantissa": (">u2", 1)}, ()))
        else:
            return numpy.dtype(">u4")

    def __repr__(self):
        args = [repr(self._low), repr(self._high), repr(self._num_bits)]
        if self._to_dims != ():
            args.append("to_dims={0}".format(repr(self._to_dims)))
        return "{0}({1})".format(type(self).__name__, ", ".join(args))

    def __eq__(self, other):
        return (
            type(self) == type(other)
            and self._low == other._low
            and self._high == other._high
            and self._num_bits == other._num_bits
            and self._to_dims == other._to_dims
        )

    @property
    def itemsize(self):
        return self.from_dtype.itemsize

    @property
    def numpy_dtype(self):
        return self.to_dtype

    @property
    def cache_key(self):
        return "{0}({1},{2},{3},{4})".format(
            type(self).__name__, self._low, self._high, self._num_bits, self._to_dims
        )

    def basket_array(self, data, byte_offsets, basket, branch, context, cursor_offset):
        self.hook_before_basket_array(
            data=data,
            byte_offsets=byte_offsets,
            basket=basket,
            branch=branch,
            context=context,
            cursor_offset=cursor_offset,
        )

        try:
            raw = data.view(self.from_dtype)
        except ValueError:
            raise ValueError(
                """basket {0} in tree/branch {1} has the wrong number of bytes ({2}) """
                """for interpretation {3} (expecting raw array of {4})
in file {5}""".format(
                    basket.basket_num,
                    branch.object_path,
                    len(data),
                    self,
                    repr(self._from_dtype),
                    branch.file.file_path,
                )
            )

        if self.truncated:
            exponent = raw["exponent"].astype(numpy.int32)
            mantissa = raw["mantissa"].astype(numpy.int32)

            exponent <<= 23
            exponent |= (mantissa & ((1 << (self.num_bits + 1)) - 1)) << (
                23 - self.num_bits
            )
            sign = ((1 << (self.num_bits + 1)) & mantissa != 0) * -2 + 1

            output = exponent.view(numpy.float32) * sign
            output = output.astype(self.to_dtype)

        else:
            d, s = _dtype_shape(self.to_dtype)
            output = raw.astype(d).reshape((-1,) + s)
            numpy.multiply(
                output,
                float(self._high - self._low) / (1 << self._num_bits),
                out=output,
            )
            numpy.add(output, self.low, out=output)

        self.hook_after_basket_array(
            data=data,
            byte_offsets=byte_offsets,
            basket=basket,
            branch=branch,
            context=context,
            cursor_offset=cursor_offset,
            raw=raw,
            output=output,
        )

        return output


class AsDouble32(TruncatedNumerical):
    def __init__(self, low, high, num_bits, to_dims=()):
        self._low = low
        self._high = high
        self._num_bits = num_bits
        self._to_dims = to_dims

        if not uproot4._util.isint(num_bits) or not 2 <= num_bits <= 32:
            raise TypeError("num_bits must be an integer between 2 and 32 (inclusive)")
        if high <= low and not self.truncated:
            raise ValueError(
                "high ({0}) must be strictly greater than low ({1})".format(high, low)
            )

    @property
    def to_dtype(self):
        return numpy.dtype((numpy.float64, self.to_dims))

    @property
    def typename(self):
        return "Double32_t" + "".join("[" + str(dim) + "]" for dim in self._to_dims)

    def awkward_form(self, file, index_format="i64", header=False, tobject_header=True):
        import awkward1

        out = awkward1.forms.NumpyForm(
            (),
            8,
            "d",
            parameters={
                "uproot": {
                    "as": "Double32",
                    "low": self._low,
                    "high": self._high,
                    "num_bits": self._num_bits,
                }
            },
        )
        for size in self._to_dims[::-1]:
            out = awkward1.forms.RegularForm(out, size)
        return out


class AsFloat16(TruncatedNumerical):
    def __init__(self, low, high, num_bits, to_dims=()):
        self._low = low
        self._high = high
        self._num_bits = num_bits
        self._to_dims = to_dims

        if not uproot4._util.isint(num_bits) or not 2 <= num_bits <= 16:
            raise TypeError("num_bits must be an integer between 2 and 16 (inclusive)")
        if high <= low and not self.truncated:
            raise ValueError(
                "high ({0}) must be strictly greater than low ({1})".format(high, low)
            )

    @property
    def to_dtype(self):
        return numpy.dtype((numpy.float32, self.to_dims))

    @property
    def typename(self):
        return "Float16_t" + "".join("[" + str(dim) + "]" for dim in self._to_dims)

    def awkward_form(self, file, index_format="i64", header=False, tobject_header=True):
        import awkward1

        out = awkward1.forms.NumpyForm(
            (),
            4,
            "f",
            parameters={
                "uproot": {
                    "as": "Float16",
                    "low": self._low,
                    "high": self._high,
                    "num_bits": self._num_bits,
                }
            },
        )
        for size in self._to_dims[::-1]:
            out = awkward1.forms.RegularForm(out, size)
        return out


class AsSTLBits(Numerical):
    def __init__(self):
        raise NotImplementedError

    @property
    def itemsize(self):
        return self._num_bytes + 4

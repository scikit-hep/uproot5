# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import uproot4.interpret

import numpy


def _dtype_shape(dtype):
    shape = ()
    while dtype.subdtype is not None:
        dtype, s = dtype.subdtype
        shape = shape + s
    return dtype, shape


class Numerical(uproot4.interpret.Interpretation):
    def empty_array(self, library):
        return library.wrap_numpy(numpy.empty(0, dtype=self.numpy_dtype))

    def fillable_array(self, num_items, num_entries):
        assert num_items == num_entries
        dtype, shape = _dtype_shape(self.to_dtype)
        quotient, remainder = divmod(num_items, numpy.prod(shape))
        if remainder != 0:
            raise ValueError(
                "cannot reshape {0} items into dimensions {1}".format(num_items, shape)
            )
        return numpy.empty(quotient, dtype=self.to_dtype)

    def fill(self, basket_array, fillable_array, item_start, item_stop, entry_start, entry_stop):
        assert item_start == entry_start and item_stop == entry_stop
        fillable_array.reshape(-1)[item_start:item_stop] = basket_array.reshape(-1)

    def trim(self, fillable_array, entry_start, entry_stop):
        return fillable_array[entry_start:entry_stop]

    def finalize(self, fillable_array, library):
        return library.wrap_numpy(fillable_array)


class AsDtype(Numerical):
    def __init__(self, from_dtype, to_dtype=None):
        self._from_dtype = numpy.dtype(from_dtype)
        if to_dtype is None:
            self._to_dtype = self._from_dtype.newbyteorder("=")
        else:
            self._to_dtype = numpy.dtype(to_dtype)

    @property
    def from_dtype(self):
        return self._from_dtype

    @property
    def to_dtype(self):
        return self._to_dtype

    _numpy_byteorder_to_cache_key = {
        "!": "B",
        ">": "B",
        "<": "L",
        "|": "L",
        "=": "B" if numpy.dtype(">f8").isnative else "L",
    }

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

        if self._from_dtype.names is None:
            from_dtype = form(self._from_dtype, "")
        else:
            from_dtype = (
                "["
                + ",".join(
                    form(self._from_dtype[n], "," + repr(n))
                    for n in self._from_dtype.names
                )
                + "]"
            )

        if self._to_dtype.names is None:
            to_dtype = form(self._to_dtype, "")
        else:
            to_dtype = (
                "["
                + ",".join(
                    form(self._to_dtype[n], "," + repr(n)) for n in self._to_dtype.names
                )
                + "]"
            )

        return "AsDtype({0},{1})".format(from_dtype, to_dtype)

    @property
    def numpy_dtype(self):
        return self._to_dtype

    @property
    def awkward_form(self):
        raise NotImplementedError

    def num_items(self, num_bytes, num_entries):
        dtype, shape = _dtype_shape(self._from_dtype)
        quotient, remainder = divmod(num_bytes, dtype.itemsize)
        assert remainder == 0
        return quotient

    def basket_array(self, data, byte_offsets):
        assert byte_offsets is None
        dtype, shape = _dtype_shape(self._from_dtype)
        return data.view(self._from_dtype).reshape((-1,) + shape)


class AsArray(NumpyDtype):
    def __init__(self):
        raise NotImplementedError


class AsDouble32(Numerical):
    def __init__(self):
        raise NotImplementedError


class AsFloat16(Numerical):
    def __init__(self):
        raise NotImplementedError


class AsSTLBits(Numerical):
    def __init__(self):
        raise NotImplementedError

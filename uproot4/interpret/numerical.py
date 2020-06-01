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
    @property
    def to_dtype(self):
        return self._to_dtype

    @property
    def numpy_dtype(self):
        return self._to_dtype

    @property
    def awkward_form(self):
        raise NotImplementedError

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

        if not entry_start < entry_stop:
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
                basket_array = basket_arrays[basket_num]
                if start <= entry_start and entry_stop <= stop:
                    local_start = entry_start - start
                    local_stop = entry_stop - start
                    output[:] = basket_array[local_start:local_stop]
                elif start <= entry_start < stop:
                    local_start = entry_start - start
                    local_stop = stop - start
                    output[: stop - entry_start] = basket_array[local_start:local_stop]
                elif start <= entry_stop <= stop:
                    local_start = 0
                    local_stop = entry_stop - start
                    output[start - entry_start :] = basket_array[local_start:local_stop]
                elif entry_start < stop and start <= entry_stop:
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

        output = library.finalize(output, branch)

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


class AsDtype(Numerical):
    def __init__(self, from_dtype, to_dtype=None):
        self._from_dtype = numpy.dtype(from_dtype)
        if to_dtype is None:
            self._to_dtype = self._from_dtype.newbyteorder("=")
        else:
            self._to_dtype = numpy.dtype(to_dtype)

    def __repr__(self):
        return "AsDtype({0}, {1})".format(repr(self._from_dtype), repr(self._to_dtype))

    @property
    def from_dtype(self):
        return self._from_dtype

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
                self._numpy_byteorder_to_cache_key[d.byteorder],
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

        return "{0}({1},{2})".format(type(self).__name__, from_dtype, to_dtype)

    def basket_array(self, basket, branch):
        self.hook_before_basket_array(basket=basket, branch=branch)

        assert basket.byte_offsets is None
        dtype, shape = _dtype_shape(self._from_dtype)
        try:
            output = basket.data.view(self._from_dtype).reshape((-1,) + shape)
        except ValueError:
            raise ValueError(
                """basket {0} in branch {1} has the wrong number of bytes ({2}) """
                """for interpretation {3}
in file {4}""".format(
                    basket.basket_num,
                    repr(branch.name),
                    len(basket.data),
                    self,
                    branch.file.file_path,
                )
            )

        self.hook_before_basket_array(basket=basket, branch=branch, output=output)

        return output


class AsArray(AsDtype):
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
